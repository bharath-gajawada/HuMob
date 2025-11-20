#!/usr/bin/env python3
"""
Optimized SVR/LightGBM/XGBoost Mobility Prediction Script

Key Optimizations:
1. User-level train-test split (instead of row-wise)
2. Memory-efficient processing with chunked operations
3. Reduced memory footprint through strategic data deletion
4. Batch processing for feature extraction

Goal: Predict user (x, y) coordinates by training a separate model for each individual user,
optimizing for large datasets and efficient execution.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import gc
import sys
from contextlib import contextmanager
import warnings

# Import LightGBM and XGBoost
import lightgbm as lgb
import xgboost as xgb

# ==============================================================================
# User-Configurable Parameters
# ==============================================================================

# 1. Input Data Configuration
DATA_DIR = "/kaggle/input/humob-data/15313913"
CITIES = ["B", "C"]

# 2. Preprocessing Parameters
HOLIDAYS = {}
INTERPOLATION_MAX_GAP_HOURS = 5

# 3. Model Training and Evaluation Parameters
TEST_SIZE_FRACTION = 0.1
MODEL_TYPE = 'LightGBM'

# SVR Model Hyperparameters
SVR_C = 100
SVR_GAMMA = 0.1

# LightGBM Model Hyperparameters
LGBM_N_ESTIMATORS = 100
LGBM_LEARNING_RATE = 0.1
LGBM_MAX_DEPTH = 7
LGBM_VERBOSE = -1

# XGBoost Model Hyperparameters
XGB_N_ESTIMATORS = 100
XGB_LEARNING_RATE = 0.1
XGB_MAX_DEPTH = 7
XGB_N_JOBS = -1

# 4. Data Sampling Parameters
SAMPLE_FRACTION_PER_CITY = 1

# 5. Memory Optimization Parameters
CHUNK_SIZE = 1000  # Process users in chunks to reduce memory usage
ENABLE_MEMORY_MONITORING = True

# ==============================================================================
# Helper Functions for Data Loading and File Selection
# ==============================================================================

def get_input_files(data_dir, cities):
    """
    Constructs a list of input CSV file paths based on the data directory and cities.
    """
    input_files = []
    for city in cities:
        file_name = f"city_{city}_challengedata.csv" if city else "data.csv"
        file_path = os.path.join(data_dir, file_name)
        if os.path.exists(file_path):
            input_files.append(file_path)
        else:
            print(f"Warning: File not found: {file_path}. Skipping.")
    return input_files

def load_data(path):
    """
    Loads mobility data from a CSV file with memory optimization.
    """
    # Use efficient data types to reduce memory usage
    dtype_dict = {
        'uid': 'int32',
        'd': 'int16', 
        't': 'int8',
        'x': 'float32',
        'y': 'float32'
    }
    
    df = pd.read_csv(path, dtype=dtype_dict)
    df = df.sort_values(['uid', 'd', 't'])
    return df

def monitor_memory():
    """Monitor memory usage if enabled."""
    if ENABLE_MEMORY_MONITORING:
        import psutil
        memory_percent = psutil.virtual_memory().percent
        print(f"  Memory usage: {memory_percent:.1f}%")

# ==============================================================================
# Optimized Feature Extraction
# ==============================================================================

class OptimizedFeatureExtractor:
    """
    Memory-optimized feature extractor with chunked processing.
    """
    def __init__(self, holidays, interpolation_max_gap_hours):
        self.holidays = holidays
        self.interpolation_max_gap_slots = int(interpolation_max_gap_hours * 2)

    def _to_datetime(self, day, t):
        """Converts day and time slot to a datetime object."""
        minutes_offset = int(t - 1) * 30
        if day <= 60:
            base_date = datetime(2025, 1, 1)
            return base_date + timedelta(days=int(day)-1, minutes=minutes_offset)
        else:
            base_date_after_gap = datetime(2025, 5, 1)
            day_offset_after_gap = int(day) - 61
            return base_date_after_gap + timedelta(days=day_offset_after_gap, minutes=minutes_offset)

    def _interpolate_missing_data(self, group_df):
        """Applies linear interpolation with memory optimization."""
        min_day = group_df['d'].min()
        max_day = group_df['d'].max()
        
        # Create time slots more efficiently
        days_range = np.arange(min_day, max_day + 1, dtype='int16')
        time_slots = np.arange(1, 49, dtype='int8')
        
        # Use memory-efficient approach for creating full time grid
        full_df = pd.DataFrame({
            'd': np.repeat(days_range, len(time_slots)),
            't': np.tile(time_slots, len(days_range))
        })
        
        merged_df = pd.merge(full_df, group_df[['d', 't', 'x', 'y']], on=['d', 't'], how='left')
        merged_df['uid'] = group_df['uid'].iloc[0]
        merged_df = merged_df.sort_values(['d', 't'])
        
        # Interpolate with limit
        merged_df['x'] = merged_df['x'].interpolate(
            method='linear', 
            limit=self.interpolation_max_gap_slots, 
            limit_direction='both'
        )
        merged_df['y'] = merged_df['y'].interpolate(
            method='linear', 
            limit=self.interpolation_max_gap_slots, 
            limit_direction='both'
        )
        
        merged_df.dropna(subset=['x', 'y'], inplace=True)
        return merged_df

    def extract_user_features(self, user_data):
        """Extract features for a single user with memory optimization."""
        interpolated_data = self._interpolate_missing_data(user_data.copy())
        if interpolated_data.empty:
            return pd.DataFrame()

        # Time-based features (vectorized operations)
        datetimes = interpolated_data.apply(
            lambda r: self._to_datetime(r['d'], r['t']), axis=1
        )
        
        interpolated_data['activity_time'] = (
            datetimes - datetimes.iloc[0]
        ).dt.total_seconds() / 60
        interpolated_data['day_of_week'] = datetimes.dt.weekday
        interpolated_data['is_holiday'] = interpolated_data['d'].isin(self.holidays).astype('int8')
        interpolated_data['is_weekday'] = (interpolated_data['day_of_week'] < 5).astype('int8')
        interpolated_data['am_pm'] = (datetimes.dt.hour >= 12).astype('int8')

        # Mobility-related features (optimized)
        coords = interpolated_data[['x', 'y']].values
        if len(coords) > 1:
            moves = np.diff(coords, axis=0)
            travel_distances = np.linalg.norm(moves, axis=1)
            travel_angles = np.degrees(np.arctan2(moves[:,1], moves[:,0]))
            travel_speeds = travel_distances / 30  # 30-minute intervals
            
            # Aggregate statistics
            num_moves = len(moves)
            avg_dist = travel_distances.mean() if num_moves > 0 else 0.0
            std_dist = travel_distances.std() if num_moves > 0 else 0.0
            avg_angle = travel_angles.mean() if num_moves > 0 else 0.0
            avg_speed = travel_speeds.mean() if num_moves > 0 else 0.0
            std_speed = travel_speeds.std() if num_moves > 0 else 0.0
        else:
            num_moves = avg_dist = std_dist = avg_angle = avg_speed = std_speed = 0.0

        # Assign mobility features efficiently
        interpolated_data = interpolated_data.assign(
            num_moves=num_moves,
            avg_travel_dist=avg_dist,
            std_travel_dist=std_dist,
            avg_travel_angle=avg_angle,
            avg_speed=avg_speed,
            std_speed=std_speed
        )

        return interpolated_data

    def extract(self, df):
        """
        Extract features with chunked processing for memory optimization.
        """
        features = []
        unique_uids = df['uid'].unique()
        
        # Process users in chunks to manage memory
        for i in range(0, len(unique_uids), CHUNK_SIZE):
            chunk_uids = unique_uids[i:i+CHUNK_SIZE]
            print(f"  Processing user chunk {i//CHUNK_SIZE + 1}/{(len(unique_uids)-1)//CHUNK_SIZE + 1}")
            
            chunk_features = []
            for uid in chunk_uids:
                user_data = df[df['uid'] == uid]
                user_features = self.extract_user_features(user_data)
                if not user_features.empty:
                    chunk_features.append(user_features)
            
            if chunk_features:
                features.extend(chunk_features)
            
            # Memory cleanup after each chunk
            del chunk_features
            gc.collect()
            
            if ENABLE_MEMORY_MONITORING and i % (CHUNK_SIZE * 5) == 0:
                monitor_memory()

        return pd.concat(features, ignore_index=True) if features else pd.DataFrame()

# ==============================================================================
# Optimized Model Training with User-Level Split
# ==============================================================================

@contextmanager
def suppress_stdout_stderr():
    """A context manager to suppress both stdout and stderr."""
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = devnull
        sys.stderr = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr

def user_level_train_test_split(features_df, test_size_fraction):
    """
    Perform user-level train-test split instead of row-wise split.
    This ensures temporal continuity and proper evaluation.
    """
    # Get unique users in prediction period (d >= 61)
    prediction_users = features_df[features_df['d'] >= 61]['uid'].unique()
    
    if len(prediction_users) < 2:
        # If too few users, fall back to temporal split for each user
        return temporal_split_per_user(features_df, test_size_fraction)
    
    # Split users into train and test sets
    train_users, test_users = train_test_split(
        prediction_users, 
        test_size=test_size_fraction, 
        random_state=42
    )
    
    return train_users, test_users

def temporal_split_per_user(features_df, test_size_fraction):
    """
    Fallback: temporal split for each user individually.
    Uses the latest portion of prediction period for testing.
    """
    train_data = []
    test_data = []
    
    for uid in features_df['uid'].unique():
        user_data = features_df[features_df['uid'] == uid]
        pred_period = user_data[user_data['d'] >= 61].sort_values(['d', 't'])
        
        if len(pred_period) < 2:
            continue
            
        # Use temporal split - latest data for testing
        split_idx = int(len(pred_period) * (1 - test_size_fraction))
        train_portion = pred_period.iloc[:split_idx]
        test_portion = pred_period.iloc[split_idx:]
        
        if len(train_portion) > 0 and len(test_portion) > 0:
            train_data.append((uid, train_portion))
            test_data.append((uid, test_portion))
    
    return train_data, test_data

def train_and_predict_model_optimized(features_df, model_type, test_size_fraction, 
                                    svr_c, svr_gamma, lgbm_n_estimators, lgbm_learning_rate, 
                                    lgbm_max_depth, lgbm_verbose, xgb_n_estimators, 
                                    xgb_learning_rate, xgb_max_depth, xgb_n_jobs):
    """
    Optimized training with user-level split and memory management.
    """
    all_actual_data = []
    all_predicted_data = []
    
    unique_uids = features_df['uid'].unique()
    print(f"  Training models for {len(unique_uids)} unique users...")
    
    # Check if we have enough users for user-level split
    prediction_users = features_df[features_df['d'] >= 61]['uid'].unique()
    
    if len(prediction_users) >= 10:  # User-level split
        train_users, test_users = user_level_train_test_split(features_df, test_size_fraction)
        print(f"  Using user-level split: {len(train_users)} train users, {len(test_users)} test users")
        
        # Process test users only (they have both train and test data)
        for uid in test_users:
            user_features_df = features_df[features_df['uid'] == uid].copy()
            
            # Historical data + training portion of prediction period
            historical_data = user_features_df[user_features_df['d'] <= 60]
            pred_period_data = user_features_df[user_features_df['d'] >= 61]
            
            if pred_period_data.empty or len(pred_period_data) < 2:
                continue
                
            # Use temporal split within user's prediction period for test data
            split_idx = int(len(pred_period_data) * 0.8)  # Use 80% for training, 20% for testing
            train_pred_portion = pred_period_data.iloc[:split_idx]
            test_pred_portion = pred_period_data.iloc[split_idx:]
            
            if test_pred_portion.empty:
                continue
                
            # Train on historical + train portion of prediction period
            train_data = pd.concat([historical_data, train_pred_portion], ignore_index=True)
            
            # Train and predict for this user
            actual, predicted = train_user_model(
                train_data, test_pred_portion, model_type, svr_c, svr_gamma,
                lgbm_n_estimators, lgbm_learning_rate, lgbm_max_depth, lgbm_verbose,
                xgb_n_estimators, xgb_learning_rate, xgb_max_depth, xgb_n_jobs
            )
            
            if actual is not None and predicted is not None:
                all_actual_data.append(actual)
                all_predicted_data.append(predicted)
                
    else:  # Temporal split per user (fallback)
        print("  Using temporal split per user (insufficient users for user-level split)")
        
        for uid in unique_uids:
            user_features_df = features_df[features_df['uid'] == uid].copy()
            
            historical_data = user_features_df[user_features_df['d'] <= 60]
            prediction_period_data = user_features_df[user_features_df['d'] >= 61]
            
            if prediction_period_data.empty or len(prediction_period_data) < 2:
                continue
                
            # Temporal split within prediction period
            prediction_period_data = prediction_period_data.sort_values(['d', 't'])
            split_idx = int(len(prediction_period_data) * (1 - test_size_fraction))
            
            train_pred_period = prediction_period_data.iloc[:split_idx]
            test_pred_period = prediction_period_data.iloc[split_idx:]
            
            if test_pred_period.empty:
                continue
                
            # Combine historical with training portion
            train_data = pd.concat([historical_data, train_pred_period], ignore_index=True)
            
            # Train and predict
            actual, predicted = train_user_model(
                train_data, test_pred_period, model_type, svr_c, svr_gamma,
                lgbm_n_estimators, lgbm_learning_rate, lgbm_max_depth, lgbm_verbose,
                xgb_n_estimators, xgb_learning_rate, xgb_max_depth, xgb_n_jobs
            )
            
            if actual is not None and predicted is not None:
                all_actual_data.append(actual)
                all_predicted_data.append(predicted)
    
    # Combine results
    final_actual_data = pd.concat(all_actual_data, ignore_index=True) if all_actual_data else pd.DataFrame()
    final_predicted_data = pd.concat(all_predicted_data, ignore_index=True) if all_predicted_data else pd.DataFrame()
    
    return final_actual_data, final_predicted_data

def train_user_model(train_data, test_data, model_type, svr_c, svr_gamma,
                    lgbm_n_estimators, lgbm_learning_rate, lgbm_max_depth, lgbm_verbose,
                    xgb_n_estimators, xgb_learning_rate, xgb_max_depth, xgb_n_jobs):
    """
    Train model for a single user with memory optimization.
    """
    try:
        # Define features
        feature_columns = [col for col in train_data.columns if col not in ['uid', 'd', 't', 'x', 'y']]
        
        X_train = train_data[feature_columns].values.astype('float32')
        y_x_train = train_data['x'].values.astype('float32')
        y_y_train = train_data['y'].values.astype('float32')
        
        X_test = test_data[feature_columns].values.astype('float32')
        y_x_test = test_data['x'].values.astype('float32')
        y_y_test = test_data['y'].values.astype('float32')
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Initialize models
        if model_type == 'SVR':
            model_x = SVR(kernel='rbf', C=svr_c, gamma=svr_gamma)
            model_y = SVR(kernel='rbf', C=svr_c, gamma=svr_gamma)
        elif model_type == 'LightGBM':
            model_x = lgb.LGBMRegressor(
                n_estimators=lgbm_n_estimators, 
                learning_rate=lgbm_learning_rate, 
                max_depth=lgbm_max_depth, 
                random_state=42, 
                verbose=lgbm_verbose
            )
            model_y = lgb.LGBMRegressor(
                n_estimators=lgbm_n_estimators, 
                learning_rate=lgbm_learning_rate, 
                max_depth=lgbm_max_depth, 
                random_state=42, 
                verbose=lgbm_verbose
            )
        elif model_type == 'XGBoost':
            model_x = xgb.XGBRegressor(
                n_estimators=xgb_n_estimators, 
                learning_rate=xgb_learning_rate, 
                max_depth=xgb_max_depth, 
                n_jobs=xgb_n_jobs, 
                random_state=42
            )
            model_y = xgb.XGBRegressor(
                n_estimators=xgb_n_estimators, 
                learning_rate=xgb_learning_rate, 
                max_depth=xgb_max_depth, 
                n_jobs=xgb_n_jobs, 
                random_state=42
            )
        else:
            raise ValueError(f"Unknown MODEL_TYPE: {model_type}")
        
        # Train models
        with suppress_stdout_stderr():
            model_x.fit(X_train_scaled, y_x_train)
            model_y.fit(X_train_scaled, y_y_train)
        
        # Predict
        y_x_pred = model_x.predict(X_test_scaled)
        y_y_pred = model_y.predict(X_test_scaled)
        
        # Prepare results
        actual_data = pd.DataFrame({
            'uid': test_data['uid'].values,
            'd': test_data['d'].values,
            't': test_data['t'].values,
            'x_actual': y_x_test,
            'y_actual': y_y_test
        })
        
        predicted_data = pd.DataFrame({
            'uid': test_data['uid'].values,
            'd': test_data['d'].values,
            't': test_data['t'].values,
            'x_predicted': y_x_pred,
            'y_predicted': y_y_pred
        })
        
        return actual_data, predicted_data
        
    except Exception as e:
        print(f"    Error training model for user: {e}")
        return None, None

# ==============================================================================
# Main Execution Function
# ==============================================================================

def main():
    """
    Main function with optimizations for memory usage and user-level splits.
    """
    warnings.filterwarnings('ignore', category=UserWarning, module='lightgbm')
    
    try:
        from geobleu import calc_geobleu_single, calc_dtw_single
    except ImportError as e:
        print(f"Error: Failed to import 'geobleu' functions. {e}")
        print("Please ensure the 'geobleu' library is installed correctly.")
        return
    
    input_csv_files = get_input_files(DATA_DIR, CITIES)
    
    if not input_csv_files:
        print("No input CSV files found. Exiting.")
        return
    
    extractor = OptimizedFeatureExtractor(
        holidays=HOLIDAYS, 
        interpolation_max_gap_hours=INTERPOLATION_MAX_GAP_HOURS
    )
    
    all_geobleu_scores = []
    all_dtw_scores = []
    all_predicted_data_overall = []
    
    for input_csv in input_csv_files:
        print(f"\n--- Processing data from: {input_csv} ---")
        monitor_memory()
        
        try:
            df = load_data(input_csv)
            
            # Apply sampling if configured
            if SAMPLE_FRACTION_PER_CITY < 1.0 and not df.empty:
                original_rows = len(df)
                unique_uids_to_sample = df['uid'].unique()
                sampled_uids = np.random.choice(
                    unique_uids_to_sample, 
                    size=int(len(unique_uids_to_sample) * SAMPLE_FRACTION_PER_CITY), 
                    replace=False
                )
                df = df[df['uid'].isin(sampled_uids)].reset_index(drop=True)
                print(f"  Sampled {len(df)} rows ({SAMPLE_FRACTION_PER_CITY*100:.1f}%) from original {original_rows} rows by UID.")
            
            gc.collect()
            monitor_memory()
            
            print("  Extracting features...")
            features_df = extractor.extract(df)
            
            del df
            gc.collect()
            monitor_memory()
            
            if features_df.empty:
                print(f"No features extracted for {input_csv}. Skipping.")
                continue
            
            print(f"--- Starting {MODEL_TYPE} Model Training and Prediction ---")
            actual_test_data_city, predicted_test_data_city = train_and_predict_model_optimized(
                features_df, MODEL_TYPE, TEST_SIZE_FRACTION, SVR_C, SVR_GAMMA,
                LGBM_N_ESTIMATORS, LGBM_LEARNING_RATE, LGBM_MAX_DEPTH, LGBM_VERBOSE,
                XGB_N_ESTIMATORS, XGB_LEARNING_RATE, XGB_MAX_DEPTH, XGB_N_JOBS
            )
            
            del features_df
            gc.collect()
            monitor_memory()
            
            if not actual_test_data_city.empty and not predicted_test_data_city.empty:
                # Evaluation
                true_trajectories_list = []
                pred_trajectories_list = []
                
                merged_test_data = pd.merge(
                    actual_test_data_city,
                    predicted_test_data_city,
                    on=['uid', 'd', 't'],
                    how='inner'
                ).sort_values(['uid', 'd', 't']).reset_index(drop=True)
                
                unique_uids_in_test = merged_test_data['uid'].unique()
                
                for uid in unique_uids_in_test:
                    user_data = merged_test_data[merged_test_data['uid'] == uid]
                    
                    true_traj = [(int(row['uid']), int(row['d']), int(row['t']), 
                                int(row['x_actual']), int(row['y_actual']))
                               for _, row in user_data.iterrows()]
                    pred_traj = [(int(row['uid']), int(row['d']), int(row['t']), 
                                int(row['x_predicted']), int(row['y_predicted']))
                               for _, row in user_data.iterrows()]
                    
                    if true_traj and pred_traj:
                        true_trajectories_list.append(true_traj)
                        pred_trajectories_list.append(pred_traj)
                
                if true_trajectories_list and pred_trajectories_list:
                    current_geobleu_scores = []
                    current_dtw_scores = []
                    
                    for i in range(len(pred_trajectories_list)):
                        pred_traj_single = pred_trajectories_list[i]
                        true_traj_single = true_trajectories_list[i]
                        
                        if pred_traj_single and true_traj_single:
                            current_geobleu_scores.append(calc_geobleu_single(pred_traj_single, true_traj_single))
                            current_dtw_scores.append(calc_dtw_single(pred_traj_single, true_traj_single))
                    
                    if current_geobleu_scores:
                        all_geobleu_scores.append(np.mean(current_geobleu_scores))
                        print(f"  GEO-BLEU Score: {all_geobleu_scores[-1]:.4f}")
                    
                    if current_dtw_scores:
                        all_dtw_scores.append(np.mean(current_dtw_scores))
                        print(f"  DTW Score: {all_dtw_scores[-1]:.4f}")
                    
                    all_predicted_data_overall.append(predicted_test_data_city)
                else:
                    print(f"No valid trajectories for metrics calculation.")
            else:
                print(f"Model training resulted in empty data. Skipping evaluation.")
                
        except Exception as e:
            print(f"Error processing {input_csv}: {e}")
    
    # Final results
    print("\n" + "="*50)
    print("Overall Evaluation Results")
    print("="*50)
    
    if all_geobleu_scores:
        overall_avg_geobleu = np.mean(all_geobleu_scores)
        print(f"Overall Average GEO-BLEU Score: {overall_avg_geobleu:.4f}")
    else:
        print("No GEO-BLEU scores calculated.")
    
    if all_dtw_scores:
        overall_avg_dtw = np.mean(all_dtw_scores)
        print(f"Overall Average DTW Score: {overall_avg_dtw:.4f}")
    else:
        print("No DTW scores calculated.")
    
    # Save predictions
    if all_predicted_data_overall:
        final_predicted_df = pd.concat(all_predicted_data_overall, ignore_index=True)
        
        if len(CITIES) == 1 and CITIES[0] != "":
            output_name = f"predicted_mobility_{MODEL_TYPE}_{CITIES[0]}_optimized_sampled{int(SAMPLE_FRACTION_PER_CITY*100)}.csv"
        elif len(CITIES) == 1 and CITIES[0] == "":
            output_name = f"predicted_mobility_{MODEL_TYPE}_optimized_sampled{int(SAMPLE_FRACTION_PER_CITY*100)}.csv"
        else:
            output_name = f"combined_predicted_mobility_{MODEL_TYPE}_optimized_sampled{int(SAMPLE_FRACTION_PER_CITY*100)}.csv"
        
        output_path = os.path.join("/kaggle/working", output_name)
        final_predicted_df.to_csv(output_path, index=False)
        print(f"\nPredicted data saved to {output_path}")
    else:
        print("\nNo predicted data generated.")
    
    print("\nProcessing complete.")

if __name__ == "__main__":
    main()
