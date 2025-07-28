import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import gc # Import garbage collection module
import sys # Import sys module for stdout redirection
from contextlib import contextmanager # Import contextmanager for a cleaner way to redirect stdout
import warnings # Import warnings module

# Import LightGBM and XGBoost
import lightgbm as lgb
import xgboost as xgb

# IMPORTANT: To run this script in a Kaggle Notebook, ensure 'geobleu' is installed.
#
# You typically run installation commands in a separate code cell at the beginning of your notebook:
# !pip install pandas numpy scikit-learn lightgbm xgboost
# !pip install git+https://github.com/yahoojapan/geobleu.git
#
# If you encounter "AttributeError: module 'geobleu' has no attribute 'geobleu'"
# or "ModuleNotFoundError", please ensure the library is installed correctly.

# ==============================================================================
# User-Configurable Parameters
# ==============================================================================

# 1. Input Data Configuration
# Path to the directory containing your mobility data CSV files within Kaggle.
# This path is common for competition datasets.
DATA_DIR = "../data" # <--- USER: Set your data directory path here

# List of cities to process. Files are expected to be named like 'city_B_challengedata.csv', etc.
# If your files are just 'data.csv', set CITIES = [""] and ensure DATA_DIR points directly to 'data.csv'
CITIES = ["F"] # <--- USER: Specify cities to process (e.g., ["B"], ["B", "C"])
# For very large files, consider processing one city at a time, e.g., CITIES = ["D"]

# 2. Preprocessing Parameters
# A set of day numbers (d) that are considered holidays.
HOLIDAYS = {70, 71} # <--- USER: Adjust holiday day numbers as needed

# The maximum gap in hours for which linear interpolation will be applied.
# Gaps larger than this will result in NaNs. (20 30-min time slots = 10 hours)
INTERPOLATION_MAX_GAP_HOURS = 10 # <--- USER: Adjust interpolation gap

# 3. Model Training and Evaluation Parameters
# Fraction of day 61-75 data to use for testing/validation.
TEST_SIZE_FRACTION = 0.5 # <--- USER: Adjust train/test split for prediction period

# Choose the model type: 'SVR', 'LightGBM', 'XGBoost'
MODEL_TYPE = 'LightGBM' # <--- USER: Select your desired model (e.g., 'SVR', 'LightGBM', 'XGBoost')

# SVR Model Hyperparameters (only used if MODEL_TYPE is 'SVR')
SVR_C = 100    # Regularization parameter
SVR_GAMMA = 0.1 # Kernel coefficient
# <--- USER: Tune SVR_C and SVR_GAMMA for SVR

# LightGBM Model Hyperparameters (only used if MODEL_TYPE is 'LightGBM')
LGBM_N_ESTIMATORS = 100
LGBM_LEARNING_RATE = 0.1
LGBM_MAX_DEPTH = 7
# To suppress "No further splits with positive gain" warnings, set verbose=-1.
# These warnings are common when training on small/homogeneous per-user datasets.
# Note: For complete suppression, we will also redirect stdout/stderr during fitting.
LGBM_VERBOSE = -1 # <--- USER: Set to 0 for info, >0 for more detail, -1 to suppress warnings
# <--- USER: Tune LGBM hyperparameters

# XGBoost Model Hyperparameters (only used if MODEL_TYPE is 'XGBoost')
XGB_N_ESTIMATORS = 100
XGB_LEARNING_RATE = 0.1
XGB_MAX_DEPTH = 7
XGB_N_JOBS = -1 # Use all available CPU cores
# <--- USER: Tune XGB hyperparameters

# 4. Data Sampling Parameters
# Fraction of data to sample per city before feature extraction.
# Set to 1.0 to use all data (might cause Out-Of-Memory for large files).
# Start with a small fraction (e.g., 0.1 or 0.05) if you encounter OOM errors.
SAMPLE_FRACTION_PER_CITY = 0.1 # <--- USER: Fraction of data to sample per city (e.g., 0.1 for 10%)


# ==============================================================================
# Helper Functions for Data Loading and File Selection
# ==============================================================================

def get_input_files(data_dir, cities):
    """
    Constructs a list of input CSV file paths based on the data directory and cities.
    Assumes file naming convention like 'city_B_challengedata.csv'.
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

# ---- Step 1: Load Data ----
def load_data(path):
    """
    Loads mobility data from a CSV file, ensuring correct types and sorting.
    """
    df = pd.read_csv(path)
    df['uid'] = df['uid'].astype(int)
    df['d'] = df['d'].astype(int)
    df['t'] = df['t'].astype(int)
    df['x'] = df['x'].astype(float)
    df['y'] = df['y'].astype(float)
    df = df.sort_values(['uid', 'd', 't'])
    return df

# ---- Step 2: Feature Extraction ----
class FeatureExtractor:
    """
    Extracts mobility and time-based features, including linear interpolation for missing data.
    """
    def __init__(self, holidays, interpolation_max_gap_hours):
        self.holidays = holidays
        self.interpolation_max_gap_slots = int(interpolation_max_gap_hours * 2)

    def _to_datetime(self, day, t):
        """
        Converts day and time slot to a datetime object, accounting for the dataset's time gap.
        """
        minutes_offset = int(t - 1) * 30
        if day <= 60:
            base_date = datetime(2025, 1, 1) # Assumes day 1 is 2025-01-01
            return base_date + timedelta(days=int(day)-1, minutes=minutes_offset)
        else:
            base_date_after_gap = datetime(2025, 5, 1) # Day 61 corresponds to 2025-05-01
            day_offset_after_gap = int(day) - 61
            return base_date_after_gap + timedelta(days=day_offset_after_gap, minutes=minutes_offset)

    def _interpolate_missing_data(self, group_df):
        """Applies linear interpolation to fill missing coordinates within a user's trajectory."""
        min_day = group_df['d'].min()
        max_day = group_df['d'].max()
        full_time_slots = pd.MultiIndex.from_product(
            [range(min_day, max_day + 1), range(1, 49)], names=['d', 't']
        )
        full_df = pd.DataFrame(index=full_time_slots).reset_index()
        merged_df = pd.merge(full_df, group_df[['d', 't', 'x', 'y']], on=['d', 't'], how='left')
        merged_df['uid'] = group_df['uid'].iloc[0]
        merged_df = merged_df.sort_values(['d', 't'])
        merged_df['x'] = merged_df['x'].interpolate(method='linear', limit=self.interpolation_max_gap_slots, limit_direction='both')
        merged_df['y'] = merged_df['y'].interpolate(method='linear', limit=self.interpolation_max_gap_slots, limit_direction='both')
        merged_df.dropna(subset=['x', 'y'], inplace=True)
        return merged_df

    def extract(self, df):
        """
        Extracts features for all users in the input DataFrame, applying interpolation and feature engineering.
        """
        features = []
        df = df.sort_values(['uid', 'd', 't'])

        for uid, group in df.groupby('uid'):
            interpolated_group = self._interpolate_missing_data(group.copy())
            if interpolated_group.empty:
                print(f"Warning: No valid data for uid {uid} after interpolation. Skipping.")
                continue

            current_group = interpolated_group.reset_index(drop=True)

            # Time-based features
            current_group['datetime'] = current_group.apply(lambda r: self._to_datetime(r['d'], r['t']), axis=1)
            current_group['activity_time'] = (current_group['datetime'] - current_group['datetime'].iloc[0]).dt.total_seconds() / 60
            current_group['day_of_week'] = current_group['datetime'].dt.weekday
            current_group['is_holiday'] = current_group['d'].apply(lambda d: 1 if d in self.holidays else 0)
            current_group['is_weekday'] = current_group['day_of_week'].apply(lambda dow: 1 if dow < 5 else 0)
            current_group['am_pm'] = current_group['datetime'].dt.hour.apply(lambda h: 0 if h < 12 else 1)

            # Mobility-related features
            coords = current_group[['x', 'y']].values
            moves = np.diff(coords, axis=0)
            num_moves = len(moves)

            avg_dist = 0.0
            std_dist = 0.0
            avg_angle = 0.0
            avg_speed = 0.0
            std_speed = 0.0

            if num_moves > 0:
                travel_distances = np.linalg.norm(moves, axis=1)
                avg_dist = travel_distances.mean()
                std_dist = travel_distances.std()
                travel_angles = np.degrees(np.arctan2(moves[:,1], moves[:,0]))
                avg_angle = travel_angles.mean()
                time_interval_minutes = 30
                travel_speeds = travel_distances / time_interval_minutes
                avg_speed = travel_speeds.mean()
                std_speed = travel_speeds.std()

            current_group['num_moves'] = num_moves
            current_group['avg_travel_dist'] = avg_dist
            current_group['std_travel_dist'] = std_dist
            current_group['avg_travel_angle'] = avg_angle
            current_group['avg_speed'] = avg_speed
            current_group['std_speed'] = std_speed

            current_group = current_group.drop(columns=['datetime'])
            features.append(current_group)

        return pd.concat(features, ignore_index=True)

# ---- Step 3: Model Training and Prediction (Per User) ----
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

def train_and_predict_model(features_df, model_type, test_size_fraction, svr_c, svr_gamma,
                            lgbm_n_estimators, lgbm_learning_rate, lgbm_max_depth, lgbm_verbose,
                            xgb_n_estimators, xgb_learning_rate, xgb_max_depth, xgb_n_jobs):
    """
    Trains the specified model for x and y coordinates for each user independently and makes predictions.
    """
    all_actual_data = []
    all_predicted_data = []

    unique_uids = features_df['uid'].unique()
    print(f"  Training models for {len(unique_uids)} unique users...")

    for uid in unique_uids:
        user_features_df = features_df[features_df['uid'] == uid].copy()

        # Define features (X) and targets (y) for the current user
        feature_columns = [col for col in user_features_df.columns if col not in ['uid', 'd', 't', 'x', 'y']]
        X_user = user_features_df[feature_columns]
        y_x_user = user_features_df['x']
        y_y_user = user_features_df['y']

        historical_data_user = user_features_df[user_features_df['d'] <= 60]
        prediction_period_data_user = user_features_df[user_features_df['d'] >= 61]

        if prediction_period_data_user.empty:
            continue

        # Ensure enough samples for splitting, especially for small user trajectories
        if len(prediction_period_data_user) < 2:
            continue
        
        # Ensure enough samples for stratification if needed, though for single user, stratify is not applied
        if len(prediction_period_data_user['uid'].unique()) < 2:
            train_pred_period_user, test_pred_period_user = train_test_split(
                prediction_period_data_user, test_size=test_size_fraction, random_state=42
            )
        else: # This case is technically not hit for single user, but kept for robustness
            train_pred_period_user, test_pred_period_user = train_test_split(
                prediction_period_data_user, test_size=test_size_fraction, random_state=42, stratify=prediction_period_data_user['uid']
            )

        # Combine historical data with the training fraction of the prediction period data for this user
        X_train_combined_user = pd.concat([historical_data_user[feature_columns], train_pred_period_user[feature_columns]], ignore_index=True)
        y_x_train_combined_user = pd.concat([historical_data_user['x'], train_pred_period_user['x']], ignore_index=True)
        y_y_train_combined_user = pd.concat([historical_data_user['y'], train_pred_period_user['y']], ignore_index=True)

        X_test_user = test_pred_period_user[feature_columns]
        y_x_test_user = test_pred_period_user['x']
        y_y_test_user = test_pred_period_user['y']

        # Skip if training data is empty after split (e.g., if all data is in test_pred_period_user)
        if X_train_combined_user.empty or X_test_user.empty:
            continue

        # Scale features for the current user
        scaler_X_user = StandardScaler()
        X_train_scaled_user = scaler_X_user.fit_transform(X_train_combined_user)
        X_test_scaled_user = scaler_X_user.transform(X_test_user)

        # Initialize models based on MODEL_TYPE
        if model_type == 'SVR':
            model_x_user = SVR(kernel='rbf', C=svr_c, gamma=svr_gamma)
            model_y_user = SVR(kernel='rbf', C=svr_c, gamma=svr_gamma)
        elif model_type == 'LightGBM':
            model_x_user = lgb.LGBMRegressor(n_estimators=lgbm_n_estimators, learning_rate=lgbm_learning_rate, max_depth=lgbm_max_depth, random_state=42, verbose=lgbm_verbose)
            model_y_user = lgb.LGBMRegressor(n_estimators=lgbm_n_estimators, learning_rate=lgbm_learning_rate, max_depth=lgbm_max_depth, random_state=42, verbose=lgbm_verbose)
        elif model_type == 'XGBoost':
            model_x_user = xgb.XGBRegressor(n_estimators=xgb_n_estimators, learning_rate=xgb_learning_rate, max_depth=xgb_max_depth, n_jobs=xgb_n_jobs, random_state=42)
            model_y_user = xgb.XGBRegressor(n_estimators=xgb_n_estimators, learning_rate=xgb_learning_rate, max_depth=xgb_max_depth, n_jobs=xgb_n_jobs, random_state=42)
        else:
            raise ValueError(f"Unknown MODEL_TYPE: {model_type}. Choose 'SVR', 'LightGBM', or 'XGBoost'.")

        # Train models for the current user
        # Suppress all output during fit for cleaner console
        with suppress_stdout_stderr():
            model_x_user.fit(X_train_scaled_user, y_x_train_combined_user)
            model_y_user.fit(X_train_scaled_user, y_y_train_combined_user)

        # Predict on the test set for the current user
        y_x_pred_user = model_x_user.predict(X_test_scaled_user)
        y_y_pred_user = model_y_user.predict(X_test_scaled_user)

        # Collect actual and predicted data for the current user
        all_actual_data.append(pd.DataFrame({
            'uid': test_pred_period_user['uid'],
            'd': test_pred_period_user['d'],
            't': test_pred_period_user['t'],
            'x_actual': y_x_test_user,
            'y_actual': y_y_test_user
        }).reset_index(drop=True))

        all_predicted_data.append(pd.DataFrame({
            'uid': test_pred_period_user['uid'],
            'd': test_pred_period_user['d'],
            't': test_pred_period_user['t'],
            'x_predicted': y_x_pred_user,
            'y_predicted': y_y_pred_user
        }).reset_index(drop=True))
        
        # Explicitly delete models and data for current user to free memory
        del model_x_user, model_y_user, X_user, y_x_user, y_y_user, historical_data_user, \
            prediction_period_data_user, train_pred_period_user, test_pred_period_user, \
            X_train_combined_user, y_x_train_combined_user, y_y_train_combined_user, \
            X_test_user, y_x_test_user, y_y_test_user, scaler_X_user, X_train_scaled_user, X_test_scaled_user, \
            y_x_pred_user, y_y_pred_user
        gc.collect()


    # Concatenate all collected data from all users
    final_actual_data = pd.concat(all_actual_data, ignore_index=True) if all_actual_data else pd.DataFrame()
    final_predicted_data = pd.concat(all_predicted_data, ignore_index=True) if all_predicted_data else pd.DataFrame()

    return final_actual_data, final_predicted_data

# ---- Main Execution Function ----
def main():
    """
    Main function to orchestrate data loading, feature extraction, model training,
    prediction, and evaluation using GEO-BLEU and DTW, processing data city by city.
    """
    # Filter out specific LightGBM warnings that might still appear
    warnings.filterwarnings('ignore', category=UserWarning, module='lightgbm')
    
    # Attempt to import geobleu functions. If unsuccessful, print error and return.
    try:
        from geobleu import calc_geobleu_single, calc_dtw_single
    except ImportError as e:
        print(f"Error: Failed to import 'geobleu' functions. {e}")
        print("Please ensure the 'geobleu' library is installed correctly in your active Python environment.")
        print("Refer to the installation instructions at the top of this script.")
        return # Exit main function gracefully

    # Get input files based on user configuration
    input_csv_files = get_input_files(DATA_DIR, CITIES)

    if not input_csv_files:
        print("No input CSV files found based on the provided DATA_DIR and CITIES. Exiting.")
        return

    extractor = FeatureExtractor(holidays=HOLIDAYS, interpolation_max_gap_hours=INTERPOLATION_MAX_GAP_HOURS)

    # Lists to store scores from each city
    all_geobleu_scores = []
    all_dtw_scores = []
    all_predicted_data_overall = [] # To optionally save all predicted data from all cities

    for input_csv in input_csv_files:
        print(f"\n--- Processing data from: {input_csv} ---")
        try:
            df = load_data(input_csv)

            # Apply sampling if configured
            if SAMPLE_FRACTION_PER_CITY < 1.0 and not df.empty:
                original_rows = len(df)
                # Sample by UID to keep user trajectories intact
                unique_uids_to_sample = df['uid'].unique()
                sampled_uids = np.random.choice(unique_uids_to_sample, size=int(len(unique_uids_to_sample) * SAMPLE_FRACTION_PER_CITY), replace=False)
                df = df[df['uid'].isin(sampled_uids)].reset_index(drop=True)
                print(f"  Sampled {len(df)} rows ({SAMPLE_FRACTION_PER_CITY*100:.1f}%) from original {original_rows} rows by UID.")
            
            # Explicitly clear memory before heavy processing for current city
            gc.collect()

            features_df = extractor.extract(df)
            
            # Clear original DataFrame to free memory
            del df
            gc.collect()


            if features_df.empty:
                print(f"No features extracted for {input_csv}. Skipping model training for this file.")
                continue

            print(f"--- Starting {MODEL_TYPE} Model Training and Prediction for {os.path.basename(input_csv)} ---")
            actual_test_data_city, predicted_test_data_city = train_and_predict_model(
                features_df, MODEL_TYPE, TEST_SIZE_FRACTION, SVR_C, SVR_GAMMA,
                LGBM_N_ESTIMATORS, LGBM_LEARNING_RATE, LGBM_MAX_DEPTH, LGBM_VERBOSE, # Pass verbose here
                XGB_N_ESTIMATORS, XGB_LEARNING_RATE, XGB_MAX_DEPTH, XGB_N_JOBS
            )
            
            # Clear features_df to free memory
            del features_df
            gc.collect()


            if not actual_test_data_city.empty and not predicted_test_data_city.empty:
                # Prepare trajectories for GEO-BLEU and DTW calculation for the current city
                true_trajectories_list = []
                pred_trajectories_list = []

                # Merge actual and predicted data to ensure alignment by uid, d, t
                merged_test_data = pd.merge(
                    actual_test_data_city,
                    predicted_test_data_city,
                    on=['uid', 'd', 't'],
                    how='inner' # Use inner join to only keep common (uid, d, t) points
                ).sort_values(['uid', 'd', 't']).reset_index(drop=True)

                unique_uids_in_test = merged_test_data['uid'].unique()

                for uid in unique_uids_in_test:
                    user_data = merged_test_data[merged_test_data['uid'] == uid]

                    # Constructing trajectories as (uid, d, t, x, y) as required by geobleu library
                    # Ensure coordinates are integers for grid-based calculations.
                    true_traj = [(int(row['uid']), int(row['d']), int(row['t']), int(row['x_actual']), int(row['y_actual']))
                                 for index, row in user_data.iterrows()]
                    pred_traj = [(int(row['uid']), int(row['d']), int(row['t']), int(row['x_predicted']), int(row['y_predicted']))
                                 for index, row in user_data.iterrows()]

                    if true_traj and pred_traj: # Only add if trajectories are not empty
                        true_trajectories_list.append(true_traj)
                        pred_trajectories_list.append(pred_traj)
                
                if true_trajectories_list and pred_trajectories_list:
                    current_geobleu_scores_per_traj = []
                    current_dtw_scores_per_traj = []
                    
                    # Iterate through each paired trajectory and calculate individual scores
                    for i in range(len(pred_trajectories_list)):
                        pred_traj_single = pred_trajectories_list[i]
                        true_traj_single = true_trajectories_list[i]

                        if not pred_traj_single or not true_traj_single:
                            continue

                        current_geobleu_scores_per_traj.append(calc_geobleu_single(pred_traj_single, true_traj_single))
                        current_dtw_scores_per_traj.append(calc_dtw_single(pred_traj_single, true_traj_single))
                    
                    if current_geobleu_scores_per_traj:
                        all_geobleu_scores.append(np.mean(current_geobleu_scores_per_traj))
                        print(f"  GEO-BLEU Score for {os.path.basename(input_csv)}: {all_geobleu_scores[-1]:.4f}")
                    else:
                        print(f"  No valid GEO-BLEU scores for {os.path.basename(input_csv)}.")

                    if current_dtw_scores_per_traj:
                        all_dtw_scores.append(np.mean(current_dtw_scores_per_traj))
                        print(f"  DTW Score for {os.path.basename(input_csv)}: {all_dtw_scores[-1]:.4f}")
                    else:
                        print(f"  No valid DTW scores for {os.path.basename(input_csv)}.")

                    all_predicted_data_overall.append(predicted_test_data_city)
                else:
                    print(f"No valid trajectories for metrics calculation in {os.path.basename(input_csv)} after splitting.")

            else:
                print(f"Model training or prediction resulted in empty data for {os.path.basename(input_csv)}. Skipping evaluation.")

        except Exception as e:
            print(f"Error processing {input_csv}: {e}")

    # Final overall average scores
    print("\n==================================================")
    print("Overall Evaluation Results")
    print("==================================================")
    if all_geobleu_scores:
        overall_avg_geobleu = np.mean(all_geobleu_scores)
        print(f"Overall Average GEO-BLEU Score: {overall_avg_geobleu:.4f}")
    else:
        print("No GEO-BLEU scores calculated across all files.")

    if all_dtw_scores:
        overall_avg_dtw = np.mean(all_dtw_scores)
        print(f"Overall Average DTW Score: {overall_avg_dtw:.4f}")
    else:
        print("No DTW scores calculated across all files.")

    # Save all predicted data if any was generated
    if all_predicted_data_overall:
        final_predicted_df = pd.concat(all_predicted_data_overall, ignore_index=True)
        # Construct the output file path for Kaggle
        if len(CITIES) == 1 and CITIES[0] != "":
            predicted_output_file_name = f"predicted_mobility_{MODEL_TYPE}_{CITIES[0]}_sampled{int(SAMPLE_FRACTION_PER_CITY*100)}.csv"
        elif len(CITIES) == 1 and CITIES[0] == "": # Case for single 'data.csv'
            predicted_output_file_name = f"predicted_mobility_{MODEL_TYPE}_sampled{int(SAMPLE_FRACTION_PER_CITY*100)}.csv"
        else:
            predicted_output_file_name = f"combined_predicted_mobility_{MODEL_TYPE}_sampled{int(SAMPLE_FRACTION_PER_CITY*100)}.csv"
        
        predicted_output_path = os.path.join("/kaggle/working", predicted_output_file_name)
        final_predicted_df.to_csv(predicted_output_path, index=False)
        print(f"\nAll predicted mobility data saved to {predicted_output_path}")
    else:
        print("\nNo predicted mobility data generated across all files.")

    print("\nAll processing, training, prediction, and evaluation complete.")

# Entry point for the script
if __name__ == "__main__":
    main()
