# import pandas as pd
# import glob
# import os

# # Get all CSV files that match the pattern
# file_list = glob.glob("data/city_*_challengedata.csv")

# all_users = set()  # to track unique users across all files

# for file in file_list:
#     print(f"\nProcessing file: {os.path.basename(file)}")
#     df = pd.read_csv(file, compression='gzip')

#     # Method 1: Total number of values (all rows × columns)
#     num_data_points = df.size

#     # Method 2: Number of rows (observations)
#     num_rows = len(df)

#     # Method 3: Number of columns
#     num_columns = df.shape[1]

#     # Method 4: Number of unique users (if 'uid' exists)
#     if 'uid' in df.columns:
#         num_users = df['uid'].nunique()
#         all_users.update(df['uid'].unique())  # add to global set
#     else:
#         num_users = "Column 'uid' not found"

#     print("  Total data points (cells):", num_data_points)
#     print("  Number of rows:", num_rows)
#     print("  Number of columns:", num_columns)
#     print("  Number of unique users (uid):", num_users)

# # Optional: overall unique users across all files
# if all_users:
#     print(f"\nTotal unique users across all files: {len(all_users)}")

# Processing file: city_A_challengedata.csv
#   Total data points (cells): 435212090
#   Number of rows: 87042418
#   Number of columns: 5
#   Number of unique users (uid): 150000

# Processing file: city_B_challengedata.csv
#   Total data points (cells): 90984965
#   Number of rows: 18196993
#   Number of columns: 5
#   Number of unique users (uid): 30000

# Processing file: city_C_challengedata.csv
#   Total data points (cells): 72375735
#   Number of rows: 14475147
#   Number of columns: 5
#   Number of unique users (uid): 25000

# Processing file: city_D_challengedata.csv
#   Total data points (cells): 60150390
#   Number of rows: 12030078
#   Number of columns: 5
#   Number of unique users (uid): 20000


import pandas as pd
import glob
import os

# Get all CSV files that match the pattern
file_list = glob.glob("data/city_*_challengedata.csv")

all_user_stats = []  # to store per-user stats across all files

for file in file_list:
    print(f"\nProcessing file: {os.path.basename(file)}")
    df = pd.read_csv(file, compression='gzip')

    # ----------------------------
    # Base stats: counts per user
    # ----------------------------
    user_counts = df.groupby("uid").size().reset_index(name="total_rows")
    user_counts["city"] = os.path.basename(file)

    # ----------------------------
    # Mask density (days 61–75)
    # ----------------------------
    day_window = df[(df["d"] >= 61) & (df["d"] <= 75)]
    mask_flags = (day_window["x"] == 999) & (day_window["y"] == 999)

    mask_density = (
        day_window.assign(mask=mask_flags)
        .groupby("uid")
        .agg(mask_rows=("mask", "sum"), total=("mask", "count"))
        .reset_index()
    )
    mask_density["mask_density"] = mask_density["mask_rows"] / mask_density["total"]

    # Merge into user stats
    user_counts = user_counts.merge(mask_density[["uid", "mask_density"]], on="uid", how="left")

    # ----------------------------
    # Distinct cells per user
    # ----------------------------
    distinct_cells = (
        df.groupby("uid")[["x", "y"]]
        .apply(lambda g: len(set(zip(g["x"], g["y"]))))
        .reset_index(name="distinct_cells")
    )

    user_counts = user_counts.merge(distinct_cells, on="uid", how="left")

    # ----------------------------
    # Summary statistics per file
    # ----------------------------
    print("  Number of unique users:", user_counts["uid"].nunique())
    print("  Min rows per user:", user_counts["total_rows"].min())
    print("  Max rows per user:", user_counts["total_rows"].max())
    print("  Mean rows per user:", user_counts["total_rows"].mean())
    print("  Median rows per user:", user_counts["total_rows"].median())

    print("  Mask density (days 61–75):")
    print("    Min:", user_counts["mask_density"].min(skipna=True))
    print("    Max:", user_counts["mask_density"].max(skipna=True))
    print("    Mean:", user_counts["mask_density"].mean(skipna=True))
    print("    Median:", user_counts["mask_density"].median(skipna=True))

    print("  Distinct cells per user:")
    print("    Min:", user_counts["distinct_cells"].min())
    print("    Max:", user_counts["distinct_cells"].max())
    print("    Mean:", user_counts["distinct_cells"].mean())
    print("    Median:", user_counts["distinct_cells"].median())

    all_user_stats.append(user_counts)

# ----------------------------
# Overall stats across all files
# ----------------------------
if all_user_stats:
    combined = pd.concat(all_user_stats, ignore_index=True)

    print("\n=== Overall Distribution Across All Files ===")
    print("Total unique users:", combined["uid"].nunique())
    print("Min rows per user:", combined["total_rows"].min())
    print("Max rows per user:", combined["total_rows"].max())
    print("Mean rows per user:", combined["total_rows"].mean())
    print("Median rows per user:", combined["total_rows"].median())

    print("Mask density (days 61–75):")
    print("  Min:", combined["mask_density"].min(skipna=True))
    print("  Max:", combined["mask_density"].max(skipna=True))
    print("  Mean:", combined["mask_density"].mean(skipna=True))
    print("  Median:", combined["mask_density"].median(skipna=True))

    print("Distinct cells per user:")
    print("  Min:", combined["distinct_cells"].min())
    print("  Max:", combined["distinct_cells"].max())
    print("  Mean:", combined["distinct_cells"].mean())
    print("  Median:", combined["distinct_cells"].median())

    combined.to_csv("user_distribution_with_masks_cells.csv", index=False)
    print("\nSaved detailed per-user stats to 'user_distribution_with_masks_cells.csv'")
