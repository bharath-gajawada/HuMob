# import pandas as pd

# # Read the CSV with gzip compression
# df = pd.read_csv('predictions/Humobility_city_D_humob25.csv')

# # Check for required columns
# required_columns = {'x', 'y', 'uid'}
# if not required_columns.issubset(df.columns):
#     raise ValueError(f"CSV file must contain columns: {required_columns}")

# # Define masked condition
# masked_condition = (df['x'] == 999) & (df['y'] == 999)

# # Count masked data points
# masked_count = masked_condition.sum()

# # Get number of unique 'uid's in masked data
# unique_uids_in_masked = df.loc[masked_condition, 'uid'].nunique()

# print(f"Number of masked data points: {masked_count}")
# print(f"Number of unique 'uid's in masked data points: {unique_uids_in_masked}")


import pandas as pd

# Read the CSV file
df = pd.read_csv('predictions/Humobility_city_C_humob25.csv')

# Check for required columns
required_columns = {'x', 'y', 'uid'}
if not required_columns.issubset(df.columns):
    raise ValueError(f"CSV file must contain columns: {required_columns}")

# Total number of data points
total_data_points = len(df)

# Total number of unique 'uid's
total_unique_uids = df['uid'].nunique()

print(f"Total number of data points: {total_data_points}")
print(f"Total number of unique 'uid's: {total_unique_uids}")
