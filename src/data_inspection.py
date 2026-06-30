import pandas as pd

# 1. Column names based on the C-MAPSS documentation
index_names = ['unit_number', 'time_in_cycles']
setting_names = ['op_setting_1', 'op_setting_2', 'op_setting_3']
sensor_names = [f'sensor_{i}' for i in range(1, 22)]
col_names = index_names + setting_names + sensor_names

# 2. Loading FD002 and FD004 training datasets 
# Note: They are space-separated, and using delim_whitespace handles irregular spacing.
print("Loading FD002 and FD004 training data...")
fd002_raw = pd.read_csv("data/CMAPSS/train_FD002.txt",sep=r"\s+",header=None)
fd004_raw = pd.read_csv("data/CMAPSS/train_FD004.txt",sep=r"\s+",header=None)

fd002_raw = fd002_raw.iloc[:, :26]
fd002_raw.columns = col_names

fd004_raw = fd004_raw.iloc[:, :26]
fd004_raw.columns = col_names

# 3. Combine them to analyze the full distribution of operating settings
# We concatenate them because our final clustering model needs to cover all conditions.
df_combined = pd.concat([fd002_raw, fd004_raw], axis=0, ignore_index=True)
print(df_combined.columns.tolist())
print(setting_names)

# 4. Extract only the three operational settings
print(df_combined.columns)
df_op_settings = df_combined[setting_names]

print("\n" + "="*40)
print("CHECKPOINT: OPERATIONAL SETTINGS INSPECTION")
print("="*40)

# Printing Shape
print(f"\n[SHAPE]: {df_op_settings.shape}")

# Printing First Five Rows
print("\n[FIRST 5 ROWS]:")
print(df_op_settings.head())

# Printing Summary & Statistics
print("\n[SUMMARY STATISTICS]:")
print(df_op_settings.describe())