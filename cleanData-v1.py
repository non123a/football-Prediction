import pandas as pd

# 1. Load the original dataset
df_original = pd.read_csv('Matches.csv')

# Define the list of columns to KEEP (Targets and Pre-Match Predictors)
columns_to_keep = [
    'FTHome', 'FTAway',                  # Your Target Variables
    'Division', 'HomeTeam', 'AwayTeam', 
    'HomeElo', 'AwayElo', 
    'Form3Home', 'Form5Home', 'Form3Away', 'Form5Away',
    'OddHome', 'OddDraw', 'OddAway', 
    'MaxHome', 'MaxDraw', 'MaxAway', 
    'Over25', 'Under25', 'MaxOver25', 'MaxUnder25',
    'HandiSize', 'HandiHome', 'HandiAway', 
    'C_LTH', 'C_LTA', 'C_VHD', 'C_VAD', 'C_HTB', 'C_PHB'
]

# 2. Create the new, cleaned DataFrame by selecting only the columns to keep
df_cleaned = df_original[columns_to_keep].copy()

# -------------------------------------------------------------------
## 3. IDENTIFY AND HANDLE MISSING DATA

print("--- Missing Data Check (Before Cleaning) ---")
# Calculate the total number of missing values per column
missing_counts = df_cleaned.isnull().sum()
# Print only columns that have missing data
print(missing_counts[missing_counts > 0])
print("-" * 40)

# Identify numerical columns for imputation
# Note: We assume all columns *except* the team/division names are numerical for this step
# You can get numerical columns easily by dropping the string/enum ones
numerical_cols = df_cleaned.drop(columns=['Division', 'HomeTeam', 'AwayTeam']).columns

# Imputation Strategy: Use the Median
# The median is robust to outliers and is a good default for filling numerical football stats.
for col in numerical_cols:
    if df_cleaned[col].isnull().any():
        median_value = df_cleaned[col].median()
        df_cleaned[col].fillna(median_value, inplace=True)

# For categorical columns (Division, HomeTeam, AwayTeam), you might fill with 'Unknown' 
# or simply leave them for now, as missing team names are rare and often indicate bad data.
# df_cleaned[['Division', 'HomeTeam', 'AwayTeam']] = df_cleaned[['Division', 'HomeTeam', 'AwayTeam']].fillna('Unknown')

print("\n--- Missing Data Check (After Cleaning) ---")
print(df_cleaned.isnull().sum().sum()) # Should be 0 if all missing data was imputed
print("-" * 40)

# 4. Save the new DataFrame to a new file
df_cleaned.to_csv('football_data_cleaned_goals.csv', index=False)

print("Process complete. Cleaned file saved as 'football_data_cleaned_goals.csv'.")