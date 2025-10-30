import pandas as pd

# Define the name of your cleaned file
CLEANED_FILE = 'football_data_cleaned_goals-v1.csv'

try:
    # 1. Load the cleaned dataset
    df = pd.read_csv(CLEANED_FILE)
    print(f"Successfully loaded '{CLEANED_FILE}' with {len(df)} rows.")
    print("=" * 60)

    # 2. Check for missing values across the entire DataFrame
    missing_counts = df.isnull().sum()
    
    # Filter to show only columns with *at least one* missing value
    missing_cols = missing_counts[missing_counts > 0]
    
    # Calculate the total number of rows that have *any* missing value
    total_rows_with_missing_data = df.isnull().any(axis=1).sum()

    # 3. Output Results
    
    print("\n--- Missing Data Summary ---")
    
    if total_rows_with_missing_data == 0:
        print("✅ Excellent! The dataset has NO missing values in any row.")
    else:
        # A. Total Rows Missing Data
        print(f"⚠️ Total rows in the dataset: {len(df)}")
        print(f"⚠️ Total rows with at least one missing value: {total_rows_with_missing_data}")
        print(f"   ({(total_rows_with_missing_data / len(df) * 100):.2f}% of rows)")

        print("-" * 30)

        # B. Columns with Missing Values
        print("Columns that still have missing values and their counts:")
        
        # Sort and display the columns with missing data, highest count first
        most_missing_cols = missing_cols.sort_values(ascending=False)
        print(most_missing_cols)

        # C. Identify the single most affected column
        if not most_missing_cols.empty:
            most_affected_col = most_missing_cols.index[0]
            count = most_missing_cols.iloc[0]
            print(f"\n💡 The most affected column is: '{most_affected_col}' with {count} missing values.")
            
    print("=" * 60)

except FileNotFoundError:
    print(f"Error: The file '{CLEANED_FILE}' was not found. Please ensure the cleaning script ran successfully first.")
except Exception as e:
    print(f"An unexpected error occurred: {e}")