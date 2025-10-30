import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.multioutput import MultiOutputRegressor # REQUIRED for two targets!
import joblib
import numpy as np

# --- 0. Configuration ---
ORIGINAL_FILE = 'Matches.csv'
MODEL_FILENAME = 'football_goal_predictor_final-v2.joblib'

# Columns to keep (targets, IDs, and all pre-match predictors)
COLUMNS_TO_KEEP = [
    'FTHome', 'FTAway', 'Division', 'HomeTeam', 'AwayTeam', 
    'HomeElo', 'AwayElo', 'Form3Home', 'Form5Home', 'Form3Away', 'Form5Away',
    'OddHome', 'OddDraw', 'OddAway', 'MaxHome', 'MaxDraw', 'MaxAway', 
    'Over25', 'Under25', 'MaxOver25', 'MaxUnder25',
    'HandiSize', 'HandiHome', 'HandiAway', 
    'C_LTH', 'C_LTA', 'C_VHD', 'C_VAD', 'C_HTB', 'C_PHB'
]
FORM_COLS = ['Form3Home', 'Form5Home', 'Form3Away', 'Form5Away']
ELO_COLS = ['HomeElo', 'AwayElo']

print("--- Starting Full ML Pipeline ---")

# --- 1. Load and Initial Column Selection ---
try:
    df = pd.read_csv(ORIGINAL_FILE)
    df_cleaned = df[COLUMNS_TO_KEEP].copy()
    print(f"Initial data loaded: {len(df_cleaned)} rows.")
except FileNotFoundError:
    print(f"Error: '{ORIGINAL_FILE}' not found. Please ensure the file is present.")
    exit()

# --- 2. Data Cleaning and Imputation ---

# A. Drop rows missing FORM data
rows_before = len(df_cleaned)
df_cleaned.dropna(subset=FORM_COLS, inplace=True)
print(f"Dropped {rows_before - len(df_cleaned)} rows with missing FORM data.")

# B. Impute remaining numerical columns (Elo, Odds, Clusters) with Median
numerical_cols = [col for col in df_cleaned.columns if df_cleaned[col].dtype in ['float64', 'int64']]
for col in numerical_cols:
    if df_cleaned[col].isnull().any():
        df_cleaned[col].fillna(df_cleaned[col].median(), inplace=True)

# --- 3. Feature Engineering (Crucial for better R² score) ---

df_cleaned['Elo_Diff'] = df_cleaned['HomeElo'] - df_cleaned['AwayElo']
df_cleaned['Form5_Diff'] = df_cleaned['Form5Home'] - df_cleaned['Form5Away']
df_cleaned['MaxOdd_Home_Ratio'] = df_cleaned['MaxHome'] / (df_cleaned['MaxHome'] + df_cleaned['MaxDraw'] + df_cleaned['MaxAway'])
df_cleaned['Total_Odd_Prob'] = 1/df_cleaned['OddHome'] + 1/df_cleaned['OddDraw'] + 1/df_cleaned['OddAway']
print("Feature Engineering complete: Added 4 difference/ratio features.")

# --- 4. Final Data Validation and Cleanup (Fixes ValueError: infinity) ---

print("-" * 40)
print("Final Data Validation (Checking for NaNs and Infs)")

# Drop ID/Context columns and the targets to isolate features (X)
X_temp = df_cleaned.drop(columns=['FTHome', 'FTAway', 'Division', 'HomeTeam', 'AwayTeam']).copy()

# A. Handle the infinity error: Replace any infinite values with NaN
X_temp.replace([np.inf, -np.inf], np.nan, inplace=True)

# B. Re-impute any NaNs created by the infinity replacement
for col in X_temp.columns:
    if X_temp[col].isnull().any():
        X_temp[col].fillna(X_temp[col].median(), inplace=True)
        # Note: Added print to confirm which column was cleaned, if any
        print(f"   > Cleaned residual NaNs in: {col}")

# C. Final assignment and type conversion
X = X_temp.astype(float)
Y = df_cleaned[['FTHome', 'FTAway']].astype(float) 

# Final check for missing data: should be 0
final_missing_count = X.isnull().sum().sum()
print(f"Total missing values in features (X) before split: {final_missing_count}")
print("-" * 40)

# --- 5. Split Data (70% Train, 30% Test) ---
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, 
    test_size=0.3, 
    random_state=42
)
print(f"Data split: Train={len(X_train)} samples, Test={len(X_test)} samples")
print("-" * 40)

# --- 6. Train the Model (Multi-Output Gradient Boosting Regressor) ---
print("Training Multi-Output Gradient Boosting Regressor (Wrapped)...")

# 1. Define the base model
base_estimator = GradientBoostingRegressor(
    n_estimators=100, 
    max_depth=5,
    learning_rate=0.1,
    random_state=42
)

# 2. Use MultiOutputRegressor to handle the two target columns (FTHome, FTAway)
model = MultiOutputRegressor(base_estimator, n_jobs=-1)

# The wrapped model can now accept your 2-column Y_train
model.fit(X_train, Y_train)
print("Training complete.")
print("-" * 40)

# --- 7. Evaluation and Saving ---

Y_pred = model.predict(X_test)
Y_pred_df = pd.DataFrame(Y_pred, columns=['PredHome', 'PredAway'], index=Y_test.index)

# Calculate Metrics
mae_home = mean_absolute_error(Y_test['FTHome'], Y_pred_df['PredHome'])
mae_away = mean_absolute_error(Y_test['FTAway'], Y_pred_df['PredAway'])
r2_home = r2_score(Y_test['FTHome'], Y_pred_df['PredHome'])
r2_away = r2_score(Y_test['FTAway'], Y_pred_df['PredAway'])

# Calculate Result Accuracy (H/D/A)
def get_result(home, away):
    if home > away: return 'H'
    if home < away: return 'A'
    return 'D'

actual_results = Y_test.apply(lambda row: get_result(row['FTHome'], row['FTAway']), axis=1)
predicted_results = Y_pred_df.apply(lambda row: get_result(row['PredHome'], row['PredAway']), axis=1)
correct_result_predictions = (actual_results == predicted_results).sum()
total_test_samples = len(Y_test)

# --- Output Results ---
print("--- Final Model Metrics (Gradient Boosting) ---")
print(f"Home Goals (FTHome) MAE: {mae_home:.3f}")
print(f"Away Goals (FTAway) MAE: {mae_away:.3f}")
print(f"Average Goals MAE: {((mae_home + mae_away) / 2):.3f}")
print("-" * 20)
print(f"Home Goals (FTHome) R²: {r2_home:.3f}")
print(f"Away Goals (FTAway) R²: {r2_away:.3f}")
print("-" * 20)
print(f"Result Accuracy (H/D/A): {(correct_result_predictions / total_test_samples * 100):.2f}%")
print("-" * 40)

# Save the trained model
joblib.dump(model, MODEL_FILENAME)
print(f"✅ Training completed and model saved as '{MODEL_FILENAME}'.")