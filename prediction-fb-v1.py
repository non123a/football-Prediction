import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# --- Configuration ---
CLEANED_FILE = 'football_data_cleaned_goals-v1.csv'

# --- 1. Load and Prepare Data ---
try:
    df = pd.read_csv(CLEANED_FILE)
except FileNotFoundError:
    print(f"Error: The file '{CLEANED_FILE}' was not found. Please run the cleaning script first.")
    exit()

# Define Targets (Y) and Features (X)
# Y is a matrix of two columns (FTHome and FTAway)
Y = df[['FTHome', 'FTAway']]

# X is everything else (your pre-match predictors).
# We must drop the target columns from the features.
X = df.drop(columns=['FTHome', 'FTAway', 'Division', 'HomeTeam', 'AwayTeam'])

# Handle Categorical Columns (if any non-numerical survived)
# We One-Hot Encode the 'Division' column if it was not dropped during the initial selection
# In this specific setup, X should already be purely numerical, but we run this check for robustness.
if not all(dtype != object for dtype in X.dtypes):
    X = pd.get_dummies(X, drop_first=True)

# --- 2. Split Data (70% Train, 30% Test) ---
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, 
    test_size=0.3, # 30% of data for testing
    random_state=42 # Set for reproducibility
)

print(f"Training set size: {len(X_train)} samples")
print(f"Testing set size: {len(X_test)} samples")
print("-" * 40)

# --- 3. Train the Model (Random Forest Regressor) ---
print("Training Multi-Output Random Forest Regressor...")

# The Random Forest Regressor naturally handles multi-output regression.
model = RandomForestRegressor(
    n_estimators=100, # Number of trees in the forest
    random_state=42, 
    n_jobs=-1         # Use all processor cores
)

model.fit(X_train, Y_train)
print("Training complete.")
print("-" * 40)

# --- 4. Evaluate the Model ---

# Make predictions on the test set
Y_pred = model.predict(X_test)

# Convert predictions to a DataFrame for easier handling
Y_pred_df = pd.DataFrame(Y_pred, columns=['PredHome', 'PredAway'], index=Y_test.index)

# A. Evaluation Metrics (Measure of Accuracy for Regression)

# Mean Absolute Error (MAE): Average absolute difference between predicted and actual goals
mae_home = mean_absolute_error(Y_test['FTHome'], Y_pred_df['PredHome'])
mae_away = mean_absolute_error(Y_test['FTAway'], Y_pred_df['PredAway'])

# R-squared (R²): Proportion of the variance in the dependent variable that is predictable from the independent variables
r2_home = r2_score(Y_test['FTHome'], Y_pred_df['PredHome'])
r2_away = r2_score(Y_test['FTAway'], Y_pred_df['PredAway'])

print("--- Regression Metrics on the Test Set ---")
print(f"Home Goals (FTHome) MAE: {mae_home:.3f}")
print(f"Away Goals (FTAway) MAE: {mae_away:.3f}")
print(f"Average MAE (Goals): {((mae_home + mae_away) / 2):.3f}")
print("-" * 20)
print(f"Home Goals (FTHome) R²: {r2_home:.3f}")
print(f"Away Goals (FTAway) R²: {r2_away:.3f}")
print("-" * 40)


# B. What were the exact goal predictions? (For the first 5 test matches)
results_df = Y_test.copy()
results_df['PredHome'] = Y_pred_df['PredHome']
results_df['PredAway'] = Y_pred_df['PredAway']

print("\n--- Example Predictions (First 5 Test Matches) ---")
print("Actual goals are integers (e.g., 2), Predicted goals are floats (e.g., 2.34)")
print(results_df.head())
print("-" * 40)


# C. What was the *Match Result* that it got right? (Classification Check)

# 1. Determine Actual Result (H, D, A)
def get_result(home, away):
    if home > away: return 'H'
    if home < away: return 'A'
    return 'D'

# Apply the function to the actual and predicted goals
actual_results = results_df.apply(lambda row: get_result(row['FTHome'], row['FTAway']), axis=1)
predicted_results = results_df.apply(lambda row: get_result(row['PredHome'], row['PredAway']), axis=1)

# 2. Compare and Count
correct_result_predictions = (actual_results == predicted_results).sum()
total_test_samples = len(results_df)

print("--- Match Result Accuracy (Derived from Goal Predictions) ---")
print(f"Total Test Matches: {total_test_samples}")
print(f"Matches where the predicted RESULT (H/D/A) was correct: {correct_result_predictions}")
print(f"Result Accuracy: {(correct_result_predictions / total_test_samples * 100):.2f}%")
print("-" * 40)