import pandas as pd
import numpy as np
import ast
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, VotingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

print("--- Starting Regression Training ---")
df = pd.read_csv('Metal_Oxide_Perovskites_1000_embedded_output.csv')

cols_to_check = ['Volume', 'Density', 'Formation_Energy_per_atom', 'Energy_Above_Hull', 'Band_Gap_eV']
for col in cols_to_check:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
df = df.dropna(subset=cols_to_check)

df['formula_embedding'] = df['formula_embedding'].apply(ast.literal_eval)
embed_df = pd.DataFrame(df['formula_embedding'].tolist(), index=df.index)
embed_df.columns = [f'embed_{i}' for i in range(embed_df.shape[1])]
df = pd.concat([df, embed_df], axis=1)

feature_cols = ['Volume', 'Density', 'Formation_Energy_per_atom', 'Energy_Above_Hull'] + list(embed_df.columns)
X = df[feature_cols].values
y = df['Band_Gap_eV'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.008, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

svm_model = SVR(kernel='rbf', C=13.0, epsilon=0.001)
rf_model = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
reg_ensemble = VotingRegressor(estimators=[('SVM', svm_model), ('Random Forest', rf_model)])

reg_ensemble.fit(X_train_scaled, y_train)

train_preds = reg_ensemble.predict(X_train_scaled)
test_preds = reg_ensemble.predict(X_test_scaled)

# --- Added Metrics Calculations ---

# Training Metrics
train_r2 = r2_score(y_train, train_preds)
train_rmse = np.sqrt(mean_squared_error(y_train, train_preds))
train_mae = mean_absolute_error(y_train, train_preds)

# Testing Metrics
test_r2 = r2_score(y_test, test_preds)
test_rmse = np.sqrt(mean_squared_error(y_test, test_preds))
test_mae = mean_absolute_error(y_test, test_preds)

print("\n--- Training Metrics ---")
print(f"R²:   {train_r2:.4f}")
print(f"RMSE: {train_rmse:.4f}")
print(f"MAE:  {train_mae:.4f}")

print("\n--- Testing Metrics ---")
print(f"R²:   {test_r2:.4f}")
print(f"RMSE: {test_rmse:.4f}")
print(f"MAE:  {test_mae:.4f}")

# --- Save models ---
print("\nSaving models...")
joblib.dump(reg_ensemble, 'perovskite_regressor.pkl')
joblib.dump(scaler, 'feature_scaler.pkl') # Save scaler here so both apps can use the same one
print("Saved perovskite_regressor.pkl and feature_scaler.pkl\n")
