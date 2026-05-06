import pandas as pd
import numpy as np
import ast
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score

print("--- Starting Classification Training ---")
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

threshold = 2.0
df['Target_Class'] = (df['Band_Gap_eV'] > threshold).astype(int)

feature_cols = ['Volume', 'Density', 'Formation_Energy_per_atom', 'Energy_Above_Hull'] + list(embed_df.columns)
X = df[feature_cols].values
y = df['Target_Class'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.008, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

svc_model = SVC(kernel='rbf', C=13.0, probability=True, random_state=42)
rf_model = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
clf_ensemble = VotingClassifier(estimators=[('SVC', svc_model), ('Random Forest', rf_model)], voting='soft')

clf_ensemble.fit(X_train_scaled, y_train)

# Generate predictions for both datasets
train_preds = clf_ensemble.predict(X_train_scaled)
test_preds = clf_ensemble.predict(X_test_scaled)

# --- Added Metrics Calculations ---

# Training Metrics
train_acc = accuracy_score(y_train, train_preds)
train_f1 = f1_score(y_train, train_preds)
train_recall = recall_score(y_train, train_preds)
train_precision = precision_score(y_train, train_preds)

# Testing Metrics
test_acc = accuracy_score(y_test, test_preds)
test_f1 = f1_score(y_test, test_preds)
test_recall = recall_score(y_test, test_preds)
test_precision = precision_score(y_test, test_preds)

print("\n--- Training Metrics ---")
print(f"Accuracy:  {train_acc:.4f}")
print(f"F1-Score:  {train_f1:.4f}")
print(f"Recall:    {train_recall:.4f}")
print(f"Precision: {train_precision:.4f}")

print("\n--- Testing Metrics ---")
print(f"Accuracy:  {test_acc:.4f}")
print(f"F1-Score:  {test_f1:.4f}")
print(f"Recall:    {test_recall:.4f}")
print(f"Precision: {test_precision:.4f}")

# --- Save model ---
print("\nSaving model...")
joblib.dump(clf_ensemble, 'perovskite_classifier.pkl')
print("Saved perovskite_classifier.pkl\n")
