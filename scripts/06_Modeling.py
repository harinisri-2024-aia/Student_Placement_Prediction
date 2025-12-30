# 06_Modeling.py
# Clean & deployment-safe modeling

import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# -------------------------------
# Load data
# -------------------------------
df = pd.read_csv('../data/03_EDA_Cleaned.csv')

# -------------------------------
# DROP leakage / non-input columns
# -------------------------------
drop_cols = ['sl_no', 'salary', 'total_score', 'average_score']
df = df.drop(columns=[c for c in drop_cols if c in df.columns])

# -------------------------------
# Target
# -------------------------------
X = df.drop('status', axis=1)
y = df['status']

y = LabelEncoder().fit_transform(y)

# -------------------------------
# One-hot encode categoricals
# -------------------------------
X = pd.get_dummies(X, drop_first=True)

# -------------------------------
# Train-test split
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------------
# Scaling
# -------------------------------
scaler = StandardScaler()
num_cols = X_train.select_dtypes(include=['int64', 'float64']).columns

X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
X_test[num_cols] = scaler.transform(X_test[num_cols])

# -------------------------------
# Train model
# -------------------------------
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

print("Accuracy:", accuracy_score(y_test, model.predict(X_test)))

# -------------------------------
# SAVE artifacts (CRITICAL)
# -------------------------------
joblib.dump(model, '../models/placement_model.pkl')
joblib.dump(scaler, '../models/scaler.pkl')
joblib.dump(X_train.columns.tolist(), '../models/feature_columns.pkl')

print("✅ Model, scaler, and feature columns saved")
