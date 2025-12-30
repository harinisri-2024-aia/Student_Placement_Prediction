# 05_Data_Preprocessing.py
# Purpose: Clean and prepare placement dataset for modeling

import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

# -------------------------------
# Step 0: Load the dataset
# -------------------------------
df = pd.read_csv('../data/03_EDA_Cleaned.csv')
print("Initial Data Shape:", df.shape)
print(df.head())

# -------------------------------
# Step 1: Handle Missing Values
# -------------------------------
print("\nMissing values per column:\n", df.isnull().sum())

# Numeric columns
numeric_cols = ['ssc_p','hsc_p','degree_p','etest_p','mba_p','salary','total_score','average_score','workex_binary']
for col in numeric_cols:
    if col in df.columns:
        df[col] = df[col].fillna(df[col].mean())

# Categorical columns
categorical_cols = ['gender','degree_category','mba_category']
for col in categorical_cols:
    if col in df.columns:
        df[col] = df[col].fillna(df[col].mode()[0])

# -------------------------------
# Step 2: Handle Outliers (IQR method)
# -------------------------------
for col in numeric_cols:
    if col in df.columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        df = df[~((df[col] < Q1 - 1.5*IQR) | (df[col] > Q3 + 1.5*IQR))]

# -------------------------------
# Step 3: Encode Categorical Variables
# -------------------------------
# Binary encoding for 'gender'
if 'gender' in df.columns:
    le = LabelEncoder()
    df['gender'] = le.fit_transform(df['gender'])

# One-Hot Encoding for 'degree_category' and 'mba_category'
df = pd.get_dummies(df, columns=[col for col in ['degree_category','mba_category'] if col in df.columns], drop_first=True)

# -------------------------------
# Step 4: Feature Scaling / Normalization
# -------------------------------
scaler = StandardScaler()
df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

# -------------------------------
# Step 5: Split Data into Train/Test
# -------------------------------
# Target: 'status' (Placed / Not Placed)
X = df.drop('status', axis=1)
y = df['status']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("\nTraining shape:", X_train.shape)
print("Testing shape:", X_test.shape)

# -------------------------------
# Step 6: Save cleaned dataset
# -------------------------------
df.to_csv('../data/03_EDA_Cleaned.csv', index=False)
print("\nCleaned dataset saved as '../data/03_EDA_Cleaned.csv'")