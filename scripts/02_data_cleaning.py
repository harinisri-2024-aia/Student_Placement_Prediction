# 02_data_cleaning.py
import pandas as pd
import numpy as np

# Load the dataset
df = pd.read_csv("../data/Placement_Data_Full_Class.csv")

# ------------------------------
# 1️⃣ Check missing values
print("Missing values per column:\n", df.isnull().sum())
print("\n")

# ------------------------------
# 2️⃣ Check basic statistics
print("Numeric columns summary:\n", df.describe())
print("\nCategorical columns value counts:\n")
categorical_cols = ['gender','ssc_b','hsc_b','hsc_s','degree_t','specialisation','workex','status']
for col in categorical_cols:
    print(f"Value counts for {col}:")
    print(df[col].value_counts())
    print("\n")

# ------------------------------
# 3️⃣ Convert categorical columns to 'category' dtype
for col in categorical_cols:
    df[col] = df[col].astype('category')

print("Updated data types:\n", df.dtypes)
print("\n")

# ------------------------------
# 4️⃣ Check for duplicates
duplicates = df.duplicated().sum()
print(f"Number of duplicate rows: {duplicates}\n")

# ------------------------------
# 5️⃣ Handle salary column
# Salary is only for Placed students. Keep NaN for Not Placed.
placed_df = df[df['status']=='Placed']
print("Salary summary for placed students:\n", placed_df['salary'].describe())

# Optional: Check for outliers in numeric columns
numeric_cols = ['ssc_p','hsc_p','degree_p','mba_p','etest_p','salary']
for col in numeric_cols:
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5*iqr
    upper = q3 + 1.5*iqr
    outliers = df[(df[col] < lower) | (df[col] > upper)]
    print(f"{col} - Outliers count: {outliers.shape[0]}")
