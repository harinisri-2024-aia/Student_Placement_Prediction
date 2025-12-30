# 04_Categorical_Visuals.py
# Purpose: Generate category-wise visualizations from cleaned dataset

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

# -----------------------
# Step 1: Load CLEANED dataset (CORRECTED)
# -----------------------
df = pd.read_csv('../data/03_EDA_Cleaned.csv')

# -----------------------
# Step 2: Prepare output folder
# -----------------------
output_folder = 'categorical_plots'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# -----------------------
# Step 3: Identify categorical columns
# -----------------------
categorical_columns = df.select_dtypes(include=['object', 'category']).columns
print("Categorical columns:", list(categorical_columns))

# -----------------------
# Step 4: Count plots (category distributions)
# -----------------------
for col in categorical_columns:
    plt.figure(figsize=(8, 5))
    sns.countplot(data=df, x=col, order=df[col].value_counts().index)
    plt.title(f'Distribution of {col}')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f'{output_folder}/{col}_countplot.png')
    plt.show()

# -----------------------
# Step 5: Boxplots (numerical vs categorical)
# -----------------------
numerical_columns = df.select_dtypes(include=['int64', 'float64']).columns

for num_col in numerical_columns:
    for cat_col in categorical_columns:
        plt.figure(figsize=(8, 5))
        sns.boxplot(data=df, x=cat_col, y=num_col)
        plt.title(f'{num_col} across {cat_col}')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f'{output_folder}/{num_col}_vs_{cat_col}_boxplot.png')
        plt.show()

# -----------------------
# Step 6: Pie charts (category proportions)
# -----------------------
for col in categorical_columns:
    plt.figure(figsize=(6, 6))
    df[col].value_counts().plot.pie(autopct='%1.1f%%', startangle=90)
    plt.title(f'Proportion of {col}')
    plt.ylabel('')
    plt.tight_layout()
    plt.savefig(f'{output_folder}/{col}_piechart.png')
    plt.show()

# -----------------------
# Step 7: Heatmaps (categorical vs categorical)
# -----------------------
for i in range(len(categorical_columns)):
    for j in range(i + 1, len(categorical_columns)):
        col1 = categorical_columns[i]
        col2 = categorical_columns[j]

        cross_tab = pd.crosstab(df[col1], df[col2])

        plt.figure(figsize=(10, 6))
        sns.heatmap(cross_tab, annot=True, fmt='d')
        plt.title(f'{col1} vs {col2}')
        plt.tight_layout()
        plt.savefig(f'{output_folder}/{col1}_vs_{col2}_heatmap.png')
        plt.show()

print("✅ All categorical visualizations generated and saved in:", output_folder)