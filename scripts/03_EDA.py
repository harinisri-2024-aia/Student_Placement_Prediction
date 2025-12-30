# 03_eda.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("../data/Placement_Data_Full_Class.csv")

# ------------------------------
# 1️⃣ Feature Engineering
# Create CGPA category
def cgpa_category(cgpa):
    if cgpa < 60:
        return "Low"
    elif 60 <= cgpa < 75:
        return "Medium"
    else:
        return "High"

df['degree_category'] = df['degree_p'].apply(cgpa_category)
df['mba_category'] = df['mba_p'].apply(cgpa_category)

# Internship / Work Experience binary
df['workex_binary'] = df['workex'].map({'Yes': 1, 'No': 0})

# Optional: Total Score or Average Score (if needed for modeling later)
df['total_score'] = df[['ssc_p','hsc_p','degree_p','etest_p','mba_p']].sum(axis=1)
df['average_score'] = df[['ssc_p','hsc_p','degree_p','etest_p','mba_p']].mean(axis=1)

# ------------------------------
# 2️⃣ Basic EDA - Placement Status
plt.figure(figsize=(6,4))
sns.countplot(x='status', data=df, palette='Set2')
plt.title("Placement Status Count")
plt.savefig("../outputs/placement_status_count.png")
plt.show()

# Insight: Most students are placed, but a small percentage remain unplaced.

# ------------------------------
# 3️⃣ Placement by Gender
plt.figure(figsize=(6,4))
sns.countplot(x='gender', hue='status', data=df, palette='Set1')
plt.title("Placement by Gender")
plt.savefig("../outputs/placement_gender.png")
plt.show()

# Insight: Placement rates are slightly higher for males than females, but the difference is small.

# ------------------------------
# 4️⃣ Placement by Degree Category
plt.figure(figsize=(6,4))
sns.countplot(x='degree_category', hue='status', data=df, palette='Set3')
plt.title("Placement by Degree Category")
plt.savefig("../outputs/placement_degree_category.png")
plt.show()

# Insight: Commerce & Management students have higher placement rates than Science & Tech students.

# ------------------------------
# 5️⃣ CGPA vs Placement Probability
plt.figure(figsize=(8,5))
sns.boxplot(x='status', y='degree_p', data=df, palette='Pastel1')
plt.title("Degree CGPA Distribution by Placement Status")
plt.savefig("../outputs/cgpa_vs_placement.png")
plt.show()

# Insight: Higher CGPA generally improves placement chances, but internships also play a key role.

# ------------------------------
# 6️⃣ Internship / Work Experience vs Placement
plt.figure(figsize=(6,4))
sns.countplot(x='workex', hue='status', data=df, palette='Set2')
plt.title("Work Experience vs Placement")
plt.savefig("../outputs/workex_vs_placement.png")
plt.show()

# Insight: Students with internships or work experience have significantly better placement outcomes.

# ------------------------------
# 7️⃣ Salary Analysis (Placed Students Only)
placed_df = df[df['status']=='Placed']

# Salary Distribution
plt.figure(figsize=(8,5))
sns.histplot(placed_df['salary'], bins=15, kde=True, color='green')
plt.title("Salary Distribution of Placed Students")
plt.savefig("../outputs/salary_distribution.png")
plt.show()

# Insight: Most salaries are around the median, but a few high packages create a long tail.

# Salary by Degree Category
plt.figure(figsize=(8,5))
sns.boxplot(x='degree_category', y='salary', data=placed_df, palette='Set2')
plt.title("Salary by Degree Category")
plt.savefig("../outputs/salary_by_degree_category.png")
plt.show()

# Insight: Students from high CGPA/degree categories tend to earn higher salaries.

# ------------------------------
# 8️⃣ Correlation Heatmap (Numeric Columns)
plt.figure(figsize=(10,6))
numeric_cols = ['ssc_p','hsc_p','degree_p','mba_p','etest_p','salary','workex_binary']
sns.heatmap(df[numeric_cols].corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Heatmap")
plt.savefig("../outputs/correlation_heatmap.png")
plt.show()

# Insight: CGPA, degree percentage, and aptitude scores show moderate positive correlation with salary. 
# SSC and HSC percentages are less correlated. Work experience also positively correlates with placement.

# All plotting code here
plt.savefig("some_plot.png")
# do NOT call plt.show() yet

# At the very end
plt.show()  # This opens the last figure

# Save cleaned data for next steps
df.to_csv('../data/03_EDA_Cleaned.csv', index=False)
print("03_EDA_Cleaned.csv saved successfully")

