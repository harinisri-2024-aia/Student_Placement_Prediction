import pandas as pd

# Correct relative path because script is in scripts/ folder
df = pd.read_csv("../data/Placement_Data_Full_Class.csv")

print(df.head())
