import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

df = pd.read_csv("test (2).csv")
print("Original Shape:", df.shape)
print(f"Missing values : {df.isnull().sum()}")
df.drop("ID", axis=1, inplace=True)
cat_cols = ['Gender','Ever_Married','Graduated','Profession','Spending_Score']
num_cols = ['Age','Work_Experience','Family_Size']

num_imputer = SimpleImputer(strategy='median')
df[num_cols] = num_imputer.fit_transform(df[num_cols])
cat_imputer = SimpleImputer(strategy='most_frequent')
df[cat_cols] = cat_imputer.fit_transform(df[cat_cols])
print(f"\nMissing Values After Cleaning:\n{df.isnull().sum()}")
df = pd.get_dummies(df,columns=cat_cols,drop_first=True)
df['Age_Group'] = pd.cut(df['Age'],bins=[0,25,40,60,100],labels=[0,1,2,3])
df['Family_Category'] = pd.cut(df['Family_Size'],bins=[0,2,5,20],labels=[0,1,2])
df['Age_Group'] = df['Age_Group'].astype(int)
df['Family_Category'] = df['Family_Category'].astype(int)
df.drop_duplicates(inplace=True)
output_file = "customer_segmentation_processed.csv"
df.to_csv(output_file, index=False)
print("\nProcessed Shape:", df.shape)
print(f"\nDataset Saved As: {output_file}")

