# 1. Import required libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# 2. Read the Titanic dataset
df = pd.read_csv("Titanic.csv")

print("First 5 rows of the dataset:")
display(df.head())

# 3. Check dataset shape and data types
print("\nShape of dataset:", df.shape)

print("\nData types:")
print(df.dtypes)

print("\nDataset information:")
df.info()

# 4. Check missing values
print("\nMissing values:")
print(df.isnull().sum())

missing_percent = (df.isnull().sum() / len(df)) * 100

print("\nPercentage of missing values:")
print(missing_percent.round(2))

# 5. Handle missing values
df["Age"] = df["Age"].fillna(df["Age"].mean())
df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

# Drop Cabin because it contains many missing values
df.drop("Cabin", axis=1, inplace=True)

print("\nMissing values after handling:")
print(df.isnull().sum())

# 6. Data formatting
print("\nEmbarked values before formatting:")
print(df["Embarked"].unique())

df["Embarked"] = df["Embarked"].str.strip().str.upper()

print("\nEmbarked values after formatting:")
print(df["Embarked"].unique())

# 7. Convert categorical variables to category data type
df["Sex"] = df["Sex"].astype("category")
df["Embarked"] = df["Embarked"].astype("category")

print("\nCategorical column types:")
print(df[["Sex", "Embarked"]].dtypes)

# 8. Convert categorical variables into quantitative variables
df["Sex"] = df["Sex"].map({"male": 0, "female": 1}).astype(int)
df["Embarked"] = df["Embarked"].map({"S": 0, "C": 1, "Q": 2}).astype(int)

print("\nCategorical variables converted to numeric values:")
display(df[["Sex", "Embarked"]].head())

# 9. Normalize numerical variables using Min-Max Scaling
scaler = MinMaxScaler()

df["Age"] = scaler.fit_transform(df[["Age"]])
df["Fare"] = scaler.fit_transform(df[["Fare"]])

print("\nBefore normalization, original values are no longer available in df.")
print("Normalized Age and Fare:")
display(df[["Age", "Fare"]].head())

# 10. Verify final data types
print("\nFinal data types:")
print(df.dtypes)

# 11. Verify missing values
print("\nFinal missing values:")
print(df.isnull().sum())

# 12. Display final preprocessed dataset
print("\nFinal preprocessed dataset:")
display(df.head(10))

print("\nFinal dataset information:")
df.info()