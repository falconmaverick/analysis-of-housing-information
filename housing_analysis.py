import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno
from scipy.stats import skew, kurtosis
import os

# Ensure dataset exists
file_path = "train.csv"
if not os.path.exists(file_path):
    raise FileNotFoundError(f"The file '{file_path}' was not found. Please check the path and try again.")

# Problem 1: Acquiring a dataset
df = pd.read_csv(file_path)

# Problem 2: Examining the dataset
dataset_info = {
    "Rows": df.shape[0],
    "Columns": df.shape[1],
    "Features": df.columns.tolist(),
    "Target Variable": "SalePrice",
}
print(dataset_info)
print(df.head())

# Problem 3: Checking the data
print(df.info())  # Check data types and null values
print(df.describe())  # Summary statistics

# Problem 4: Dealing with missing values
missing_values = df.isnull().sum()
missing_values = missing_values[missing_values > 0].sort_values(ascending=False)
print("Missing Values:")
print(missing_values)

# Visualizing missing values
msno.matrix(df)
plt.show()

# Removing features with more than 5 missing values
df_cleaned = df.drop(columns=missing_values[missing_values > 5].index)

# Dropping rows with remaining missing values
df_cleaned.dropna(inplace=True)
print("Data after handling missing values:")
print(df_cleaned.info())

# Problem 5: Terminology research
# Kurtosis: Measures the "tailedness" of a distribution
# Skewness: Measures the asymmetry of a distribution

# Problem 6: Checking the distribution
sns.histplot(df["SalePrice"], kde=True)
plt.title("Original SalePrice Distribution")
plt.show()

print("Skewness:", skew(df["SalePrice"]))
print("Kurtosis:", kurtosis(df["SalePrice"]))

# Log transformation
df["SalePrice_log"] = np.log1p(df["SalePrice"])
sns.histplot(df["SalePrice_log"], kde=True)
plt.title("Log-transformed SalePrice Distribution")
plt.show()

print("Skewness after log transformation:", skew(df["SalePrice_log"]))
print("Kurtosis after log transformation:", kurtosis(df["SalePrice_log"]))

# Problem 7: Checking correlation coefficients
correlation_matrix = df.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm')
plt.title("Heatmap of Feature Correlations")
plt.show()

# Selecting 10 highly correlated features
top_corr_features = correlation_matrix["SalePrice"].abs().sort_values(ascending=False)[1:11]
print("Top 10 correlated features with SalePrice:")
print(top_corr_features)

# Heatmap of selected features
plt.figure(figsize=(8,6))
sns.heatmap(df[top_corr_features.index].corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Matrix of Top Features")
plt.show()

# Identifying highly correlated feature pairs
high_corr_pairs = df[top_corr_features.index].corr().abs()
high_corr_pairs = high_corr_pairs[high_corr_pairs > 0.7]
np.fill_diagonal(high_corr_pairs.values, 0)
print("Highly correlated feature pairs:")
print(high_corr_pairs.dropna(how='all').dropna(axis=1, how='all'))
