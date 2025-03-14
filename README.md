# **Housing Information Analysis**  

## **Project Overview**  
This project analyzes housing prices in Ames, Iowa, using a dataset from Kaggle. The analysis focuses on **data cleaning, exploratory data analysis (EDA), feature selection, and correlation analysis** to better understand how different housing attributes influence sale prices.  

## **Dataset**  
- The dataset used is **Ames Housing Data** (`train.csv`).  
- It contains **79 features** describing various aspects of houses, with **`SalePrice`** as the target variable.  

## **Installation & Requirements**  
### **Prerequisites**  
Ensure you have the following libraries installed before running the notebook:  
```bash
pip install pandas numpy matplotlib seaborn missingno scipy
```

### **Dataset Download**  
1. Download the dataset from [Kaggle](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data).  
2. Place `train.csv` in the working directory before running the script.  

## **Key Analysis Steps**  
### 1️⃣ **Data Loading & Overview**  
- The dataset is loaded using **Pandas** (`pd.read_csv()`).  
- Basic structure and statistical summaries are inspected.  

### 2️⃣ **Handling Missing Values**  
- Features with **more than 5 missing values** are dropped.  
- Remaining missing values are removed to clean the dataset.  
- **Missingno** is used for visualizing missing data.  

### 3️⃣ **Exploratory Data Analysis (EDA)**  
- **Distribution Analysis:**  
  - The target variable (`SalePrice`) is visualized using **seaborn histograms**.  
  - **Skewness & Kurtosis** are computed to assess normality.  
  - **Log transformation** is applied to `SalePrice` to normalize its distribution.  

### 4️⃣ **Correlation Analysis**  
- A **heatmap** of the correlation matrix is generated to identify strong relationships.  
- The **top 10 features** with the highest correlation to `SalePrice` are selected.  
- Highly correlated feature pairs are identified to avoid redundancy.  

## **Results & Insights**  
- **Cleaning missing values improves dataset quality.**  
- **Log transformation helps normalize the price distribution.**  
- **Feature selection aids in reducing dimensionality for predictive modeling.**  

## **Usage**  
Run the Python script in Jupyter Notebook or a Python environment:  
```python
python housing_analysis.py
```

## **Future Improvements**  
- Implement **feature engineering** to create new meaningful attributes.  
- Apply **machine learning models** (e.g., Linear Regression, XGBoost) to predict house prices.  
