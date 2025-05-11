import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency, shapiro

# Load the dataset
df = pd.read_csv('data/raw/loan_data_set.csv')

# Drop the 'Loan_ID' column as it is non-numeric
df = df.drop(columns=['Loan_ID'])

# Data Preprocessing
categorical_columns = ['Gender', 'Married', 'Self_Employed', 'Property_Area', 'Dependents', 'Education']
numerical_columns = ['LoanAmount', 'ApplicantIncome', 'CoapplicantIncome', 'Loan_Amount_Term', 'Credit_History']

# Impute missing categorical values with mode
categorical_imputer = SimpleImputer(strategy='most_frequent')
df[categorical_columns] = categorical_imputer.fit_transform(df[categorical_columns])

# Impute missing numerical values with median
numerical_imputer = SimpleImputer(strategy='median')
df[numerical_columns] = numerical_imputer.fit_transform(df[numerical_columns])

# DEBUG: Check for any missing values after imputation
print("\nMissing values after imputation:\n", df.isnull().sum())

# Outlier detection using IQR method
Q1 = df[numerical_columns].quantile(0.25)
Q3 = df[numerical_columns].quantile(0.75)
IQR = Q3 - Q1
outliers = ((df[numerical_columns] < (Q1 - 1.5 * IQR)) | (df[numerical_columns] > (Q3 + 1.5 * IQR)))
print("\nOutliers detected:\n", outliers.sum())

# Optional: Remove outliers
df = df[~((df[numerical_columns] < (Q1 - 1.5 * IQR)) | (df[numerical_columns] > (Q3 + 1.5 * IQR))).any(axis=1)]

# Descriptive statistics for numerical columns
print("\nDescriptive statistics (Numerical):\n", df.describe())

# Descriptive statistics for categorical columns
print("\nDescriptive statistics (Categorical):\n", df[categorical_columns].describe())

# Correlation matrix
correlation_matrix = df[numerical_columns].corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', cbar=True)
plt.title('Correlation Matrix')
plt.show()

# Chi-square test for 'Gender' vs 'Loan_Status'
contingency_table = pd.crosstab(df['Gender'], df['Loan_Status'])
chi2, p, dof, expected = chi2_contingency(contingency_table)
print(f'Chi-Square Test: p-value = {p}')

# Shapiro-Wilk test for normality on LoanAmount
stat, p_value = shapiro(df['LoanAmount'])
print(f'Shapiro-Wilk Test for normality on LoanAmount: p-value = {p_value}')

# Encode categorical variables
le = LabelEncoder()
df['Gender'] = le.fit_transform(df['Gender'])
df['Married'] = le.fit_transform(df['Married'])
df['Self_Employed'] = le.fit_transform(df['Self_Employed'])
df['Dependents'] = le.fit_transform(df['Dependents'])
df['Loan_Status'] = le.fit_transform(df['Loan_Status'])

# OneHotEncoding for multi-category features like 'Education' and 'Property_Area'
df = pd.get_dummies(df, columns=['Education', 'Property_Area'], drop_first=True)

# Define the features and target
X = df.drop('Loan_Status', axis=1)
y = df['Loan_Status']

# Final check for NaNs
assert X.isnull().sum().sum() == 0, "Still missing values in features!"

# Apply StandardScaler to scale numeric features
scaler = StandardScaler()
X[numerical_columns] = scaler.fit_transform(X[numerical_columns])

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'\nAccuracy of the model: {accuracy:.2f}')

# Save model and scaler
joblib.dump(model, 'model.joblib')
joblib.dump(scaler, 'scaler.joblib')
print("Model saved as 'model.joblib'")
print("Scaler saved as 'scaler.joblib'")
