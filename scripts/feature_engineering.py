import pandas as pd

# Load the dataset
df = pd.read_csv('data/raw/loan_data_set.csv')

# Convert 'Dependents' column to numeric, forcing errors to NaN
df['Dependents'] = pd.to_numeric(df['Dependents'], errors='coerce')

# Feature Engineering: Creating a new feature for income per family member
df['Income_per_Family_Member'] = (df['ApplicantIncome'] + df['CoapplicantIncome']) / (df['Dependents'].fillna(0) + 1)  # Avoid division by zero

# Feature tuning: Create a new feature combining loan amount and applicant income
df['Loan_to_Income_Ratio'] = df['LoanAmount'] / (df['ApplicantIncome'] + 1)

# Fill NaN values in 'Loan_to_Income_Ratio' with 0
df['Loan_to_Income_Ratio'] = df['Loan_to_Income_Ratio'].fillna(0)

# Show the new features in the dataframe
print(df[['Income_per_Family_Member', 'Loan_to_Income_Ratio']].head())

# Optionally, save the modified dataframe to check the results later
df.to_csv('data/processed/loan_data_with_features.csv', index=False)
