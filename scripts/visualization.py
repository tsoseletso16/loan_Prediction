import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns  # Import seaborn

# Load your dataset
df = pd.read_csv('../data/raw/loan_data_set.csv')  # Adjust path if needed




# Plot 4: Education vs Loan Status (Bar Plot)
edu_loan = pd.crosstab(df['Education'], df['Loan_Status'])
edu_loan.plot(kind='bar', stacked=True, colormap='coolwarm', figsize=(6, 4))
plt.title('Education Level vs Loan Status')
plt.xlabel('Education')
plt.ylabel('Count')
plt.tight_layout()
plt.show()
