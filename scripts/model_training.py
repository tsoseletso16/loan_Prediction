import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# Load dataset
df = pd.read_csv('./data/processed/loan_data_with_features.csv')

# Drop rows with missing values
df.dropna(inplace=True)

# Restrict to selected input features + target
selected_features = [
    'Gender', 'Married', 'Dependents', 'Education', 'Self_Employed',
    'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term',
    'Credit_History', 'Property_Area'
]

X = df[selected_features].copy()
y = df['Loan_Status']
y = y.map({'N': 0, 'Y': 1})  # Fix for XGBoost

# Encode categorical variables
label_encoders = {}
for column in X.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    X[column] = le.fit_transform(X[column])
    label_encoders[column] = le

# Split train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale numerical data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Logistic Regression + Tuning
logreg = LogisticRegression(max_iter=1000, class_weight='balanced')
logreg_params = {
    'C': [0.01, 0.1, 1, 10],
    'solver': ['liblinear', 'lbfgs']
}
logreg_grid = GridSearchCV(logreg, logreg_params, cv=5)
logreg_grid.fit(X_train_scaled, y_train)

logreg_best = logreg_grid.best_estimator_
y_pred_logreg = logreg_best.predict(X_test_scaled)

print("Logistic Regression:")
print(f"Best Params: {logreg_grid.best_params_}")
print("Accuracy:", accuracy_score(y_test, y_pred_logreg))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_logreg))

# Random Forest + Tuning
rf = RandomForestClassifier(random_state=42, class_weight='balanced')
rf_params = {
    'n_estimators': [100, 200],
    'max_depth': [None, 5, 10],
    'min_samples_split': [2, 5]
}
rf_grid = GridSearchCV(rf, rf_params, cv=5)
rf_grid.fit(X_train, y_train)

rf_best = rf_grid.best_estimator_
y_pred_rf = rf_best.predict(X_test)

print("\nRandom Forest Classifier:")
print(f"Best Params: {rf_grid.best_params_}")
print("Accuracy:", accuracy_score(y_test, y_pred_rf))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_rf))

# XGBoost + Tuning
scale_pos_weight = (y == 0).sum() / (y == 1).sum()  # For imbalance handling
xgb = XGBClassifier(eval_metric='logloss', 
                    random_state=42, 
                    scale_pos_weight=scale_pos_weight)  # Removed use_label_encoder
xgb_params = {
    'n_estimators': [50, 100],
    'max_depth': [3, 5],
    'learning_rate': [0.01, 0.1]
}
xgb_grid = GridSearchCV(xgb, xgb_params, cv=5)
xgb_grid.fit(X_train, y_train)

xgb_best = xgb_grid.best_estimator_
y_pred_xgb = xgb_best.predict(X_test)

print("\nXGBoost Classifier:")
print(f"Best Params: {xgb_grid.best_params_}")
print("Accuracy:", accuracy_score(y_test, y_pred_xgb))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_xgb))

# Save models and preprocessing objects
with open('./models/logistic_model_clean.pkl', 'wb') as f:
    pickle.dump(logreg_best, f)

with open('./models/random_forest_model_clean.pkl', 'wb') as f:
    pickle.dump(rf_best, f)

with open('./models/xgboost_model_clean.pkl', 'wb') as f:
    pickle.dump(xgb_best, f)

with open('./models/scaler_clean.pkl', 'wb') as f:
    pickle.dump(scaler, f)

with open('./models/label_encoders_clean.pkl', 'wb') as f:
    pickle.dump(label_encoders, f)

print("\n✅ Clean models, scaler, and label encoders have been saved.")

# Accuracy comparison plot
accuracies = {
    'Logistic Regression': accuracy_score(y_test, y_pred_logreg),
    'Random Forest': accuracy_score(y_test, y_pred_rf),
    'XGBoost': accuracy_score(y_test, y_pred_xgb)
}

plt.figure(figsize=(8, 5))
sns.barplot(x=list(accuracies.keys()), y=list(accuracies.values()))
plt.ylabel('Accuracy')
plt.ylim(0, 1)
plt.title('Model Accuracy Comparison')
plt.xticks(rotation=15)
plt.tight_layout()
plt.show()

# Plot confusion matrices
models = {
    'Logistic Regression': confusion_matrix(y_test, y_pred_logreg),
    'Random Forest': confusion_matrix(y_test, y_pred_rf),
    'XGBoost': confusion_matrix(y_test, y_pred_xgb)
}

for name, cm in models.items():
    plt.figure(figsize=(4, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title(f'{name} Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.show()
