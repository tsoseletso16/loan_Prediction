# Loan Prediction System Using Machine Learning

## Project Overview

The **Loan Prediction System** is a machine learning-based application developed as part of the **Artificial Intelligence (AI)** course. The primary intent of this project is to automate the process of predicting whether a loan application should be approved or not, using machine learning algorithms. By leveraging the power of machine learning, the system aims to assist financial institutions in making data-driven, faster, and more accurate decisions during the loan approval process.

The model predicts loan approval status based on applicant details like income, credit history, and property area. By eliminating human biases and speeding up the decision-making process, this system will contribute to more efficient loan processing.

## Problem Statement

Traditional loan approval processes are often slow, prone to human error, and involve subjective decision-making. By using machine learning to automate this process, the system helps financial institutions reduce delays, improve consistency, and ultimately enhance customer satisfaction. Automating loan prediction also allows institutions to scale their operations without the need for additional human intervention.

## Dataset

The dataset used for this project was provided by the course instructor and consists of historical loan application data. Key features in the dataset include:

- **Gender** (Male/Female)
- **Marital Status** (Single/Married)
- **Education** (Graduate/Not Graduate)
- **Applicant Income**
- **Loan Amount**
- **Credit History** (Good/Bad)
- **Property Area** (Urban/Semiurban/Rural)
- **Loan Status** (target variable: Approved/Not Approved)

### Dataset Source
- Provided by the course lecturer for academic purposes.
- Similar datasets can be found on Kaggle for similar machine learning projects.

## Intent of the Project

The intent of the project is to create a robust loan prediction system that:

1. Automates the loan approval process by predicting whether a loan should be approved based on the applicant’s data.
2. Compares multiple machine learning models to identify the best model for loan prediction.
3. Provides a clear, user-friendly interface using **Streamlit** that allows stakeholders to input real-time data and receive loan approval predictions.

## Features of the System

- **Loan Approval Prediction:** Predicts whether a loan will be approved or denied based on the input data.
- **Model Comparison:** Allows users to compare the performance of three different machine learning models: Logistic Regression, Random Forest, and XGBoost.
- **Feature Importance Visualization:** Displays a bar chart representing the importance of each feature in predicting loan approval.
- **Confusion Matrix Visualization:** A graphical representation of the model’s performance using confusion matrices.
- **Interactive Streamlit Dashboard:** A web interface built with Streamlit that allows users to interact with the models and visualize predictions.

## Machine Learning Models Used

The system is based on three commonly used classification algorithms:

1. **Logistic Regression:** A simple and widely used algorithm for binary classification tasks like loan approval (approved/not approved).
2. **Random Forest Classifier:** A powerful ensemble learning algorithm that combines multiple decision trees to improve prediction accuracy and handle overfitting.
3. **XGBoost Classifier:** An optimized gradient boosting model known for its performance and efficiency in handling classification tasks.

### Model Evaluation Metrics

- **Accuracy:** Measures the overall accuracy of the model in predicting loan approval.
- **Confusion Matrix:** Visualizes the true positives, true negatives, false positives, and false negatives to understand the performance of the models.
- **Feature Importance:** Highlights which features contribute the most to predicting loan approval, aiding interpretability.

## Technologies Used

- **Python:** The primary programming language used for data processing, modeling, and developing the Streamlit app.
- **Pandas, NumPy:** For data manipulation and analysis.
- **Scikit-learn:** For building, training, and evaluating machine learning models.
- **XGBoost:** For implementing the XGBoost gradient boosting model.
- **Matplotlib, Seaborn:** For creating data visualizations.
- **Streamlit:** For developing an interactive web interface that displays model predictions and performance metrics.
- **GitHub:** For version control and sharing the code.

## Installation Instructions

### Prerequisites

Ensure you have Python 3.6+ installed on your system. You will also need to install the required dependencies.

### Steps to Run the Project

1. **Clone the Repository:**

   First, clone the repository to your local machine using the following command:

   ```bash
   git clone https://github.com/tsoseletso16/loan-Prediction-system.git
