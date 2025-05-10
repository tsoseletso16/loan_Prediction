# Loan Prediction System Using Machine Learning

## Project Overview

This project is a Loan Prediction System developed as part of the **Artificial Intelligence (AI)** course.
It utilizes machine learning algorithms to predict whether a loan application should be approved based on various applicant information.
The goal is to assist financial institutions in making more accurate and data-driven decisions during the loan approval process.

## Problem Statement

Loan approval processes are traditionally based on manual judgment, which can be slow, inconsistent,
and prone to human errors. By automating the loan prediction process using machine learning,
this system aims to reduce delays and increase consistency in decision-making,
ultimately helping financial institutions make faster and more accurate decisions.

## Dataset

The dataset used for this project was provided by the course instructor and contains various features about loan applicants,
such as:

- **Gender** (Male/Female)
- **Marital Status** (Single/Married)
- **Education** (Graduate/Not Graduate)
- **Applicant Income**
- **Loan Amount**
- **Credit History** (Good/Bad)
- **Property Area** (Urban/Semiurban/Rural)
- **Loan Status** (target variable: Approved/Not Approved)

### Dataset Source
- The dataset was provided by the course lecturer.

## Features of the System

The Loan Prediction System includes the following features:

- **Loan Approval Prediction:** Predicts whether a loan should be approved (Yes/No) based on applicant data.
- **Model Comparison:** Compares the performance of three machine learning models: Logistic Regression, Random Forest, and XGBoost.
- **Feature Importance Visualization:** Displays a graphical representation of feature importance based on the trained models.
- **Streamlit Dashboard:** An interactive web interface built with Streamlit to allow users to input data and see predictions in real time.

## Machine Learning Models Used

The following machine learning models were trained and evaluated for the loan prediction task:

1. **Logistic Regression**  
   A fundamental classification algorithm that models the relationship between the features and the probability of the target class.

2. **Random Forest Classifier**  
   A robust ensemble learning method that combines multiple decision trees to improve accuracy and avoid overfitting.

3. **XGBoost Classifier**  
   A highly effective gradient boosting model that has proven to be extremely powerful for classification tasks.

### Model Evaluation Metrics

- **Accuracy:** Measures the percentage of correct predictions.
- **Confusion Matrix:** Shows the true positives, true negatives, false positives, and false negatives.
- **Feature Importance:** Identifies the most important features in predicting loan approval.

## Technologies Used

- **Python:** Programming language for implementing the machine learning models and web interface.
- **Pandas, NumPy:** Libraries for data manipulation and analysis.
- **Scikit-learn:** For building and evaluating machine learning models.
- **XGBoost:** For the advanced gradient boosting model.
- **Matplotlib, Seaborn:** For data visualization.
- **Streamlit:** For creating an interactive web dashboard to visualize predictions and model performance.
- **GitHub:** For version control and project management.

## Installation Instructions

### Prerequisites

To run this project on your local machine, ensure you have Python 3.6+ installed. You also need to install the required libraries.

### Steps to Run the Project

1. **Clone the Repository:**

   Open a terminal or command prompt and run the following command to clone the repository:

   ```bash
   git clone https://github.com/tsoseletso16/loan-prediction-system.git
