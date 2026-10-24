# Credit Card Fraud Detection

## Project Overview

The objective of this project is to detect fraudulent credit card transactions using machine learning techniques.

## Business Problem

Credit card fraud can cause significant financial losses for banks and their customers.

Detecting fraudulent transactions is essential to reduce financial risks and protect customers.

The main challenge is to identify as many fraudulent transactions as possible while minimizing false alarms.

## Dataset

The dataset was obtained from Kaggle and contains 284,807 credit card transactions.

The target variable is `Class`:
- `0`: Legitimate transaction
- `1`: Fraudulent transaction

The dataset is highly imbalanced, with fraudulent transactions representing approximately 0.17% of all transactions.

## Data Preprocessing

- Checked for missing values.
- Removed duplicate transactions using `drop_duplicates(keep='first')`.
- Split the dataset into training, validation, and test sets using stratified sampling.
- Applied `StandardScaler` to standardize the features.
- Used SMOTE to balance the training data only.

## Machine Learning Models

Two machine learning algorithms were evaluated:

- Logistic Regression
- Random Forest

To address class imbalance, two approaches were explored:
- Logistic Regression with `class_weight="balanced"`
- Logistic Regression with SMOTE

Different decision thresholds were also tested to minimize the estimated financial cost of classification errors.

## Model Results

The final Logistic Regression model with SMOTE was evaluated on the test dataset using a decision threshold of 0.99.

| Metric | Result |
|---|---|
| Fraud Detection Recall | 78.95% |
| Fraud Detection Precision | 55.56% |
| F1-Score | 65% |
| Fraudulent Transactions Detected | 75 |
| Fraudulent Transactions Missed | 20 |
| False Positives | 60 |
| Estimated Financial Cost | €21,200 |

### Confusion Matrix

| | Predicted Legitimate | Predicted Fraud |
|---|---:|---:|
| Actual Legitimate | 56,591 | 60 |
| Actual Fraud | 20 | 75 |

The model detected 75 out of 95 fraudulent transactions.

Although the model achieved a recall above 85% on the validation set, it did not reach the target on the test set.

## Model Limitations

The model's performance must be evaluated according to the bank's business requirements.

Although the model achieved promising results, its recall on the test set was 78.95%, below the bank's target of 85%.

This means that some fraudulent transactions remain undetected.

Future improvements could include:
- Testing other machine learning algorithms.
- Optimizing model hyperparameters.
- Exploring additional techniques for handling class imbalance.
- Improving fraud detection while reducing false positives.

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Imbalanced-learn (SMOTE)
- Matplotlib
- Google Colab

## Conclusion

This project demonstrates how machine learning can be applied to credit card fraud detection.

It highlights the challenges of imbalanced datasets and the importance of evaluating models using business-oriented metrics rather than accuracy alone.
