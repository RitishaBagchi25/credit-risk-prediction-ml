# Credit Risk Prediction

Predicts loan default risk (good/bad) using the German Credit dataset (1,000 applicants), with an interactive Streamlit app for real-time predictions.

## Approach
- EDA and feature engineering (age brackets, credit-amount-to-duration ratio, log transform for skew)
- Addressed class imbalance (70:30 good/bad) using SMOTE
- Compared Logistic Regression and XGBoost
- Chose Logistic Regression for deployment — higher recall on defaulters (65% vs 60%), which matters more than raw accuracy in credit risk
- Used SHAP for model explainability at both global and individual-prediction level

## Results
- AUC: 0.73 (Logistic Regression)
- Recall on defaulters: 65%

## How to run
\`\`\`
pip install -r requirements.txt
streamlit run app.py
\`\`\`

## Tech stack
Python, pandas, scikit-learn, imbalanced-learn, XGBoost, SHAP, Streamlit