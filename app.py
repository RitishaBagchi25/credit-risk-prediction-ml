import streamlit as st
import pandas as pd
import numpy as np
import pickle

model = pickle.load(open('model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))
columns = pickle.load(open('columns.pkl', 'rb'))

st.title("Credit Risk Predictor")
st.write("Enter applicant details to predict loan risk")

age = st.slider("Age", 18, 75, 30)
sex = st.selectbox("Sex", ["male", "female"])
job = st.selectbox("Job", [0, 1, 2, 3])
housing = st.selectbox("Housing", ["own", "rent", "free"])
saving = st.selectbox("Saving accounts", ["little", "moderate", "quite rich", "rich", "none"])
checking = st.selectbox("Checking account", ["little", "moderate", "rich", "none"])
credit_amount = st.number_input("Credit amount", min_value=100, max_value=20000, value=3000)
duration = st.slider("Duration (months)", 4, 72, 24)
purpose = st.selectbox("Purpose", ["car", "radio/TV", "furniture/equipment", "business", "education", "repairs", "vacation/others", "domestic appliances"])

if st.button("Predict Risk"):
    new_data = pd.DataFrame({
        'Age': [age], 'Sex': [sex], 'Job': [job], 'Housing': [housing],
        'Saving accounts': [saving], 'Checking account': [checking],
        'Credit amount': [credit_amount], 'Duration': [duration], 'Purpose': [purpose]
    })
    new_data['Credit amount'] = np.log(new_data['Credit amount'])
    new_data['Age_cat'] = pd.cut(new_data['Age'], bins=[18,25,35,60,120], labels=['Student','Young','Adult','Senior'])
    new_data['Amount_per_duration'] = new_data['Credit amount'] / new_data['Duration']
    new_data = pd.get_dummies(new_data, columns=['Sex','Housing','Saving accounts','Checking account','Purpose','Age_cat'])
    new_data = new_data.reindex(columns=columns, fill_value=0)
    new_data_scaled = scaler.transform(new_data)

    pred = model.predict(new_data_scaled)[0]
    prob = model.predict_proba(new_data_scaled)[0][1]

    if pred == 1:
        st.error(f"⚠️ Bad Risk — Probability of default: {prob:.1%}")
    else:
        st.success(f"✅ Good Risk — Probability of default: {prob:.1%}")