import streamlit as st
import pandas as pd
import joblib

#load modal 
modal = joblib.load('heart_attack_modal.pkl')
scaler = joblib.load('scaler.pkl')
expected_columns = joblib.load('columns.pkl')

st.title("Heart strok Prediction by AI")
st.markdown("provide the following details to cheak youe heart stroke risk:")

#collect user input
age = st.slider("Age", 18,100,40)
sex = st.selectbox("Sex",["MALE","FEMALE"])
chest_pain = st.selectbox("Chests Pain Type",["ATS","NAP","TA","ASY"])
resting_bp = st.number_input("Resting Blood Pressure (mm Hg)",80 , 100,100)
cholesterol = st.number_input("cholesterol (mg/dL)",100,600,200)
fasting_bs = st.selectbox("Fasting Blood suger > 120 mg/dl", [0,1])
resting_ecg = st.selectbox("Resting ECG",["Normal", "ST","LVH"])
max_hr  = st.slider("Max Heart Rate",60,220,150)
exercise_angina = st.selectbox("Exercise_Induced Anging",["Y","N"])
oldpeak = st.slider("Oldpeak (ST depression)",0.0,6.+0,1.0)
st_slope = st.selectbox("ST Slope",["UP","Flat","Down"])

#when predict is clicked

if st.button("Predict"):

    #create a raw input dictionary
    raw_input = {
        'Age':age,
        'ReatingBP': resting_bp,
        'cholesterol': cholesterol,
        'FastingBS' : fasting_bs,
        'MaxHR' : max_hr,
        'Oldpeak': oldpeak,
        'Sex_'+ sex:1,
        'ChestPainType_'+ chest_pain:1,
        'RestingECG_' + exercise_angina:1,
        'ST_Slope_' + st_slope:1
    }

    #create input datafram
    input_df = pd.DataFrame([raw_input])

    # Fill in missing columns with 0s
    for col in expected_columns:
        if col not in input_df.columns:
            input_df[col] = 0

    #Reorder colums 
    input_df = input_df[expected_columns]

    # scale the input
    scaled_input = scaler.transform(input_df)

    # Make prediction 
    prediction = modal.predict(scaled_input)[0]

    #show result
    if prediction == 1:
        st.error("High Risk of Heat Disease")
    else:
        st.success("Low Risk of Heat Disease")