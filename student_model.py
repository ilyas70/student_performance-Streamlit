import pickle
import streamlit as st
import pandas as pd
import numpy as np
import sklearn
from sklearn.preprocessing import LabelEncoder




st.title('Student performence Prediction')
st.subheader('Please enter your data:')

# input fields for the user
Hours_Studied = st.number_input("Hours Studied",min_value = 0, max_value = 120, value = 45)

Attendance = st.number_input("Attendance",min_value = 0, max_value = 200, value = 45)

Access_to_Resources_m= st.selectbox("Access_to_Resources_m",["Low","Medium","High"])

Motivation_Level_m = st.selectbox("Motivation_Level_m",["Low","Medium","High"])


# prepare the input data as a dictoary
input_data={
    "Hours_Studied":Hours_Studied,
    "Attendance":Attendance,
    "Access_to_Resources_m":Access_to_Resources_m,
    "Motivation_Level_m":Motivation_Level_m
    }

df = pd.read_csv('features.csv')
columns_list = df.columns.to_list()
# convert input data to dataframe
new_data = pd.DataFrame([input_data])

# load saved labelencoders
lmh={
    'Low':1,
    'Medium':2,
    'High':3
}

new_data['Access_to_Resources_m'] = new_data['Access_to_Resources_m'].map(lmh)
new_data['Motivation_Level_m'] = new_data['Motivation_Level_m'].map(lmh)


# Reindex to match the original column order
new_data = new_data.reindex(columns=columns_list, fill_value=0)


# Load the RandomForest model
with open('linear_model.pkl', 'rb') as model_file:
    loaded_model = pickle.load(model_file)

# Make predictions
prediction = loaded_model.predict(new_data)



if st.button('Predict Exam Score❤️'):
    # Output the prediction
    if prediction[0] >55:
        st.balloons() 
        st.success(f"Pass ! : your score is {prediction[0]}")
    else:
        st.snow()
        st.error(f"Fail ! : your score is {prediction[0]}")
