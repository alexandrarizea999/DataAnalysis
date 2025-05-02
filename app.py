import streamlit as st
import joblib
import numpy as np


# Example: 
# Suppose the model learned the following parameters:
#     intercept = 30,000
#     coef1 = 2,000 (per year at the company)
#     coef2 = 5,000 (per job rate point)
# If the user inputs years = 5 and jobrate = 4.0, the prediction would be:
# prediction = 30,000 + (2,000 * 5) + (5,000 * 4.0)
#            = 30,000 + 10,000 + 20,000
#            = 60,000



st.title("Salary Prediction App")
st.divider()
st.write("With this app, you can get estimations for the salaries of the company employees")

# take the inputs
years = st.number_input("Enter the year at company: ", value = 1, step = 1, min_value=0)
jobrate = st.number_input("Enter the job rate: ", value = float(3.5), step = float(0.5), min_value=0.0)

X = [years, jobrate]
# load the model
model = joblib.load("linearmodel.pkl")
st.divider()

# we add a button for predict
predict = st.button("Press the button for salary prediction")
st.divider()

if predict:
    st.balloons()
    X1 = np.array([X])
    prediction = model.predict(X1)[0]
    st.write(f"Salary prediction: {prediction:,.2f}")
else: "Please press the button for app to make the prediction"