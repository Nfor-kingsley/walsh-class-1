
import streamlit as st
import pandas as pd
import joblib
import os

# Set page title
st.set_page_config(page_title="Wellness Tourism Package Prediction")

st.title("Wellness Tourism Package Purchase Prediction")
st.write("This app predicts whether a customer will purchase the Wellness Tourism Package.")

# Load the trained model
model_path = "model.joblib"
if os.path.exists(model_path):
    model = joblib.load(model_path)
else:
    st.error("Model file not found. Please ensure model.joblib is in the same directory.")

# Create input fields for features
st.header("Customer Information")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=18, max_value=100, value=35)
    type_of_contact = st.selectbox("Type of Contact", ["Self Enquiry", "Company Invited"])
    city_tier = st.selectbox("City Tier", [1, 2, 3])
    occupation = st.selectbox("Occupation", ["Salaried", "Small Business", "Large Business", "Free Lancer"])
    gender = st.selectbox("Gender", ["Male", "Female"])
    num_person = st.number_input("Number of Person Visiting", min_value=1, max_value=10, value=2)
    prop_star = st.selectbox("Preferred Property Star", [3, 4, 5])
    marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced", "Unmarried"])

with col2:
    num_trips = st.number_input("Number of Trips", min_value=1, max_value=20, value=3)
    passport = st.selectbox("Has Passport?", [0, 1])
    own_car = st.selectbox("Owns Car?", [0, 1])
    num_children = st.number_input("Number of Children Visiting", min_value=0, max_value=5, value=0)
    designation = st.selectbox("Designation", ["Manager", "Executive", "Senior Manager", "AVP", "VP"])
    monthly_income = st.number_input("Monthly Income", min_value=0, value=25000)
    pitch_score = st.slider("Pitch Satisfaction Score", 1, 5, 3)
    prod_pitched = st.selectbox("Product Pitched", ["Basic", "Deluxe", "Standard", "Super Deluxe", "King"])
    followups = st.number_input("Number of Followups", 1, 10, 3)
    duration = st.number_input("Duration of Pitch", 1, 120, 15)

# Prediction button
if st.button("Predict"):
    input_data = pd.DataFrame([{
        "Age": age, "TypeofContact": type_of_contact, "CityTier": city_tier,
        "Occupation": occupation, "Gender": gender, "NumberOfPersonVisiting": num_person,
        "PreferredPropertyStar": prop_star, "MaritalStatus": marital_status,
        "NumberOfTrips": num_trips, "Passport": passport, "OwnCar": own_car,
        "NumberOfChildrenVisiting": num_children, "Designation": designation,
        "MonthlyIncome": monthly_income, "PitchSatisfactionScore": pitch_score,
        "ProductPitched": prod_pitched, "NumberOfFollowups": followups,
        "DurationOfPitch": duration
    }])

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    if prediction == 1:
        st.success(f"Targeted: The customer is likely to purchase the package. (Probability: {probability:.2f})")
    else:
        st.warning(f"Not Targeted: The customer is unlikely to purchase the package. (Probability: {probability:.2f})")
