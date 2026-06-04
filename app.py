import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# Load model
model = joblib.load("car_price_model.pkl")

st.title("🚗 Car Price Prediction Dashboard")

# User Inputs
name = st.selectbox(
    "Car Brand",
    ["Maruti","Hyundai","Toyota","Ford"]
)

year = st.number_input(
    "Manufacturing Year",
    2000,
    2025,
    2018
)

km_driven = st.number_input(
    "Kilometers Driven",
    1000,
    500000,
    50000
)

fuel = st.selectbox(
    "Fuel Type",
    ["Petrol","Diesel","CNG","LPG"]
)

seller_type = st.selectbox(
    "Seller Type",
    ["Individual","Dealer"]
)

transmission = st.selectbox(
    "Transmission",
    ["Manual","Automatic"]
)

owner = st.selectbox(
    "Owner",
    ["First Owner","Second Owner"]
)

seats = st.number_input(
    "Seats",
    2,
    10,
    5
)

power = st.number_input(
    "Max Power",
    30.0,
    300.0,
    80.0
)

mileage_unit = st.selectbox(
    "Mileage Unit",
    ["kmpl"]
)

mileage = st.number_input(
    "Mileage",
    5.0,
    40.0,
    20.0
)

engine = st.number_input(
    "Engine CC",
    500,
    5000,
    1200
)

# Predict Button
if st.button("Predict Price"):

    input_data = pd.DataFrame({
        "name":[name],
        "year":[year],
        "km_driven":[km_driven],
        "fuel":[fuel],
        "seller_type":[seller_type],
        "transmission":[transmission],
        "owner":[owner],
        "seats":[seats],
        "max_power (in bph)":[power],
        "Mileage Unit":[mileage_unit],
        "Mileage":[mileage],
        "Engine (CC)":[engine]
    })

    prediction = model.predict(input_data)

    st.success(
        f"Estimated Car Price: ₹ {prediction[0]:,.0f}"
    )

    # Chart
    fig, ax = plt.subplots()

    ax.bar(
        ["Predicted Price"],
        [prediction[0]]
    )

    ax.set_ylabel("Price")

    st.pyplot(fig)