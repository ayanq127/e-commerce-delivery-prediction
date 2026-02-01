import streamlit as st
import pandas as pd
import joblib


model = joblib.load("delivery_model.pkl")
scaler = joblib.load("scaler.pkl")
feature_order = joblib.load("feature_order.pkl")

st.set_page_config(page_title="E-Commerce Delivery Prediction", layout="centered")

st.title(" E-Commerce Product Delivery Prediction")
st.info(" Enter order details and click **Predict Delivery Status**")



warehouse = st.selectbox("Warehouse Block", [0, 1, 2, 3, 4])
shipment = st.selectbox("Mode of Shipment", [0, 1, 2])
importance = st.selectbox("Product Importance", [0, 1, 2])
gender = st.selectbox("Gender", [0, 1])

calls = st.slider("Customer Care Calls", 0, 10)
rating = st.slider("Customer Rating", 1, 5)
cost = st.number_input("Cost of Product", min_value=1)
prior = st.number_input("Prior Purchases", min_value=0)
discount = st.slider("Discount Offered", 0, 100)
weight = st.number_input("Product Weight (grams)", min_value=1)



discount_ratio = discount / (cost + 1)

if weight <= 1000:
    weight_category = 0
elif weight <= 3000:
    weight_category = 1
else:
    weight_category = 2



if st.button("🔍 Predict Delivery Status"):

    input_dict = {
        "Warehouse_block": warehouse,
        "Mode_of_Shipment": shipment,
        "Product_importance": importance,
        "Gender": gender,
        "Customer_care_calls": calls,
        "Customer_rating": rating,
        "Cost_of_the_Product": cost,
        "Prior_purchases": prior,
        "Discount_offered": discount,
        "Weight_in_gms": weight,
        "discount_ratio": discount_ratio,
        "weight_category": weight_category
    }

    input_data = pd.DataFrame([input_dict])

  
    input_data = input_data[feature_order]

  
    input_scaled = scaler.transform(input_data)

  
    prediction = model.predict(input_scaled)[0]

    if prediction == 0:
        st.success(" Product will be delivered **ON TIME**")
    else:
        st.error(" Product is likely to be **DELAYED**")
