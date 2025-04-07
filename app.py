import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Load trained model
model = joblib.load("rf_model.pkl")  # Trained on one-hot encoded company names

# Extract expected feature names from model
expected_features = model.feature_names_in_

# Define one-hot encoded company columns
company_columns = ['Name_AAPL', 'Name_AMZN', 'Name_GOOGL', 'Name_IBM', 'Name_JPM']
company_names = [col.split("_")[1] for col in company_columns]

st.title("📈 Stock Closing Price Prediction")

# Sidebar input
st.sidebar.header("Enter Stock Features")
selected_company = st.sidebar.selectbox("Select Company", company_names)

open_price = st.sidebar.number_input("Open Price", min_value=0.0)
high_price = st.sidebar.number_input("High Price", min_value=0.0)
low_price = st.sidebar.number_input("Low Price", min_value=0.0)
volume = st.sidebar.number_input("Volume", min_value=0.0)
year = st.sidebar.slider("Year", 2005, 2025)
month = st.sidebar.slider("Month", 1, 12)
day = st.sidebar.slider("Day", 1, 31)
day_of_week = st.sidebar.slider("Day of Week (0=Mon, 6=Sun)", 0, 6)

# Predict button
if st.sidebar.button("Predict Closing Price"):
    # Start with all company one-hot columns as 0
    company_data = dict.fromkeys(company_columns, 0)
    company_col_name = f"Name_{selected_company}"
    if company_col_name in company_data:
        company_data[company_col_name] = 1

    # Create input dictionary with all features
    input_dict = {
        'Open': open_price,
        'High': high_price,
        'Low': low_price,
        'Volume': volume,
        'Year': year,
        'Month': month,
        'Day': day,
        'DayOfWeek': day_of_week,
    }
    input_dict.update(company_data)

    # Convert to DataFrame
    input_df = pd.DataFrame([input_dict])

    # Ensure all expected columns are present
    for col in expected_features:
        if col not in input_df.columns:
            input_df[col] = 0  # Fill missing with 0 (e.g., for unseen one-hot columns)

    # Reorder columns to match model training
    input_df = input_df[expected_features]

    # Predict
    prediction = model.predict(input_df)[0]
    st.success(f"Predicted Closing Price for {selected_company}: ${prediction:.2f}")

    # Show chart
    st.subheader("Prediction Input Overview")
    st.bar_chart(input_df.T)

# Show EDA
if st.checkbox("Show EDA & Insights"):
    st.subheader("Feature Importance")
    importance = model.feature_importances_
    features = model.feature_names_in_
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.barplot(x=importance, y=features, ax=ax)
    st.pyplot(fig)
