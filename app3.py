# import streamlit as st
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# import joblib
# import yfinance as yf
# from datetime import datetime, timedelta

# # Load trained models
# models = {
#     "Random Forest depth6": joblib.load("rf_model_depth6.pkl"),
#     "Random Forest depth10": joblib.load("rf_model_depth10.pkl"),
#     "Decision Tree": joblib.load("dt_model.pkl"),    
# }

# # Extract expected feature names (assumed same across models)
# expected_features = models["Random Forest depth6"].feature_names_in_

# # Define one-hot encoded company columns
# company_columns = ['Name_AAPL', 'Name_AMZN', 'Name_GOOGL', 'Name_IBM', 'Name_JPM']
# company_names = [col.split("_")[1] for col in company_columns]

# st.title("📈 Multi-Model Stock Closing Price Prediction")

# # Sidebar input
# st.sidebar.header("Enter Stock Features")
# selected_company = st.sidebar.selectbox("Select Company", company_names)
# year = st.sidebar.slider("Year", 2005, 2025)
# month = st.sidebar.slider("Month", 1, 12)
# day = st.sidebar.slider("Day", 1, 31)
# auto_fill = st.sidebar.checkbox("Auto-fill from Yahoo Finance")

# # Initialize feature variables
# open_price = high_price = low_price = volume = 0.0
# day_of_week = datetime(year, month, day).weekday()

# if auto_fill:
#     try:
#         date_obj = datetime(year, month, day)
#         end_date = date_obj + timedelta(days=1)
#         start_date = date_obj - timedelta(days=7)

#         ticker_map = {
#             "AAPL": "AAPL",
#             "AMZN": "AMZN",
#             "GOOGL": "GOOGL",
#             "IBM": "IBM",
#             "JPM": "JPM"
#         }

#         ticker_symbol = ticker_map[selected_company]
#         stock = yf.Ticker(ticker_symbol)
#         hist = stock.history(start=start_date.strftime('%Y-%m-%d'), end=end_date.strftime('%Y-%m-%d'))
#         if hist.index.tz is not None:
#             hist.index = hist.index.tz_localize(None)

#         if date_obj in hist.index:
#             row = hist.loc[date_obj]
#         elif not hist.empty:
#             closest_date = hist.index[hist.index <= date_obj][-1]
#             row = hist.loc[closest_date]
#             st.sidebar.info(f"Market closed on selected date. Using data from {closest_date.strftime('%Y-%m-%d')}")
#         else:
#             row = None
#             st.sidebar.warning("No data found in range.")

#         if row is not None:
#             open_price = float(row['Open'])
#             high_price = float(row['High'])
#             low_price = float(row['Low'])
#             volume = float(row['Volume'])

#     except Exception as e:
#         st.sidebar.error(f"Failed to auto-fetch data: {e}")

# # Inputs with pre-filled values
# open_price = st.sidebar.number_input("Open Price", min_value=0.0, value=open_price)
# high_price = st.sidebar.number_input("High Price", min_value=0.0, value=high_price)
# low_price = st.sidebar.number_input("Low Price", min_value=0.0, value=low_price)
# volume = st.sidebar.number_input("Volume", min_value=0.0, value=volume)
# day_of_week = st.sidebar.slider("Day of Week (0=Mon, 6=Sun)", 0, 6, value=day_of_week)

# # Predict button
# if st.sidebar.button("Predict Closing Price"):
#     # One-hot encode company
#     company_data = dict.fromkeys(company_columns, 0)
#     company_col_name = f"Name_{selected_company}"
#     if company_col_name in company_data:
#         company_data[company_col_name] = 1

#     # Create input dictionary
#     input_dict = {
#         'Open': open_price,
#         'High': high_price,
#         'Low': low_price,
#         'Volume': volume,
#         'Year': year,
#         'Month': month,
#         'Day': day,
#         'DayOfWeek': day_of_week,
#     }
#     input_dict.update(company_data)

#     # Convert to DataFrame
#     input_df = pd.DataFrame([input_dict])

#     # Ensure all expected columns are present
#     for col in expected_features:
#         if col not in input_df.columns:
#             input_df[col] = 0

#     # Reorder columns
#     input_df = input_df[expected_features]

#     # Make predictions
#     predictions = {}
#     for model_name, model in models.items():
#         predictions[model_name] = model.predict(input_df)[0]
#         st.success(f"{model_name} Prediction for {selected_company}: ${predictions[model_name]:.2f}")

#     # Actual vs Predicted
#     st.subheader("📊 Actual vs Predicted Closing Prices")

#     try:
#         selected_date_obj = datetime(year, month, day)
#         end_date = selected_date_obj + timedelta(days=1)
#         start_date = selected_date_obj - timedelta(days=7)

#         ticker_map = {
#             "AAPL": "AAPL",
#             "AMZN": "AMZN",
#             "GOOGL": "GOOGL",
#             "IBM": "IBM",
#             "JPM": "JPM"
#         }

#         ticker_symbol = ticker_map[selected_company]
#         stock = yf.Ticker(ticker_symbol)
#         hist = stock.history(start=start_date.strftime('%Y-%m-%d'), end=end_date.strftime('%Y-%m-%d'))

#         hist.index = hist.index.tz_localize(None)

#         if selected_date_obj in hist.index:
#             actual_close = hist.loc[selected_date_obj]['Close']
#             actual_date = selected_date_obj.strftime('%Y-%m-%d')
#         elif not hist.empty:
#             closest_date = hist.index[hist.index <= selected_date_obj][-1]
#             actual_close = hist.loc[closest_date]['Close']
#             actual_date = closest_date.strftime('%Y-%m-%d')
#             st.info(f"⚠️ Market closed on selected date. Showing closest previous date.")
#         else:
#             actual_close = None

#         if actual_close is not None:
#             st.info(f"📅 Actual Closing Price on {actual_date}: ${actual_close:.2f}")

#             # Plot
#             st.subheader("📉 Model-wise Actual vs Predicted Closing Price")
#             fig, axes = plt.subplots(1, len(models), figsize=(18, 5))

#             for i, (model_name, predicted_value) in enumerate(predictions.items()):
#                 ax = axes[i]
#                 ax.bar(["Predicted", "Actual"], [predicted_value, actual_close], color=["blue", "green"])
#                 ax.set_title(model_name, fontsize=10)
#                 ax.set_ylabel("Closing Price ($)")
#                 ax.set_ylim(min(predicted_value, actual_close) * 0.9, max(predicted_value, actual_close) * 1.1)

#             st.pyplot(fig)

#         else:
#             st.warning("No actual data available for selected range.")

#     except Exception as e:
#         st.error(f"Error fetching actual stock price: {e}")

# # Show EDA
# if st.checkbox("Show EDA & Insights"):
#     st.subheader("Feature Importance (Random Forest)")
#     rf_model = models["Random Forest depth6"]
#     importance = rf_model.feature_importances_
#     features = rf_model.feature_names_in_
#     fig, ax = plt.subplots(figsize=(8, 6))
#     sns.barplot(x=importance, y=features, ax=ax)
#     ax.set_title("Feature Importances")
#     st.pyplot(fig)


import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import yfinance as yf
from datetime import datetime, timedelta

# Load trained models
models = {
    "Random Forest depth6": joblib.load("rf_model_depth6.pkl"),
    "Random Forest depth10": joblib.load("rf_model_depth10.pkl"),
    "Decision Tree": joblib.load("dt_model.pkl"),    
}

# Extract expected feature names (assumed same across models)
expected_features = models["Random Forest depth6"].feature_names_in_

# Define one-hot encoded company columns
company_columns = ['Name_AAPL', 'Name_AMZN', 'Name_GOOGL', 'Name_IBM', 'Name_JPM']
company_names = [col.split("_")[1] for col in company_columns]

st.title("📈 Multi-Model Stock Closing Price Prediction")

# Sidebar input
st.sidebar.header("Enter Stock Features")
selected_company = st.sidebar.selectbox("Select Company", company_names)
year = st.sidebar.slider("Year", 2005, 2025)
month = st.sidebar.slider("Month", 1, 12)
day = st.sidebar.slider("Day", 1, 31)
auto_fill = st.sidebar.checkbox("Auto-fill from Yahoo Finance")

# Initialize feature variables
open_price = high_price = low_price = volume = 0.0
day_of_week = datetime(year, month, day).weekday()

if auto_fill:
    try:
        date_obj = datetime(year, month, day)
        end_date = date_obj + timedelta(days=1)
        start_date = date_obj - timedelta(days=7)

        ticker_map = {
            "AAPL": "AAPL",
            "AMZN": "AMZN",
            "GOOGL": "GOOGL",
            "IBM": "IBM",
            "JPM": "JPM"
        }

        ticker_symbol = ticker_map[selected_company]
        stock = yf.Ticker(ticker_symbol)
        hist = stock.history(start=start_date.strftime('%Y-%m-%d'), end=end_date.strftime('%Y-%m-%d'))
        if hist.index.tz is not None:
            hist.index = hist.index.tz_localize(None)

        if date_obj in hist.index:
            row = hist.loc[date_obj]
        elif not hist.empty:
            closest_date = hist.index[hist.index <= date_obj][-1]
            row = hist.loc[closest_date]
            st.sidebar.info(f"Market closed on selected date. Using data from {closest_date.strftime('%Y-%m-%d')}")
        else:
            row = None
            st.sidebar.warning("No data found in range.")

        if row is not None:
            open_price = float(row['Open'])
            high_price = float(row['High'])
            low_price = float(row['Low'])
            volume = float(row['Volume'])

    except Exception as e:
        st.sidebar.error(f"Failed to auto-fetch data: {e}")

# Inputs with pre-filled values
open_price = st.sidebar.number_input("Open Price", min_value=0.0, value=open_price)
high_price = st.sidebar.number_input("High Price", min_value=0.0, value=high_price)
low_price = st.sidebar.number_input("Low Price", min_value=0.0, value=low_price)
volume = st.sidebar.number_input("Volume", min_value=0.0, value=volume)
day_of_week = st.sidebar.slider("Day of Week (0=Mon, 6=Sun)", 0, 6, value=day_of_week)

# Predict button
if st.sidebar.button("Predict Closing Price"):
    # One-hot encode company
    company_data = dict.fromkeys(company_columns, 0)
    company_col_name = f"Name_{selected_company}"
    if company_col_name in company_data:
        company_data[company_col_name] = 1

    # Create input dictionary
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
            input_df[col] = 0

    # Reorder columns
    input_df = input_df[expected_features]

    # Make predictions
    predictions = {}
    for model_name, model in models.items():
        predictions[model_name] = model.predict(input_df)[0]
        st.success(f"{model_name} Prediction for {selected_company}: ${predictions[model_name]:.2f}")

    # Actual vs Predicted
    st.subheader("📊 Actual vs Predicted Closing Prices")

    try:
        selected_date_obj = datetime(year, month, day)
        end_date = selected_date_obj + timedelta(days=1)
        start_date = selected_date_obj - timedelta(days=7)

        ticker_map = {
            "AAPL": "AAPL",
            "AMZN": "AMZN",
            "GOOGL": "GOOGL",
            "IBM": "IBM",
            "JPM": "JPM"
        }

        ticker_symbol = ticker_map[selected_company]
        stock = yf.Ticker(ticker_symbol)
        hist = stock.history(start=start_date.strftime('%Y-%m-%d'), end=end_date.strftime('%Y-%m-%d'))

        if hist.index.tz is not None:
            hist.index = hist.index.tz_localize(None)

        if selected_date_obj in hist.index:
            actual_close = hist.loc[selected_date_obj]['Close']
            actual_date = selected_date_obj.strftime('%Y-%m-%d')
        elif not hist.empty:
            closest_date = hist.index[hist.index <= selected_date_obj][-1]
            actual_close = hist.loc[closest_date]['Close']
            actual_date = closest_date.strftime('%Y-%m-%d')
            st.info(f"⚠️ Market closed on selected date. Showing closest previous date.")
        else:
            actual_close = None

        if actual_close is not None:
            st.info(f"📅 Actual Closing Price on {actual_date}: ${actual_close:.2f}")

            # Plot actual vs predicted for each model
            st.subheader("📉 Model-wise Actual vs Predicted Closing Price")

            model_count = len(models)
            fig, axes = plt.subplots(1, model_count, figsize=(6 * model_count, 5))

            # Ensure axes is iterable
            if model_count == 1:
                axes = [axes]

            for i, (model_name, predicted_value) in enumerate(predictions.items()):
                ax = axes[i]
                ax.bar(["Predicted", "Actual"], [predicted_value, actual_close], color=["blue", "green"])
                ax.set_title(model_name, fontsize=10)
                ax.set_ylabel("Closing Price ($)")
                ax.set_ylim(min(predicted_value, actual_close) * 0.9, max(predicted_value, actual_close) * 1.1)

            st.pyplot(fig)

        else:
            st.warning("No actual data available for selected range.")

    except Exception as e:
        st.error(f"Error fetching actual stock price: {e}")

# Show EDA
if st.checkbox("Show EDA & Insights"):
    st.subheader("Feature Importance (Random Forest)")
    rf_model = models["Random Forest depth6"]
    importance = rf_model.feature_importances_
    features = rf_model.feature_names_in_
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.barplot(x=importance, y=features, ax=ax)
    ax.set_title("Feature Importances")
    st.pyplot(fig)
