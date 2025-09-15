
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing




st.set_page_config(page_title=" Energy Consumption Forecasting", layout="wide")

st.title(" Energy Consumption Forecasting App")

# Upload CSV file
uploaded_file = st.file_uploader("https://github.com/Varshini2355/AICTE-INTERNSHIP-PROJECT/blob/main/Energy_Consumption_Forecasting.csv")

if uploaded_file is not None:
    # Load dataset
    data = pd.read_csv(uploaded_file)

    st.write("### Step 1: Data Preview")
    st.dataframe(data.head())

    # Select columns
    date_col = st.selectbox("Select Date/Time Column", data.columns)
    consumption_col = st.selectbox("Select Energy Consumption Column", data.columns)

    if date_col and consumption_col:
        st.write("### Step 2: Process Data")
        try:
            data[date_col] = pd.to_datetime(data[date_col])
            data = data.sort_values(by=date_col)
            data = data.set_index(date_col)
            st.success(" Date column converted successfully")
        except Exception as e:
            st.error(f"Error converting date column: {e}")

        st.write("### Step 3: Energy Consumption Over Time")
        fig, ax = plt.subplots(figsize=(10, 4))
        data[consumption_col].plot(ax=ax, label="Energy Consumption")
        plt.legend()
        st.pyplot(fig)

        # Forecasting
        st.write("### Step 4: Forecast Future Energy Consumption")

        n_periods = st.number_input(
            "Enter number of future periods to forecast:",
            min_value=7, max_value=365, value=30
        )

        try:
            # Holt-Winters model
            model = ExponentialSmoothing(
                data[consumption_col],
                trend="add",
                seasonal="add",
                seasonal_periods=12  # adjust depending on dataset frequency
            )
            model_fit = model.fit()

            forecast = model_fit.forecast(n_periods)

            # Plot actual vs forecast
            fig, ax = plt.subplots(figsize=(10, 4))
            data[consumption_col].plot(ax=ax, label="Actual")
            forecast.plot(ax=ax, label="Forecast", color="red")
            plt.legend()
            st.pyplot(fig)

            # Show forecast table
            st.write("### Forecasted Values")
            st.dataframe(forecast.reset_index())

        except Exception as e:
            st.error(f"⚠ Forecasting failed: {e}")
else:
    st.info("https://github.com/Varshini2355/AICTE-INTERNSHIP-PROJECT/blob/main/Energy_Consumption_Forecasting.csv")