import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load model and cleaned dataset
model = joblib.load("optimized_rf_model.pkl")
df = pd.read_csv("cleaned_sales_data.csv", parse_dates=['Date'])  # Preprocessed data

st.title("📊 Sales Prediction App")

# Inputs
input_date = st.date_input("Select Date")
input_country = st.selectbox("Select Country", sorted(df['Country'].unique()))
input_segment = st.selectbox("Select Segment", sorted(df['Segment'].unique()))

# Prediction logic
if st.button("Predict Sales"):
    # Filter historical data
    history = df[(df['Country'] == input_country) & (df['Segment'] == input_segment)]
    history = history[history['Date'] < pd.to_datetime(input_date)].sort_values(by='Date')

    if len(history) < 3:
        st.warning("⚠️ Not enough historical data to compute rolling features.")
    else:
        # Feature Engineering
        lag_1 = history.iloc[-1]['Sales Amount']
        rolling_avg = history['Sales Amount'].iloc[-3:].mean()
        quarter = pd.to_datetime(input_date).quarter
        is_q4 = 1 if quarter == 4 else 0

        input_features = np.array([[lag_1, rolling_avg, is_q4]])
        prediction = model.predict(input_features)[0]

        # Output Prediction
        st.success(f"✅ Predicted Sales Amount: {prediction:.2f}")

        # Show Feature Values
        st.subheader("📌 Features Used in Prediction")
        st.write(f"**Lag 1 Sales:** {lag_1:.2f}")
        st.write(f"**Rolling Average (Last 3 Days):** {rolling_avg:.2f}")
        st.write(f"**Is Quarter 4:** {'Yes' if is_q4 == 1 else 'No'}")

        # Line Chart of Historical Sales
        st.subheader("📈 Historical Sales Trend")
        sales_trend = history[['Date', 'Sales Amount']].set_index('Date')
        st.line_chart(sales_trend)

        # Show feature importances (if model supports it)
        if hasattr(model, 'feature_importances_'):
            st.subheader("🔍 Feature Importance (Random Forest)")
            features = ['Lag_1_Sales', 'RollingAvg_3M', 'Is_Q4']
            importances = model.feature_importances_
            importance_df = pd.DataFrame({'Feature': features, 'Importance': importances})
            st.bar_chart(importance_df.set_index('Feature'))

