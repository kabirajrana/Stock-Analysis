import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from streamlit_option_menu import option_menu
from streamlit_lottie import st_lottie
import requests

st.set_page_config(page_title="Stock Market Investment Prediction", layout="wide")


@st.cache_data
def load_lottie_url(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()


lottie_stock = load_lottie_url("https://assets7.lottiefiles.com/packages/lf20_3rwasyjy.json")

# Load models
model1 = joblib.load('model1.pkl')  # Closing price prediction
model2 = joblib.load('model2.pkl')  # Volume classification
rf_model = joblib.load('rf_model.pkl')  # Feature importance
model3 = joblib.load('model3.pkl')  # Feature ranking
model4 = joblib.load('model4.pkl')  # Trade prediction

# Load dataset
df = pd.read_csv('Stock_cleaned_data.csv')

with st.sidebar:
    selected = option_menu(
        "Stock Market ML App",
        ["Home", "Price Prediction", "Volume Classification", "Feature Importance", "Feature Ranking",
         "Trade Prediction", "About"],
        icons=["house", "graph-up", "bar-chart", "list-task", "trophy", "arrow-up-right", "info-circle"],
        menu_icon="cast",
        default_index=0,
    )

st.markdown("<h1 style='text-align: center;'>📈 Stock Market Investment Prediction Using ML</h1>",
            unsafe_allow_html=True)

if selected == "Home":
    st_lottie(lottie_stock, speed=1, height=300, key="stock_animation")
    st.subheader("Welcome to the Stock Market Prediction Dashboard")
    st.markdown("""
    This project showcases 5 Machine Learning use cases on stock market data:
    1. 📉 Predict closing stock price
    2. 📊 Classify high vs low trading volume days
    3. 🔥 Feature importance (Random Forest)
    4. 🏆 Rank key features affecting price
    5. 📈 Predict total trades for upcoming days
    """)
    st.markdown("### Preview Dataset")
    st.dataframe(df.head())

elif selected == "Price Prediction":
    st.subheader("🔮 Predict Closing Stock Price")

    try:
        features = ['high_price', 'low_price', 'total_traded_quantity', 'total_traded_value', 'total_trades',
                    'day', 'month', 'weekday', 'volume_class', 'anomaly_label']
        X = df[features]
        y_actual = df['close_price']
        y_pred = model1.predict(X)

        # Show actual vs predicted line chart
        chart_df = pd.DataFrame({"Actual": y_actual, "Predicted": y_pred})
        st.line_chart(chart_df)

        # Buy / Sell / Hold recommendation based on predicted price change
        pred_diff = np.diff(y_pred)
        recommendation = []
        for diff in pred_diff:
            if diff > 0.5:  # Threshold for price increase
                recommendation.append("Buy")
            elif diff < -0.5:  # Threshold for price decrease
                recommendation.append("Sell")
            else:
                recommendation.append("Hold")
        recommendation.insert(0, "Hold")  # First day no diff

        rec_df = pd.DataFrame({
            "Date": df['business_date'],
            "Actual Close Price": y_actual,
            "Predicted Close Price": y_pred,
            "Recommendation": recommendation
        })

        st.markdown("### Buy/Sell/Hold Recommendations Based on Predicted Price Movement")
        st.dataframe(rec_df.tail(20).set_index("Date"))

    except Exception as e:
        st.error(f"❌ Error in prediction: {e}")

elif selected == "Volume Classification":
    st.subheader("📊 Classify High vs Low Volume Trading Days")
    try:
        features = ['high_price', 'low_price', 'total_traded_value', 'total_trades',
                    'day', 'month', 'weekday', 'anomaly_label']
        X = df[features]
        y_pred = model2.predict(X)
        fig, ax = plt.subplots()
        sns.countplot(x=y_pred, ax=ax)
        ax.set_title("Predicted Volume Class")
        st.pyplot(fig)
    except Exception as e:
        st.error(f"❌ Error in classification: {e}")

elif selected == "Feature Importance":
    st.subheader("🔥 Feature Importance")
    try:
        importance = rf_model.feature_importances_
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric_cols) != len(importance):
            numeric_cols = numeric_cols[:len(importance)]

        feat_df = pd.DataFrame({"Feature": numeric_cols, "Importance": importance})
        feat_df = feat_df.sort_values(by="Importance", ascending=False)
        st.bar_chart(feat_df.set_index("Feature"))
    except Exception as e:
        st.error(f"❌ Error in feature importance: {e}")

elif selected == "Feature Ranking":
    st.subheader("🏆 Feature Ranking by Importance")
    try:
        importance = model3.feature_importances_
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric_cols) != len(importance):
            numeric_cols = numeric_cols[:len(importance)]

        feat_df = pd.DataFrame({"Feature": numeric_cols, "Ranking": importance})
        feat_df = feat_df.sort_values(by="Ranking", ascending=False)

        fig, ax = plt.subplots()
        sns.barplot(x="Ranking", y="Feature", data=feat_df, ax=ax)
        ax.set_title("Ranked Features")
        st.pyplot(fig)
    except Exception as e:
        st.error(f"❌ Error in ranking features: {e}")

elif selected == "Trade Prediction":
    st.subheader("📈 Predict Total Trades for Upcoming Days")

    try:
        days_ahead = st.number_input("Enter number of upcoming days to predict", min_value=1, max_value=30, value=5)

        last_date = pd.to_datetime(df['business_date']).max()
        future_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=days_ahead)

        future_df = pd.DataFrame()
        future_df['business_date'] = future_dates
        future_df['day'] = future_df['business_date'].dt.day
        future_df['month'] = future_df['business_date'].dt.month
        future_df['weekday'] = future_df['business_date'].dt.weekday

        recent_days = 10
        recent_data = df.tail(recent_days)

        future_df['high_price'] = recent_data['high_price'].mean()
        future_df['low_price'] = recent_data['low_price'].mean()
        future_df['close_price'] = recent_data['close_price'].mean()
        future_df['total_traded_quantity'] = recent_data['total_traded_quantity'].mean()
        future_df['total_traded_value'] = recent_data['total_traded_value'].mean()
        future_df['prev_total_trades'] = recent_data['total_trades'].iloc[-1]
        future_df['avg_traded_qty_3'] = recent_data['total_traded_quantity'].rolling(3).mean().iloc[-1]

        features = ['high_price', 'low_price', 'close_price', 'total_traded_quantity', 'total_traded_value',
                    'day', 'month', 'weekday', 'prev_total_trades', 'avg_traded_qty_3']

        X_future = future_df[features]

        y_pred = model4.predict(X_future)

        pred_df = pd.DataFrame({
            'Date': future_df['business_date'].dt.strftime('%Y-%m-%d'),
            'Predicted Total Trades': y_pred
        })

        st.line_chart(pred_df.set_index('Date'))
        st.dataframe(pred_df)

    except Exception as e:
        st.error(f"❌ Error in trade prediction: {e}")

elif selected == "About":
    st.subheader("📬 About this Project")
    st.markdown("""
    Created by **Kabiraj Rana**, Data Science Student 🧑‍💻  
    This dashboard presents 5 advanced stock market ML use cases:
    - 📉 Stock Price Forecasting (XGBoost)
    - 📊 Volume Classification (Logistic Regression)
    - 🔥 Feature Analysis (Random Forest)
    - 🏆 Feature Ranking
    - 📈 Trade Volume Forecasting
    """)

