import streamlit as st
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import requests
import os
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("OPENWEATHER_API_KEY")

st.set_page_config(page_title="Smart AQI System", layout="wide")

# ================= REAL-TIME AQI FUNCTION =================

def get_live_aqi(city_name):
    try:
        geo_url = f"https://api.openweathermap.org/geo/1.0/direct?q={city_name}&limit=1&appid={API_KEY}"
        geo_response = requests.get(geo_url).json()

        if len(geo_response) == 0:
            return None

        lat = geo_response[0]['lat']
        lon = geo_response[0]['lon']

        aqi_url = f"https://api.openweathermap.org/data/2.5/air_pollution?lat={lat}&lon={lon}&appid={API_KEY}"
        aqi_response = requests.get(aqi_url).json()

        live_raw = aqi_response['list'][0]['main']['aqi']

        scale_conversion = {
            1: 25,
            2: 75,
            3: 150,
            4: 250,
            5: 350
        }

        return scale_conversion.get(live_raw, None)

    except:
        return None


# ================= AQI CATEGORY FUNCTION =================
def get_aqi_category(aqi_value):
    if aqi_value <= 50:
        return "Good", "#2ecc71"
    elif aqi_value <= 100:
        return "Moderate", "#f1c40f"
    elif aqi_value <= 200:
        return "Unhealthy", "#e67e22"
    elif aqi_value <= 300:
        return "Very Unhealthy", "#e74c3c"
    else:
        return "Hazardous", "#2c3e50"


# ================= LOAD ARTIFACTS =================
model = joblib.load("models/best_model.pkl")
feature_columns = joblib.load("models/feature_columns.pkl")
le = joblib.load("models/city_encoder.pkl")
metrics = joblib.load("models/model_metrics.pkl")

df = pd.read_csv("data/air_quality_data.csv")
df['Date'] = pd.to_datetime(df['Date'])

# ================= HEADER =================
st.markdown("""
<div style="text-align:center; padding:20px;">
    <h1 style="color:#3ffa3c;">🌍 Smart AQI Monitoring & Health Advisory System</h1>
    <p style="color:#bdc3c7; font-size:18px;">
        Real-Time Air Quality Intelligence Platform
    </p>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

st.sidebar.markdown("## 🌫 Smart AQI System")
page = st.sidebar.radio("", ["Public Dashboard", "Advanced Analytics"])

# ==========================================================
# PUBLIC DASHBOARD
# ==========================================================
if page == "Public Dashboard":

    # -------- AQI LEGEND --------
    st.markdown("## 📊 AQI Color Guide")

    st.markdown("""
    <div style="display:flex; justify-content:space-between; padding:10px;">
    <div style="text-align:center;">
    <div style="width:20px;height:20px;background:#2ecc71;border-radius:50%;margin:auto;"></div>
    0–50<br>Good
    </div>
    <div style="text-align:center;">
    <div style="width:20px;height:20px;background:#f1c40f;border-radius:50%;margin:auto;"></div>
    51–100<br>Moderate
    </div>
    <div style="text-align:center;">
    <div style="width:20px;height:20px;background:#e67e22;border-radius:50%;margin:auto;"></div>
    101–200<br>Unhealthy
    </div>
    <div style="text-align:center;">
    <div style="width:20px;height:20px;background:#e74c3c;border-radius:50%;margin:auto;"></div>
    201–300<br>Very Unhealthy
    </div>
    <div style="text-align:center;">
    <div style="width:20px;height:20px;background:#2c3e50;border-radius:50%;margin:auto;"></div>
    300+<br>Hazardous
    </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    city_names = le.classes_
    selected_city = st.selectbox("📍 Select City", city_names)

    if st.button("Check Air Quality"):

        city_encoded = le.transform([selected_city])[0]
        live_aqi = get_live_aqi(selected_city)

        city_data = df[df['City'] == selected_city]

        if city_data.empty:
            st.error("No data available for selected city.")
        else:
            latest_data = city_data.sort_values("Date").iloc[-1]

            input_features = []
            for col in feature_columns:
                if col == "City":
                    input_features.append(city_encoded)
                else:
                    input_features.append(latest_data[col])

            input_array = np.array(input_features).reshape(1, -1)
            prediction = model.predict(input_array)
            predicted_aqi = int(prediction[0])

            predicted_status, predicted_color = get_aqi_category(predicted_aqi)

            if live_aqi:
                live_status, live_color = get_aqi_category(live_aqi)


            # AQI Card
            if predicted_aqi <= 50:
                color = "#2ecc71"
                status = "Good"
            elif predicted_aqi <= 100:
                color = "#f1c40f"
                status = "Moderate"
            elif predicted_aqi <= 200:
                color = "#e67e22"
                status = "Unhealthy"
            elif predicted_aqi <= 300:
                color = "#e74c3c"
                status = "Very Unhealthy"
            else:
                color = "#2c3e50"
                status = "Hazardous"

            st.markdown(f"""
            <div style="
                background-color:{color};
                padding:30px;
                border-radius:15px;
                text-align:center;
                color:white;
                font-size:30px;
                font-weight:bold;">
                🤖 Predicted AQI in {selected_city}: {predicted_aqi} <br>
                Status: {status}
            </div>
            """, unsafe_allow_html=True)

            # -------- AQI CARDS --------
            st.markdown("## 🌍 AQI Overview")
            col1, col2 = st.columns(2)

            with col1:
                st.markdown(f"""
                <div style="
                    background: linear-gradient(135deg, {predicted_color}, #2c3e50);
                    padding:30px;
                    border-radius:20px;
                    text-align:center;
                    color:white;
                    box-shadow: 0px 4px 20px rgba(0,0,0,0.3);
                ">
                    <h3>🤖 Predicted AQI</h3>
                    <h1 style="font-size:55px;">{predicted_aqi}</h1>
                    <h4>{predicted_status}</h4>
                </div>
                """, unsafe_allow_html=True)

            with col2:
                if live_aqi:
                    st.markdown(f"""
                    <div style="
                        background: linear-gradient(135deg, {live_color}, #34495e);
                        padding:30px;
                        border-radius:20px;
                        text-align:center;
                        color:white;
                        box-shadow: 0px 4px 20px rgba(0,0,0,0.3);
                    ">
                        <h3>📡 Live AQI</h3>
                        <h1 style="font-size:55px;">{live_aqi}</h1>
                        <h4>{live_status}</h4>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.warning("Live AQI unavailable.")

            # -------- DIFFERENCE --------
            if live_aqi:
                difference = abs(live_aqi - predicted_aqi)
                st.markdown("### 📊 Model Accuracy Check")
                st.metric("Difference (Predicted vs Live)", difference)

            # -------- HEALTH ADVISORY --------
# -------- HEALTH ADVISORY (Based on LIVE AQI) --------
                st.markdown("## 🏥 Health Advisory")

                advisory_value = live_aqi if live_aqi else predicted_aqi

                if advisory_value <= 50:
                    advice = """
                    ✅ Air quality is GOOD.
                    • Safe for outdoor activities
                    • Ideal for jogging & exercise
                    • No special precautions needed
                    """

                elif advisory_value <= 100:
                    advice = """
                    ⚠ Moderate air quality.
                    • Sensitive people should reduce prolonged outdoor exposure
                    • Carry mask if needed
                    """

                elif advisory_value <= 200:
                    advice = """
                    🚨 Unhealthy air quality.
                    • Wear N95 mask outdoors
                    • Avoid heavy exercise outside
                    • Children & elderly stay indoors
                    """

                elif advisory_value <= 300:
                    advice = """
                    ❗ Very Unhealthy.
                    • Avoid outdoor activities
                    • Keep windows closed
                    • Use air purifier if available
                    """

                else:
                    advice = """
                    ☠ Hazardous air quality.
                    • Stay indoors completely
                    • Avoid physical activity
                    • Seek medical help if breathing discomfort occurs
                    """

            st.markdown(f"""
            <div style="
                background-color:#1f2c3a;
                padding:25px;
                border-radius:15px;
                color:#ecf0f1;
                font-size:18px;
                line-height:1.6;
            ">
            {advice}
            </div>
            """, unsafe_allow_html=True)


                # -------- TREND --------
            st.markdown("## 📈 Last 30 Days AQI Trend")
            trend_data = city_data.sort_values("Date").tail(30)
            fig = px.line(trend_data, x="Date", y="AQI")
            fig.update_layout(template="plotly_white")
            st.plotly_chart(fig, use_container_width=True)

# ==========================================================
# ADVANCED ANALYTICS
# ==========================================================
elif page == "Advanced Analytics":

    st.title("🔬 Advanced Analytics Dashboard")

    col1, col2 = st.columns(2)
    col1.metric("R² Score", round(metrics["R2"], 3))
    col2.metric("RMSE", round(metrics["RMSE"], 2))

    st.markdown("---")

    # Stage Comparison
    st.subheader("📊 Stage-1 vs Stage-2 Comparison")

    stage1_rmse = 55
    stage2_rmse = metrics["RMSE"]

    comparison_df = pd.DataFrame({
        "Stage": ["Stage-1", "Stage-2"],
        "RMSE": [stage1_rmse, stage2_rmse]
    })

    fig = px.bar(comparison_df, x="Stage", y="RMSE",
                 color="Stage", title="Model Improvement Comparison")

    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # Feature Importance
    st.subheader("📌 Feature Importance")

    importances = model.feature_importances_
    feature_importance_df = pd.DataFrame({
        "Feature": feature_columns,
        "Importance": importances
    }).sort_values(by="Importance", ascending=False)

    fig2 = px.bar(feature_importance_df,
                  x="Importance",
                  y="Feature",
                  orientation='h',
                  title="Pollutant Contribution to AQI")

    st.plotly_chart(fig2, use_container_width=True)

    st.markdown("---")

    # Correlation Heatmap
    st.subheader("📈 Pollutant Correlation Heatmap")

    corr = df.corr(numeric_only=True)

    heatmap = ff.create_annotated_heatmap(
        z=corr.values,
        x=list(corr.columns),
        y=list(corr.index),
        colorscale="Viridis"
    )

    st.plotly_chart(heatmap, use_container_width=True)

    st.markdown("---")

    # City-wise AQI Comparison
    st.subheader("🏙 City-wise Average AQI")

    city_avg = df.groupby("City")["AQI"].mean().reset_index()

    fig3 = px.bar(city_avg,
                  x="City",
                  y="AQI",
                  title="Average AQI by City")

    st.plotly_chart(fig3, use_container_width=True)

    st.markdown("---")

    # AQI Distribution
    st.subheader("📊 AQI Distribution")

    fig4 = px.histogram(df, x="AQI", nbins=50,
                        title="Distribution of AQI Values")

    st.plotly_chart(fig4, use_container_width=True)


# ===================== FOOTER =====================
st.markdown(" ")
st.markdown(" ")
st.markdown(" ")
st.markdown(" ")
st.markdown(" ")
st.markdown(" ")
st.markdown("""
<style>
.footer {
    background-color: #232c38;
    color: #e7e7e7;
    padding: 30px 0 10px 0;
    font-size: 16px;
    margin-top: 40px;
}
.footer hr {
    border-color: #555;
    margin-top: 18px;
    margin-bottom: 12px;
}
.footer-container {
    display: flex;
    flex-wrap: wrap;
    justify-content: space-between;
    max-width: 1100px;
    margin: 0 auto;
    padding: 0 32px;
}
.footer-section {
    flex: 1;
    min-width: 260px;
    margin-bottom: 16px;
}
.footer-section h3 {
    color: #fff;
    margin-bottom: 7px;
}
.footer-icons a {
    color: #44aaff;
    margin-right: 12px;
    font-size: 26px;
    text-decoration: none;
}
.footer-icons img {
    vertical-align: middle;
}
.footer-bottom {
    text-align: center;
    font-size: 15px;
    color: #bbb;
    margin-top: 10px;
}
</style>

<div class="footer">
  <div class="footer-container">
    <div class="footer-section">
      <h3>Smart AQI Monitoring & Health Advisory System</h3>
      <div>
        Real-time Air Quality Insights for Safer Living<br>
        Designed to help citizens make informed decisions about their health and environment.
      </div>
    </div>
    <div class="footer-section">
      <h3>🚀Key Features</h3>
      <div>
       > Machine Learning Based AQI Prediction<br>
       > Real-time Air Quality Monitoring<br>
       > Trend Analysis & Visualization <br>
       > Advanced Model Analytics Dashboard<br>
       > Health Risk Advisory Engine
      </div>
    </div>
    <div class="footer-section">
      <h3>Connect</h3>
      <div class="footer-icons">
        <a href="https://twitter.com/" target="_blank" title="Twitter">
            <img src="https://cdn.jsdelivr.net/npm/simple-icons@v9/icons/twitter.svg" width="24" height="24"/>
        </a>
        <a href="https://www.linkedin.com/in/muralikrishna-banoth/" target="_blank" title="LinkedIn">
            <img src="https://cdn.jsdelivr.net/npm/simple-icons@v9/icons/linkedin.svg" width="24" height="24"/>
        </a>
        <a href="https://github.com/Murali-2569" target="_blank" title="GitHub">
            <img src="https://cdn.jsdelivr.net/npm/simple-icons@v9/icons/github.svg" width="24" height="24"/>
        </a>
      </div>
    </div>
  </div>
  <hr>
  <div class="footer-bottom">
    <div style="text-align:center; color:#bdc3c7; font-size:14px; padding:15px;">
    © 2026 Smart AQI Monitoring & Health Advisory System <br>
     | Developed by Murali Krishna & Team... <br>
    Powered by Machine Learning & Real-Time OpenWeather API
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

