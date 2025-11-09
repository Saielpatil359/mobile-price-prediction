import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt

# Load trained model
with open("mobile_price_model.pkl", "rb") as f:
    model = pickle.load(f)

st.title("📱 Mobile Price Range Prediction App")
st.write("Fill in the phone specifications to estimate the price range.")

# UI Inputs
battery_power = st.number_input("Battery Power (mAh)", min_value=500, max_value=5000, value=1500)
blue = st.selectbox("Bluetooth", [0, 1])
clock_speed = st.number_input("Clock Speed (GHz)", min_value=0.5, max_value=3.5, value=1.8)
dual_sim = st.selectbox("Dual SIM", [0, 1])
fc = st.number_input("Front Camera (MP)", min_value=0, max_value=25, value=5)
four_g = st.selectbox("4G Support", [0, 1])
int_memory = st.number_input("Internal Memory (GB)", min_value=2, max_value=256, value=64)
m_dep = st.number_input("Mobile Depth (cm)", min_value=0.1, max_value=1.5, value=0.5)
mobile_wt = st.number_input("Mobile Weight (grams)", min_value=80, max_value=300, value=150)
n_cores = st.number_input("Number of Cores", min_value=1, max_value=8, value=4)
pc = st.number_input("Primary Camera (MP)", min_value=0, max_value=25, value=12)
px_height = st.number_input("Pixel Height", min_value=0, max_value=1960, value=900)
px_width = st.number_input("Pixel Width", min_value=0, max_value=1960, value=1080)
ram = st.number_input("RAM (MB)", min_value=256, max_value=8000, value=4096)
sc_h = st.number_input("Screen Height (cm)", min_value=3, max_value=20, value=12)
sc_w = st.number_input("Screen Width (cm)", min_value=2, max_value=15, value=6)
talk_time = st.number_input("Talk Time (Hours)", min_value=2, max_value=20, value=12)
three_g = st.selectbox("3G Support", [0, 1])
touch_screen = st.selectbox("Touch Screen", [0, 1])
wifi = st.selectbox("WiFi Support", [0, 1])

# Create DataFrame for prediction
df = pd.DataFrame([[battery_power, blue, clock_speed, dual_sim, fc, four_g, int_memory, m_dep,
                    mobile_wt, n_cores, pc, px_height, px_width, ram, sc_h, sc_w, talk_time,
                    three_g, touch_screen, wifi]],
                  columns=['battery_power','blue','clock_speed','dual_sim','fc','four_g','int_memory','m_dep',
                           'mobile_wt','n_cores','pc','px_height','px_width','ram','sc_h','sc_w','talk_time',
                           'three_g','touch_screen','wifi'])

# Predict
if st.button("Predict Price Range"):
    prediction = model.predict(df)[0]

    price_text = {
        0: "💰 Low Cost",
        1: "💵 Medium Cost",
        2: "💎 High Cost",
        3: "🏆 Very High Cost"
    }

    price_range_value = {
        0: "₹3,000 – ₹7,000",
        1: "₹8,000 – ₹15,000",
        2: "₹16,000 – ₹25,000",
        3: "₹26,000+"
    }

    st.success(f"Predicted Price Range: {price_text[prediction]}")
    st.info(f"**Estimated Price:** {price_range_value[prediction]}")

# Feature Importance Button
if st.button("Show Feature Importance"):
    importances = model.feature_importances_
    feature_names = df.columns

    importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances}).sort_values(
        by="Importance", ascending=False)

    st.subheader("📊 Feature Importance")
    st.write("This shows which features contribute most to the price range prediction.")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(importance_df["Feature"], importance_df["Importance"])
    ax.set_xlabel("Importance Score")
    ax.set_ylabel("Feature")
    ax.invert_yaxis()
    st.pyplot(fig)
