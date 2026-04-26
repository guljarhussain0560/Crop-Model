import streamlit as st
import numpy as np
import joblib

# ─── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Kisaan AI 🌾",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ─── Custom Styling ───────────────────────────────────────────────────────────
st.markdown("""
    <style>
    /* Main background gradient */
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Main text color - white for visibility */
    body, p, div, span, label {
        color: #ffffff !important;
    }
    
    /* Header styling */
    h1 {
        color: #ffffff !important;
        text-shadow: 2px 2px 6px rgba(0,0,0,0.3);
        text-align: center !important;
        font-weight: bold !important;
    }
    
    h2 {
        color: #ffffff !important;
        border-left: 5px solid #FFD700;
        padding-left: 15px !important;
        text-shadow: 1px 1px 3px rgba(0,0,0,0.2);
    }
    
    h3 {
        color: #ffffff !important;
        text-shadow: 1px 1px 3px rgba(0,0,0,0.2);
    }
    
    /* Markdown text */
    .stMarkdown {
        color: #ffffff !important;
    }
    
    /* Container styling */
    .stContainer {
        background-color: rgba(255, 255, 255, 0.95);
        border-radius: 15px;
        padding: 20px;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(90deg, #FFD700, #FFC700) !important;
        color: #2d5a3d !important;
        font-weight: bold !important;
        border-radius: 8px !important;
        border: none !important;
        box-shadow: 0 4px 15px rgba(255, 215, 0, 0.3) !important;
        transition: all 0.3s ease !important;
        padding: 12px 20px !important;
    }
    
    .stButton > button:hover {
        background: linear-gradient(90deg, #FFC700, #FFB700) !important;
        box-shadow: 0 6px 20px rgba(255, 215, 0, 0.4) !important;
        transform: translateY(-2px) !important;
    }
    
    /* Input field styling */
    .stNumberInput input {
        border: 2px solid #FFD700 !important;
        border-radius: 8px !important;
        background-color: rgba(255, 255, 255, 0.9) !important;
        color: #2d5a3d !important;
        padding: 10px !important;
        font-weight: bold !important;
    }
    
    .stNumberInput label {
        color: #ffffff !important;
        font-weight: bold !important;
    }
    
    /* Success message styling */
    .stSuccess {
        background-color: #d4edda !important;
        border: 2px solid #28a745 !important;
        border-radius: 8px !important;
        padding: 15px !important;
    }
    
    .stSuccess > div > div > p {
        color: #155724 !important;
        font-weight: bold !important;
    }
    
    /* Info message styling */
    .stInfo {
        background-color: #d1ecf1 !important;
        border: 2px solid #0c5460 !important;
        border-radius: 8px !important;
        padding: 15px !important;
    }
    
    .stInfo > div > div > p {
        color: #0c5460 !important;
        font-weight: bold !important;
    }
    
    /* Column styling */
    [data-testid="column"] {
        background-color: rgba(255, 255, 255, 0.1);
        border-radius: 10px;
        padding: 15px;
        border: 2px solid #FFD700;
        margin: 5px;
    }
    </style>
""", unsafe_allow_html=True)

# ─── Load Models ───────────────────────────────────────────────────────────────
@st.cache_resource
def load_crop_model():
    crop_model = joblib.load("crop_model.pkl")
    crop_encoder = joblib.load("crop_encoder.pkl")
    return crop_model, crop_encoder

crop_model, crop_encoder = load_crop_model()

# ─── Header ────────────────────────────────────────────────────────────────────
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.markdown("""
        <p style='text-align: center; color: #ffffff; font-size: 2.1em; margin-top: -8px;'>
        🌱 Crop Recommendation System 🌱
        </p>
    """, unsafe_allow_html=True)

st.markdown("---")

# ─── Main Container ───────────────────────────────────────────────────────────
with st.container():

    st.markdown("### 📋 Feature: Crop Recommendation Engine", unsafe_allow_html=True)
    st.markdown("✅ Enter your soil and climate details to find the best crop to grow.", unsafe_allow_html=True)
    st.markdown("")
    
    # Create two columns for input
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 🌍 Soil Nutrients", unsafe_allow_html=True)
        N           = st.number_input("🔵 Nitrogen (N) - ppm",           0, 200, 50, help="Nitrogen content in soil")
        P           = st.number_input("🟡 Phosphorus (P) - ppm",         0, 200, 50, help="Phosphorus content in soil")
        K           = st.number_input("🟢 Potassium (K) - ppm",          0, 200, 50, help="Potassium content in soil")
        ph          = st.number_input("⚗️ Soil pH (Acidity/Basicity)",    0.0, 14.0,  6.5, help="Soil pH level (0-14)")

    with col2:
        st.markdown("### 🌤️ Climate Conditions", unsafe_allow_html=True)
        temperature = st.number_input("🌡️ Temperature (°C)",             0.0, 60.0, 25.0, help="Average temperature")
        humidity    = st.number_input("💧 Humidity (%)",                 0.0, 100.0, 60.0, help="Moisture in air")
        rainfall    = st.number_input("🌧️ Rainfall (mm)",                0.0, 500.0, 100.0, help="Annual rainfall")

    st.markdown("")
    
    # Recommendation Button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("🎯 Get Crop Recommendation", use_container_width=True):
            features = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
            prediction = crop_model.predict(features)
            crop_name = crop_encoder.inverse_transform(prediction)[0]

            # Success display with colorful styling
            st.markdown("""
                <div style='background: linear-gradient(135deg, #4CAF50, #45a049); padding: 20px; border-radius: 10px; text-align: center; color: white;'>
                    <h2 style='color: white; margin: 0; font-size: 2em;'>✅ Recommended Crop</h2>
                    <h1 style='color: #fff9e6; margin: 10px 0; font-size: 3em;'>🌾 """ + crop_name.upper() + """ 🌾</h1>
                </div>
            """, unsafe_allow_html=True)
            
            st.balloons()
            
            # Additional info
            crop_info = {
                "RICE": "🍚 Best for humid, tropical climates. Requires standing water.",
                "MAIZE": "🌽 Versatile crop. Suitable for warm, well-drained soils.",
                "WHEAT": "🌾 Ideal for temperate climates. Requires moderate rainfall.",
                "COTTON": "🤍 Heat-loving crop. Needs well-draining soil with moderate nutrients.",
                "SUGARCANE": "🍂 Thrives in warm climate. Requires high water and nitrogen.",
                "POTATO": "🥔 Suitable for cool climates. Needs well-drained, loose soil.",
                "PULSES": "🫘 Nitrogen-fixing crop. Improves soil health naturally.",
                "FRUITS": "🍎 Depends on specific type. Generally need warm, sunny locations.",
                "VEGETABLES": "🥬 Diverse crop. Most prefer moderate temperatures and humidity.",
            }
            
            crop_upper = crop_name.upper()
            if crop_upper in crop_info:
                st.markdown(f"<div style='background-color: rgba(255, 215, 0, 0.15); border-left: 5px solid #FFD700; padding: 15px; border-radius: 8px;'><h3 style='color: #FFD700; margin: 0;'>💡 Quick Info</h3><p style='color: #ffffff; margin: 10px 0 0 0; font-weight: 500;'>{crop_info[crop_upper]}</p></div>", unsafe_allow_html=True)

st.markdown("---")

# ─── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
    <div style='text-align: center; padding: 20px; background: linear-gradient(135deg, #667eea, #764ba2); border-radius: 10px; color: white; margin-top: 30px;'>
        <h3 style='color: #FFD700; margin: 0; font-size: 1.3em;'>🌾 Kisaan AI 🌾</h3>
        <p style='color: #ffffff; margin: 8px 0 0 0; font-size: 0.95em;'>Crop Recommendation System - Part of Kisaan AI Platform</p>
        <p style='margin: 10px 0 0 0; font-size: 0.85em; color: #FFD700;'>Made with ❤️ for Indian Farmers | Empowering Agriculture with AI</p>
    </div>
""", unsafe_allow_html=True)
