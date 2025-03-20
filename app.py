import streamlit as st
import numpy as np
import joblib
import pandas as pd




try:
    model = joblib.load('best_model.pkl')  
    scaler = joblib.load('scaler_model.pkl')  
except Exception as e:
    st.error(f"⚠️ Error loading model or scaler: {e}")
    st.stop()


FEATURES = ['Open','High','Low','Adj Close','Close_1']


st.markdown("""
    <style>
        /* Background Styling */
        body {
            background: linear-gradient(to bottom, Pink, #ffffff);
        }

        /* Center the header */
        .main-title {
            font-size: 42px;
            font-weight: bold;
            color: #ff4b4b;
            text-align: center;
        }
        .sub-header {
            font-size: 22px;
            color: #444;
            text-align: center;
        }

        /* Input fields styling */
        .stTextInput>div>div>input {
            border: 2px solid #ff4b4b;
            border-radius: 5px;
            padding: 8px;
        }

        /* Button Styling */
        .stButton>button {
            background-color: #ff4b4b;
            color: white;
            font-size: 18px;
            border-radius: 10px;
            padding: 12px;
            transition: 0.3s ease;
        }
        .stButton>button:hover {
            background-color: #d33f3f;
        }

        /* Prediction Box */
        .prediction-box {
            background-color: #fff3f3;
            padding: 15px;
            border-radius: 10px;
            text-align: center;
            font-size: 20px;
            font-weight: bold;
            color: #333;
        }
    </style>
""", unsafe_allow_html=True)


st.markdown("<h1 class='main-title'> 📈 StockVision </h1>", unsafe_allow_html=True)
st.markdown("<p class='sub-header'>Enter the required details to check the prediction</p>", unsafe_allow_html=True)


col1, col2 = st.columns(2)
user_input = {}

for i, feature in enumerate(FEATURES):
    with col1 if i % 2 == 0 else col2:
        user_input[feature] = st.number_input(f"🔹 {feature.replace('_', ' ').title()}", value=0.0, format="%.5f")


if st.button("🔮 Predict"):
    try:
      
        input_df = pd.DataFrame([list(user_input.values())], columns=FEATURES)
        scaled_data = scaler.transform(input_df)
        prediction = model.predict(scaled_data)
        result_text = f'THE FORECASTED PRICE IS ${prediction.item():.2f}'
        result_color = "#ff4b4b" 
   
        st.markdown(f"<div class='prediction-box' style='color: {result_color};'>{result_text}</div>", unsafe_allow_html=True)

        with st.expander("📊 Model Insights"):
            st.write("✅ This prediction is based on a trained Long Short Term Memory.")
            
    
    except Exception as e:
        st.error(f"⚠️ Error during prediction: {str(e)}")



st.markdown("---")
st.markdown("<p style='text-align: center; font-size: 16px;'>Made By MehakPreet Singh</p>", unsafe_allow_html=True)
