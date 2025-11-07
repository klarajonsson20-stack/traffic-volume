""" AI disclosure: 
Tool: ChatGPT-5 
Purpose: Brainstormed session state for aplha slider; Idea for min/max under slider; 
        Guidance on imports and libomp path for M1 Mac; Guidance on XGBoost import; 
        Brainstormed encoding for datetime features  
Usage: Html code for titles; Implemented session state for alpha slider; ctypes libomp load line; 
        XGBoost import line; Used pd.to_datetime but corrected ChatGPT suggestion to our project.
Location: Documented here and in app.py comments (top of file)."""

# Import necessary libraries
import streamlit as st
import pandas as pd
import pickle
import warnings
import numpy as np
warnings.filterwarnings("ignore")

# Page configuration
st.set_page_config(page_title="Traffic Volume Predictor", layout="wide")

# Title
st.markdown(
        """
        <div style="text-align:center;">
            <h1 style="display:inline-block; margin-bottom:0.25rem; font-size:52px; font-weight:800;
                background: linear-gradient(90deg, #FF8C00 0%, #FFB300 40%, #ADFF2F 70%, #00CC00 100%);
                -webkit-background-clip: text; background-clip: text; -webkit-text-fill-color: transparent;">
                Traffic Volume Predictor
            </h1>
        </div>
        """,
        unsafe_allow_html=True,
)
st.markdown("<p style='text-align:center; color:#ffffff; font-size:20px; margin-top:0.25rem;'>Utilize our advanced Machine Learning application to predict traffic volume.</p>", unsafe_allow_html=True)
st.image("traffic_image.gif", use_container_width =True)

# Load the trained model
@st.cache_resource
def load_model(path="traffic_volume.pickle"):
    with open(path, "rb") as f:
        return pickle.load(f)

clf = load_model()

# Load the raw features used during training
@st.cache_resource
def load_default_raw(path="traffic_volume_raw.csv"):
    return pd.read_csv(path)

default_df = load_default_raw()

# Raw training columns
feature_cols = list(default_df.columns)

# Alpha slider (show min/max labels underneath)
slider_min = 0.01
slider_max = 0.5
alpha = st.slider(
    "Select alpha value for prediction interval",
    min_value=slider_min,
    max_value=slider_max,
    value=0.10,
    step=0.01,
)

# Display the min and max values under the slider, left and right aligned.
st.markdown(
            f"""
            <div style="margin-top:-30px; display:flex; justify-content:space-between; align-items:center; color:#ffffff; font-size:13px; line-height:1; padding:0 4px; margin-bottom:0;">
                <span style="text-align:left; padding:0;">{slider_min:.2f}</span>
                <span style="text-align:right; padding:0;">{slider_max:.2f}</span>
            </div>
            """,
        unsafe_allow_html=True,
)

# Sidebar for user input
st.sidebar.image("traffic_sidebar.jpg", use_container_width=True, caption="Traffic Volume Predictior")
st.sidebar.header("Input Features")
st.sidebar.write("You can either upload your data file or manually input features.")

# Option 1: Asking users to input their data as a file
with st.sidebar.expander("Option 1: Upload CSV file"):
    st.write("Upload a CSV file containing diamond details")
    traffic_file = st.file_uploader('Choose a CSV file', type=['csv'])
    
    # Show sample data format
    sample_data = pd.read_csv("Traffic_Volume.csv").head(4)
    st.write("Sample data format for upload:")
    st.dataframe(sample_data)
    st.warning("⚠️ Ensure your CSV file has the same columns as shown above.")  

# Option 2: Manual input of feature
with st.sidebar.expander("Option 2: Manually input features"):
    holiday_options = sorted(default_df["holiday"].fillna("None").unique().tolist())
    holiday_options = ["None"] + [opt for opt in holiday_options if opt != "None"]
    weather_options = sorted(default_df["weather_main"].dropna().unique().tolist())
    month_options = sorted(default_df["month"].dropna().unique().tolist())
    weekday_options = sorted(default_df["weekday"].dropna().unique().tolist())

    # Sidebar numeric + categorical inputs
    holiday = st.selectbox("US Holiday", options = holiday_options, index = 0, help="Choose whether today is a desginated holiday or not")
    holiday = holiday if holiday != "None" else np.nan

    temp = st.number_input("Temperature", min_value=200.0, max_value=350.0, value=281.0, step=0.1, help = "Average temperature in Kelvin")
    rain_1h = st.number_input("Rain", min_value=0.0, max_value=200.0, value=0.0, step=0.01, help = "Amount in mm of rain that occurred in the hour")
    snow_1h = st.number_input("Snow", min_value=0.0, max_value=200.0, value=0.0, step=0.01, help = "Amount in mm of snow that occurred in the hour")
    clouds_all = st.number_input("Clouds", min_value=0, max_value=100, value=40, step=1, help = "Percentage of cloud cover")
    weather_main = st.selectbox("Current weather", options=weather_options, help = "Choose the current weather")
    
    month = st.selectbox("Month", options=month_options, help = "Choose month")
    weekday = st.selectbox("Day of the week", options=weekday_options, help = "Choose day of the week")
    hour = st.number_input("Hour", min_value=0, max_value=23, value=12, step=1, help = "Choose hour of the day (0–23)")

    submit = st.button("Submit Form Data")

# Initialize session state for storing prediction data
if 'user_encoded' not in st.session_state:
    st.session_state.user_encoded = None
    st.session_state.prediction = None

# Prediction 
if traffic_file is None and submit:
    encode_df = default_df.copy()
    encode_df.loc[len(encode_df)] = [holiday, temp, rain_1h, snow_1h, clouds_all, weather_main, hour, weekday, month]

    encode_dummy_df = pd.get_dummies(encode_df, columns=["holiday", "weather_main", "weekday", "month"], drop_first=True)

    st.session_state.user_encoded = encode_dummy_df.tail(1).astype("float32")
    predictions, prediction_intervals = clf.predict(st.session_state.user_encoded, alpha=alpha)
    st.session_state.prediction = float(predictions[0])

    st.subheader("Predicting Traffic Volume...")
    st.metric("Predicted Traffic Volume", f"{st.session_state.prediction:.0f}")
    st.caption(f"Prediction Interval ({int((1 - alpha) * 100)}%): "
               f"[{float(prediction_intervals[0,0]):.0f}, {float(prediction_intervals[0,1]):.0f}]")
               
# Update confidence interval if we have a prediction and alpha changes
elif st.session_state.user_encoded is not None:
    predictions, prediction_intervals = clf.predict(st.session_state.user_encoded, alpha=alpha)
    
    st.subheader("Predicting Traffic Volume...")
    st.metric("Predicted Traffic Volume", f"{st.session_state.prediction:.0f}")
    st.caption(f"Prediction Interval ({int((1 - alpha) * 100)}%): "
               f"[{float(prediction_intervals[0,0]):.0f}, {float(prediction_intervals[0,1]):.0f}]")

elif traffic_file is not None:
    input_df = pd.read_csv(traffic_file)

    input_df = input_df[feature_cols].copy()

    combined_raw = pd.concat([default_df, input_df], ignore_index=True)

    combined_encoded = pd.get_dummies(combined_raw, columns=["holiday", "weather_main", "weekday", "month"], drop_first=True).astype("float32")

    user_encoded = combined_encoded.tail(len(input_df))

    predictions, prediction_intervals = clf.predict(user_encoded, alpha=alpha)

    # Present alongside original input columns
    out = input_df.copy()
    out["Predicted Traffic Volume"] = predictions
    out["Lower Bound"] = prediction_intervals[:, 0]
    out["Upper Bound"] = prediction_intervals[:, 1]

    st.subheader("Predictions")
    st.dataframe(out)

else: # Default view shown when no input method selected yet 
    st.info("ℹ️ Please upload a CSV file or use the form to make predictions.")
    st.subheader("Predicting Traffic Volume...") 
    st.metric("Predicted Traffic Volume", "0") 
    st.caption(f"Prediction Interval ({int((1 - alpha) * 100)}%): [0, 0]") 


# Showing additional items in tabs
st.markdown("<h3 style='color:white;'>Model Performance and Inference</h3>", unsafe_allow_html=True)
tab1, tab2, tab3, tab4 = st.tabs(["Feature Importance", "Histogram of residuals", "Predicted vs Actual", "Coverage Plot"])
with tab1:
    st.write("### Feature Importance")
    st.image('feature_importance.png', width=700, caption="Relative importance of features in predicition.")
with tab2:
    st.write("### Histogram of Residuals")
    st.image('histogram_of_residuals.png', width=700, caption="Distribution of residuals to evaluate predcition quality.")
with tab3:
    st.write("### Predicted vs Actual")
    st.image('predicted_vs_actual.png', width=700, caption="Visual comparison of predicted and actual values.")
with tab4:
    st.write("### Coverage Plot")
    st.image('coverage_plot.png', width=700, caption="Range of prediction with confidence intervals.")