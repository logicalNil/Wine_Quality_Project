import streamlit as st
import requests
import pandas as pd
import json

# Set page config
st.set_page_config(
    page_title="Wine Quality Predictor",
    page_icon="🍷",
    layout="wide"
)

# Title and description
st.title("🍷 Wine Quality Prediction")
st.markdown("""
This app predicts the quality of wine based on its physicochemical properties.
Fill in the values below and click 'Predict' to get a quality prediction.
""")

# API endpoint
API_URL = "http://localhost:8000/predict"

# Create form for input
with st.form("wine_features_form"):
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Wine Properties")
        fixed_acidity = st.slider("Fixed Acidity", 3.8, 15.9, 7.0, 0.1)
        volatile_acidity = st.slider("Volatile Acidity", 0.08, 1.58, 0.5, 0.01)
        citric_acid = st.slider("Citric Acid", 0.0, 1.66, 0.25, 0.01)
        residual_sugar = st.slider("Residual Sugar", 0.6, 65.8, 2.0, 0.1)
        chlorides = st.slider("Chlorides", 0.009, 0.611, 0.05, 0.001)
        free_sulfur_dioxide = st.slider("Free Sulfur Dioxide", 1.0, 289.0, 30.0, 1.0)

    with col2:
        st.subheader("")
        total_sulfur_dioxide = st.slider("Total Sulfur Dioxide", 6.0, 440.0, 120.0, 1.0)
        density = st.slider("Density", 0.98711, 1.03898, 0.995, 0.0001)
        pH = st.slider("pH", 2.72, 4.01, 3.2, 0.01)
        sulphates = st.slider("Sulphates", 0.22, 2.0, 0.5, 0.01)
        alcohol = st.slider("Alcohol", 8.0, 14.9, 10.0, 0.1)
        wine_type = st.radio("Wine Type", ["red", "white"])

    # Submit button
    submitted = st.form_submit_button("Predict Quality")

# When form is submitted
if submitted:
    # Prepare data for API request
    data = {
        "fixed_acidity": fixed_acidity,
        "volatile_acidity": volatile_acidity,
        "citric_acid": citric_acid,
        "residual_sugar": residual_sugar,
        "chlorides": chlorides,
        "free_sulfur_dioxide": free_sulfur_dioxide,
        "total_sulfur_dioxide": total_sulfur_dioxide,
        "density": density,
        "pH": pH,
        "sulphates": sulphates,
        "alcohol": alcohol,
        "type": wine_type
    }

    # Make API request
    try:
        response = requests.post(API_URL, json=data)

        if response.status_code == 200:
            result = response.json()

            # Display results
            st.success(f"Prediction completed successfully!")

            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Prediction Result")

                # Display predicted class with color coding
                if result["predicted_class"] == "excellent":
                    st.markdown(f"**Quality:** :green[{result['predicted_class'].upper()}] 🎉")
                elif result["predicted_class"] == "average":
                    st.markdown(f"**Quality:** :orange[{result['predicted_class'].upper()}] 👍")
                else:
                    st.markdown(f"**Quality:** :red[{result['predicted_class'].upper()}] 👎")

                st.markdown(f"**Confidence:** {result['confidence']:.2%}")
                st.markdown(f"**Latency:** {result['latency']:.4f} seconds")

            with col2:
                st.subheader("Probability Distribution")

                # Create a bar chart of probabilities
                prob_df = pd.DataFrame({
                    'Quality': list(result['probabilities'].keys()),
                    'Probability': list(result['probabilities'].values())
                })

                st.bar_chart(prob_df.set_index('Quality'))

                # Display probabilities as percentages
                for quality, prob in result['probabilities'].items():
                    st.markdown(f"{quality.title()}: {prob:.2%}")

        else:
            st.error(f"Error: {response.json()['detail']}")

    except requests.exceptions.ConnectionError:
        st.error("Could not connect to the API. Please make sure the API server is running on localhost:8000")
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")

# Add sidebar with information
with st.sidebar:
    st.header("About")
    st.markdown("""
    This application uses a machine learning model to predict wine quality based on physicochemical properties.

    The model was trained on the Wine Quality dataset and classifies wines into three categories:
    - **Poor** (quality score 0-3)
    - **Average** (quality score 4-7)
    - **Excellent** (quality score 8-10)

    Adjust the sliders to input wine properties and get a prediction.
    """)

    st.header("API Health Check")
    try:
        health_response = requests.get("http://localhost:8000/health")
        if health_response.status_code == 200:
            st.success("API is healthy ✅")
        else:
            st.error("API is not responding ❌")
    except:
        st.error("API is not responding ❌")

# Add some sample data for testing
with st.expander("Sample Data for Testing"):
    st.markdown("""
    Try these sample values for quick testing:

    **Red Wine (Good Quality):**
    - Fixed Acidity: 7.4
    - Volatile Acidity: 0.7
    - Citric Acid: 0.0
    - Residual Sugar: 1.9
    - Chlorides: 0.076
    - Free Sulfur Dioxide: 11.0
    - Total Sulfur Dioxide: 34.0
    - Density: 0.9978
    - pH: 3.51
    - Sulphates: 0.56
    - Alcohol: 9.4
    - Type: red

    **White Wine (Excellent Quality):**
    - Fixed Acidity: 7.0
    - Volatile Acidity: 0.27
    - Citric Acid: 0.36
    - Residual Sugar: 20.7
    - Chlorides: 0.045
    - Free Sulfur Dioxide: 45.0
    - Total Sulfur Dioxide: 170.0
    - Density: 1.001
    - pH: 3.0
    - Sulphates: 0.45
    - Alcohol: 8.8
    - Type: white
    """)