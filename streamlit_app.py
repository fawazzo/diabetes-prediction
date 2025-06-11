import streamlit as st
import numpy as np
import pickle
import tensorflow as tf
import os
import sklearn

# Streamlit app configuration MUST be the first Streamlit command
st.set_page_config(page_title="Diabetes Prediction", page_icon=":hospital:", layout="wide")

# Model and scaler paths
MODELS_DIR = "models/"  # Replace with the actual path if different
SCALER_PATH = os.path.join(MODELS_DIR, "scaler.pkl")
MODEL_PATHS = {
    "Decision Tree": os.path.join(MODELS_DIR, "decision_tree_model.pkl"),
    "SVM": os.path.join(MODELS_DIR, "svm_model.pkl"),
    "k-NN": os.path.join(MODELS_DIR, "knn_model.pkl"),
    "ANN": os.path.join(MODELS_DIR, "ann_model.h5"),  # Make sure this is .h5 for Keras models
}

# Input ranges for validation
INPUT_RANGES = {
    "Pregnancies": (0.0, 10.0),
    "Glucose": (0.0, 200.0),
    "Blood Pressure": (0.0, 150.0),
    "Skin Thickness": (0.0, 100.0),
    "Insulin": (0.0, 1000.0),
    "BMI": (0.0, 70.0),
    "Diabetes Pedigree Function": (0.000, 3.000),
    "Age": (0.0, 100.0),
}

# Input field formats
INPUT_FORMATS = {
    "Diabetes Pedigree Function": "%.3f",
    "BMI": "%.1f"
}

# Check if models and scaler exist
if not all(os.path.exists(path) for path in list(MODEL_PATHS.values()) + [SCALER_PATH]):
    st.error("Error: Model files not found. Please run 'train.py' first.")
    st.stop()

# Load scaler
try:
    with open(SCALER_PATH, "rb") as scaler_file:
        scaler = pickle.load(scaler_file)
except Exception as e:
    st.error(f"Error loading scaler: {e}")
    st.stop()


# Bootstrap CSS and styling
st.markdown(
    """
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet" integrity="sha384-9ndCyUaIbzAi2FUVXJi0CjmCapSmO7SnpJef0Sk0Wl1Awi+z9Wl/v/x6V6yWxNq" crossorigin="anonymous">
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js" integrity="sha384-geWF76w3IJwLiqEO0/mJqR8pPojYyKV9bwEv/yEKovSyuJTxGJ5pEN7j4oWS8w8" crossorigin="anonymous"></script>
    <style>
        body { background-color: #f8f9fa; }
        .container {
            border-radius: 10px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
            padding: 30px;
            background-color: white;
            margin-top: 20px;
        }
        h1 { text-align: center; margin-bottom: 20px; }
    </style>
    """,
    unsafe_allow_html=True,
)

# Main container for title and subtitle
with st.container():
    st.title("Diabetes Prediction Expert System")
    st.subheader("Choose a model and enter patient details to get a prediction")


# Model Selection
selected_model_name = st.selectbox("Select Model:", list(MODEL_PATHS.keys()), format_func=lambda x: x.replace("_", " ").title())
model_path = MODEL_PATHS[selected_model_name]

# Load Model
try:
    if selected_model_name == "ANN":
        model = tf.keras.models.load_model(model_path)
    else:
        with open(model_path, "rb") as model_file:
            model = pickle.load(model_file)
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()


# Input fields and Prediction
input_data = []
col1, col2 = st.columns(2)
for i, (label, (min_val, max_val)) in enumerate(INPUT_RANGES.items()):
    format_str = INPUT_FORMATS.get(label, "%.0f")
    with col1 if i < len(INPUT_RANGES) // 2 else col2:
        input_value = st.number_input(label, min_value=min_val, max_value=max_val, value=0.0, step=1.0 if format_str == "%.0f" else 0.1, format=format_str)
        input_data.append(input_value)



if st.button("Predict"): 
    try:
        input_data = np.array([input_data])  # Assuming input_data is defined elsewhere
        scaled_input = scaler.transform(input_data) # Assuming scaler is defined

        if selected_model_name == "ANN":
            prediction = model.predict(scaled_input)[0][0]
        elif hasattr(model, "predict_proba"):
            prediction = model.predict_proba(scaled_input)[0, 1]
        else:
            prediction = model.predict(scaled_input)[0]

        prediction_percentage = prediction * 100 if prediction <= 1 else prediction

        if 0 <= prediction_percentage <= 25:
            color = "green"  # Low risk - green
            message = "Low Risk"
            icon = "🟢"  # Green circle emoji
        elif 26 <= prediction_percentage <= 60:
            color = "orange"
            message = "Medium Risk"
            icon = "🟠"  # Orange circle emoji
        elif 61 <= prediction_percentage <= 100:
            color = "red"  # High risk - red
            message = "High Risk"
            icon = "🔴"  # Red circle emoji
        else:
            color = "black"
            message = "Unexpected Value"  # Handle appropriately
            icon = "⚫️"  # Black circle emoji
            st.warning(f"Unexpected prediction value: {prediction_percentage:.2f}%")

        # Enhanced Display with st.markdown and HTML:
        st.markdown(
            f"""
                    <div style='text-align: center; margin-top: 20px;'>
                        <div style='font-size: 36px; font-weight: bold; color: {color};'>
                            {icon} {prediction_percentage:.2f}%
                        </div>
                        <div style='font-size: 24px; color: {color}; margin-top: 10px;'>
                            {message} of Diabetes
                        </div>
                    </div>
                    """,
            unsafe_allow_html=True,
        )


    except Exception as e:
        st.error(f"Error during prediction: {e}")