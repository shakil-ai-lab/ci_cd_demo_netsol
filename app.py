
import streamlit as st
import joblib

from sklearn.datasets import load_breast_cancer

# Configure the Streamlit application
st.set_page_config(
    page_title="Breast Cancer Prediction",
    page_icon="🏥"
)

st.title("Breast Cancer Prediction")
st.write("ML deployment demonstration using Random Forest")

# Load the trained model
@st.cache_resource
def load_model():
    return joblib.load("models/breast_cancer_model.joblib")

model = load_model()

# Load sample dataset
data = load_breast_cancer()

# Allow user to select a sample patient
sample_index = st.slider(
    "Select Patient Sample",
    min_value=0,
    max_value=len(data.data) - 1,
    value=0
)

# Get selected patient's features
patient_data = data.data[sample_index].reshape(1, -1)

# Make prediction
prediction = model.predict(patient_data)[0]

# Convert prediction into readable label
prediction_label = data.target_names[prediction]

st.subheader("Prediction Result")

if prediction_label == "malignant":
    st.error("Prediction: Malignant")
else:
    st.success("Prediction: Benign")

# Show actual label for educational comparison
actual_label = data.target_names[data.target[sample_index]]

st.write(f"Actual class: **{actual_label}**")

# Show selected patient features
with st.expander("View patient features"):
    st.write(data.feature_names)
    st.write(data.data[sample_index])
