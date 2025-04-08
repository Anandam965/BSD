import streamlit as st
import pickle
import numpy as np
from PIL import Image
from tensorflow.keras.models import model_from_json


# Load models
def load_model(pkl_path):
    with open(pkl_path, "rb") as f:
        model_data = pickle.load(f)
    model = model_from_json(model_data["architecture"])
    model.set_weights(model_data["weights"])
    return model


# Load MRI and CT models
mri_model = load_model("cnn_MRI_brain_stroke_model.pkl")
ct_model = load_model("cnn_CT_brain_stroke_model.pkl")


# Streamlit UI
st.set_page_config(page_title="Brain Stroke Detection", page_icon="🧠", layout="wide")

# Custom CSS for Styling
st.markdown("""
    <style>
    .main {background-color: #f5f5f5;}
    .stButton>button {background-color: #4CAF50; color: white; border-radius: 8px;}
    .stSelectbox {color: black;}
    .stImage {border-radius: 10px;}
    .stProgress > div > div {background-color: #4CAF50;}
    </style>
    """, unsafe_allow_html=True)


# Sidebar

st.sidebar.title("🩺 Medical AI")
st.sidebar.write("This tool uses **Deep Learning** to detect brain stroke from MRI and CT scan images.")
scan_type = st.sidebar.radio("📷 Select Scan Type:", ["MRI", "CT"], index=0)


# Main Section
st.title("🧠 Brain Stroke Detection")
st.write("Upload an MRI or CT scan image to analyze stroke detection using **Deep Learning Models**.")

uploaded_file = st.file_uploader("📤 Upload an Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Display image
    image = Image.open(uploaded_file)
    st.image(image, caption="🖼 Uploaded Image", width=300)


    # Preprocess image
    image = image.resize((224, 224)).convert('RGB')
    image_array = np.array(image, dtype=np.float32) / 255.0  # Normalize
    image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension

    # Select model
    selected_model = mri_model if scan_type == "MRI" else ct_model

    # Prediction progress bar
    progress_bar = st.progress(0)
    for percent_complete in range(100):
        progress_bar.progress(percent_complete + 1)

    # Make prediction
    prediction = selected_model.predict(image_array)
    predicted_class = "🟥 Stroke Detected" if prediction[0][0] > 0.5 else "🟩 No Stroke (Normal)"
    confidence = prediction[0][0] if prediction[0][0] > 0.5 else 1 - prediction[0][0]

    # Display Results
    st.success(f"✅ **Prediction: {predicted_class}**")
   # st.write(f"🧪 **Confidence Score:** `{confidence:.4f}`")

    # Show extra message based on prediction
    if prediction[0][0] > 0.5:
        st.error("⚠️ Please consult a doctor immediately for further medical advice.")
    else:
        st.info("✅ No stroke detected. Keep monitoring your health regularly.")

