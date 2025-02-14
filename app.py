import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image

# Load the trained model
model = load_model("military_vs_transport_model.h5")

# Streamlit UI
st.title("🚀 Military vs. Transport Vehicle Classifier")
st.write("Upload an image, and the AI model will classify it as either **Military** or **Other Transport**.")

# File uploader for user input
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

# Prediction function
def predict_image(image):
    image = image.resize((224, 224))  # Resize to match model input
    image = img_to_array(image) / 255.0  # Normalize
    image = np.expand_dims(image, axis=0)  # Add batch dimension

    # Make prediction
    result = model.predict(image)
    confidence = result[0][0]

    if confidence < 0.5:
        return "🚔 Military Vehicle", confidence
    else:
        return "🚗 Other Transport", confidence

# Process uploaded image
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Make prediction
    prediction, confidence = predict_image(image)

    # Display result
    st.markdown(f"### **Prediction: {prediction}**")
    st.markdown(f"### **Confidence: {confidence:.2f}**")
