import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf

# Load the trained model
model = tf.keras.models.load_model("flower_model.keras")

# Define class names 
class_names = ['daisy', 'dandelion', 'roses', 'sunflowers',
               'tulips']

st.title("Flower AI 🌸")
st.write("Welcome! Upload an image to start.")

uploaded_file = st.file_uploader(
    "choose a flower image"
    ,type=["jpg", "png", "jpeg", "jfif"])
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image",
              use_container_width = True)
    
# prepare image for model
    img = image.resize((224, 224))
    img_array = np.array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    predictions = model.predict(img_array)
    predicted_class = class_names[np.argmax(predictions)]
    st.success(f"Predicted Flower: {predicted_class}")

