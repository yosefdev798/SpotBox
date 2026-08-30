import streamlit as st
from ultralytics import YOLO
from PIL import Image

st.title("SpotBox - Object Detection")
st.write("Upload an image to detect objects!")

model = YOLO('yolov8n.pt')

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    st.write("Detecting...")
    
    results = model(image)
    
    for result in results:
        im_array = result.plot()
        im = Image.fromarray(im_array[..., ::-1])
        st.image(im, caption='Detected Image', use_column_width=True)
