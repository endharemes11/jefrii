
import streamlit as st
from PIL import Image
import numpy as np
from ultralytics import YOLO
import os

# Path to the trained YOLOv8 model
# Assuming 'best (7).pt' will be in the root directory of the deployed app
model_path = 'best (7).pt'

# Check if the model file exists
if not os.path.exists(model_path):
    st.error(f"Error: Model file not found at {model_path}. Please ensure it's in the same directory as app.py")
else:
    # Load the trained YOLOv8 model
    model = YOLO(model_path)

    st.title("Deteksi Objek dengan YOLOv8")

    uploaded_file = st.file_uploader("Unggah gambar...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Read the image file
        image = Image.open(uploaded_file)
        st.image(image, caption="Gambar yang Diunggah", use_column_width=True)

        # Perform inference
        results = model(image)

        # Display the results (image with bounding boxes)
        for r in results:
            im_array = r.plot()  # plot a BGR numpy array of predictions
            im = Image.fromarray(im_array[..., ::-1])  # RGB PIL image
            st.image(im, caption="Hasil Deteksi", use_column_width=True)

        # Optional: Display prediction details
        # st.write("Detail Prediksi:")
        # for r in results:
        #     for box in r.boxes:
        #         st.write(f"Class: {model.names[int(box.cls)]}, Confidence: {box.conf.item():.2f}, Box: {box.xyxy[0].tolist()}")
