import streamlit as st
from PIL import Image
import numpy as np
from ultralytics import YOLO
import os
import base64  # untuk membuat file HTML hasil deteksi

# --- Path model ---
model_path = 'best.pt'

# --- Cek model ---
if not os.path.exists(model_path):
    st.error(f"❌ Error: Model file tidak ditemukan di path '{model_path}'. Pastikan file model ada di folder yang sama dengan app.py.")
else:
    # --- Load model YOLOv8 ---
    model = YOLO(model_path)
    st.title("🧠 Deteksi Objek dengan YOLOv8")

    # --- Upload gambar ---
    uploaded_file = st.file_uploader("📤 Unggah gambar...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Baca gambar
        image = Image.open(uploaded_file)
        st.image(image, caption="🖼️ Gambar yang Diunggah", use_column_width=True)

        # --- Deteksi objek ---
        results = model(image)

        # --- Tampilkan hasil ---
        for r in results:
            im_array = r.plot()  # hasil prediksi dalam array BGR
            im = Image.fromarray(im_array[..., ::-1])  # ubah ke RGB
            st.image(im, caption="✅ Hasil Deteksi", use_column_width=True)

            # --- Simpan hasil deteksi ke HTML ---
            result_image_path = "result_image.png"
            im.save(result_image_path)

            # Encode gambar ke base64
            with open(result_image_path, "rb") as img_file:
                encoded_string = base64.b64encode(img_file.read()).decode()

            # Buat konten HTML
            html_content = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <meta charset="UTF-8">
                <title>Hasil Deteksi Objek</title>
                <style>
                    body {{ font-family: Arial, sans-serif; text-align: center; background: #f9f9f9; }}
                    h1 {{ color: #333; }}
                    img {{ max-width: 90%; height: auto; border-radius: 10px; box-shadow: 0 0 10px rgba(0,0,0,0.2); }}
                </style>
            </head>
            <body>
                <h1>Hasil Deteksi Objek (YOLOv8)</h1>
                <img src="data:image/png;base64,{encoded_string}" alt="Detected Image">
            </body>
            </html>
            """

            # Tombol untuk download hasil HTML
            st.download_button(
                label="💾 Unduh Hasil Deteksi (HTML)",
                data=html_content,
                file_name="hasil_deteksi_yolov8.html",
                mime="text/html"
            )

        # --- (Opsional) tampilkan detail prediksi ---
        # st.write("📋 Detail Prediksi:")
        # for r in results:
        #     for box in r.boxes:
        #         st.write(f"Class: {model.names[int(box.cls)]}, Confidence: {box.conf.item():.2f}, Box: {box.xyxy[0].tolist()}")
