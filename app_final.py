import os
import streamlit as st
from PIL import Image
import numpy as np
import base64
import torch
from ultralytics import YOLO

# --- Konfigurasi environment agar aman di server headless (tanpa GUI) ---
os.environ["QT_QPA_PLATFORM"] = "offscreen"
os.environ["OPENCV_VIDEOIO_PRIORITY_MSMF"] = "0"

# --- Cegah error OpenCV di Streamlit Cloud ---
try:
    import cv2
except Exception as e:
    st.warning("⚠️ OpenCV tidak tersedia sepenuhnya di environment ini.")

# --- Pastikan model ada ---
model_path = "best_safe (1).pt"  # disarankan gunakan model aman
if not os.path.exists(model_path):
    st.error(f"❌ Model file tidak ditemukan di path '{model_path}'. Pastikan file model ada di folder yang sama dengan app.py.")
    st.stop()

# --- Tambahkan safe_globals (wajib di PyTorch >=2.6) ---
torch.serialization.add_safe_globals([
    torch.nn.modules.container.Sequential,
    torch.nn.modules.conv.Conv2d,
    torch.nn.modules.batchnorm.BatchNorm2d,
    torch.nn.modules.activation.ReLU,
    torch.nn.modules.activation.SiLU,
])

# --- Load YOLO model ---
try:
    model = YOLO(model_path)
    st.title("🧠 Deteksi Objek dengan YOLOv8")
except Exception as e:
    st.error(f"❌ Gagal memuat model: {e}")
    st.stop()

# --- Upload gambar ---
uploaded_file = st.file_uploader("📤 Unggah gambar...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Baca gambar
    image = Image.open(uploaded_file)
    st.image(image, caption="🖼️ Gambar yang Diunggah", use_column_width=True)

    # --- Jalankan deteksi ---
    with st.spinner("🚀 Mendeteksi objek..."):
        results = model(image)

    # --- Tampilkan hasil ---
    for r in results:
        im_array = r.plot()  # hasil dalam BGR
        im = Image.fromarray(im_array[..., ::-1])  # ubah ke RGB
        st.image(im, caption="✅ Hasil Deteksi", use_column_width=True)

        # Simpan hasil sementara
        result_image_path = "result_image.png"
        im.save(result_image_path)

        # Encode ke base64 untuk HTML
        with open(result_image_path, "rb") as f:
            encoded_img = base64.b64encode(f.read()).decode()

        # Buat file HTML hasil deteksi
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <title>Hasil Deteksi Objek</title>
            <style>
                body {{ font-family: Arial, sans-serif; background: #f4f4f4; text-align: center; }}
                h1 {{ color: #333; }}
                img {{ max-width: 90%; height: auto; border-radius: 10px; box-shadow: 0 0 15px rgba(0,0,0,0.2); }}
            </style>
        </head>
        <body>
            <h1>Hasil Deteksi Objek (YOLOv8)</h1>
            <img src="data:image/png;base64,{encoded_img}" alt="Hasil Deteksi">
        </body>
        </html>
        """

        # Tombol download HTML
        st.download_button(
            label="💾 Unduh Hasil Deteksi (HTML)",
            data=html_content,
            file_name="hasil_deteksi_yolov8.html",
            mime="text/html"
        )

    # (Opsional) tampilkan hasil prediksi detail
    if st.checkbox("📋 Tampilkan detail prediksi"):
        for r in results:
            for box in r.boxes:
                cls = int(box.cls)
                conf = box.conf.item()
                xyxy = box.xyxy[0].tolist()
                st.write(f"- Class: {model.names[cls]}, Confidence: {conf:.2f}, Box: {xyxy}")
