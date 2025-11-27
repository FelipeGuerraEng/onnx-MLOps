import os
import sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import io
import time
import numpy as np
import streamlit as st
from PIL import Image, ImageOps

from app.onnx_inference import OnnxMNISTModel
from app.storage import append_prediction_log

st.set_page_config(page_title="MNIST ONNX - Demo", page_icon="🔮", layout="centered")

st.title("MNIST ONNX - Proyecto MLOPS")
st.write("Sube una imagen de un dígito (0–9). La app la normaliza a 28x28 y predice con el modelo ONNX.")

@st.cache_resource
def get_model():
    return OnnxMNISTModel()

model = get_model()

uploaded = st.file_uploader("Sube una imagen (PNG/JPG)", type=["png", "jpg", "jpeg"])

def image_to_mnist_pixels(img: Image.Image) -> list[float]:
    """
    Convierte una imagen cualquiera a formato MNIST:
    - escala de grises
    - 28x28
    - normalización [0,1]
    - flatten a 784 floats
    """
    img = ImageOps.grayscale(img)
    img = img.resize((28, 28))
    arr = np.array(img).astype("float32")

    # Normalizar a [0,1]
    arr = arr / 255.0

    # Muchos MNIST esperan "blanco sobre negro" o viceversa.
    # Si tu modelo queda al revés, descomenta la inversión:
    # arr = 1.0 - arr

    # Aplanar a 784
    return arr.reshape(-1).tolist()

if uploaded is not None:
    try:
        img_bytes = uploaded.read()
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")

        st.subheader("Imagen cargada")
        st.image(img, caption="Imagen original", use_container_width=True)

        if st.button("🔮 Predecir"):
            t0 = time.time()
            pixels = image_to_mnist_pixels(img)
            pred = model.predict(pixels)
            latency_ms = int((time.time() - t0) * 1000)

            # Log en Blob (igual que API)
            append_prediction_log(prediction=pred, inputs=pixels)

            st.success(f"Predicción: **{pred}**")
            st.caption(f"Latencia aprox: {latency_ms} ms")

            st.write("Vista previa (28x28 en grises):")
            img28 = ImageOps.grayscale(img).resize((28, 28))
            st.image(img28, width=200)

    except Exception as e:
        st.error(f"Error procesando la imagen: {e}")
else:
    st.info("Sube una imagen para comenzar.")
