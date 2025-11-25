# app/onnx_inference.py
import os
from typing import List

import numpy as np
import onnxruntime as ort
import onnx
from onnx import numpy_helper

from .storage import get_blob_bytes

def load_example_pixels_from_blob():
    """
    Carga el tensor de ejemplo (input_0.pb) desde Blob y lo devuelve
    como lista de 784 píxeles (float) para usar en /predict_example.
    """
    input_blob_path = os.getenv("TEST_INPUT_BLOB_PATH", "data/test_data_set_0/input_0.pb")
    input_bytes = get_blob_bytes(input_blob_path)

    tensor = onnx.TensorProto()
    tensor.ParseFromString(input_bytes)

    arr = numpy_helper.to_array(tensor).astype("float32")  # (1, 1, 28, 28)
    pixels = arr.reshape(-1).tolist()  # 784 valores
    return pixels

class OnnxMNISTModel:
    """
    MNIST ONNX cargado desde Blob Storage de Azure.
    """

    def __init__(self) -> None:
    
        self.model_blob_path = os.getenv("MODEL_BLOB_PATH", "models/mnist-12.onnx")
        model_bytes = get_blob_bytes(self.model_blob_path)

        # Cargar sesión desde bytes
        self.session = ort.InferenceSession(model_bytes, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    def predict(self, pixels: List[float]) -> int:
        arr = np.array(pixels, dtype=np.float32)
        if arr.shape[0] != 784:
            raise ValueError(f"Se esperaban 784 valores, se recibieron {arr.shape[0]}")

        arr = arr.reshape(1, 1, 28, 28)

        outputs = self.session.run([self.output_name], {self.input_name: arr})
        logits = outputs[0]
        predicted_class = int(np.argmax(logits, axis=1)[0])
        return predicted_class
