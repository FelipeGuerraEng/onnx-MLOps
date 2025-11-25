# tests/test_model.py
import os
import sys

# Añadimos la raíz del proyecto al sys.path para que 'app' sea importable
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import onnx
from onnx import numpy_helper

from app.onnx_inference import OnnxMNISTModel
from app.storage import get_blob_bytes


def _load_test_tensors_from_blob():
    """
    Lee los tensores de input y output de prueba desde Azure Blob en memoria.
    Espera:
      - TEST_INPUT_BLOB_PATH  -> data/test_data_set_0/input_0.pb
      - TEST_OUTPUT_BLOB_PATH -> data/test_data_set_0/output_0.pb
    """
    input_blob_path = os.getenv("TEST_INPUT_BLOB_PATH", "data/test_data_set_0/input_0.pb")
    output_blob_path = os.getenv("TEST_OUTPUT_BLOB_PATH", "data/test_data_set_0/output_0.pb")

    input_bytes = get_blob_bytes(input_blob_path)
    output_bytes = get_blob_bytes(output_blob_path)

    input_tensor = onnx.TensorProto()
    input_tensor.ParseFromString(input_bytes)

    output_tensor = onnx.TensorProto()
    output_tensor.ParseFromString(output_bytes)

    input_array = numpy_helper.to_array(input_tensor)   # (1, 1, 28, 28)
    output_array = numpy_helper.to_array(output_tensor) # (1, 10) normalmente

    return input_array, output_array


def test_model_predicts_for_fixed_input():
    """
    Prueba 1: el modelo responde para una entrada fija (la de input_0.pb).
    """
    model = OnnxMNISTModel()
    input_array, _ = _load_test_tensors_from_blob()

    # Aplanamos a 784 pixeles como espera OnnxMNISTModel.predict
    pixels = input_array.reshape(-1).astype("float32").tolist()

    pred = model.predict(pixels)

    assert isinstance(pred, int)
    assert 0 <= pred <= 9


def test_model_output_close_to_reference():
    """
    Prueba 2: la salida del modelo no se desvía significativamente
    de la salida esperada (output_0.pb).
    """
    model = OnnxMNISTModel()
    input_array, expected_output = _load_test_tensors_from_blob()

    # Ejecutamos el modelo usando el tensor tal cual (1, 1, 28, 28)
    input_array = input_array.astype("float32")
    outputs = model.session.run([model.output_name], {model.input_name: input_array})
    model_output = outputs[0]  # shape (1, 10)

    # Calculamos el error máximo absoluto entre las dos salidas
    diff = np.max(np.abs(model_output - expected_output))

    max_allowed_diff = float(os.getenv("MAX_TEST_OUTPUT_DIFF", "0.001"))
    assert (
        diff <= max_allowed_diff
    ), f"Diferencia máxima {diff:.6f} > umbral permitido {max_allowed_diff}"
