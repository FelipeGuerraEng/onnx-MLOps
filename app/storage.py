# app/storage.py
import os
import json
from datetime import datetime
from typing import List

from azure.storage.blob import BlobServiceClient, BlobClient, ContentSettings
from dotenv import load_dotenv

load_dotenv()


def _get_blob_service_client() -> BlobServiceClient:
    connection_string = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
    if not connection_string:
        raise RuntimeError("AZURE_STORAGE_CONNECTION_STRING no está definido")
    return BlobServiceClient.from_connection_string(connection_string)


def _get_blob_client(blob_path: str) -> BlobClient:
    container_name = os.getenv("AZURE_STORAGE_CONTAINER_NAME", "proyecto-ml")
    service = _get_blob_service_client()
    container_client = service.get_container_client(container_name)
    return container_client.get_blob_client(blob_path)


def get_blob_bytes(blob_path: str) -> bytes:
    """
    Devuelve el contenido del blob como bytes
    """
    blob_client = _get_blob_client(blob_path)
    downloader = blob_client.download_blob()
    return downloader.readall()


def get_blob_text(blob_path: str, encoding: str = "utf-8") -> str:
    """
    Devuelve el contenido del blob como texto
    """
    data = get_blob_bytes(blob_path)
    return data.decode(encoding)


def append_log_line(blob_path: str, line: str) -> None:
    """
    Agrega una línea al blob de logs.
    Implementado leyendo el contenido actual en memoria y re-subiendo.
    """
    blob_client = _get_blob_client(blob_path)

    try:
        existing = blob_client.download_blob().readall().decode("utf-8")
    except Exception:
        existing = ""

    new_content = existing + line

    blob_client.upload_blob(
        new_content.encode("utf-8"),
        overwrite=True,
        content_settings=ContentSettings(content_type="text/plain"),
    )


def append_prediction_log(prediction: int, inputs: List[float]) -> None:
    """
    Registra la predicción en el blob correspondiente (dev o prod).
    """
    environment = os.getenv("ENVIRONMENT", "dev").lower()

    if environment == "prod":
        default_path = "logs/predicciones_prod.txt"
    else:
        default_path = "logs/predicciones_dev.txt"

    blob_path = os.getenv("PREDICTIONS_LOG_BLOB_PATH", default_path)

    line_dict = {
        "timestamp_utc": datetime.utcnow().isoformat(),
        "environment": environment,
        "prediction": int(prediction),
        "input_length": len(inputs),
    }

    line = json.dumps(line_dict) + "\n"
    append_log_line(blob_path, line)
