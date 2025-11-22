from typing import List

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from .onnx_inference import OnnxMNISTModel
from .storage import append_prediction_log

app = FastAPI(title="MNIST ONNX API", version="1.0.0")

model = OnnxMNISTModel()


class PredictRequest(BaseModel):
    pixels: List[float]


class PredictResponse(BaseModel):
    prediction: int


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest) -> PredictResponse:
    try:
        pred = model.predict(request.pixels)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error interno al inferir: {e}")

    # Log de la predicción (solo metadatos sencillos)
    append_prediction_log(prediction=pred, inputs=request.pixels)

    return PredictResponse(prediction=pred)
