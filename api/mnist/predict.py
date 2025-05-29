from fastapi import APIRouter
from pydantic import BaseModel
import numpy as np
from models.mnist.mlp import MLP
import os

router = APIRouter()

class PredictRequest(BaseModel):
    input: list[float]

class PredictResponse(BaseModel):
    prediction: int
    top3: list[dict]

model = MLP()

model_path = "models/mnist/model.npz"
# Check if the pre-trained model exists
if (os.path.exists(model_path)):
    print("MLP - MNIST : Loading pre-trained model...")
    model.load_model(model_path)
else:
    print("MLP - MNIST : Pre-trained model not found, please train the model first.")


@router.post("/mlp/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    if len(req.input) != 784:
        return PredictResponse(prediction=-2, top3=[])

    if not model.ready:
        return PredictResponse(prediction=-1, top3=[])

    X = np.array(req.input).reshape(1, -1).astype(np.float32)
    pred_probs = model.forward(X)
    pred_class = int(np.argmax(pred_probs, axis=1)[0])

    # Calcul du top 3
    top3_indices = np.argsort(pred_probs[0])[::-1][:3]
    top3 = [
        {"class": int(idx), "probability": float(pred_probs[0][idx])}
        for idx in top3_indices
    ]
    
    return PredictResponse(prediction=pred_class, top3=top3)