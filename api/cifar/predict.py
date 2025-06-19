from fastapi import APIRouter
from pydantic import BaseModel
import numpy as np
from models.cifar10.cnn.cnn import CNN
import os

router = APIRouter()

class PredictRequest(BaseModel):
    input: list[float]

class PredictResponse(BaseModel):
    prediction: str
    top3: list[dict]

model = CNN()  # Instanciation du modèle CNN

model_path = "models/cifar10/cnn/model.bin"
# Check if the pre-trained model exists
if os.path.exists(model_path):
    print("CNN - CIFAR : Loading pre-trained CNN model...")
    model.load_model(model_path)
else:
    print("CNN - CIFAR : Pre-trained CNN model not found, please train the model first.")

CIFAR10_LABELS = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

@router.post("/cnn/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    if len(req.input) != 32 * 32 * 3:
        return PredictResponse(prediction="stub - 2", top3=[])

    if not model.is_ready():
        return PredictResponse(prediction="stub - 1", top3=[])

    mean = np.load("models/cifar10/cnn/mean_cifar10.npy")
    std = np.load("models/cifar10/cnn/std_cifar10.npy")

    X = np.array(req.input).reshape(1, 32, 32, 3).astype(np.float32)
    X = (X - mean) / std
    pred_probs = model.forward(X)

    pred_class_idx = int(np.argmax(pred_probs, axis=1)[0])
    pred_class_label = CIFAR10_LABELS[pred_class_idx]

    # Calcul du top 3
    top3_indices = np.argsort(pred_probs[0])[::-1][:3]
    top3 = [
        {
            "class": CIFAR10_LABELS[int(idx)],
            "probability": float(pred_probs[0][idx])
        }
        for idx in top3_indices
    ]
    
    return PredictResponse(prediction=pred_class_label, top3=top3)