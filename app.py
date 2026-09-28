
import io

import torch
import torch.nn as nn
import torch.nn.functional as F

from PIL import Image
from torchvision import models, transforms

from fastapi import FastAPI, File, UploadFile


# ============================================================
# Configuration
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

classes = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck"
]


# ============================================================
# Model
# ============================================================

model = models.resnet50(weights=None)

model.fc = nn.Linear(
    model.fc.in_features,
    10
)

model.load_state_dict(
    torch.load(
        "resnet50_cifar10_best.pth",
        map_location=device
    )
)

model = model.to(device)

model.eval()


# ============================================================
# Preprocessing
# ============================================================

transform = transforms.Compose([
    transforms.Resize((224, 224)),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ============================================================
# Prediction Function
# ============================================================

def predict_image(image):

    # PIL Image
    image = image.convert("RGB")

    # Resize + Tensor + Normalize
    image = transform(image)

    # [3, 224, 224]
    # →
    # [1, 3, 224, 224]
    image = image.unsqueeze(0)

    image = image.to(device)

    # Inference
    with torch.no_grad():

        output = model(image)

        probabilities = F.softmax(
            output,
            dim=1
        )

        confidence, predicted_index = torch.max(
            probabilities,
            dim=1
        )

    prediction = classes[
        predicted_index.item()
    ]

    confidence = confidence.item() * 100

    return prediction, confidence


# ============================================================
# FastAPI
# ============================================================

app = FastAPI(
    title="CIFAR-10 ResNet50 API"
)


# ============================================================
# Health Check
# ============================================================

@app.get("/")
def home():

    return {
        "message": "CIFAR-10 ResNet50 API is running"
    }


# ============================================================
# Prediction API
# ============================================================

@app.post("/predict")
async def predict(
    file: UploadFile = File(...)
):

    # Read uploaded image
    image_bytes = await file.read()

    # Convert bytes → PIL Image
    image = Image.open(
        io.BytesIO(image_bytes)
    )

    # Reuse our prediction function
    prediction, confidence = predict_image(image)

    return {
        "prediction": prediction,
        "confidence": round(
            confidence,
            2
        )
    }
