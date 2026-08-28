import io
import os
from pathlib import Path

import torch
import torch.nn.functional as F
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
from torchvision import transforms

import sys
sys.path.insert(0, str(Path(__file__).parent))
from model import get_model

app = FastAPI(title="CIFAR-10 Classifier")

CHECKPOINT_PATH = os.environ.get("CHECKPOINT_PATH", "/app/checkpoints/classifier_v1.pt")
CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]

_model = None
_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.4914, 0.4822, 0.4465],
        std=[0.2470, 0.2435, 0.2616],
    ),
])


def load_model() -> None:
    global _model
    ckpt_path = Path(CHECKPOINT_PATH)
    if not ckpt_path.exists():
        return
    checkpoint = torch.load(ckpt_path, map_location=_device)
    model = get_model(architecture="resnet18", num_classes=10).to(_device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    _model = model


@app.on_event("startup")
def startup_event() -> None:
    load_model()


@app.get("/health")
def health() -> JSONResponse:
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return JSONResponse({"status": "ok"})


@app.post("/predict")
async def predict(image: UploadFile = File(...)) -> JSONResponse:
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    data = await image.read()
    img = Image.open(io.BytesIO(data)).convert("RGB")
    tensor = _transform(img).unsqueeze(0).to(_device)
    with torch.no_grad():
        logits = _model(tensor)
        probs = F.softmax(logits, dim=1).squeeze(0).tolist()
    return JSONResponse({
        "predicted_class": CIFAR10_CLASSES[int(torch.tensor(probs).argmax())],
        "probabilities": {cls: round(p, 6) for cls, p in zip(CIFAR10_CLASSES, probs)},
    })


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
