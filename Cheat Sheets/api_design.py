from fastapi import FastAPI, File, UploadFile
import torch
from PIL import Image
import io

app = FastAPI()

# Load the pre-trained CIFAR-10 model (Example: ResNet)
model = torch.load("cifar10_model.pth")
model.eval()

CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # 1. Read image bytes
    request_object_content = await file.read()
    img = Image.open(io.BytesIO(request_object_content)).convert("RGB")
    
    # 2. Preprocess (Resize to 32x32 for CIFAR-10)
    img = img.resize((32, 32))
    # ... apply other transforms (tensor conversion, normalization) ...

    # 3. Inference
    with torch.no_grad():
        prediction = model(img_tensor)
        label_idx = torch.argmax(prediction).item()
    
    return {"class": CLASSES[label_idx], "confidence": float(prediction.max())}