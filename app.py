import torch
import torch.nn as nn
import gradio as gr
from PIL import Image
from torchvision import models, transforms

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "quantized_efficientnet.pth"

# Hardcoded class names for trash classification
class_names = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash', 'compost']

# Load the quantized model
model = torch.load(MODEL_PATH, map_location=DEVICE)
model.eval()

# Image preprocessing
normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    normalize
])

def predict(image):
    if image is None:
        return {"error": "No image uploaded"}

    image = image.convert("RGB")
    tensor = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.softmax(outputs, dim=1)[0]

    confidences = {class_names[i]: float(probs[i]) for i in range(len(class_names))}
    predicted_idx = torch.argmax(probs).item()
    predicted_class = class_names[predicted_idx]

    return confidences

demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="Upload Trash Image"),
    outputs=gr.Label(num_top_classes=7, label="Predictions"),
    title="Trash Classification with EfficientNet-B0",
    description="Upload an image and the model will classify it as cardboard, e-waste, glass, medical, metal, paper, or plastic."
)

if __name__ == "__main__":
    demo.launch(ssr_mode=False)