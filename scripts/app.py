import os
import gradio as gr
import torch
import torchvision.transforms as T
import torch.nn as nn
from torchvision.models import resnet18
from PIL import Image
import numpy as np

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

model = resnet18()
model.fc = nn.Sequential(
    nn.Linear(model.fc.in_features, 128),
    nn.ReLU(),
    nn.Linear(128, 1)
)
model.to(DEVICE)

# Charger les poids seulement s'ils existent
model_path = os.path.join("models", "resnet_weights.pt")
if os.path.exists(model_path):
    state_dict = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.eval()
    print("Model loaded successfully")
else:
    print("Warning: Model weights not found, using random weights")
    model.eval()

transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])
])

def predict(image):
    if image is None:
        return "Please upload an image.", None
    
    
    print(f"Input type: {type(image)}")
    print(f"Input shape/size: {getattr(image, 'shape', getattr(image, 'size', 'unknown'))}")
    
    
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image.astype('uint8'), 'RGB')
    elif not isinstance(image, Image.Image):
        return "Error: Invalid image format", None
    
    info = f"Original size: {image.size[0]}x{image.size[1]} pixels\n"
    
    # Predict
    small = image.convert("RGB").resize((224, 224))
    crop_px = int(224 * 0.11)
    input_image = small.crop((crop_px, crop_px, 224 - crop_px, 224 - crop_px)).resize((224, 224))
    tensor = transform(input_image).unsqueeze(0).to(DEVICE)
    
    with torch.no_grad():
        y = model(tensor).item()
        y = max(0.0, min(100.0, y))
    
    result = f"{info}\n{y:.1f}% of the surface is impervious"
    
    return result, image


iface = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs=[
        gr.Textbox(label="Result"),
        gr.Image(label="Uploaded Image", type="pil")
    ],
    title="Imperviousness Estimator",
    description="Upload an aerial image to analyze surface imperviousness."
)

# Lancer l'interface
if __name__ == "__main__":
    
    iface.launch(debug=True, share=False)