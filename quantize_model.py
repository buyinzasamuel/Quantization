import torch
from torchvision import models

# Load pretrained EfficientNet-B0
model = models.efficientnet_b0(weights='DEFAULT')

# Modify the classifier for 7 classes (assuming trash classification)
in_features = model.classifier[1].in_features
model.classifier[1] = torch.nn.Linear(in_features, 7)

# Set model to evaluation mode
model.eval()

print("Original model size:")
print(f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

# Apply dynamic quantization
quantized_model = torch.quantization.quantize_dynamic(
    model, 
    {torch.nn.Linear},  # Quantize Linear layers
    dtype=torch.qint8
)

print("Quantized model size:")
print(f"Model parameters: {sum(p.numel() for p in quantized_model.parameters() if p.requires_grad)}")

# Save the quantized model
torch.save(quantized_model, 'quantized_efficientnet.pth')

print("Quantized model saved as 'quantized_efficientnet.pth'")