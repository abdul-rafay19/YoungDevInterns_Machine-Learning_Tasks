import streamlit as st
import torch
from torchvision import transforms, models
from PIL import Image
import os

# Define transformations
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
])

# Define the model (using the pretrained ResNet18)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = models.resnet18(pretrained=True)
model.fc = torch.nn.Linear(model.fc.in_features, 2)  # 2 classes: cat and dog
model = model.to(device)

# Load the trained model
model.load_state_dict(torch.load("best_model.pth", map_location=device))  # Adjust path if necessary
model.eval()

# Streamlit app UI
st.title("Cat vs Dog Image Classification")
st.write("Upload an image of a cat or dog to classify!")

# Image upload widget
uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_image:
    # Open the image
    image = Image.open(uploaded_image).convert('RGB')

    # Apply the same transformations as during training
    image = transform(image).unsqueeze(0).to(device)

    # Perform prediction
    with torch.no_grad():
        outputs = model(image)
        _, preds = torch.max(outputs, 1)
        result = 'Dog' if preds.item() == 1 else 'Cat'

    # Display the image and result
    st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)
    st.write(f"Prediction: {result}")
