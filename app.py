import streamlit as st
from PIL import Image
import torch
from torchvision import transforms, models

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Classes (should match your training)
classes = ['vial', 'Biriyani', 'Burger', 'chocobar', 'fried rice', 'gulab jamun', 
           'ice cream', 'Noodles', 'Pasta', 'puttu and kadala']

# Load model once, cache it for faster reloads
@st.cache_resource
def load_model():
    model = models.resnet50(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Linear(num_ftrs, len(classes))
    model.load_state_dict(torch.load("food_classifier.pth", map_location=device))
    model.eval()
    return model.to(device)

model = load_model()

# Transform - same as training
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


st.title("SnackFit: Outfit to Food AI 🥖🍦🥗")

uploaded_file = st.file_uploader("Upload your outfit photo", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="Your Outfit Photo", use_container_width=True)

    # Preprocess and predict
    input_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.nn.functional.softmax(output, dim=1)
        for i, p in enumerate(probs[0]):
            print(f"{classes[i]}: {p.item():.4f}")
        conf, pred_class = torch.max(probs, 1)

    st.markdown(f"### You resemble: **{classes[pred_class.item()]}**")
    st.markdown(f"Confidence: **{conf.item():.2%}**")
    st.caption("This AI judges your outfit like a food item — just for fun!")

