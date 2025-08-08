import torch
from torchvision import transforms, models
from PIL import Image
import os

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Classes - should match your training dataset classes
classes = ['vial', 'Biriyani', 'Burger', 'chocobar', 'fried rice', 'gulab jamun', 
           'ice cream', 'Noodles', 'Pasta', 'puttu and kadala']

# Load model
model = models.resnet50(weights=None)
num_ftrs = model.fc.in_features
model.fc = torch.nn.Linear(num_ftrs, len(classes))
model.load_state_dict(torch.load("food_classifier.pth", map_location=device))
model.eval()
model = model.to(device)

# Image transform - same as training
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406], 
        std=[0.229, 0.224, 0.225]
    )
])

def predict(image_path, topk=3):
    image = Image.open(image_path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.nn.functional.softmax(output, dim=1)
        top_probs, top_idxs = torch.topk(probs, topk)

    print(f"Top {topk} predictions:")
    for i in range(topk):
        print(f"  {classes[top_idxs[0][i]]}: {top_probs[0][i].item():.4f}")

if __name__ == "__main__":
    # Replace with path to your test image
    test_image_path = "Datatset/Burger/14.jpeg"  
    if os.path.exists(test_image_path):
        predict(test_image_path)
    else:
        print(f"Image path {test_image_path} does not exist.")
