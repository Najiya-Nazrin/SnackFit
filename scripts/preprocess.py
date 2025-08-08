# scripts/preprocess.py
import os
import pickle
import numpy as np
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPModel
import torch
from utils import get_dominant_colors_cv2, load_image_pil

device = "cuda" if torch.cuda.is_available() else "cpu"

def extract_clip_embedding(image_pil, model, processor):
    inputs = processor(images=image_pil, return_tensors="pt").to(device)
    with torch.no_grad():
        img_emb = model.get_image_features(**inputs)
    img_emb = img_emb / img_emb.norm(p=2, dim=-1, keepdim=True)
    return img_emb.cpu().numpy()[0]

def main(dataset_dir="Datatset", out_file="data/food_features.pkl"):
    # Check dataset directory before processing
    abs_path = os.path.abspath(dataset_dir)
    print(f"Looking for dataset folder at: {abs_path}")
    print(f"Folder exists? {os.path.exists(dataset_dir)}")

    if not os.path.exists(dataset_dir):
        print(f"Dataset directory '{dataset_dir}' not found. Please check the path.")
        return

    # List folder contents for debugging
    try:
        contents = os.listdir(dataset_dir)
        print(f"Contents of dataset folder: {contents}")
    except Exception as e:
        print(f"Error reading dataset folder contents: {e}")
        return

    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    os.makedirs(os.path.dirname(out_file), exist_ok=True)

    data = []  # list of dicts: {label, image_path, colors, embedding}

    total_labels = 0
    total_images = 0
    processed_images = 0

    for label in sorted(os.listdir(dataset_dir)):
        label_dir = os.path.join(dataset_dir, label)
        if not os.path.isdir(label_dir):
            print(f"Skipping non-directory: {label_dir}")
            continue
        total_labels += 1
        print(f"Processing label: {label} ...")
        files = [f for f in os.listdir(label_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        print(f"  Found {len(files)} image files.")
        for fname in files:
            total_images += 1
            p = os.path.join(label_dir, fname)
            try:
                colors = get_dominant_colors_cv2(p, k=3)
                img = load_image_pil(p)
                emb = extract_clip_embedding(img, model, processor)
                data.append({"label": label, "path": p, "colors": colors, "embedding": emb})
                processed_images += 1
                if processed_images % 10 == 0:
                    print(f"  Processed {processed_images} images so far...")
            except Exception as e:
                print(f"Skipping {p}, error: {e}")

    print(f"Total labels found: {total_labels}")
    print(f"Total images found: {total_images}")
    print(f"Total images processed successfully: {processed_images}")

    # Save to pickle
    with open(out_file, "wb") as f:
        pickle.dump(data, f)
    print("Saved features to", out_file)

if __name__ == "__main__":
    main(dataset_dir="Datatset")  # use your actual dataset folder name here
