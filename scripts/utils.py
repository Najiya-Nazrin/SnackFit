# scripts/utils.py
import cv2
import numpy as np
from sklearn.cluster import KMeans
from PIL import Image
import os

def load_image_pil(path):
    return Image.open(path).convert("RGB")

def get_dominant_colors_cv2(image_path, k=3):
    """Return list of k RGB dominant colors (0-255 ints)."""
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Cannot read {image_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]
    # downsample if very large
    sample = img.reshape(-1, 3).astype(float)
    # KMeans
    kmeans = KMeans(n_clusters=k, random_state=0)
    kmeans.fit(sample)
    colors = kmeans.cluster_centers_.astype(int)
    return [tuple(c.tolist()) for c in colors]

def color_distance(c1, c2):
    """Euclidean distance between RGB tuples."""
    return np.linalg.norm(np.array(c1) - np.array(c2))
