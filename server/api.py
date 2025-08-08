import io
import pickle
import numpy as np
from PIL import Image
from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from transformers import CLIPProcessor, CLIPModel
import torch
import faiss
from scripts.utils import get_dominant_colors_cv2, color_distance

app = FastAPI()
app.mount("/static", StaticFiles(directory="server/static"), name="static")
templates = Jinja2Templates(directory="server/templates")

# Load model
device = "cuda" if torch.cuda.is_available() else "cpu"
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Load faiss index and metadata
index = faiss.read_index("data/faiss.index")
with open("data/meta.pkl", "rb") as f:
    meta = pickle.load(f)

def get_embedding_from_pil(img_pil):
    inputs = processor(images=img_pil, return_tensors="pt").to(device)
    with torch.no_grad():
        emb = model.get_image_features(**inputs)
    emb = emb / emb.norm(p=2, dim=-1, keepdim=True)
    return emb.cpu().numpy().astype('float32')

@app.get("/", response_class=HTMLResponse)
def index_page(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "result": None})

@app.post("/match", response_class=HTMLResponse)
async def match_image(request: Request, file: UploadFile = File(...)):
    contents = await file.read()
    img_pil = Image.open(io.BytesIO(contents)).convert("RGB")
    tmp_path = "tmp_uploaded.jpg"
    img_pil.save(tmp_path)

    colors = get_dominant_colors_cv2(tmp_path, k=3)  # colors are BGR numpy arrays

    # Convert BGR to RGB tuples (int)
    rgb_colors = [(int(c[2]), int(c[1]), int(c[0])) for c in colors]

    # Format as strings like "(R, G, B)"
    rgb_color_strings = [f"({r}, {g}, {b})" for r, g, b in rgb_colors]

    # Compose description
    color_desc = (
        f"The uploaded image displays dominant colors approximately: {', '.join(rgb_color_strings)}. "
        "It has a rich and balanced color palette with notable detailing."
    )

    emb = get_embedding_from_pil(img_pil)
    faiss.normalize_L2(emb)
    D, I = index.search(emb, 20)  # search more to get unique labels

    results = []
    seen_labels = set()

    for score, idx in zip(D[0], I[0]):
        item = meta[idx]
        label = item["label"]
        if label in seen_labels:
            continue
        seen_labels.add(label)
        distances = [min([color_distance(c, fc) for fc in item["colors"]]) for c in colors]
        color_score = float(np.mean(distances))
        results.append({
            "label": label,
            "score": float(score),
            "color_dist": color_score,
            "path": item["path"]
        })
        if len(results) >= 3:
            break

    # Explanation based on the most accurate match (first item)
    top = results[0]
    explanation = f"You look like **{top['label']}** — CLIP similarity {top['score']:.3f}, color distance {top['color_dist']:.1f}."

    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "result": results,
            "explanation": explanation,
            "color_desc": color_desc
        }
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server.api:app", host="0.0.0.0", port=8000, reload=True)
