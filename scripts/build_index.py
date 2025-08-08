# scripts/build_index.py
import pickle
import numpy as np
import faiss
import os

def build_index(features_pkl="data/food_features.pkl",
                out_index="data/faiss.index",
                out_meta="data/meta.pkl"):

    # Load data once
    with open(features_pkl, "rb") as f:
        data = pickle.load(f)

    print(f"Loaded {len(data)} total entries from {features_pkl}")

    # Filter valid embeddings (NumPy safe check)
    valid_data = []
    for d in data:
        emb = d.get("embedding")
        if emb is not None:
            emb = np.array(emb, dtype="float32")
            if emb.size > 0 and not np.isnan(emb).any():
                valid_data.append({**d, "embedding": emb})

    print(f"Found {len(valid_data)} valid embeddings")

    if not valid_data:
        raise ValueError(
            f"No valid embeddings found in {features_pkl}. "
            "Please check preprocess.py output and dataset paths."
        )

    # Create FAISS index
    embeddings = np.stack([d["embedding"] for d in valid_data])
    d_dim = embeddings.shape[1]

    index = faiss.IndexFlatIP(d_dim)  # cosine similarity
    faiss.normalize_L2(embeddings)
    index.add(embeddings)

    # Save index
    os.makedirs(os.path.dirname(out_index), exist_ok=True)
    faiss.write_index(index, out_index)

    # Save metadata
    meta = [
        {"label": d["label"], "path": d["path"], "colors": d["colors"]}
        for d in valid_data
    ]
    with open(out_meta, "wb") as f:
        pickle.dump(meta, f)

    print(f"Saved FAISS index to {out_index}")
    print(f"Saved metadata to {out_meta}")
    print(f"Total embeddings indexed: {len(valid_data)}")


if __name__ == "__main__":
    build_index()
