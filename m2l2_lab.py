"""
M2L2 - Lab 2

Auto-extracted verbatim from: M2L2_Lab_2.ipynb
Code cells are unchanged; notebook magics are commented out.
"""


# ============================================================================
# **Similarity Retrieval with Metadata Filtering** | Introduction | Objectives
# ============================================================================

# ================================
# Import dependencies
# ================================

# Standard library
import os
from pathlib import Path

# Third-party library
import numpy as np
import torch
from PIL import Image
from langchain_chroma import Chroma
from sentence_transformers import SentenceTransformer
from transformers import CLIPModel, CLIPProcessor

print("✅ Environment ready")


# ============================================================================
# Environment and database validation
# ============================================================================

# ================================
# Verify vector database
# ================================

DB_DIR = str((Path.home() / "chroma_multimodal").resolve())

if not os.path.isdir(DB_DIR):
    raise RuntimeError(
        f"Vector database directory not found: '{DB_DIR}'. "
        "Please run Lesson 1 (Multimodal Vector Index Construction) first."
    )

article_db = Chroma(
    collection_name="restaurant_articles",
    persist_directory=DB_DIR,
)

image_db = Chroma(
    collection_name="food_images",
    persist_directory=DB_DIR,
)

n_articles = article_db._collection.count()
n_images = image_db._collection.count()

if n_articles <= 0 or n_images <= 0:
    raise RuntimeError(
        "One or more collections are empty. Please rerun Lesson 1 to rebuild the index."
    )

print(f"✅ Article vectors: {n_articles}")
print(f"✅ Image vectors:   {n_images}")


# ============================================================================
# Embedding model initialization
# ============================================================================

# ================================
# Initialize embedding models
# ================================

# ---- Text embedding model (384-d) ----
text_model = SentenceTransformer("all-MiniLM-L6-v2")

def embed_texts(texts, batch_size=64):
    return text_model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=False,
        normalize_embeddings=True,  # cosine-ready
    ).astype(np.float32)

print("✅ Text embedder ready")


# ---- Image embedding model (512-d) ----
device = "cpu"
clip_name = "openai/clip-vit-base-patch32"
clip_model = CLIPModel.from_pretrained(clip_name).to(device)
clip_processor = CLIPProcessor.from_pretrained(clip_name, use_fast=True)
clip_model.eval()

@torch.no_grad()
def embed_images(paths, batch_size=16):
    vecs = []
    for i in range(0, len(paths), batch_size):
        batch = paths[i:i+batch_size]
        imgs = [Image.open(p).convert("RGB") for p in batch]
        inputs = clip_processor(images=imgs, return_tensors="pt").to(device)
        feats = clip_model.get_image_features(**inputs)          # (B,512)
        feats = feats / feats.norm(dim=-1, keepdim=True)         # cosine-ready
        vecs.append(feats.cpu().numpy().astype(np.float32))
    return np.vstack(vecs)

print("✅ Image embedder ready")


# ============================================================================
# Retrieval utilities
# ============================================================================



def _unwrap(res: dict):  
    ids = res.get("ids", [[]])[0]
    docs = res.get("documents", [[]])[0]
    metas = res.get("metadatas", [[]])[0]
    dists = res.get("distances", [[]])[0]
    return ids, docs, metas, dists


def print_hits(ids, docs, metas, dists, title: str, max_chars: int = 180):
    print(f"\n=== {title} ===")
    for i in range(len(ids)):
        meta = metas[i] if i < len(metas) else {}
        dist = float(dists[i]) if i < len(dists) else None

        snippet = (docs[i] or "").replace("\n", " ").strip()
        if len(snippet) > max_chars:
            snippet = snippet[:max_chars].rstrip() + "..."

        # compact metadata view
        cuisine = meta.get("cuisine", "N/A") if isinstance(meta, dict) else "N/A"
        location = meta.get("location", "N/A") if isinstance(meta, dict) else "N/A"
        doc_id = meta.get("doc_id", "N/A") if isinstance(meta, dict) else "N/A"
        source = meta.get("source", "N/A") if isinstance(meta, dict) else "N/A"

        print(f"[{i+1}] id={doc_id} | cuisine={cuisine} | location={location} | source={source} | distance={dist:.4f}")
        print(f"{snippet}")


# ============================================================================
# Article similarity retrieval
# ============================================================================

# ================================
# Article retrieval
# ================================

# Similarity retrieval over restaurant articles with optional metadata filtering.
def retrieve_articles(query: str, k: int = 5, where: dict | None = None):

    q_vec = embed_texts([query])[0]  # 384-d, cosine-ready

    res = article_db._collection.query(
        query_embeddings=[q_vec.tolist()],
        n_results=k,
        where=where,
        include=["documents", "metadatas", "distances"],
    )
    return _unwrap(res)

print("✅ Article retrieval ready")


# ============================================================================
# Image similarity retrieval
# ============================================================================

# ================================
# Image retrieval
# ================================

# Similarity retrieval over food images using an image query.
def retrieve_images_by_image(query_image_path: str, k: int = 5, where: dict | None = None):

    q_vec = embed_images([query_image_path])[0]  # 512-d, cosine-ready

    res = image_db._collection.query(
        query_embeddings=[q_vec.tolist()],
        n_results=k,
        where=where,
        include=["documents", "metadatas", "distances"],
    )
    return _unwrap(res)

print("✅ Image retrieval ready")


# ============================================================================
# Demo 1 — Article similarity search (no filter)
# ============================================================================

# ================================
# Demo 1 — Article similarity search (no filter)
# ================================

q = "cozy restaurant with noodles and warm atmosphere"

ids, docs, metas, dists = retrieve_articles(q, k=5, where=None)
print_hits(ids, docs, metas, dists, title="Demo 1 — Article similarity search (no filter)")

print("✅ Demo 1 complete")


# ============================================================================
# Demo 2 — Article similarity search + metadata filter
# ============================================================================

# ================================
# Demo 2 — Article similarity search + metadata filter
# ================================

q = "handmade pasta and romantic dinner"

# ---- metadata constraint (must exist in your dataset) ----
where_filter = {"location": "Pasadena"}  # adjust if needed

ids, docs, metas, dists = retrieve_articles(q, k=5, where=where_filter)

if len(ids) == 0:
    print("⚠️ No results found with current filter.")
else:
    print_hits(ids, docs, metas, dists, title="Demo 2 — Article similarity search + metadata filter")
    
print("✅ Demo 2 complete")


# ============================================================================
# Demo 3 — Image similarity search (image→image)
# ============================================================================

# ================================
# Demo 3 — Image similarity search (image→image)
# ================================

meta_all = image_db._collection.get(include=["metadatas"])["metadatas"]

QUERY_INDEX = 0  # ← change this (0 … N-1) to try different images

if QUERY_INDEX >= len(meta_all):
    raise ValueError("QUERY_INDEX out of range.")

query_img = meta_all[QUERY_INDEX]["image_path"]

print(f"Query image: {query_img}")
img = Image.open(query_img)
img.thumbnail((300, 300))
display(img)
# display(Image.open(query_img))

# TODO:
# 1. Create an optional metadata filter for recipe images
# 2. Retrieve the top-5 most similar images using query_img
# 3. Display the retrieved results with metadata (use title="Demo 3 — Image similarity search (image→image)" when printing results)

# your code here

# 1. Optional metadata filter: restrict the search to recipe images only
image_where_filter = {"source": "recipe_image"}   # set to None to search without any constraint

# 2. Retrieve the top-5 most similar images to the query image
ids, docs, metas, dists = retrieve_images_by_image(
    query_image_path=query_img,
    k=5,
    where=image_where_filter,
)

# 3. Display the retrieved results together with their metadata
if len(ids) == 0:
    print("⚠️ No results found with current filter.")
else:
    print_hits(
        ids, docs, metas, dists,
        title="Demo 3 — Image similarity search (image→image)"
    )

print("✅ Demo 3 complete")
print("🎉 Similarity Retrieval with Metadata Filtering COMPLETE")

