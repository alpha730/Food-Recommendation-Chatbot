"""
M2L3 - Lab 2

Auto-extracted verbatim from: M2L3_Lab_2.ipynb
Code cells are unchanged; notebook magics are commented out.
"""


# ============================================================================
# **Multimodal Similarity Fusion and Retrieval Ranking** | Introduction | Objectives
# ============================================================================

# ================================
# Import environment
# (Dependencies were installed in Lesson 1)
# ================================

# Standard library
import os
from pathlib import Path

# Third-party
import numpy as np
import torch
from PIL import Image
from langchain_chroma import Chroma
from sentence_transformers import SentenceTransformer
from transformers import CLIPModel, CLIPProcessor

print("✅ Environment ready")


# ============================================================================
# Verify vector databases
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

article_db = Chroma(collection_name="restaurant_articles", persist_directory=DB_DIR)
image_db   = Chroma(collection_name="food_images",          persist_directory=DB_DIR)

n_articles = article_db._collection.count()
n_images   = image_db._collection.count()

if n_articles <= 0 or n_images <= 0:
    raise RuntimeError(
        "One or more collections are empty. Please rerun Lesson 1 to rebuild the index."
    )

print(f"✅ Article vectors: {n_articles}")
print(f"✅ Image vectors:   {n_images}")


# ============================================================================
# Initialize embedding models
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


# ---- CLIP embedding model (512-d) for image + query text ----
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

@torch.no_grad()
def embed_query_clip_text(query: str):
    inputs = clip_processor(text=[query], return_tensors="pt", padding=True).to(device)
    feats = clip_model.get_text_features(**inputs)              # (1,512)
    feats = feats / feats.norm(dim=-1, keepdim=True)            # cosine-ready
    return feats[0].cpu().numpy().astype(np.float32)

print("✅ CLIP embedders ready")


# ============================================================================
# Prepare utility functions
# ============================================================================

# ================================
# Utilities
# ================================

def _unwrap(res: dict):
    """Chroma returns lists-of-lists; unwrap the first query."""
    ids   = res.get("ids", [[]])[0]
    docs  = res.get("documents", [[]])[0]
    metas = res.get("metadatas", [[]])[0]
    dists = res.get("distances", [[]])[0]
    return ids, docs, metas, dists

def _to_similarity(dists):
    """Convert 'smaller is better' distance to 'larger is better' similarity."""
    d = np.array(dists, dtype=np.float32)
    return 1.0 - d

def _minmax(x):
    """Min-max normalize to [0, 1] with safe handling for constant arrays."""
    x = np.array(x, dtype=np.float32)
    if x.size == 0:
        return x
    lo, hi = float(x.min()), float(x.max())
    if abs(hi - lo) < 1e-8:
        return np.ones_like(x)  # all equal -> treat as same confidence
    return (x - lo) / (hi - lo)

def print_hits(ids, docs, metas, scores, title: str, max_chars: int = 140):
    print(f"\n=== {title} ===")
    for i in range(len(ids)):
        meta = metas[i] if i < len(metas) else {}
        score = float(scores[i]) if i < len(scores) else None

        snippet = (docs[i] or "").replace("\n", " ").strip()
        if len(snippet) > max_chars:
            snippet = snippet[:max_chars].rstrip() + "..."

        cuisine = meta.get("cuisine", "N/A") if isinstance(meta, dict) else "N/A"
        location = meta.get("location", "N/A") if isinstance(meta, dict) else "N/A"
        doc_id = meta.get("doc_id", ids[i]) if isinstance(meta, dict) else ids[i]
        source = meta.get("source", "N/A") if isinstance(meta, dict) else "N/A"

        print(f"[{i+1}] id={doc_id} | cuisine={cuisine} | location={location} | source={source} | score={score:.4f}")
        print(f"{snippet}")


# ============================================================================
# Define Retrieval functions (text → articles, text → images)
# ============================================================================

# ================================
# Retrieval functions
# ================================

def retrieve_articles(query: str, k: int = 5, where: dict | None = None):
    q_vec = embed_texts([query])[0]  # 384-d
    res = article_db._collection.query(
        query_embeddings=[q_vec.tolist()],
        n_results=k,
        where=where,
        include=["documents", "metadatas", "distances"],
    )
    ids, docs, metas, dists = _unwrap(res)
    sims = _to_similarity(dists)
    return ids, docs, metas, sims

def retrieve_images_by_text(query: str, k: int = 5, where: dict | None = None):
    q_vec = embed_query_clip_text(query)  # 512-d
    res = image_db._collection.query(
        query_embeddings=[q_vec.tolist()],
        n_results=k,
        where=where,
        include=["documents", "metadatas", "distances"],
    )
    ids, docs, metas, dists = _unwrap(res)
    sims = _to_similarity(dists)
    return ids, docs, metas, sims

print("✅ Retrieval functions ready")


# ============================================================================
# Implement multimodal fusion and reranking
# ============================================================================

# ================================
# Multimodal fusion
# ================================

def fuse_rank(
    query: str,
    k_text: int = 5,
    k_img: int = 5,
    w_text: float = 0.6,
    w_img: float = 0.4,
    where_text: dict | None = None,
    where_img: dict | None = None,
    top_n: int = 5
):
    # Retrieve per modality
    t_ids, t_docs, t_metas, t_sims = retrieve_articles(query, k=k_text, where=where_text)
    i_ids, i_docs, i_metas, i_sims = retrieve_images_by_text(query, k=k_img, where=where_img)

    # Normalize within modality
    t_norm = _minmax(t_sims)
    i_norm = _minmax(i_sims)

    # Build one mixed candidate list with fused scores
    rows = []
    for j in range(len(t_ids)):
        rows.append({
            "modality": "article",
            "id": t_metas[j].get("doc_id", t_ids[j]) if isinstance(t_metas[j], dict) else t_ids[j],
            "cuisine": t_metas[j].get("cuisine", "N/A") if isinstance(t_metas[j], dict) else "N/A",
            "location": t_metas[j].get("location", "N/A") if isinstance(t_metas[j], dict) else "N/A",
            "source": t_metas[j].get("source", "N/A") if isinstance(t_metas[j], dict) else "N/A",
            "text_score": float(t_norm[j]),
            "img_score": 0.0,
            "fused": float(w_text * t_norm[j]),
            "snippet": (t_docs[j] or "").replace("\n", " ").strip(),
        })

    for j in range(len(i_ids)):
        rows.append({
            "modality": "image",
            "id": i_metas[j].get("doc_id", i_ids[j]) if isinstance(i_metas[j], dict) else i_ids[j],
            "cuisine": i_metas[j].get("cuisine", "N/A") if isinstance(i_metas[j], dict) else "N/A",
            "location": i_metas[j].get("location", "N/A") if isinstance(i_metas[j], dict) else "N/A",
            "source": i_metas[j].get("source", "N/A") if isinstance(i_metas[j], dict) else "N/A",
            "text_score": 0.0,
            "img_score": float(i_norm[j]),
            "fused": float(w_img * i_norm[j]),
            "snippet": (i_docs[j] or "").replace("\n", " ").strip(),
        })

    # Sort by fused score (desc rerank)
    rows.sort(key=lambda r: r["fused"], reverse=True)
    
    # if top_n not specified, return full pool (k_text + k_img)
    if top_n is None:
        return rows

    top_n = max(0, min(int(top_n), len(rows)))
    return rows[:top_n]

def print_fused(rows, title: str, max_chars: int = 90):
    print(f"\n=== {title} ===")
    for idx, r in enumerate(rows, start=1):
        snippet = r["snippet"]
        if len(snippet) > max_chars:
            snippet = snippet[:max_chars].rstrip() + "..."
        print(
            f"[{idx}] {r['modality']} | id={r['id']} | cuisine={r['cuisine']} | "
            f"location={r['location']} | fused={r['fused']:.4f} "
            f"(text={r['text_score']:.4f}, img={r['img_score']:.4f})"
        )
        print(snippet)


# ============================================================================
# Demo 1 — Multimodal fusion (no filters)
# ============================================================================

# ================================
# Demo 1 — Multimodal fusion (no filters)
# ================================

q = "cozy noodles with warm atmosphere"

rows = fuse_rank(
    q,
    k_text=5,
    k_img=5,
    w_text=0.6,
    w_img=0.4,
    where_text=None,
    where_img=None,
    top_n=5
)

print_fused(rows, title="Demo 1 — Multimodal fusion (no filters)")
print("✅ Demo 1 complete")


# ============================================================================
# Demo 2 — Multimodal fusion (metadata filters)
# ============================================================================

# ================================
# Demo 2 — Multimodal fusion (metadata filters)
# ================================

q = "handmade pasta and romantic dinner"

where_articles = {"location": "Pasadena"}   # change to any location present in your dataset
where_images   = {"source": "recipe_image"} # optional

rows = fuse_rank(
    q,
    k_text=5,
    k_img=5,
    w_text=0.6,
    w_img=0.4,
    where_text=where_articles,
    where_img=where_images,
    top_n=5
)

if len(rows) == 0:
    print("⚠️ No results found. Try relaxing filters (location/cuisine/source).")
else:
    print_fused(rows, title="Demo 2 — Multimodal fusion (metadata filters)")

print("✅ Demo 2 complete")


# ============================================================================
# Demo 3 — Weight tuning (reranking behavior)
# ============================================================================

# ================================
# Demo 3 — Weight tuning
# ================================

q = "fresh sushi and minimalist presentation"

# TODO:
# 1. Run fusion ranking with a text-heavy setting and print results (use title="Demo 3A — Text-heavy fusion (w_text=0.8, w_img=0.2)")
# 2. Run fusion ranking with an image-heavy setting and print results (use title="Demo 3B — Image-heavy fusion (w_text=0.3, w_img=0.7)")
# 3. Select 5 results from each modality, but only show the top 5 fused results

# your code here

# 1. Text-heavy fusion: article evidence dominates the ranking
rows_text_heavy = fuse_rank(
    q,
    k_text=5,          # 5 candidates from the article index
    k_img=5,           # 5 candidates from the image index
    w_text=0.8,
    w_img=0.2,
    where_text=None,
    where_img=None,
    top_n=5            # but only show the top 5 fused results
)

print_fused(
    rows_text_heavy,
    title="Demo 3A — Text-heavy fusion (w_text=0.8, w_img=0.2)"
)

# 2. Image-heavy fusion: visual evidence dominates the ranking
rows_image_heavy = fuse_rank(
    q,
    k_text=5,
    k_img=5,
    w_text=0.3,
    w_img=0.7,
    where_text=None,
    where_img=None,
    top_n=5
)

print_fused(
    rows_image_heavy,
    title="Demo 3B — Image-heavy fusion (w_text=0.3, w_img=0.7)"
)

print("✅ Demo 3 complete")
print("🎉 Multimodal Similarity Fusion and Retrieval Ranking COMPLETE")

