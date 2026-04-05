"""
Utility functions for CLIP-based satellite image search.
Handles model loading, image embedding, and similarity search.
Uses HuggingFace Datasets Server API to load images (no pyarrow needed).
"""

import numpy as np
import requests
import torch
import streamlit as st
from io import BytesIO
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

from config import MODEL_REPO, DATASET_NAME, NUM_DEMO_IMAGES, BATCH_SIZE


# ─── Model Loading ───────────────────────────────────────────────────────────


@st.cache_resource(show_spinner=False)
def load_model():
    """Load the fine-tuned CLIP model and processor from HuggingFace Hub."""
    model = CLIPModel.from_pretrained(MODEL_REPO)
    processor = CLIPProcessor.from_pretrained(MODEL_REPO)
    model.eval()
    return model, processor


# ─── Dataset Loading (via HuggingFace Datasets Server API) ───────────────────


DATASETS_API = "https://datasets-server.huggingface.co"


@st.cache_data(show_spinner=False)
def load_demo_images(num_images=NUM_DEMO_IMAGES):
    """
    Load RSICD images using the HuggingFace Datasets Server API.
    This avoids requiring pyarrow / datasets library.
    """
    images = []
    captions = []
    filenames = []

    batch_size = 100  # API max per request
    offset = 0

    while len(images) < num_images:
        remaining = min(batch_size, num_images - len(images))
        url = (
            f"{DATASETS_API}/rows"
            f"?dataset={DATASET_NAME}&config=default&split=valid"
            f"&offset={offset}&length={remaining}"
        )

        try:
            resp = requests.get(url, timeout=60)
            if resp.status_code != 200:
                st.warning(f"Dataset API returned status {resp.status_code}. Using {len(images)} images.")
                break
            data = resp.json()
        except Exception as e:
            st.warning(f"Could not reach dataset API: {e}. Using {len(images)} images.")
            break

        rows = data.get("rows", [])
        if not rows:
            break

        for row in rows:
            if len(images) >= num_images:
                break

            item = row.get("row", {})

            # ── Download image from CDN URL ──
            img_info = item.get("image")
            if not img_info:
                continue

            img_url = img_info.get("src", "")
            if not img_url:
                continue

            try:
                img_resp = requests.get(img_url, timeout=15)
                img = Image.open(BytesIO(img_resp.content)).convert("RGB")
            except Exception:
                continue

            images.append(img)

            # ── Caption ──
            cap = item.get("captions", item.get("caption", item.get("text", "")))
            if isinstance(cap, list):
                cap = cap[0] if cap else ""
            captions.append(str(cap))

            # ── Filename ──
            fn = item.get("filename", f"image_{len(filenames):04d}.jpg")
            if "/" in str(fn):
                fn = str(fn).split("/")[-1]
            filenames.append(str(fn))

        offset += len(rows)

    if not images:
        st.error("❌ Could not load any images from the RSICD dataset. Please try again later.")
        st.stop()

    return images, captions, filenames


# ─── Embedding Computation ───────────────────────────────────────────────────


@st.cache_data(show_spinner=False)
def compute_image_embeddings(_model, _processor, _images):
    """Compute normalized embeddings for all images (batched)."""
    all_embeddings = []

    for i in range(0, len(_images), BATCH_SIZE):
        batch = _images[i : i + BATCH_SIZE]
        inputs = _processor(images=batch, return_tensors="pt", padding=True)

        with torch.no_grad():
            emb = _model.get_image_features(**inputs)
            
            # Handle anomalous HuggingFace returns
            if not isinstance(emb, torch.Tensor):
                if hasattr(emb, "pooler_output") and hasattr(_model, "visual_projection"):
                    emb = _model.visual_projection(emb.pooler_output)
                else:
                    emb = getattr(emb, "image_embeds", emb[0] if isinstance(emb, tuple) else emb)
            
            import torch.nn.functional as F
            emb = F.normalize(emb, p=2, dim=-1)

        all_embeddings.append(emb.cpu().numpy())

    return np.vstack(all_embeddings)


# ─── Search Functions ────────────────────────────────────────────────────────


def search_by_text(query: str, model, processor, image_embeddings, top_k=5):
    """Text-to-image search: encode query, compute cosine similarity, return top-K."""
    inputs = processor(text=query, return_tensors="pt", padding=True, truncation=True)

    with torch.no_grad():
        text_emb = model.get_text_features(**inputs)
        
        if not isinstance(text_emb, torch.Tensor):
            if hasattr(text_emb, "pooler_output") and hasattr(model, "text_projection"):
                text_emb = model.text_projection(text_emb.pooler_output)
            else:
                text_emb = getattr(text_emb, "text_embeds", text_emb[0] if isinstance(text_emb, tuple) else text_emb)
        
        import torch.nn.functional as F
        text_emb = F.normalize(text_emb, p=2, dim=-1)

    text_emb_np = text_emb.cpu().numpy()
    similarities = (text_emb_np @ image_embeddings.T).squeeze(0)

    top_indices = similarities.argsort()[::-1][:top_k]
    top_scores = similarities[top_indices]

    return top_indices.tolist(), top_scores.tolist()


def search_by_image(uploaded_image: Image.Image, model, processor, image_embeddings, top_k=5):
    """Image-to-image search: encode uploaded image, find similar ones."""
    uploaded_image = uploaded_image.convert("RGB")
    inputs = processor(images=uploaded_image, return_tensors="pt")

    with torch.no_grad():
        img_emb = model.get_image_features(**inputs)
        
        if not isinstance(img_emb, torch.Tensor):
            if hasattr(img_emb, "pooler_output") and hasattr(model, "visual_projection"):
                img_emb = model.visual_projection(img_emb.pooler_output)
            else:
                img_emb = getattr(img_emb, "image_embeds", img_emb[0] if isinstance(img_emb, tuple) else img_emb)
        
        import torch.nn.functional as F
        img_emb = F.normalize(img_emb, p=2, dim=-1)

    img_emb_np = img_emb.cpu().numpy()
    similarities = (img_emb_np @ image_embeddings.T).squeeze(0)

    top_indices = similarities.argsort()[::-1][:top_k]
    top_scores = similarities[top_indices]

    return top_indices.tolist(), top_scores.tolist()
