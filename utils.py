"""
Utility functions for CLIP-based satellite image search.
Handles model loading, image embedding, and similarity search.
"""

import numpy as np
import torch
import streamlit as st
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


# ─── Dataset Loading ─────────────────────────────────────────────────────────


@st.cache_data(show_spinner=False)
def load_demo_images(num_images=NUM_DEMO_IMAGES):
    """Load a subset of RSICD images from HuggingFace Datasets."""
    from datasets import load_dataset

    try:
        ds = load_dataset(DATASET_NAME, split="valid")
    except Exception:
        try:
            ds = load_dataset(DATASET_NAME, split="validation")
        except Exception:
            ds = load_dataset(DATASET_NAME, split="test")

    # Limit to requested number
    if len(ds) > num_images:
        ds = ds.select(range(num_images))

    images = []
    captions = []
    filenames = []

    for i, item in enumerate(ds):
        # Handle image field
        img = item.get("image")
        if img is None:
            continue
        if not isinstance(img, Image.Image):
            img = Image.open(img).convert("RGB")
        else:
            img = img.convert("RGB")
        images.append(img)

        # Handle captions field (may be list or string)
        cap = item.get("captions", item.get("caption", item.get("text", "")))
        if isinstance(cap, list):
            cap = cap[0] if cap else ""
        captions.append(str(cap))

        # Handle filename
        fn = item.get("filename", f"image_{i:04d}.jpg")
        if "/" in str(fn):
            fn = str(fn).split("/")[-1]
        filenames.append(str(fn))

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
            emb = emb / emb.norm(dim=-1, keepdim=True)

        all_embeddings.append(emb.cpu().numpy())

    return np.vstack(all_embeddings)


# ─── Search Functions ────────────────────────────────────────────────────────


def search_by_text(query: str, model, processor, image_embeddings, top_k=5):
    """Text-to-image search: encode query, compute cosine similarity, return top-K."""
    inputs = processor(text=query, return_tensors="pt", padding=True, truncation=True)

    with torch.no_grad():
        text_emb = model.get_text_features(**inputs)
        text_emb = text_emb / text_emb.norm(dim=-1, keepdim=True)

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
        img_emb = img_emb / img_emb.norm(dim=-1, keepdim=True)

    img_emb_np = img_emb.cpu().numpy()
    similarities = (img_emb_np @ image_embeddings.T).squeeze(0)

    top_indices = similarities.argsort()[::-1][:top_k]
    top_scores = similarities[top_indices]

    return top_indices.tolist(), top_scores.tolist()
