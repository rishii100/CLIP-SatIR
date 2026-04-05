"""
🛰️ CLIP-SatIR — CLIP-powered satellite image retrieval
"""

import streamlit as st
from PIL import Image

from config import (
    APP_TITLE,
    APP_SUBTITLE,
    APP_DESCRIPTION,
    EXAMPLE_QUERIES,
    DEFAULT_TOP_K,
    MAX_TOP_K,
    NUM_DEMO_IMAGES,
)
from utils import (
    load_model,
    load_demo_images,
    compute_image_embeddings,
    search_by_text,
    search_by_image,
)

# ─── Page Config ─────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="CLIP-SatIR — CLIP Image Retrieval",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ──────────────────────────────────────────────────────────────

st.markdown(
    """
<style>
/* ── Import Google Font ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

/* ── Global ── */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* ── Hero Section ── */
.hero-container {
    text-align: center;
    padding: 2rem 1rem 1.5rem 1rem;
    margin-bottom: 1rem;
}
.hero-title {
    font-size: 3rem;
    font-weight: 800;
    background: linear-gradient(135deg, #00b4d8, #90e0ef, #caf0f8);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 0.3rem;
    letter-spacing: -1px;
}
.hero-subtitle {
    font-size: 1.15rem;
    color: #94a3b8;
    font-weight: 400;
    margin-bottom: 0.5rem;
}

/* ── Result Card ── */
.result-card {
    background: rgba(255, 255, 255, 0.04);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 16px;
    padding: 0.75rem;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    position: relative;
    overflow: hidden;
}
.result-card:hover {
    border-color: rgba(0, 180, 216, 0.4);
    box-shadow: 0 8px 32px rgba(0, 180, 216, 0.15);
    transform: translateY(-2px);
}
.result-card img {
    border-radius: 10px;
    width: 100%;
}

/* ── Score Badge ── */
.score-badge {
    display: inline-block;
    background: linear-gradient(135deg, #0077b6, #00b4d8);
    color: white;
    font-weight: 700;
    font-size: 0.85rem;
    padding: 0.25rem 0.75rem;
    border-radius: 999px;
    margin-top: 0.5rem;
    letter-spacing: 0.5px;
}

/* ── Filename Label ── */
.filename-label {
    color: #64748b;
    font-size: 0.78rem;
    margin-top: 0.35rem;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}

/* ── Example Query Chips ── */
.chip-container {
    display: flex;
    flex-wrap: wrap;
    gap: 0.5rem;
    justify-content: center;
    margin: 0.75rem 0 1.5rem 0;
}

/* ── Sidebar Styling ── */
[data-testid="stSidebar"] {
    background: rgba(15, 23, 42, 0.95) !important;
    border-right: 1px solid rgba(255, 255, 255, 0.06);
}

/* ── Divider ── */
.subtle-divider {
    border: none;
    border-top: 1px solid rgba(255, 255, 255, 0.06);
    margin: 1.5rem 0;
}

/* ── Info Box ── */
.info-box {
    background: rgba(0, 180, 216, 0.08);
    border: 1px solid rgba(0, 180, 216, 0.2);
    border-radius: 12px;
    padding: 1rem;
    font-size: 0.9rem;
    color: #cbd5e1;
    line-height: 1.6;
}

/* ── Caption preview ── */
.caption-preview {
    color: #94a3b8;
    font-size: 0.8rem;
    font-style: italic;
    margin-top: 0.25rem;
    display: -webkit-box;
    -webkit-line-clamp: 2;
    -webkit-box-orient: vertical;
    overflow: hidden;
}

/* ── Hide default Streamlit branding for cleaner look ── */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style>
""",
    unsafe_allow_html=True,
)


# ─── Sidebar ─────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("### ⚙️ Settings")

    search_mode = st.radio(
        "Search mode",
        ["🔍 Text → Image", "📷 Image → Image"],
        index=0,
        help="Choose how to search: type a description or upload a satellite image.",
    )

    top_k = st.slider(
        "Number of results",
        min_value=1,
        max_value=MAX_TOP_K,
        value=DEFAULT_TOP_K,
        step=1,
    )

    st.markdown("<hr class='subtle-divider'>", unsafe_allow_html=True)

    st.markdown("### 🛰️ About")
    st.markdown(
        """
    <div class='info-box'>
        <strong>SatelliteSearch</strong> uses a CLIP model fine-tuned on
        the RSICD dataset to match text queries with satellite imagery.
        <br><br>
        <b>Model:</b> CLIP ViT-B/32<br>
        <b>Dataset:</b> RSICD (10K+ images)<br>
        <b>Embeddings:</b> 512-dimensional<br>
        <b>Search:</b> Cosine similarity
    </div>
    """,
        unsafe_allow_html=True,
    )

    st.markdown("<hr class='subtle-divider'>", unsafe_allow_html=True)
    st.markdown(
        "<p style='color:#475569; font-size:0.75rem; text-align:center;'>",
        unsafe_allow_html=True,
    )


# ─── Hero Section ────────────────────────────────────────────────────────────

st.markdown(
    f"""
<div class='hero-container'>
    <div class='hero-title'>{APP_TITLE}</div>
    <div class='hero-subtitle'>{APP_SUBTITLE}</div>
</div>
""",
    unsafe_allow_html=True,
)
st.markdown(
    f"<p style='text-align:center; color:#64748b; font-size:0.9rem; margin-top:-0.5rem;'>{APP_DESCRIPTION}</p>",
    unsafe_allow_html=True,
)


# ─── Load Model & Data ──────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def _cached_embeddings(_model, _processor, _images):
    return compute_image_embeddings(_model, _processor, _images)


def initialize():
    """Load model, images, and compute embeddings with progress indicators."""
    with st.spinner("🤖  Loading CLIP model from HuggingFace Hub..."):
        model, processor = load_model()

    with st.spinner(f"🖼️  Loading {NUM_DEMO_IMAGES} satellite images..."):
        images, captions, filenames = load_demo_images()

    with st.spinner("⚡  Computing image embeddings (this may take a minute on first load)..."):
        image_embeddings = _cached_embeddings(model, processor, images)

    return model, processor, images, captions, filenames, image_embeddings


model, processor, images, captions, filenames, image_embeddings = initialize()

st.markdown("<hr class='subtle-divider'>", unsafe_allow_html=True)


# ─── Search Interface ───────────────────────────────────────────────────────

if search_mode == "🔍 Text → Image":
    # Text search input
    query = st.text_input(
        "Describe what you're looking for:",
        placeholder="e.g. an airport with multiple airplanes and runways",
        label_visibility="collapsed",
    )

    # Example query chips
    st.markdown("<p style='text-align:center; color:#64748b; font-size:0.85rem; margin-top: -0.25rem;'>Try an example:</p>", unsafe_allow_html=True)

    chip_cols = st.columns(5)
    selected_example = None
    for i, ex in enumerate(EXAMPLE_QUERIES):
        col_idx = i % 5
        if chip_cols[col_idx].button(ex, key=f"chip_{i}", use_container_width=True):
            selected_example = ex

    # Use selected example if clicked
    if selected_example:
        query = selected_example

    # Perform search
    if query:
        st.markdown(
            f"<p style='color:#94a3b8; margin: 1rem 0 0.5rem 0;'>Results for: <strong style=\"color:#00b4d8;\">\"{query}\"</strong></p>",
            unsafe_allow_html=True,
        )

        with st.spinner("Searching..."):
            indices, scores = search_by_text(query, model, processor, image_embeddings, top_k)

        # Display results grid
        cols = st.columns(min(top_k, 5))
        for rank, (idx, score) in enumerate(zip(indices, scores)):
            with cols[rank % min(top_k, 5)]:
                st.markdown("<div class='result-card'>", unsafe_allow_html=True)
                st.image(images[idx], use_container_width=True)
                st.markdown(
                    f"<div class='score-badge'>🎯 {score:.3f}</div>"
                    f"<div class='filename-label'>📁 {filenames[idx]}</div>"
                    f"<div class='caption-preview'>💬 {captions[idx]}</div>",
                    unsafe_allow_html=True,
                )
                st.markdown("</div>", unsafe_allow_html=True)

elif search_mode == "📷 Image → Image":
    st.markdown(
        "<p style='text-align:center; color:#94a3b8; font-size:0.9rem;'>"
        "Upload a satellite image to find visually similar ones in the dataset.</p>",
        unsafe_allow_html=True,
    )

    uploaded_file = st.file_uploader(
        "Upload satellite image",
        type=["jpg", "jpeg", "png", "webp"],
        label_visibility="collapsed",
    )

    if uploaded_file is not None:
        uploaded_image = Image.open(uploaded_file).convert("RGB")

        col_upload, col_spacer = st.columns([1, 3])
        with col_upload:
            st.markdown("**Your image:**")
            st.image(uploaded_image, width=200)

        st.markdown("<hr class='subtle-divider'>", unsafe_allow_html=True)

        with st.spinner("Finding similar images..."):
            indices, scores = search_by_image(uploaded_image, model, processor, image_embeddings, top_k)

        st.markdown(
            "<p style='color:#94a3b8; margin-bottom:0.5rem;'>Most similar satellite images:</p>",
            unsafe_allow_html=True,
        )

        cols = st.columns(min(top_k, 5))
        for rank, (idx, score) in enumerate(zip(indices, scores)):
            with cols[rank % min(top_k, 5)]:
                st.markdown("<div class='result-card'>", unsafe_allow_html=True)
                st.image(images[idx], use_container_width=True)
                st.markdown(
                    f"<div class='score-badge'>🎯 {score:.3f}</div>"
                    f"<div class='filename-label'>📁 {filenames[idx]}</div>"
                    f"<div class='caption-preview'>💬 {captions[idx]}</div>",
                    unsafe_allow_html=True,
                )
                st.markdown("</div>", unsafe_allow_html=True)


# ─── Footer ──────────────────────────────────────────────────────────────────

st.markdown("<hr class='subtle-divider'>", unsafe_allow_html=True)
st.markdown(
    """
<div style='text-align:center; padding: 1rem 0;'>
    <p style='color:#475569; font-size:0.8rem;'>
        Built with
        <a href='https://huggingface.co/rishii100/clip-rsicd-finetuned' target='_blank' style='color:#00b4d8; text-decoration:none;'>CLIP (ViT-B/32)</a>
        fine-tuned on
        <a href='https://huggingface.co/datasets/arampacha/rsicd' target='_blank' style='color:#00b4d8; text-decoration:none;'>RSICD</a>
        · Deployed on
        <a href='https://streamlit.io/cloud' target='_blank' style='color:#00b4d8; text-decoration:none;'>Streamlit Cloud</a>
    </p>
</div>
""",
    unsafe_allow_html=True,
)
