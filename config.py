# ─── Configuration ───────────────────────────────────────────────────────────

# HuggingFace Hub
MODEL_REPO = "rishii100/clip-rsicd-finetuned"
DATASET_NAME = "arampacha/rsicd"

# App settings
NUM_DEMO_IMAGES = 200
DEFAULT_TOP_K = 5
MAX_TOP_K = 20
BATCH_SIZE = 16

# Example queries for quick-search chips
EXAMPLE_QUERIES = [
    "an airport with multiple airplanes",
    "a baseball field near residential area",
    "dense buildings in a city",
    "a bridge over a river",
    "green farmland with roads",
    "a parking lot full of cars",
    "a beach with blue water",
    "industrial buildings near roads",
    "a river flowing through the city",
    "a large stadium",
]

# Branding
APP_TITLE = "🛰️ SatelliteSearch"
APP_SUBTITLE = "Search satellite imagery using natural language"
APP_DESCRIPTION = (
    "Powered by **CLIP ViT-B/32** fine-tuned on the "
    "[RSICD dataset](https://huggingface.co/datasets/arampacha/rsicd) "
    "— a collection of 10,000+ remote sensing images with captions."
)
