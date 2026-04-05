# ─── Configuration ───────────────────────────────────────────────────────────

# HuggingFace Hub
MODEL_REPO = "rishii100/clip-rsicd-finetuned"
DATASET_NAME = "arampacha/rsicd"

# App settings
NUM_DEMO_IMAGES = 50
DEFAULT_TOP_K = 5
MAX_TOP_K = 20
BATCH_SIZE = 4

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
APP_TITLE = "🛰️ CLIP-SatIR"
APP_SUBTITLE = "Search satellite imagery using natural language RSICD dataset"
APP_DESCRIPTION = (
    "CLIP ViT-B/32 is fine-tuned on the "
)
