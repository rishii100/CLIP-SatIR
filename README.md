# CLIP-SatIR (Satellite Image Retrieval)

Live Link: [https://clip-satir-efmoq52houyrimwepa9hvx.streamlit.app/](https://clip-satir-efmoq52houyrimwepa9hvx.streamlit.app/)

This project fine-tunes the **CLIP (Contrastive Language–Image Pretraining)** model on the **RSICD Remote Sensing Image Caption Dataset** to improve alignment between satellite images and natural language descriptions, and includes a **fully deployed interactive web application** for performing zero-shot semantic search.

The trained model can perform both **text-to-image retrieval** (e.g., retrieving images for *"an airport with multiple airplanes and runway"*) and **image-to-image similarity search**.

## Live Web Application

The retrieval model is deployed as a fully interactive dashboard on Streamlit Cloud!
**Web UI Stack:** Streamlit, HuggingFace Dataset Server API, PyTorch

Features:
* **Text-to-Image:** Describe a geography, object, or scene, and the app will instantly search the dataset for visually matching satellite pictures.
* **Image-to-Image:** Upload a satellite picture or select a demo image, and the app will find visually similar satellite scenes nearby.
* **Large Search Pool:** Configured to search through a curated set of 350 satellite images from the RSICD dataset.
* Built with custom CSS to provide a responsive, grid-based, space-themed user interface.

## Quick Links
* **Fine-Tuned Model Weights:** [rishii100/clip-rsicd-finetuned](https://huggingface.co/rishii100/clip-rsicd-finetuned)
* **Dataset Used:** [RSICD – Remote Sensing Image Caption Dataset](https://huggingface.co/datasets/arampacha/rsicd)

---

# Training Pipeline & Architecture

<img width="1536" height="1024" alt="image" src="https://github.com/user-attachments/assets/69123129-65e1-4ff3-a1ff-ef7fcf31d916" />


### Model Details
Base model: `openai/clip-vit-base-patch32`

Components:
* **Vision Encoder** → ViT (Vision Transformer)
* **Text Encoder** → Transformer language encoder
* **Projection Heads** → Shared 512-dimensional embedding space

### Training Setup & Optimization
Training uses contrastive learning with the CLIP loss function (`CrossEntropy(sim(image_i, text_j))`), explicitly learning to maximize similarity between correct image/caption pairs while minimizing incorrect ones.

| Parameter     | Value         |
| ------------- | ------------- |
| Model         | CLIP ViT-B/32 |
| Batch size    | 32            |
| Optimizer     | AdamW         |
| Learning rate | 1e-5          |
| Weight decay  | 1e-4          |
| Epochs        | 5             |
| Hardware      | GPU (CUDA)    |

*Multi-GPU training supported via `torch.nn.DataParallel`.*

---

# Data Preprocessing

RSICD captions are originally stored as lists corresponding to a single image. These are parsed and flattened into individual `(image_path, caption)` training pairs to maximize the dataset volume for contrastive framing.

Example:
```
dataset/images/train/airport_12.jpg | "a large airport with airplanes parked near runway"
```

---

# Running & Deploying the Project

### Running the Web App Locally
To run the Streamlit frontend UI locally:
```bash
pip install -r requirements.txt
streamlit run app.py
```

### Reproducing Training
To replicate the fine-tuning process, place the RSICD dataset in `/kaggle/input/rsicd-image-caption-dataset` and execute the provided Jupyter Notebook:
```
clip-finetuning.ipynb
```
This notebook will handle building the dataset, extracting features, fine-tuning the model loop, and exporting the `config.json` and PyTorch `.bin` weights.

---

# Production Considerations & Engineering

During deployment to Streamlit Cloud, several key engineering challenges for handling HuggingFace libraries were resolved:
1. **Dependency Constraints:** Bypassed heavy C++ builds (`pyarrow`, `datasets`) failing on newer Python environments by directly accessing the HuggingFace REST `datasets-server` API to dynamically load batches of images.
2. **Environment Versioning:** Pinned execution context to PyTorch-compatible Python 3.11 using `.python-version`.
3. **Robust Feature Extraction:** Overrode standard HuggingFace `get_image_features()` pipelines with custom matrix dimensionality checks (preventing catastrophic 4x512 to 768x512 projection shape mismatches) guaranteeing compatibility regardless of underlying `transformers` library updates.

---

# Applications
This system can be used for:
* Satellite image database search
* Geospatial intelligence & Defense surveillance
* Disaster monitoring
* Urban planning and Land use classification

# Future Improvements
* Train with state-of-the-art **CLIP ViT-L/14**
* Add **hard negative mining** to the contrastive loop
* Integrate **BLIP / Flamingo style caption models**

# Technologies Used
* PyTorch
* HuggingFace Transformers & Hub
* Streamlit
* Pandas
* Pillow (PIL)
* Matplotlib
