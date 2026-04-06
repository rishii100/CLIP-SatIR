# CLIP-SatIR (Satellite Image Retrieval)

Live Link: [https://clip-satir-efmoq52houyrimwepa9hvx.streamlit.app/](https://clip-satir-efmoq52houyrimwepa9hvx.streamlit.app/)

This project fine-tunes the **CLIP (Contrastive Language–Image Pretraining)** model on the **RSICD Remote Sensing Image Caption Dataset** to improve alignment between satellite images and natural language descriptions.

The trained model can perform **text-to-image retrieval**, enabling queries like *“an airport with multiple airplanes and runway”* to retrieve relevant satellite images.

Goal:

* Learn **better vision–language alignment for satellite imagery**
* Enable **semantic image retrieval using natural language**


# Dataset

Dataset Link: [https://www.kaggle.com/datasets/thedevastator/rsicd-image-caption-dataset](https://www.kaggle.com/datasets/thedevastator/rsicd-image-caption-dataset)

Dataset used: **RSICD – Remote Sensing Image Caption Dataset**

Each image contains **multiple captions describing the scene**.

Example caption:

```
"A large airport with multiple airplanes and long runways."
```

Dataset structure after preprocessing:

```
dataset/
 ├── train.csv
 ├── valid.csv
 ├── test.csv
 └── images/
      ├── train/
      ├── val/
      └── test/
```

Each training sample contains:

```
image_path | caption
```

Example:

```
images/train/airport_12.jpg | a large airport with airplanes parked near runway
```


# Pipeline

The project follows the following workflow.

<img width="1400" height="512" alt="image" src="https://github.com/user-attachments/assets/26d6e6fc-ba2d-40e3-9253-54901a7ff38c" />



# Data Preprocessing

### 1. Caption Parsing

RSICD captions are stored as lists inside CSV files.

Example:

```
["caption1",
 "caption2",
 "caption3"]
```

They are parsed and flattened into individual samples.


### 2. Dataset Flattening

Each caption becomes a separate training pair.

Example:

```
Image A → caption1
Image A → caption2
Image A → caption3
```

Final training dataset:

```
(image_path, caption)
```


# Model

Base model:

```
openai/clip-vit-base-patch32
```

Components:

* **Vision Encoder** → ViT (Vision Transformer)
* **Text Encoder** → Transformer language encoder
* **Projection Heads** → shared embedding space

Both image and text are projected into the **same embedding space**.

Similarity is computed using **cosine similarity**.


# Training Setup

Training uses contrastive learning.

### Loss Function

CLIP contrastive loss:

```
L = CrossEntropy(sim(image_i, text_j))
```

The model learns to:

* maximize similarity between correct image–caption pairs
* minimize similarity with incorrect pairs

---

### Hyperparameters

| Parameter     | Value         |
| ------------- | ------------- |
| Model         | CLIP ViT-B/32 |
| Batch size    | 32            |
| Optimizer     | AdamW         |
| Learning rate | 1e-5          |
| Weight decay  | 1e-4          |
| Epochs        | 5             |
| Hardware      | GPU (CUDA)    |

Multi-GPU training is supported via:

```
torch.nn.DataParallel
```


# Training Loop

For each batch:

1. Load images and captions
2. Encode using CLIP processor
3. Forward pass through CLIP
4. Compute contrastive loss
5. Backpropagate gradients
6. Update weights

Pseudo workflow:

```
for epoch:
    for batch:
        image_features = model.encode_image()
        text_features = model.encode_text()
        loss = contrastive_loss()
        loss.backward()
        optimizer.step()
```


# Model Saving

After training the model is saved locally.

```
clip-rsicd/
 ├── config.json
 ├── pytorch_model.bin
 └── tokenizer files
```

Compressed export:

```
clip-rsicd.zip
```

---

# Inference: Text → Image Retrieval

After training, the model can retrieve relevant satellite images from natural language queries.

Example query:

```
"an airport with multiple airplanes and multiple runway"
```

Steps:

1. Encode query text
2. Encode validation images
3. Compute cosine similarity
4. Retrieve top-K images

Similarity:

```
similarity = text_embedding ⋅ image_embedding
```

Top matches are returned.


# Example Retrieval

Query:

```
"an airport with multiple airplanes and runway"
```

Output:

```
Top 5 relevant satellite images
```

The retrieved images correspond to **airport scenes with runways and aircraft**.

---

# Installation

Install dependencies:

```bash
pip install torch torchvision transformers accelerate tqdm pandas pillow matplotlib
```


# Running the Project

### 1. Prepare dataset

Place RSICD dataset in:

```
/kaggle/input/rsicd-image-caption-dataset
```

---

### 2. Train model

Run the notebook:

```
clip-finetuning.ipynb
```

This will:

* preprocess captions
* build dataset
* fine-tune CLIP
* save trained model

---

### 3. Run retrieval

Example query:

```python
query = "an airport with multiple airplanes and runway"
```

The model returns the **most similar satellite images**.

# Technologies Used

* PyTorch
* HuggingFace Transformers
* CLIP
* Pandas
* PIL
* Matplotlib
