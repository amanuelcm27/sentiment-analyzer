# Sentiment Analyzer

## 📌 Overview
This repository contains a **Sentiment Analysis** deep learning model trained to classify text as **positive** or **negative** sentiment.  
The model (`sentiment_modelv1.1.keras`) was trained on a labeled dataset and achieved **81.56% accuracy** on the test set.  

This model can be integrated into real-world systems where understanding customer sentiment is valuable, such as:
- **E-commerce Review Aggregation** – Analyze product reviews to highlight top-rated items and detect low-rated products.
- **Customer Support Feedback** – Automatically flag negative responses for quick attention.
- **Social Media Monitoring** – Track brand reputation by identifying positive vs. negative mentions.
- **Survey Analysis** – Summarize sentiment trends from large-scale customer feedback forms.

---

## 📂 Repository Contents
- **`sentiment_modelv1.1.keras`** – Pre-trained Keras model file.
- **`requirements.txt`** – Python dependencies.
- **README.md** – Project documentation.

---

## 🚀 Getting Started

### 1️⃣ Installation
Clone this repository and install the required dependencies:
```bash
git clone https://github.com/yourusername/sentiment-analysis-model.git
cd sentiment-analysis-model
pip install -r requirements.txt

## 2️⃣ Loading the Model
You can load the trained model in Python using:
```python
from tensorflow.keras.models import load_model

model = load_model("sentiment_modelv1.1.keras")


## 3️⃣ Preparing Text Input
You need to embed raw text using the same sentence transformer model used during training. Example:
```python
from sentence_transformers import SentenceTransformer

embedder = SentenceTransformer('all-mpnet-base-v2')

texts = [
    "I love this product! Highly recommended.",
    "This was a terrible experience."
]

embeddings = embedder.encode(texts, batch_size=32, show_progress_bar=True)

## 4️⃣ Predicting Sentiment
Once you have embeddings, feed them to the sentiment model for prediction:
```python
import numpy as np

predictions = model.predict(embeddings)

for i, score in enumerate(predictions):
    label = "Positive" if score >= 0.5 else "Negative"
    print(f"Text: {texts[i]}\nSentiment: {label} (Confidence: {score[0]:.2f})\n")

## Performance
Accuracy: 81.56% on test dataset.

Model size: ~200KB (Keras HDF5 file).

Embedding dimension: 384 (from all-mpnet-base-v2).

## Notes
The model currently supports binary classification (positive/negative).

Neutral or mixed sentiments may be classified as positive due to binary setup.

Text preprocessing (e.g., removing HTML tags) is recommended before embedding.


#Contact
For questions or collaboration, reach out to:

GitHub: github.com/amanuelcm27