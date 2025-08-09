

# 🎵 Music Genre Classification

## 📌 Objective

Classify audio tracks into musical genres using either:

* **Tabular features** extracted from audio (e.g., MFCCs)
* **Spectrogram-based** image classification models (CNNs / Transfer Learning)

---

## 📂 Dataset

* **Dataset**: [GTZAN Dataset](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification)
* Contains **10 genres**, each with **100 audio files** in `.wav` format
* Example genres: *blues, classical, country, disco, hiphop, jazz, metal, pop, reggae, rock*

---

## 🛠 Workflow Steps

1. **Load & Preprocess Audio Files**

   * Read `.wav` files
   * Normalize or trim audio length (e.g., 30 seconds)

2. **Feature Extraction**

   * **Tabular approach**: Extract **MFCCs** (Mel-frequency cepstral coefficients) using `librosa`
   * **Image approach**: Generate **Mel spectrograms** for CNN input

3. **Dataset Preparation**

   * Encode labels
   * Split into **train/test sets**

4. **Model Training**

   * **Tabular**: Random Forest, XGBoost
   * **Image-based**: CNN from scratch or Transfer Learning (VGG, ResNet)

5. **Evaluation**

   * Accuracy
   * Confusion Matrix
   * Classification Report

6. **Visualization**

   * Plot feature importance (for tabular models)
   * Display sample spectrograms

## 📦 Tools & Libraries

* **Python**
* **Librosa** – audio processing
* **NumPy, Pandas** – data handling
* **Matplotlib, Seaborn** – visualization
* **Scikit-learn** – tabular ML models
* **XGBoost** – gradient boosting
* **TensorFlow / Keras** – deep learning models

