# Core Machine Learning Models

This repository contains implementations and examples of four fundamental machine learning models, covering both supervised and unsupervised learning approaches.

## Table of Contents
- [Supervised Learning](#supervised-learning)
  - [Classification](#classification)
  - [Regression](#regression)
- [Unsupervised Learning](#unsupervised-learning)
  - [Clustering](#clustering)
  - [Dimensionality Reduction](#dimensionality-reduction)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)


## Supervised Learning

Supervised learning involves training models on labeled data to make predictions on new, unseen data.

### Classification
Email Spam Detection Implementation

**Description:**
- Classifies emails as spam or non-spam using text analysis
- Uses TF-IDF vectorization for text feature extraction
- Implements Naive Bayes classifier

**Key Features:**
- Text preprocessing and cleaning
- Feature extraction using TF-IDF
- Model validation with cross-validation
- Performance metrics (precision, recall, F1-score)

**Example Usage:**
```python
from spam_classifier import evaluate_model

# Train and evaluate the model
model = evaluate_model(X_train, X_test, y_train, y_test)

# Classify new email
result = classify_email("Sample email text")
print(f"Classification: {result['classification']}")
print(f"Confidence: {result['confidence']}")
```

### Regression
House Price Prediction Model

**Description:**
- Predicts house prices based on various features
- Implements multiple regression techniques (Linear, Ridge, Lasso)
- Includes feature scaling and validation

**Key Features:**
- Feature engineering and scaling
- Multiple regression models comparison
- Cross-validation
- Feature importance analysis

**Example Usage:**
```python
from house_price_predictor import evaluate_model

# Train and evaluate models
models = train_models(X_train, y_train)
predictions = predict_price(models, new_house_features)
```

## Unsupervised Learning

Unsupervised learning discovers patterns in unlabeled data.

### Clustering
Customer Segmentation Analysis

**Description:**
- Segments customers based on behavior and demographics
- Uses K-means clustering
- Automatically determines optimal number of clusters

**Key Features:**
- Automatic cluster number optimization
- Customer behavior analysis
- Segment interpretation
- Feature importance analysis

**Example Usage:**
```python
from customer_segmentation import perform_clustering

# Perform clustering
clusters = perform_clustering(customer_data)
segment = predict_segment(new_customer)
```

### Dimensionality Reduction
Image Compression using PCA

**Description:**
- Reduces image dimensionality while preserving key features
- Implements Principal Component Analysis (PCA)
- Includes compression ratio analysis

**Key Features:**
- Variable compression ratios
- Quality metrics (PSNR, MSE)
- Memory usage analysis
- Visual comparison tools

**Example Usage:**
```python
from image_compressor import compress_image

# Compress image
compressed = compress_image(image_data, ratio=0.5)
reconstructed = reconstruct_image(compressed)
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/ml-models.git
cd ml-models
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

## Usage

Each model is contained in its own directory with specific instructions:

```bash
ml-models/
├── classification/
├── regression/
├── clustering/
└── dimensionality_reduction/
```

Run the examples:
```bash
python classification/spam_detection.py
python regression/house_prices.py
python clustering/customer_segments.py
python dimensionality_reduction/image_compression.py
```

## Contributing

1. Fork the repository
2. Create your feature branch: `git checkout -b feature/new-feature`
3. Commit your changes: `git commit -am 'Add new feature'`
4. Push to the branch: `git push origin feature/new-feature`
5. Submit a pull request


## Requirements

```
scikit-learn>=0.24.0
pandas>=1.2.0
numpy>=1.19.0
matplotlib>=3.3.0
seaborn>=0.11.0
```

## Performance Metrics

### Classification Model
- Accuracy: ~95% on test set
- F1-Score: 0.94
- Cross-validation score: 0.93

### Regression Model
- R² Score: 0.85
- RMSE: $45,000
- Cross-validation score: 0.83

### Clustering Model
- Silhouette Score: 0.68
- Optimal clusters: 5

### Dimensionality Reduction
- Compression ratio: 5:1
- PSNR: 35dB
- Variance explained: 95%
