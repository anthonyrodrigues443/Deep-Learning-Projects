# SuperCar Image Classification using CNN

> A deep learning project that classifies supercar images using Convolutional Neural Networks with Transfer Learning

**Author:** Anthony Rodrigues  
**Model Accuracy:** 90%  
**Framework:** TensorFlow/Keras  
**Architecture:** InceptionV3 (Transfer Learning)

## 🚗 Project Overview

This project implements an end-to-end image classification system that can identify different supercar models from images. The system uses deep learning techniques with transfer learning to achieve high accuracy in classifying various luxury car brands and models.

### Key Features
- **Multi-class Classification**: Identifies multiple supercar brands and models
- **Transfer Learning**: Leverages pre-trained InceptionV3 for better performance
- **Data Augmentation**: Implements comprehensive image augmentation techniques
- **Web Scraping**: Automated data collection from multiple automotive websites
- **Production Pipeline**: Complete ML pipeline with model serialization
- **Flask Web App**: User-friendly interface for image uploads and predictions

## 📊 Model Performance

| Metric | Value |
|--------|-------|
| **Training Accuracy** | 90% |
| **Validation Accuracy** | 90% |
| **Training Epochs** | 20 (15 + 5 with early stopping) |
| **Model Size** | InceptionV3 base + custom layers |
| **Input Shape** | 299x299x3 (RGB images) |

## 🏗️ Project Architecture

```
CNN-SuperCar-Classification/
├── data/
│   ├── train_images/          # Training dataset
│   ├── test_images/           # Testing dataset
│   └── filtered_cars_data.csv # Car metadata
├── notebooks/
│   ├── Image_Classifier_Webscraper.ipynb
│   ├── Image_Classifier_Preprocessor.ipynb
│   ├── Image_Classifier_Model.ipynb
│   └── Image_Classifier_Tester.ipynb
├── checkpoint/
│   ├── weightings.h5          # Trained model weights
│   └── label_to_index.txt     # Label mappings
├── pred_model.pkl             # Serialized prediction pipeline
└── flask_app/                 # Web application
```

## 🔧 Implementation Details

### 1. Data Collection & Web Scraping
- **Sources**: Multiple automotive websites with UAE-based supercar listings
- **Methodology**: Automated web scraping using Python libraries
- **Data Volume**: 20 images per car model initially
- **Storage**: Structured CSV format with car metadata

**Key Technologies:**
- `requests` for web scraping
- `pygoogle_image` for Google Images API
- `pandas` for data manipulation

### 2. Data Preprocessing & Augmentation
- **Augmentation Techniques**:
  - Horizontal/Vertical flipping
  - Random cropping and scaling
  - Linear contrast adjustment
  - Grayscaling variations
  - Gaussian blur and noise addition
  - Shearing and rotation transforms
  - Translation (horizontal/vertical movement)

- **Augmentation Factor**: 8x per original image
- **Final Dataset Size**: ~160 images per class (20 × 8)
- **Overfitting Prevention**: Controlled augmentation to avoid duplicate patterns

### 3. Model Architecture & Training

**Base Model**: InceptionV3 (Transfer Learning)
- **Reason for Choice**: Balanced performance vs. computational efficiency
- **Modifications**: Custom classification head for supercar classes
- **Input Processing**: 299×299×3 RGB images, normalized to [0,1]

**Training Configuration:**
- **Loss Function**: Categorical Crossentropy
- **Optimizer**: Adam with learning rate scheduling
- **Batch Size**: 32 (optimized for memory constraints)
- **Regularization**: Early stopping, dropout layers
- **Data Split**: 80% training, 20% validation

**Training Strategy:**
- **Phase 1**: 15 epochs with initial learning rate
- **Phase 2**: 5 additional epochs with reduced learning rate
- **Monitoring**: Validation accuracy and loss tracking
- **Early Stopping**: Prevented overfitting, improved generalization

### 4. Model Evaluation & Deployment

**Testing Pipeline:**
- Preprocessing pipeline consistency
- Model performance on unseen data
- Prediction confidence scoring
- Error analysis and edge case handling

**Deployment:**
- Model serialization using `joblib`
- Flask web application for user interaction
- Real-time image upload and prediction
- Confidence score display for predictions

## 🚀 Getting Started

### Prerequisites
```bash
Python 3.7+
TensorFlow 2.x
Flask
OpenCV
PIL
pandas
numpy
scikit-learn
```

### Installation
```bash
git clone https://github.com/yourusername/supercar-classification
cd supercar-classification
pip install -r requirements.txt
```

### Usage

#### Training the Model
```bash
# 1. Data Collection
jupyter notebook Image_Classifier_Webscraper.ipynb

# 2. Data Augmentation
jupyter notebook Image_Classifier_Preprocessor.ipynb

# 3. Model Training
jupyter notebook Image_Classifier_Model.ipynb

# 4. Model Testing
jupyter notebook Image_Classifier_Tester.ipynb
```

#### Running the Flask App
```bash
python app.py
```

#### Making Predictions
```python
import joblib
from PIL import Image
import numpy as np

# Load the pipeline
pipeline = joblib.load('pred_model.pkl')

# Predict on new image
image = Image.open('test_car.jpg')
prediction = pipeline.predict(image)
print(f"Predicted car: {prediction}")
```

## 📈 Technical Challenges & Solutions

### Challenge 1: Limited Training Data
- **Problem**: Only 20 images per class initially
- **Solution**: Comprehensive data augmentation (8x multiplication)
- **Result**: Sufficient data diversity for robust training

### Challenge 2: Overfitting
- **Problem**: Model memorizing training data (epoch 15)
- **Solution**: Early stopping + additional 5 epochs with reduced learning rate
- **Result**: Improved generalization (77% → 90% accuracy)

### Challenge 3: Model Size & Efficiency
- **Problem**: Computational constraints for real-time inference
- **Solution**: InceptionV3 transfer learning (moderate size, good accuracy)
- **Result**: Balanced performance vs. resource utilization

## 🎯 Future Enhancements

- [ ] **Expand Dataset**: Include more car brands and models
- [ ] **Real-time Detection**: Video stream processing capabilities
- [ ] **Mobile App**: Cross-platform mobile application
- [ ] **API Development**: RESTful API for third-party integration
- [ ] **Performance Optimization**: Model quantization and pruning
- [ ] **Advanced Augmentation**: GAN-based synthetic data generation

## 📝 Technical Stack

| Component | Technology |
|-----------|------------|
| **Deep Learning** | TensorFlow, Keras |
| **Transfer Learning** | InceptionV3 |
| **Data Processing** | OpenCV, PIL, NumPy |
| **Web Scraping** | Requests, BeautifulSoup |
| **Web Framework** | Flask |
| **Data Storage** | CSV, HDF5 |
| **Model Serialization** | Joblib, TensorFlow SavedModel |

## 🔍 Model Insights

### Performance Metrics
- **Training Accuracy**: Steady improvement to 90%
- **Validation Accuracy**: Consistent with training (no overfitting)
- **Loss Convergence**: Smooth convergence after learning rate adjustment
- **Inference Time**: ~0.5 seconds per image

### Key Learnings
1. **Transfer Learning Effectiveness**: Pre-trained models significantly reduce training time
2. **Data Augmentation Impact**: 8x augmentation provided optimal balance
3. **Learning Rate Scheduling**: Critical for fine-tuning performance
4. **Early Stopping**: Essential for preventing overfitting

## 📚 References & Resources

- [InceptionV3 Paper](https://arxiv.org/abs/1512.00567)
- [Transfer Learning Guide](https://www.tensorflow.org/tutorials/images/transfer_learning)
- [Data Augmentation Techniques](https://www.tensorflow.org/tutorials/images/data_augmentation)

---

**Note**: This project demonstrates practical implementation of computer vision techniques in automotive industry applications. The model can be extended for commercial use with additional data and fine-tuning.
