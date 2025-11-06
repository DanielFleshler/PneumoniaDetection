# Pneumonia Detection from Chest X-Rays using Deep Learning

A comprehensive deep learning project comparing three different architectures (Custom CNN, ResNet50, and Vision Transformer) for automated pneumonia detection from chest X-ray images.

## 🔍 Overview

This project implements and compares three state-of-the-art deep learning architectures for pneumonia detection from chest X-ray images:

1. **Custom CNN** - Lightweight architecture designed for efficient inference
2. **ResNet50** - Transfer learning approach with pre-trained weights
3. **Vision Transformer (ViT)** - Attention-based architecture for maximum performance

The project includes comprehensive data preprocessing, augmentation, model training, and evaluation with detailed performance metrics.

## 📊 Dataset

### Data Sources

Two Kaggle datasets were combined and preprocessed:

1. [60K Augmented Chest X-Ray Pneumonia](https://www.kaggle.com/datasets/alexandrelemercier/60k-augmented-chest-x-ray-pneumonia)
2. [Chest X-Ray Pneumonia](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)

### Data Preprocessing

Our thorough Exploratory Data Analysis revealed critical issues that were systematically addressed:

#### Issues Identified:
- **Improper Data Distribution**: Imbalanced train-test split
- **Severe Class Imbalance**: Disproportion between pneumonia and healthy lung images
- **Data Redundancy**: 450 duplicate images in different formats (jpeg/jpg)
- **File Format Inconsistency**: Mixed image formats
- **Dimensional Variance**: Inconsistent image dimensions

#### Solutions Implemented:

1. **Format Standardization**: Converted all images to PNG format for lossless compression
2. **Dimension Standardization**: Resized all images to 256×256 pixels
3. **Duplicate Removal**: Eliminated 450 duplicate images
4. **Class Balancing**: Applied advanced augmentation techniques to balance the dataset
   - Rotation (±15 degrees)
   - Horizontal flips
   - Brightness and contrast adjustments
   - Zoom variations (±5%)
   - Shearing transformations
5. **Strategic Partitioning**: 80% training, 20% testing with stratified sampling
6. **Data Normalization**: Normalized pixel values to [0,1] range

### Final Dataset Statistics

- **Training Set**: ~80% of combined dataset (balanced 1:1 ratio)
- **Test Set**: ~20% of combined dataset (balanced 1:1 ratio)
- **Image Size**: 256×256 pixels
- **Format**: PNG (lossless compression)
- **Classes**: NORMAL, PNEUMONIA

## 🧠 Models

### 1. Custom CNN Architecture

A lightweight convolutional neural network designed for efficient inference and deployment on resource-constrained devices.

**Architecture Highlights:**
- Multiple convolutional blocks with batch normalization
- MaxPooling for spatial dimension reduction
- Dropout layers for regularization
- Fully connected layers for classification

**Advantages:**
- Fast inference time
- Lower computational requirements
- Suitable for edge devices and real-time screening

### 2. ResNet50 (Transfer Learning)

Pre-trained ResNet50 model fine-tuned on chest X-ray images.

**Architecture Highlights:**
- 50-layer deep residual network
- Pre-trained on ImageNet
- Fine-tuned final layers for binary classification
- Skip connections to prevent vanishing gradients

**Advantages:**
- Excellent balance between accuracy and efficiency
- Robust generalization through transfer learning
- Well-established architecture with proven performance

### 3. Vision Transformer (ViT)

Transformer-based architecture applying self-attention mechanisms to image patches.

**Architecture Highlights:**
- Patch-based image processing
- Multi-head self-attention layers
- Position embeddings for spatial awareness
- Transformer encoder blocks

**Advantages:**
- State-of-the-art accuracy
- Attention mechanisms provide interpretability
- Captures global context effectively

## 📈 Results

### Performance Summary

| Model | Test Accuracy | Precision | Recall | F1-Score | Training Time |
|-------|--------------|-----------|--------|----------|---------------|
| **ResNet50** | **~96%** | High | High | Excellent | Moderate |
| **ViT** | **~97%** | High | High | Excellent | High |
| **Custom CNN** | **~92%** | Good | Good | Good | Low |

### Confusion Matrices

All models demonstrated strong performance with detailed confusion matrices showing:
- High true positive rates for pneumonia detection
- Low false negative rates (critical for medical applications)
- Balanced performance across both classes

### Key Metrics

- **Best Overall Model**: ResNet50 (balanced performance)
- **Highest Accuracy**: Vision Transformer
- **Most Efficient**: Custom CNN
- **Best F1-Score**: ResNet50 (most balanced)


## 🎯 Key Findings

### 1. Model Performance
- **ResNet50** achieved the best balance between accuracy, computational efficiency, and generalization
- **ViT** demonstrated the highest raw accuracy but at increased computational cost
- **Custom CNN** proved viable for resource-constrained environments while maintaining good performance

### 2. Data Quality Impact
Comprehensive preprocessing and augmentation significantly improved model performance:
- Removing duplicates prevented overfitting
- Balancing classes eliminated prediction bias
- Standardizing dimensions improved training stability

### 3. Transfer Learning Effectiveness
Pre-trained weights (ResNet50) provided significant advantages:
- Faster convergence
- Better generalization
- Reduced overfitting risk

## 💡 Recommendations

### For Production Systems
**Use ResNet50**
- Best balance of accuracy and efficiency
- Robust generalization through transfer learning
- Proven reliability in medical imaging

### For Edge Devices / Real-Time Screening
**Use Custom CNN**
- Smaller model size
- Faster inference time
- Suitable for resource-limited environments

### For Research / Maximum Performance
**Use Vision Transformer (ViT)**
- Highest possible accuracy
- Better interpretability through attention mechanisms
- Ideal when computational resources are abundant

**Note**: This project is for educational and research purposes. Always consult qualified medical professionals for actual medical diagnoses.
