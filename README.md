# Pneumonia Detection from Chest X-Rays using Deep Learning

A comprehensive deep learning project comparing three different architectures (Custom CNN, ResNet50, and Vision Transformer) for automated pneumonia detection from chest X-ray images.

## 📋 Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Models](#models)
- [Results](#results)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Key Findings](#key-findings)
- [Recommendations](#recommendations)
- [Future Work](#future-work)
- [Contributing](#contributing)
- [License](#license)

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

## 🚀 Installation

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended)
- 16GB+ RAM

### Dependencies

```bash
pip install torch torchvision
pip install transformers
pip install scikit-learn
pip install matplotlib seaborn
pip install numpy pandas
pip install pillow opencv-python
pip install tqdm
pip install gdown PyDrive
```

### Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/pneumonia-detection.git
cd pneumonia-detection
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

3. Download the datasets from Kaggle:
```bash
kaggle datasets download -d alexandrelemercier/60k-augmented-chest-x-ray-pneumonia
kaggle datasets download -d paultimothymooney/chest-xray-pneumonia
```

## 💻 Usage

### Training Models

The notebook includes complete training pipelines for all three models:

```python
# Load and preprocess data
train_loader, test_loader = prepare_data()

# Train ResNet50
resnet_model = train_resnet50(train_loader, test_loader)

# Train Vision Transformer
vit_model = train_vit(train_loader, test_loader)

# Train Custom CNN
custom_model = train_custom_cnn(train_loader, test_loader)
```

### Evaluation

```python
# Evaluate models
evaluate_model(resnet_model, test_loader, "ResNet50")
evaluate_model(vit_model, test_loader, "ViT")
evaluate_model(custom_model, test_loader, "Custom CNN")

# Generate confusion matrices
plot_confusion_matrix(model, test_loader)

# Show misclassified examples
show_misclassified_examples(model, test_loader, model_name="ResNet50")
```

### Inference

```python
# Load trained model
model = load_model("resnet50_best.pth")

# Predict on new image
image = preprocess_image("path/to/xray.png")
prediction = model(image)
result = "PNEUMONIA" if prediction > 0.5 else "NORMAL"
```

## 📁 Project Structure

```
pneumonia-detection/
│
├── Pneumonia_Final_Project_3Models.ipynb  # Main notebook
├── README.md                               # This file
├── requirements.txt                        # Python dependencies
│
├── data/
│   ├── train/
│   │   ├── NORMAL/
│   │   └── PNEUMONIA/
│   └── test/
│       ├── NORMAL/
│       └── PNEUMONIA/
│
├── models/
│   ├── custom_cnn.py
│   ├── resnet50_transfer.py
│   └── vit_model.py
│
├── utils/
│   ├── data_preprocessing.py
│   ├── augmentation.py
│   └── evaluation.py
│
└── results/
    ├── confusion_matrices/
    ├── training_curves/
    └── model_checkpoints/
```

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

## 🔮 Future Work

- [ ] Implement ensemble methods combining all three models
- [ ] Add Grad-CAM visualizations for model interpretability
- [ ] Expand dataset with multi-class pneumonia types (bacterial vs viral)
- [ ] Deploy models as REST API for production use
- [ ] Optimize models for mobile deployment (TensorFlow Lite/ONNX)
- [ ] Conduct external validation on independent datasets
- [ ] Implement uncertainty quantification for predictions
- [ ] Add support for other chest X-ray abnormalities

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

1. Fork the project
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Kaggle for providing the chest X-ray datasets
- PyTorch and Hugging Face communities for excellent deep learning frameworks
- The medical imaging research community for advancing AI in healthcare

## 📧 Contact

For questions or collaboration opportunities, please open an issue in the GitHub repository.

---

**Note**: This project is for educational and research purposes. Always consult qualified medical professionals for actual medical diagnoses.
