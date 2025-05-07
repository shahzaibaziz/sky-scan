# Sky-Scan: Roof Texture Classification

A comprehensive AI system for classifying roof textures from satellite imagery using deep learning.

## 🚀 Features

- **Fast Data Processing**: Optimized patch extraction with parallel processing and caching
- **Deep Learning Model**: EfficientNet-B0 based classifier for roof texture classification
- **Comprehensive Training**: Complete training pipeline with validation and testing
- **Model Evaluation**: Detailed metrics including confusion matrix and per-class performance
- **Inference Pipeline**: Ready-to-use inference script for new predictions
- **GPU Acceleration**: Full CUDA support for faster training and inference

## 📊 Dataset

The system processes roof texture patches with the following classes:
- **Smooth**: Uniform, flat roof surfaces
- **Average**: Moderately textured roof surfaces  
- **Rough**: Highly textured or irregular roof surfaces
- **No Contour**: Areas without clear roof structures

## 🏗️ Architecture

### Data Processing Pipeline
1. **Image Loading**: Loads original, binary, and color-coded texture images
2. **Patch Extraction**: Extracts roof patches using contour detection
3. **Label Generation**: Automatically generates labels from color-coded texture images
4. **Caching**: Implements intelligent caching for faster subsequent runs

### AI Model
- **Backbone**: EfficientNet-B0 (ImageNet pre-trained)
- **Classifier Head**: Custom fully connected layers with dropout
- **Transfer Learning**: Frozen early layers for better generalization
- **Data Augmentation**: Comprehensive augmentation for robust training

## 📁 Project Structure

```
sky-scan/
├── notebooks/
│   └── traning_model.ipynb          # Main training notebook
├── src/
│   ├── inference.py                 # Inference script
│   ├── model.py                     # Model architecture
│   └── preprocessing.py             # Data preprocessing utilities
├── scripts/
│   ├── activate.sh                  # Virtual environment activation
│   ├── clean.sh                     # Clean cache and temporary files
│   ├── create_venv.sh              # Create virtual environment
│   └── run_test.sh                 # Run tests
├── requirements.txt                 # Python dependencies
└── README.md                       # This file
```

## 🛠️ Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd sky-scan
   ```

2. **Create virtual environment**:
   ```bash
   chmod +x scripts/create_venv.sh
   ./scripts/create_venv.sh
   ```

3. **Activate virtual environment**:
   ```bash
   source scripts/activate.sh
   ```

4. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## 🎯 Usage

### 1. Training the Model

Open the Jupyter notebook and run the training pipeline:

```bash
jupyter notebook notebooks/traning_model.ipynb
```

The notebook contains:
- **Cell 1**: Data extraction and preprocessing
- **Cell 2**: AI model training and testing pipeline

Key training parameters:
- `num_epochs`: Number of training epochs (default: 15)
- `batch_size`: Batch size for training (default: 32)
- `learning_rate`: Learning rate for optimizer (default: 0.001)

### 2. Model Inference

Use the inference script to make predictions on new images:

#### Single Image Prediction
```bash
python src/inference.py --model_path best_roof_texture_model.pth --image_path path/to/image.jpg --show_images
```

#### Batch Prediction
```bash
python src/inference.py --model_path best_roof_texture_model.pth --batch_dir path/to/images/ --max_images 20
```

### 3. Model Outputs

The training pipeline generates:
- `best_roof_texture_model.pth`: Best performing model checkpoint
- `model_results.pkl`: Detailed training and testing results
- Training plots: Loss and accuracy curves
- Confusion matrix: Visual representation of model performance

## 📈 Model Performance

The model provides comprehensive evaluation metrics:
- **Overall Accuracy**: Percentage of correct predictions
- **Precision**: Accuracy of positive predictions
- **Recall**: Ability to find all positive instances
- **F1-Score**: Harmonic mean of precision and recall
- **Per-class Metrics**: Detailed performance for each roof texture class

## 🔧 Configuration

### Data Paths
Update the data path in the notebook:
```python
base_path = "/path/to/your/data"
```

### Model Parameters
Adjust training parameters in the notebook:
```python
trained_model, results = train_and_test_model(
    patches=patches,
    labels=labels, 
    tile_info=tile_info,
    num_epochs=15,      # Adjust based on convergence
    batch_size=32,      # Adjust based on GPU memory
    learning_rate=0.001 # Adjust based on training stability
)
```

### GPU Settings
The system automatically detects and uses available GPUs. For optimal performance:
- Ensure CUDA is properly installed
- Monitor GPU memory usage
- Adjust batch size if needed

## 🧪 Testing

Run the test suite:
```bash
chmod +x scripts/run_test.sh
./scripts/run_test.sh
```

## 🧹 Maintenance

Clean cache and temporary files:
```bash
chmod +x scripts/clean.sh
./scripts/clean.sh
```

## 📝 Contributing

Please read [CONTRIBUTING.md](CONTRIBUTING.md) for details on our code of conduct and the process for submitting pull requests.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Acknowledgments

- EfficientNet architecture by Google Research
- PyTorch and torchvision for deep learning framework
- OpenCV for computer vision operations
- Scikit-learn for machine learning utilities

## 📞 Support

For questions and support, please open an issue in the repository or contact the development team.
