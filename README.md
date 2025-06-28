# Medical AI Assistant 🩺

## Description
A comprehensive Medical AI Assistant application that provides AI-powered diagnosis for medical images including retina scans for diabetic retinopathy detection and chest X-ray analysis for pneumonia detection. The application features a user-friendly Streamlit interface with Grad-CAM visualizations and clinical note summarization capabilities.

## Features
- 👁️ **Retina Analysis**: Diabetic retinopathy classification with 5-stage severity detection
- 🩻 **X-Ray Diagnosis**: Chest X-ray pneumonia detection
- 🔥 **Grad-CAM Visualization**: Visual explanations of AI decisions
- 📋 **Clinical Note Summarization**: AI-powered text summarization using T5 model
- 📊 **Interactive Dashboard**: Modern Streamlit web interface
- 📄 **PDF Report Generation**: Export results to PDF format
- 🎯 **Real-time Processing**: Fast inference with confidence scores

## Project Structure
```
Medical_AI_Assistant/
├── app.py                          # Main Streamlit application
├── best_model_checkpoint.pth       # Pre-trained model weights (excluded from git)
├── requirements.txt                # Python dependencies
├── README.md                       # Project documentation
├── docs/                          # Documentation and progress reports
│   └── Project_Progress_Summary.txt
├── output/                        # Generated results and visualizations
│   ├── gradcam_result.png
│   └── uploads/
├── src/                          # Source code modules
│   ├── ophthalmology/           # Retina analysis module
│   │   ├── backend/
│   │   │   ├── config.py       # Configuration settings
│   │   │   ├── model_loader.py # Model loading utilities
│   │   │   ├── inference.py    # Inference pipeline
│   │   │   ├── grad_cam.py     # Grad-CAM implementation
│   │   │   └── summarizer.py   # Clinical note summarization
│   │   ├── train_model.py      # Training pipeline
│   │   └── test_backend.py     # Backend testing
│   └── xray_classifier/         # X-ray analysis module
│       ├── backend/
│       │   ├── config.py       # X-ray specific configuration
│       │   ├── model.py        # Model architecture
│       │   ├── inference.py    # X-ray inference
│       │   ├── grad_cam.py     # X-ray Grad-CAM
│       │   └── train_model.py  # X-ray training pipeline
│       └── test_backend.py     # X-ray backend testing
└── pages/                      # Additional Streamlit pages (if any)
```

## Installation

### Prerequisites
- Python 3.8 or higher
- CUDA-compatible GPU (optional, for faster processing)
- 4GB+ RAM recommended

### Setup Instructions

1. **Clone the repository:**
```bash
git clone https://github.com/yourusername/Medical_AI_Assistant.git
cd Medical_AI_Assistant
```

2. **Create a virtual environment (recommended):**
```bash
python -m venv medical_ai_env
# On Windows:
medical_ai_env\Scripts\activate
# On Mac/Linux:
source medical_ai_env/bin/activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

4. **Download additional requirements for Streamlit:**
```bash
pip install streamlit fpdf2
```

## Usage

### Running the Application
```bash
streamlit run app.py
```

The application will open in your default web browser at `http://localhost:8501`

### Using the Interface
1. **Home Page**: Overview of features and navigation
2. **Scan & Diagnose**: Upload and analyze medical images
   - **Retina Tab**: Upload retina images for diabetic retinopathy analysis
   - **X-Ray Tab**: Upload chest X-rays for pneumonia detection
3. **Final Results**: View all scan results and download PDF reports

### Model Training (Optional)
To train models from scratch:

**For Retina Classification:**
```bash
python src/ophthalmology/train_model.py
```

**For X-Ray Classification:**
```bash
python src/xray_classifier/backend/train_model.py
```

## Technical Details

### Models Used
- **Retina Classifier**: ResNet18 with transfer learning (5 classes for DR severity)
- **X-Ray Classifier**: ResNet18/DenseNet121 with transfer learning (2 classes: Normal/Pneumonia)
- **Text Summarizer**: T5-base transformer model for clinical notes

### Key Technologies
- **PyTorch**: Deep learning framework
- **Streamlit**: Web application framework
- **OpenCV**: Image processing
- **Transformers**: NLP models
- **Grad-CAM**: Explainable AI visualizations
- **PIL/Pillow**: Image handling
- **NumPy/Matplotlib**: Data processing and visualization

### Performance Metrics
- Retina Model Accuracy: ~95.2%
- X-Ray Model Accuracy: ~93.1%
- Average Processing Time: 2-5 seconds per image

## Dependencies
See `requirements.txt` for complete list. Key dependencies include:
- `torch>=2.0.0`
- `torchvision>=0.15.0`  
- `transformers>=4.30.0`
- `streamlit`
- `opencv-python`
- `Pillow`
- `numpy`
- `matplotlib`
- `scikit-learn`
- `tqdm`

## Model Checkpoints
Pre-trained model weights are automatically downloaded when running the application for the first time. Models are stored as:
- `best_model_checkpoint.pth` (Retina DR classifier)
- Additional X-ray model weights (downloaded as needed)

## Contributing
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments
- APTOS 2019 Blindness Detection Dataset for retina images
- NIH Chest X-ray Dataset for pneumonia detection
- PyTorch and Hugging Face communities
- Streamlit development team

## Support
For issues, questions, or contributions, please:
1. Check existing GitHub issues
2. Create a new issue with detailed description
3. Contact the development team

## Changelog
- v1.0.0: Initial release with retina and X-ray analysis
- v1.1.0: Added Grad-CAM visualizations
- v1.2.0: Integrated clinical note summarization
- v1.3.0: Added PDF report generation

---
**Note**: This application is for educational and research purposes. Always consult with qualified medical professionals for actual medical diagnosis and treatment.
