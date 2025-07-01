# Medical AI Assistant 🩺

## Description
A modern, retina-only Medical AI Assistant app for AI-powered retina (ophthalmology) image analysis. Features a user-friendly Streamlit interface, Grad-CAM visualizations, PDF export, and robust theme and feedback support. Designed for research and educational use.

## Features
- 👁️ **Retina Analysis Only**: Upload retina images for diabetic retinopathy detection (5-stage severity)
- 🔥 **Grad-CAM Visualization**: Visual explanations of AI decisions
- 📄 **PDF Report Generation**: Export all results to PDF (auto-clears after download)
- 🎨 **Multiple Themes**: Choose from 5 modern themes, with persistence and custom CSS
- 📱 **Mobile Responsive**: Optimized for all devices
- 🧑‍💻 **Modern UI/UX**: Sidebar navigation, About/Help info, and a stylish footer
- 🟢 **Scan Workflow**: Button disables during scan, clear feedback with emoji, and results in a dedicated tab
- 📝 **Session History**: All scans are saved in session and viewable in expandable history
- 🧾 **Automatic Model Download**: Model checkpoint is auto-downloaded if missing
- 🛡️ **Robust Error Handling**: Unicode/emoji fixes, user-friendly messages

## What's New (v2.0.0)
- Removed all X-ray code and references; retina-only
- Refactored theme support: 5 themes, persistence, and custom CSS
- Added About/Help info as a sidebar button
- Improved mobile responsiveness
- Added a footer with copyright and GitHub link
- Scan workflow: disables button while scanning, shows feedback (emoji, info, etc.), and stores results for "Final Results" tab
- Added multiple feedback styles (st.balloons, st.snow, st.success, st.info, emoji)
- Fixed UnicodeEncodeError in subheader (now uses 📋)
- PDF export now clears results after download

## Project Structure
```
app.py                          # Main Streamlit application
best_model_checkpoint.pth       # Pre-trained retina model (auto-downloaded)
requirements.txt                # Python dependencies
README.md                       # Project documentation
docs/                          # Documentation and progress reports
  └── Project_Progress_Summary.txt
output/                        # Generated results and visualizations
  ├── gradcam_result.png
  └── uploads/                 # Uploaded images
src/
  ophthalmology/               # Retina analysis module
    backend/
      config.py
      model_loader.py
      inference.py
      grad_cam.py
      summarizer.py
    train_model.py
    test_backend.py
pages/                         # (Unused, for future expansion)
```

## Installation

### Prerequisites
- Python 3.8 or higher
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
pip install streamlit fpdf2
```

## Usage

### Running the Application
```bash
streamlit run app.py
```
The app will open in your browser at `http://localhost:8501`

### Using the Interface
- **Home**: Overview and navigation
- **Scan & Diagnose**: Upload retina images, scan, and view feedback (with emoji)
  - Scan button disables during processing
  - Feedback shown with emoji (🟢, 🥼, 🔬, etc.)
  - Results saved in session and shown in expandable history
- **Final Results**: View all scan results and download a PDF report (results auto-clear after download)
- **About/Help**: Sidebar button shows info box

### Model Training (Optional)
To train the retina model from scratch:
```bash
python src/ophthalmology/train_model.py
```

## Technical Details
- **Model**: ResNet18 (5-class DR severity)
- **Frameworks**: PyTorch, Streamlit, OpenCV, PIL, Transformers
- **Explainability**: Grad-CAM overlays
- **PDF Export**: fpdf2
- **Performance**: Retina model accuracy ~95.2%
- **Model Download**: Auto-downloads on first run
- **Session State**: Results/history stored in session
- **UI/UX**: 5 themes, mobile responsive, emoji feedback, About/Help, footer

## Dependencies
See `requirements.txt` for full list. Key packages:
- torch, torchvision, transformers, streamlit, opencv-python, Pillow, numpy, matplotlib, scikit-learn, tqdm, fpdf2

## Troubleshooting
- If you see Unicode errors, ensure your terminal and browser support emoji.
- If model download fails, check your internet connection.
- For PDF export, ensure `fpdf2` is installed.

## License & Acknowledgments
- MIT License
- APTOS 2019 Blindness Detection Dataset
- PyTorch, Hugging Face, Streamlit

---
**Disclaimer:** This tool is for research/education only. Not for clinical use.
