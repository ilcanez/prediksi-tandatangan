# 🖊️ BSI Signature Detection System

AI-powered signature recognition system untuk BSI Innovation Idea 2026. Sistem ini menggunakan Deep Learning (MobileNetV2) untuk mengidentifikasi pemilik tanda tangan dari 34 nasabah BSI.

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.16-orange.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

## 📊 Performance Metrics

| Metric | Score |
|--------|-------|
| **Test Accuracy** | 76.47% |
| **Top-3 Accuracy** | 92.65% |
| **Validation Accuracy** | 97.06% |

## ✨ Features

- 🎯 **Top-3 Predictions** dengan confidence visualization
- 📤 **Single & Batch Upload** untuk prediksi massal
- 📊 **Statistics Dashboard** dengan interactive charts
- 📜 **Prediction History** tracking
- 💾 **Export Results** ke CSV
- 🔄 **Retrain Interface** untuk update model dengan data baru
- 🎨 **Professional UI** dengan BSI branding

## 🏗️ Architecture

```
Input Image (224x224x3)
    ↓
MobileNetV2 Base (ImageNet)
    ↓
Custom Classification Head
    ↓
Top-3 Predictions (34 classes)
```

## 📁 Project Structure

```
signature-detection/
├── app.py                          # Main Streamlit application
├── requirements.txt                # Python dependencies
├── signature_model_final.keras     # Trained model (76.47% accuracy)
├── label_map.json                  # Class labels mapping
├── README.md                       # This file
└── models/                         # Model files directory
    ├── signature_model_final.keras
    ├── label_map.json
    └── model_metadata.json
```

## 🚀 Quick Start

### Option 1: Local Development

1. **Clone repository**
```bash
git clone <repository-url>
cd signature-detection
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Prepare model files**

Download atau copy file-file berikut ke directory yang sama dengan `app.py`:
- `signature_model_final.keras` (dari Google Drive)
- `label_map.json` (dari Google Drive)

4. **Run Streamlit app**
```bash
streamlit run app.py
```

5. **Open browser**
```
http://localhost:8501
```

### Option 2: Deploy to Streamlit Cloud

1. **Push to GitHub**
```bash
git init
git add .
git commit -m "Initial commit"
git remote add origin <your-repo-url>
git push -u origin main
```

2. **Deploy to Streamlit Cloud**
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Sign in with GitHub
   - Click "New app"
   - Select your repository
   - Set main file path: `app.py`
   - Click "Deploy"

3. **Upload model files**
   
   ⚠️ **IMPORTANT**: Model files terlalu besar untuk GitHub (>100MB). Options:
   
   **Option A: Git LFS (Large File Storage)**
   ```bash
   git lfs install
   git lfs track "*.keras"
   git add .gitattributes
   git add signature_model_final.keras
   git commit -m "Add model with LFS"
   git push
   ```
   
   **Option B: Google Drive + Direct Download**
   
   Modify `app.py` to download from Google Drive:
   ```python
   import gdown
   
   MODEL_URL = "https://drive.google.com/uc?id=YOUR_FILE_ID"
   gdown.download(MODEL_URL, "signature_model_final.keras", quiet=False)
   ```
   
   Add to `requirements.txt`:
   ```
   gdown==5.1.0
   ```
   
   **Option C: Hugging Face Model Hub**
   
   Upload model to Hugging Face, then download:
   ```python
   from huggingface_hub import hf_hub_download
   
   model_path = hf_hub_download(
       repo_id="your-username/signature-detection",
       filename="signature_model_final.keras"
   )
   ```

### Option 3: Deploy to Hugging Face Spaces

1. **Create new Space**
   - Go to [huggingface.co/spaces](https://huggingface.co/spaces)
   - Click "Create new Space"
   - Select "Streamlit" SDK
   - Set visibility (Public/Private)

2. **Upload files**
   - Upload `app.py`
   - Upload `requirements.txt`
   - Upload model files (no size limit!)
   - Upload `label_map.json`

3. **Space will auto-deploy**
   - Your app will be live at: `https://huggingface.co/spaces/YOUR_USERNAME/signature-detection`

## 📦 Required Files

Make sure these files are in the same directory as `app.py`:

1. **signature_model_final.keras** (~30-40 MB)
   - Trained MobileNetV2 model
   - Location: `/content/drive/MyDrive/project/signature_classfication/models/`

2. **label_map.json** (~1 KB)
   - Mapping dari class index ke nama
   - Format: `{"0": "ainun putri", "1": "andi ahmad afiq", ...}`

## 🔧 Configuration

### Model Path Configuration

By default, app mencari model di current directory. Untuk custom path:

```python
# In app.py, line ~250
MODEL_PATH = "path/to/your/signature_model_final.keras"
LABEL_MAP_PATH = "path/to/your/label_map.json"
```

### UI Customization

Edit CSS di bagian `st.markdown()` untuk customize:
- Colors (ganti `#3B82F6` dengan brand color)
- Layout (width, padding, spacing)
- Typography (font, size, weight)

## 💡 Usage Guide

### 1. Single Prediction

1. Go to "🏠 Home & Predict" page
2. Click "Single Upload" tab
3. Upload signature image (PNG/JPG)
4. Click "🔍 Predict Signature"
5. View Top-3 predictions dengan confidence scores

### 2. Batch Prediction

1. Go to "📦 Batch Upload" tab
2. Upload multiple images
3. Click "🔍 Predict All"
4. View results table
5. Download CSV hasil prediksi

### 3. View Statistics

1. Go to "📊 Statistics & History"
2. View summary metrics
3. Explore interactive charts
4. Download prediction history

### 4. Retrain Model (Advanced)

1. Go to "🔄 Retrain Model"
2. Prepare ZIP file dengan struktur:
   ```
   signatures.zip
   ├── Person1/
   │   ├── sig1.png
   │   └── sig2.png
   └── Person2/
       └── sig1.png
   ```
3. Upload ZIP
4. Configure training parameters
5. Click "🚀 Start Retraining"

⚠️ **Note**: Retrain feature adalah placeholder. Untuk actual retraining, gunakan notebook Tahap 2 di Google Colab.

## 🐛 Troubleshooting

### Error: Model file not found

**Solution**: Make sure `signature_model_final.keras` ada di directory yang sama dengan `app.py`

```bash
ls -la
# Should show:
# app.py
# signature_model_final.keras
# label_map.json
# requirements.txt
```

### Error: Module not found

**Solution**: Install dependencies

```bash
pip install -r requirements.txt
```

### Error: Out of memory

**Solution**: Model terlalu besar untuk free tier. Options:
- Use Hugging Face Spaces (better hardware)
- Optimize model (quantization, pruning)
- Use smaller batch size dalam app

### Slow predictions

**Solution**: 
- Use GPU-enabled deployment (Hugging Face Spaces Pro)
- Cache model loading dengan `@st.cache_resource`
- Reduce image size before upload

## 📊 API Documentation

### Prediction Function

```python
def predict_signature_top3(model, image, idx_to_name):
    """
    Predict top-3 signatures with confidence scores
    
    Args:
        model: Loaded Keras model
        image: PIL Image object
        idx_to_name: Dict mapping index to name
    
    Returns:
        List of dicts: [
            {'rank': 1, 'name': 'John', 'confidence': 0.85},
            {'rank': 2, 'name': 'Jane', 'confidence': 0.10},
            {'rank': 3, 'name': 'Bob', 'confidence': 0.03}
        ]
    """
```

### Preprocessing Function

```python
def preprocess_signature(image, target_size=(224, 224)):
    """
    Preprocess signature image for prediction
    
    Steps:
    1. Convert to RGB
    2. Resize to 224x224
    3. Convert to [0, 255] range
    4. Apply MobileNetV2 preprocessing
    
    Args:
        image: PIL Image
        target_size: Tuple (height, width)
    
    Returns:
        Preprocessed numpy array ready for model
    """
```

## 🔒 Security Notes

- Model files should be secured
- Don't expose API keys in code
- Validate all user inputs
- Sanitize uploaded files
- Use HTTPS in production

## 📈 Performance Optimization

1. **Model Caching**
```python
@st.cache_resource
def load_model_and_mappings(model_path, label_map_path):
    # Model loaded only once
```

2. **Image Preprocessing**
- Resize before upload (client-side)
- Use efficient image formats (PNG > JPG)
- Batch processing untuk multiple images

3. **UI Responsiveness**
- Use `st.spinner()` for long operations
- Show progress bars
- Async operations where possible

## 🚀 Future Improvements

- [ ] Real-time webcam signature capture
- [ ] Signature forgery detection
- [ ] Multi-model ensemble
- [ ] REST API endpoint
- [ ] Mobile app (React Native)
- [ ] Docker containerization
- [ ] CI/CD pipeline
- [ ] A/B testing framework

## 📝 License

MIT License - see LICENSE file for details

## 🤝 Contributing

Contributions welcome! Please:
1. Fork repository
2. Create feature branch
3. Commit changes
4. Push to branch
5. Open Pull Request

## 📞 Contact

**Project Team**  
Bank Syariah Indonesia  
📧 Email: contact@bsi.id  
🌐 Website: www.bsi.co.id

---

**BSI Innovation Idea 2024** 🏆  
*Powered by TensorFlow & Streamlit*
