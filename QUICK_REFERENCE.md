# 🎯 Quick Reference Guide - BSI Signature Detection

## 📁 File Structure

```
signature-detection/
├── 📄 app.py                       # Main Streamlit application (ADVANCED VERSION)
├── 📋 requirements.txt             # Python dependencies
├── 📖 README.md                    # Complete documentation
├── 🚀 DEPLOYMENT_GUIDE.md          # Deployment instructions
├── 🔧 run.sh                       # Quick start (Linux/Mac)
├── 🔧 run.bat                      # Quick start (Windows)
├── 📝 .gitignore                   # Git ignore rules
│
├── .streamlit/
│   └── config.toml                 # Streamlit configuration
│
└── [REQUIRED - Not included, download from Google Drive]
    ├── signature_model_final.keras # Trained model (~30-40 MB)
    └── label_map.json              # Class labels
```

## ✨ Features Summary

### Version 2 (Advanced) - What You Got

✅ **Single Upload**
- Upload satu gambar tanda tangan
- Real-time prediction
- Top-3 results dengan confidence visualization
- Interactive bar chart

✅ **Batch Upload**
- Upload multiple images sekaligus
- Progress tracking
- Detailed results table
- Export to CSV
- Expandable detailed view

✅ **Statistics Dashboard**
- Total predictions count
- Unique signatures detected
- Average confidence metrics
- High confidence predictions
- Top predicted signatures bar chart
- Confidence distribution histogram
- Prediction history table
- Export history to CSV

✅ **Retrain Interface**
- Upload ZIP file dengan new data
- Preview uploaded data structure
- Configure training parameters:
  - Epochs (10-100)
  - Batch size (8/16/32)
  - Learning rate (1e-5 to 1e-3)
  - Data augmentation toggle
- Simulated retrain process

✅ **About Page**
- System overview
- Features list
- Architecture details
- Performance metrics
- Technical stack
- How it works
- Future improvements
- Contact information

✅ **Professional UI**
- BSI-themed color scheme
- Responsive design
- Custom CSS styling
- Interactive charts (Plotly)
- Progress indicators
- Error handling
- Loading spinners

## 🚀 Quick Start

### Option 1: Linux/Mac

```bash
# Make executable
chmod +x run.sh

# Run
./run.sh
```

### Option 2: Windows

```cmd
# Double-click or run in cmd
run.bat
```

### Option 3: Manual

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt

# Run app
streamlit run app.py
```

## 📥 Before Running - IMPORTANT!

**Download these files from Google Drive:**

1. **signature_model_final.keras** (~30-40 MB)
   - Path in Drive: `/content/drive/MyDrive/project/signature_classfication/models/`
   - Place in: Same directory as `app.py`

2. **label_map.json** (~1 KB)
   - Path in Drive: Same as above
   - Place in: Same directory as `app.py`

**Your directory should look like:**
```
signature-detection/
├── app.py                          ✅
├── signature_model_final.keras     ✅ DOWNLOAD THIS!
├── label_map.json                  ✅ DOWNLOAD THIS!
├── requirements.txt                ✅
└── ...
```

## 🌐 Deployment Options

### 1. **Hugging Face Spaces** (EASIEST) ⭐

**Pros:**
- No file size limit
- Free GPU option
- One-click deploy
- Professional URL

**Steps:**
1. Create account at huggingface.co
2. Create new Space (Streamlit SDK)
3. Upload ALL files (including model!)
4. Auto-deploys in ~5 minutes

**Recommended for:** BSI demo presentation

---

### 2. **Streamlit Cloud** (FREE FOREVER)

**Pros:**
- Free forever
- Easy GitHub integration
- Good for long-term

**Cons:**
- Need to handle large model file

**Steps:**
1. Upload model to Google Drive
2. Modify app.py to download from Drive
3. Push to GitHub
4. Deploy from share.streamlit.io

**Recommended for:** Production use after demo

---

### 3. **Local/Internal Server**

**Best for:**
- Internal BSI demo
- No internet dependency
- Full control

**Requirements:**
- Server with Python 3.10+
- Port 8501 accessible
- Model files on server

## 🎨 Customization

### Change Colors (BSI Branding)

Edit in `app.py`:

```python
# Line ~30-50: Custom CSS
st.markdown("""
<style>
    .main-header {
        color: #1E3A8A;  /* ← Change this */
    }
    .stButton>button {
        background-color: #3B82F6;  /* ← And this */
    }
</style>
""", unsafe_allow_html=True)
```

### Change Logo

Replace placeholder image:

```python
# Line ~150: Sidebar
st.image("https://your-bsi-logo-url.com/logo.png", ...)
```

### Add BSI Contact Info

Edit in About page section (line ~600+)

## 📊 Usage Tips

### For Best Predictions:

1. **Image Quality**
   - Clear signature
   - Good lighting
   - Minimal background noise
   - PNG or JPG format

2. **Batch Upload**
   - Max 50 images recommended
   - Use consistent file naming
   - Monitor progress bar

3. **Statistics**
   - Clear history periodically
   - Export important results
   - Monitor confidence trends

## 🐛 Common Issues & Fixes

### Issue: "Model file not found"

**Fix:**
```bash
# Check current directory
ls -la
# Should show signature_model_final.keras

# If missing, download from Google Drive
# Then place in same directory as app.py
```

### Issue: "ModuleNotFoundError"

**Fix:**
```bash
# Reinstall dependencies
pip install -r requirements.txt
```

### Issue: Slow predictions

**Fix:**
- Deploy to Hugging Face with GPU
- Or reduce image size before upload
- Or use batch processing

## 📈 Performance Benchmarks

**Local (CPU):**
- Single prediction: ~1-2 seconds
- Batch (10 images): ~10-15 seconds

**Hugging Face (GPU):**
- Single prediction: ~0.3-0.5 seconds
- Batch (10 images): ~2-3 seconds

## 🎯 Demo Script for BSI Presentation

### 1. Introduction (1 min)
"Kami mengembangkan sistem AI untuk mengenali tanda tangan nasabah BSI..."

### 2. Live Demo (3-5 min)

**Single Upload:**
1. Upload sample signature
2. Show Top-3 predictions
3. Highlight confidence score (92.65% top-3!)

**Batch Upload:**
1. Upload 5-10 signatures
2. Show batch processing
3. Export results to CSV

**Statistics:**
1. Show prediction history
2. Display charts
3. Explain metrics

### 3. Technical Details (2 min)
- MobileNetV2 architecture
- 76.47% accuracy
- 34 signatures recognized
- Fast inference (~1 sec)

### 4. Future Plans (1 min)
- Real-time webcam capture
- Forgery detection
- Mobile app
- API integration

## 📞 Support

**Questions?**
- Check README.md for details
- See DEPLOYMENT_GUIDE.md for deployment
- Contact: [your-email]

---

**🏆 Good luck dengan BSI Innovation Idea 2024!**

*Powered by TensorFlow & Streamlit*
