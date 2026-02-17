# 🚀 Deployment Guide - BSI Signature Detection

Panduan lengkap deployment Streamlit app ke berbagai platform.

## 📋 Table of Contents

1. [Persiapan Files](#persiapan-files)
2. [Deploy ke Streamlit Cloud](#deploy-ke-streamlit-cloud)
3. [Deploy ke Hugging Face Spaces](#deploy-ke-hugging-face-spaces)
4. [Deploy ke Heroku](#deploy-ke-heroku)
5. [Deploy dengan Docker](#deploy-dengan-docker)

---

## 📦 Persiapan Files

Sebelum deploy, pastikan Anda punya files berikut:

```
signature-detection/
├── app.py                          # Main app ✅
├── requirements.txt                # Dependencies ✅
├── signature_model_final.keras     # Model (~30-40 MB) ⚠️
├── label_map.json                  # Labels ✅
└── README.md                       # Documentation ✅
```

### ⚠️ MASALAH: Model File Terlalu Besar

File `signature_model_final.keras` (~30-40 MB) terlalu besar untuk:
- GitHub free tier (max 100 MB per repo)
- Streamlit Cloud free tier

**Solusi:** Ada 3 opsi (pilih salah satu)

---

## 🌟 Option 1: Deploy ke Streamlit Cloud (RECOMMENDED)

### Step 1: Upload Model ke Google Drive

1. Upload `signature_model_final.keras` ke Google Drive
2. Set file sharing: "Anyone with the link can view"
3. Get file ID dari link:
   ```
   https://drive.google.com/file/d/1ABC123xyz456/view?usp=sharing
                                  ↑
                            This is the FILE_ID
   ```

### Step 2: Modify app.py untuk Download Model

Add this code di awal `app.py`:

```python
import os
import gdown

# Download model from Google Drive if not exists
MODEL_PATH = "signature_model_final.keras"
MODEL_GDRIVE_ID = "1ABC123xyz456"  # ← GANTI dengan file ID Anda

if not os.path.exists(MODEL_PATH):
    st.info("📥 Downloading model from Google Drive...")
    gdown.download(
        f"https://drive.google.com/uc?id={MODEL_GDRIVE_ID}",
        MODEL_PATH,
        quiet=False
    )
    st.success("✅ Model downloaded!")
```

### Step 3: Update requirements.txt

Add `gdown`:

```txt
streamlit==1.32.0
tensorflow==2.16.1
gdown==5.1.0
...
```

### Step 4: Push to GitHub

```bash
# Initialize git (if not already)
git init

# Add files (EXCLUDE .keras file!)
echo "*.keras" >> .gitignore
git add .
git commit -m "Initial commit"

# Create repo on GitHub, then:
git remote add origin https://github.com/YOUR_USERNAME/signature-detection.git
git push -u origin main
```

### Step 5: Deploy ke Streamlit Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Sign in with GitHub
3. Click **"New app"**
4. Select your repository: `YOUR_USERNAME/signature-detection`
5. Set:
   - **Main file path**: `app.py`
   - **Python version**: 3.10
6. Click **"Deploy!"**

### Step 6: Wait & Test

- Initial deployment: 5-10 minutes
- App will download model on first run
- URL: `https://YOUR_USERNAME-signature-detection.streamlit.app`

---

## 🤗 Option 2: Deploy ke Hugging Face Spaces (EASIEST!)

**Advantages:**
- ✅ No file size limit
- ✅ Free GPU option
- ✅ Easy to manage
- ✅ Great for ML apps

### Step 1: Create Hugging Face Account

1. Go to [huggingface.co](https://huggingface.co)
2. Sign up (free)

### Step 2: Create New Space

1. Go to [huggingface.co/new-space](https://huggingface.co/new-space)
2. Fill form:
   - **Space name**: `signature-detection`
   - **License**: MIT
   - **Select SDK**: Streamlit
   - **Visibility**: Public
3. Click **"Create Space"**

### Step 3: Upload Files via Web Interface

1. Click **"Files"** tab
2. Click **"Add file"** → **"Upload files"**
3. Upload ALL files:
   - `app.py`
   - `requirements.txt`
   - `signature_model_final.keras` ← DAPAT UPLOAD LANGSUNG!
   - `label_map.json`
   - `README.md`
4. Click **"Commit changes to main"**

### Step 4: Wait for Auto-Deploy

- Space will automatically build and deploy
- Check logs in "Logs" tab
- Build time: ~5-10 minutes
- Your app will be live at: `https://huggingface.co/spaces/YOUR_USERNAME/signature-detection`

### Step 5 (Optional): Enable GPU

For faster predictions:

1. Go to Space settings
2. Select **"Space hardware"**
3. Choose **"CPU Upgrade"** or **"T4 small (GPU)"**
4. Click **"Save"**

**Note:** GPU costs ~$0.60/hour, but there's free tier!

---

## 🐳 Option 3: Deploy dengan Docker (Advanced)

### Step 1: Create Dockerfile

```dockerfile
FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy files
COPY requirements.txt .
COPY app.py .
COPY signature_model_final.keras .
COPY label_map.json .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose Streamlit port
EXPOSE 8501

# Health check
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

# Run app
ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

### Step 2: Build Image

```bash
docker build -t signature-detection .
```

### Step 3: Run Container

```bash
docker run -p 8501:8501 signature-detection
```

### Step 4: Deploy to Cloud

**Google Cloud Run:**
```bash
gcloud run deploy signature-detection \
  --source . \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated
```

**AWS ECS / Azure Container Instances:**
Similar process, consult platform docs.

---

## 🎯 Comparison: Which Platform to Choose?

| Platform | Difficulty | Cost | GPU | File Size Limit | Speed |
|----------|-----------|------|-----|-----------------|-------|
| **Streamlit Cloud** | ⭐ Easy | Free | ❌ No | 1GB total | Medium |
| **Hugging Face** | ⭐ Easy | Free* | ✅ Yes | Unlimited | Fast |
| **Heroku** | ⭐⭐ Medium | $7/mo | ❌ No | 500MB slug | Medium |
| **Docker + Cloud** | ⭐⭐⭐ Hard | Varies | ✅ Yes | Unlimited | Fast |

**Recommendation untuk BSI Demo:**

1. **Quick Demo (1-2 days)**: Hugging Face Spaces
   - Fastest setup
   - No model size issues
   - Professional URL

2. **Long-term (1+ week)**: Streamlit Cloud + Google Drive
   - Free forever
   - Good branding
   - Easy maintenance

---

## 🔐 Security Best Practices

### 1. Environment Variables

Don't hardcode sensitive data. Use Streamlit secrets:

**File: `.streamlit/secrets.toml`**
```toml
[gdrive]
model_file_id = "1ABC123xyz456"

[api]
secret_key = "your-secret-key"
```

**In code:**
```python
import streamlit as st

MODEL_ID = st.secrets["gdrive"]["model_file_id"]
```

### 2. .gitignore

```
# .gitignore
*.keras
*.h5
*.pkl
.streamlit/secrets.toml
__pycache__/
*.pyc
.env
```

### 3. Input Validation

```python
# Validate uploaded file
def validate_image(file):
    if file.size > 10 * 1024 * 1024:  # 10MB limit
        return False, "File too large"
    
    if file.type not in ['image/png', 'image/jpeg']:
        return False, "Invalid file type"
    
    return True, "OK"
```

---

## 🐛 Troubleshooting Common Issues

### Issue 1: Model Download Fails

**Error:**
```
gdown.exceptions.FileURLRetrievalError
```

**Solution:**
1. Make sure Google Drive file is public
2. Check file ID is correct
3. Try alternative download method:

```python
import urllib.request

url = f"https://drive.google.com/uc?export=download&id={FILE_ID}"
urllib.request.urlretrieve(url, "signature_model_final.keras")
```

### Issue 2: Out of Memory

**Error:**
```
ResourceExhaustedError: OOM when allocating tensor
```

**Solutions:**
1. Use smaller batch size
2. Enable GPU (Hugging Face Spaces)
3. Optimize model:

```python
# Enable memory growth
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
```

### Issue 3: Slow First Load

**Cause:** Model loading on first request

**Solution:** Keep app warm with cron job:

```bash
# Ping app every 5 minutes
*/5 * * * * curl https://your-app-url.streamlit.app
```

Or use Streamlit's `@st.cache_resource`:

```python
@st.cache_resource
def load_model():
    return keras.models.load_model("signature_model_final.keras")
```

---

## 📊 Monitoring & Analytics

### Add Google Analytics

```python
# In app.py
st.markdown("""
<script async src="https://www.googletagmanager.com/gtag/js?id=G-XXXXXXXXXX"></script>
<script>
  window.dataLayer = window.dataLayer || [];
  function gtag(){dataLayer.push(arguments);}
  gtag('js', new Date());
  gtag('config', 'G-XXXXXXXXXX');
</script>
""", unsafe_allow_html=True)
```

### Track Usage

```python
import streamlit as st
from datetime import datetime

# Log predictions
if 'usage_log' not in st.session_state:
    st.session_state.usage_log = []

st.session_state.usage_log.append({
    'timestamp': datetime.now(),
    'action': 'prediction',
    'result': predictions[0]['name']
})
```

---

## 🎓 Next Steps After Deployment

1. **Test thoroughly**
   - Upload berbagai jenis gambar
   - Test batch upload
   - Check mobile responsiveness

2. **Get feedback**
   - Share dengan team
   - Demo ke stakeholders
   - Collect user feedback

3. **Monitor performance**
   - Check app logs
   - Monitor response times
   - Track error rates

4. **Iterate**
   - Fix bugs
   - Add features
   - Improve UI/UX

---

## 📞 Support

Jika ada masalah deployment, check:

1. **Streamlit Docs**: [docs.streamlit.io](https://docs.streamlit.io)
2. **Hugging Face Docs**: [huggingface.co/docs](https://huggingface.co/docs)
3. **Community Forum**: [discuss.streamlit.io](https://discuss.streamlit.io)

---

**Good luck dengan deployment! 🚀**

*BSI Innovation Idea 2024*
