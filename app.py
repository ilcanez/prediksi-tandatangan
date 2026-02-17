"""
🖊️ Signature Detection System - BSI Innovation Idea
Advanced Streamlit App dengan Multiple Features

Features:
- Single & Batch Upload
- Top-3 Predictions dengan Confidence Visualization
- Upload New Data & Retrain Model
- Prediction History & Statistics
- Export Results to CSV
- Professional UI
"""

import streamlit as st
import numpy as np
from PIL import Image
import json
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import io
import base64
from pathlib import Path
import zipfile
import os

# ✅ FIX: Compatible imports untuk TensorFlow 2.x & Keras 3
try:
    import tensorflow as tf
    import keras
    from keras import layers, models
except ImportError:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, models

# Page configuration
st.set_page_config(
    page_title="BSI Signature Detection",
    page_icon="🖊️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for BSI branding
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1E3A8A;
        text-align: center;
        padding: 1rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #4B5563;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #F3F4F6;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #3B82F6;
    }
    .prediction-card {
        background-color: #FFFFFF;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 1rem;
    }
    .confidence-high {
        color: #10B981;
        font-weight: bold;
    }
    .confidence-medium {
        color: #F59E0B;
        font-weight: bold;
    }
    .confidence-low {
        color: #EF4444;
        font-weight: bold;
    }
    .stButton>button {
        width: 100%;
        background-color: #3B82F6;
        color: white;
        border-radius: 8px;
        padding: 0.75rem;
        font-weight: bold;
        border: none;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #2563EB;
        box-shadow: 0 4px 12px rgba(59, 130, 246, 0.4);
    }
</style>
""", unsafe_allow_html=True)

# ==================== HELPER FUNCTIONS ====================

@st.cache_resource
def load_model_and_mappings(model_path, label_map_path):
    """Load trained model and label mappings"""
    try:
        model = tf.keras.models.load_model(model_path)
        with open(label_map_path, 'r') as f:
            label_map = json.load(f)
            # Convert string keys to int
            label_map = {int(k): v for k, v in label_map.items()}
        
        # Create idx_to_name mapping
        idx_to_name = {idx: name for idx, name in label_map.items()}
        
        return model, idx_to_name
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None

def preprocess_signature(image, target_size=(224, 224)):
    """
    Preprocess signature image untuk prediction
    
    Steps:
    1. Convert to RGB if needed
    2. Resize to target size
    3. Convert to numpy array
    4. Ensure [0, 255] range
    5. Apply MobileNetV2 preprocessing
    """
    # Convert to RGB
    if image.mode == 'RGBA':
        background = Image.new('RGB', image.size, (255, 255, 255))
        background.paste(image, mask=image.split()[3])
        image = background
    elif image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Resize
    image = image.resize(target_size, Image.Resampling.LANCZOS)
    
    # Convert to numpy array
    img_array = np.array(image).astype(np.float32)
    
    # Ensure [0, 255] range
    if img_array.max() <= 1.0:
        img_array = img_array * 255.0
    
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    
    # Apply MobileNetV2 preprocessing (scales to [-1, 1])
    img_preprocessed = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
    
    return img_preprocessed

def predict_signature_top3(model, image, idx_to_name):
    """Predict top-3 signatures with confidence scores"""
    # Preprocess
    img_preprocessed = preprocess_signature(image)
    
    # Predict
    predictions = model.predict(img_preprocessed, verbose=0)[0]
    
    # Get top-3 indices
    top3_idx = np.argsort(predictions)[-3:][::-1]
    
    # Create results
    results = []
    for rank, idx in enumerate(top3_idx, 1):
        name = idx_to_name[idx]
        confidence = float(predictions[idx])
        results.append({
            'rank': rank,
            'name': name,
            'confidence': confidence
        })
    
    return results

def create_confidence_bar_chart(predictions):
    """Create horizontal bar chart for top-3 predictions"""
    names = [p['name'] for p in predictions]
    confidences = [p['confidence'] * 100 for p in predictions]
    ranks = [f"#{p['rank']}" for p in predictions]
    
    # Color mapping
    colors = ['#10B981', '#3B82F6', '#9CA3AF']
    
    fig = go.Figure(data=[
        go.Bar(
            y=ranks,
            x=confidences,
            orientation='h',
            text=[f"{c:.2f}%" for c in confidences],
            textposition='outside',
            marker=dict(color=colors),
            hovertemplate='<b>%{y}</b>: %{customdata}<br>Confidence: %{x:.2f}%<extra></extra>',
            customdata=names
        )
    ])
    
    fig.update_layout(
        title="Top-3 Predictions",
        xaxis_title="Confidence (%)",
        yaxis_title="Rank",
        height=300,
        showlegend=False,
        xaxis=dict(range=[0, 100]),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
    )
    
    return fig

def save_prediction_history(predictions, image_name):
    """Save prediction to session state history"""
    if 'prediction_history' not in st.session_state:
        st.session_state.prediction_history = []
    
    history_entry = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'image_name': image_name,
        'predictions': predictions
    }
    
    st.session_state.prediction_history.append(history_entry)

def get_confidence_color(confidence):
    """Get color class based on confidence level"""
    if confidence >= 0.7:
        return "confidence-high"
    elif confidence >= 0.4:
        return "confidence-medium"
    else:
        return "confidence-low"

def export_predictions_to_csv(history):
    """Export prediction history to CSV"""
    data = []
    for entry in history:
        for pred in entry['predictions']:
            data.append({
                'Timestamp': entry['timestamp'],
                'Image': entry['image_name'],
                'Rank': pred['rank'],
                'Name': pred['name'],
                'Confidence': f"{pred['confidence']*100:.2f}%"
            })
    
    df = pd.DataFrame(data)
    return df

# ==================== MAIN APP ====================

def main():
    # Header
    st.markdown('<h1 class="main-header">🖊️ BSI Signature Detection System</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">AI-Powered Signature Recognition dengan MobileNetV2</p>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.image("picture/logo.png", use_container_width=True)

        st.markdown("---")
        
        # Navigation
        page = st.radio(
            "Navigation",
            ["🏠 Home & Predict", "📊 Statistics & History", "🔄 Retrain Model", "ℹ️ About"],
            label_visibility="collapsed"
        )
        
        st.markdown("---")
        
        # Model info
        st.markdown("### 📋 Model Info")
        st.info("""
        **Architecture:** MobileNetV2  
        **Classes:** 34 signatures  
        **Accuracy:** 76.47%  
        **Top-3 Acc:** 92.65%
        """)
        
        st.markdown("---")
        st.markdown("**🏆 BSI Innovation Idea 2024**")
        st.markdown("*Powered by TensorFlow & Streamlit*")
    
    # ==================== PAGE: HOME & PREDICT ====================
    if page == "🏠 Home & Predict":
        st.markdown("## 📤 Upload Signature untuk Prediksi")
        
        # Tab selection
        tab1, tab2 = st.tabs(["📄 Single Upload", "📦 Batch Upload"])
        
        # ===== SINGLE UPLOAD =====
        with tab1:
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.markdown("### Upload Image")
                uploaded_file = st.file_uploader(
                    "Choose a signature image",
                    type=['png', 'jpg', 'jpeg'],
                    help="Upload PNG, JPG, or JPEG format"
                )
                
                if uploaded_file is not None:
                    # Display uploaded image
                    image = Image.open(uploaded_file)
                    st.image(image, caption="Uploaded Signature", use_container_width=True)

                    if st.button("🔄 Reset", key="reset_single"):
                         if "current_predictions" in st.session_state:
                             del st.session_state["current_predictions"]
                         st.rerun()

                    
                    # Predict button
                    if st.button("🔍 Predict Signature", key="predict_single"):
                        with st.spinner("Analyzing signature..."):
                            # Load model
                            MODEL_PATH = "signature_model_final.keras"
                            LABEL_MAP_PATH = "label_map.json"
                            
                            # Check if files exist
                            if not os.path.exists(MODEL_PATH):
                                st.error(f"❌ Model file not found: {MODEL_PATH}")
                                st.info("Please upload the model file to the same directory as this app.")
                            elif not os.path.exists(LABEL_MAP_PATH):
                                st.error(f"❌ Label map file not found: {LABEL_MAP_PATH}")
                            else:
                                model, idx_to_name = load_model_and_mappings(MODEL_PATH, LABEL_MAP_PATH)
                                
                                if model is not None:
                                    # Predict
                                    predictions = predict_signature_top3(model, image, idx_to_name)
                                    
                                    # Save to history
                                    save_prediction_history(predictions, uploaded_file.name)
                                    
                                    # Store in session state for display
                                    st.session_state.current_predictions = predictions
            
            with col2:
                st.markdown("### 🎯 Prediction Results")
                
                # Display predictions if available
                if 'current_predictions' in st.session_state:
                    predictions = st.session_state.current_predictions
                    
                    # Top prediction highlight
                    top_pred = predictions[0]
                    st.success(f"**Most Likely:** {top_pred['name']}")
                    st.markdown(f"**Confidence:** {top_pred['confidence']*100:.2f}%")
                    
                    st.markdown("---")
                    
                    # All top-3 predictions
                    st.markdown("#### 📊 Top-3 Predictions")
                    
                    for pred in predictions:
                        rank_emoji = "🥇" if pred['rank'] == 1 else "🥈" if pred['rank'] == 2 else "🥉"
                        conf_class = get_confidence_color(pred['confidence'])
                        
                        with st.container():
                            col_rank, col_name, col_conf = st.columns([1, 3, 2])
                            with col_rank:
                                st.markdown(f"### {rank_emoji}")
                            with col_name:
                                st.markdown(f"**{pred['name']}**")
                            with col_conf:
                                st.markdown(f"<span class='{conf_class}'>{pred['confidence']*100:.2f}%</span>", unsafe_allow_html=True)
                            
                            st.progress(pred['confidence'])
                            st.markdown("---")
                    
                    # Confidence chart
                    fig = create_confidence_bar_chart(predictions)
                    st.plotly_chart(fig, use_container_width=True)
                    
                else:
                    st.info("👆 Upload an image and click 'Predict Signature' to see results")
        
        # ===== BATCH UPLOAD =====
        with tab2:
            st.markdown("### 📦 Batch Upload")
            st.info("Upload multiple signature images at once for batch prediction")
            
            uploaded_files = st.file_uploader(
                "Choose multiple signature images",
                type=['png', 'jpg', 'jpeg'],
                accept_multiple_files=True,
                help="Upload multiple PNG, JPG, or JPEG files"
            )
            
            if uploaded_files:
                st.success(f"✅ {len(uploaded_files)} images uploaded")
                
                if st.button("🔍 Predict All", key="predict_batch"):
                    MODEL_PATH = "signature_model_final.keras"
                    LABEL_MAP_PATH = "label_map.json"
                    
                    if os.path.exists(MODEL_PATH) and os.path.exists(LABEL_MAP_PATH):
                        model, idx_to_name = load_model_and_mappings(MODEL_PATH, LABEL_MAP_PATH)
                        
                        if model is not None:
                            # Progress bar
                            progress_bar = st.progress(0)
                            status_text = st.empty()
                            
                            batch_results = []
                            
                            for idx, uploaded_file in enumerate(uploaded_files):
                                # Update progress
                                progress = (idx + 1) / len(uploaded_files)
                                progress_bar.progress(progress)
                                status_text.text(f"Processing {idx+1}/{len(uploaded_files)}: {uploaded_file.name}")
                                
                                # Predict
                                image = Image.open(uploaded_file)
                                predictions = predict_signature_top3(model, image, idx_to_name)
                                
                                batch_results.append({
                                    'filename': uploaded_file.name,
                                    'image': image,
                                    'predictions': predictions
                                })
                                
                                # Save to history
                                save_prediction_history(predictions, uploaded_file.name)
                            
                            progress_bar.empty()
                            status_text.empty()
                            
                            # Display results
                            st.success("✅ Batch prediction completed!")
                            
                            st.markdown("### 📊 Batch Results")
                            
                            # Create dataframe
                            df_data = []
                            for result in batch_results:
                                top_pred = result['predictions'][0]
                                df_data.append({
                                    'Filename': result['filename'],
                                    'Top Prediction': top_pred['name'],
                                    'Confidence': f"{top_pred['confidence']*100:.2f}%",
                                    'Rank 2': result['predictions'][1]['name'],
                                    'Rank 3': result['predictions'][2]['name']
                                })
                            
                            df = pd.DataFrame(df_data)
                            st.dataframe(df, use_container_width=True)
                            
                            # Download results
                            csv = df.to_csv(index=False)
                            st.download_button(
                                label="📥 Download Results (CSV)",
                                data=csv,
                                file_name=f"batch_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                mime="text/csv"
                            )
                            
                            # Show detailed view
                            st.markdown("### 🔍 Detailed View")
                            for result in batch_results:
                                with st.expander(f"📄 {result['filename']}"):
                                    col1, col2 = st.columns([1, 2])
                                    with col1:
                                        st.image(result['image'], use_container_width=True)
                                    with col2:
                                        for pred in result['predictions']:
                                            st.markdown(f"**#{pred['rank']}: {pred['name']}** - {pred['confidence']*100:.2f}%")
                    else:
                        st.error("❌ Model or label map file not found")
    
    # ==================== PAGE: STATISTICS & HISTORY ====================
    elif page == "📊 Statistics & History":
        st.markdown("## 📊 Prediction Statistics & History")
        
        if 'prediction_history' not in st.session_state or len(st.session_state.prediction_history) == 0:
            st.info("📭 No predictions yet. Upload signatures in the Home page to build history.")
        else:
            history = st.session_state.prediction_history
            
            # Summary metrics
            st.markdown("### 📈 Summary")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Predictions", len(history))
            
            with col2:
                # Count unique signatures predicted
                unique_sigs = set([h['predictions'][0]['name'] for h in history])
                st.metric("Unique Signatures", len(unique_sigs))
            
            with col3:
                # Average confidence
                avg_conf = np.mean([h['predictions'][0]['confidence'] for h in history])
                st.metric("Avg Confidence", f"{avg_conf*100:.1f}%")
            
            with col4:
                # High confidence predictions (>70%)
                high_conf = sum([1 for h in history if h['predictions'][0]['confidence'] >= 0.7])
                st.metric("High Confidence", f"{high_conf}/{len(history)}")
            
            st.markdown("---")
            
            # Visualizations
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### 📊 Top Predicted Signatures")
                # Count frequency
                pred_counts = {}
                for h in history:
                    name = h['predictions'][0]['name']
                    pred_counts[name] = pred_counts.get(name, 0) + 1
                
                # Create bar chart
                df_counts = pd.DataFrame(list(pred_counts.items()), columns=['Name', 'Count'])
                df_counts = df_counts.sort_values('Count', ascending=False).head(10)
                
                fig = px.bar(df_counts, x='Name', y='Count', 
                           title="Top 10 Most Predicted Signatures",
                           color='Count',
                           color_continuous_scale='Blues')
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.markdown("### 📈 Confidence Distribution")
                confidences = [h['predictions'][0]['confidence'] * 100 for h in history]
                
                fig = go.Figure(data=[go.Histogram(x=confidences, nbinsx=20,
                                                   marker_color='#3B82F6')])
                fig.update_layout(
                    title="Distribution of Top Prediction Confidence",
                    xaxis_title="Confidence (%)",
                    yaxis_title="Frequency",
                    showlegend=False
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Prediction history table
            st.markdown("### 📜 Prediction History")
            
            # Create detailed dataframe
            history_data = []
            for h in history:
                history_data.append({
                    'Timestamp': h['timestamp'],
                    'Image': h['image_name'],
                    'Top Prediction': h['predictions'][0]['name'],
                    'Confidence': f"{h['predictions'][0]['confidence']*100:.2f}%",
                    '2nd': h['predictions'][1]['name'],
                    '3rd': h['predictions'][2]['name']
                })
            
            df_history = pd.DataFrame(history_data)
            st.dataframe(df_history, use_container_width=True)
            
            # Export options
            col1, col2, col3 = st.columns(3)
            
            with col1:
                # Export to CSV
                csv = export_predictions_to_csv(history).to_csv(index=False)
                st.download_button(
                    label="📥 Export to CSV",
                    data=csv,
                    file_name=f"prediction_history_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                )
            
            with col2:
                # Clear history
                if st.button("🗑️ Clear History"):
                    st.session_state.prediction_history = []
                    st.rerun()
            
            with col3:
                st.info(f"Total records: {len(history)}")
    
    # ==================== PAGE: RETRAIN MODEL ====================
    elif page == "🔄 Retrain Model":
        st.markdown("## 🔄 Upload New Data & Retrain Model")
        
        st.warning("⚠️ **Advanced Feature:** This will retrain the model with new data. Make sure you have the necessary computational resources.")
        
        st.markdown("### 📤 Upload New Training Data")
        st.info("""
        **Instructions:**
        1. Prepare a ZIP file containing folders for each person
        2. Each folder should contain signature images of that person
        3. Folder name = Person's name
        4. Minimum 3-4 images per person recommended
        
        **Example structure:**
        ```
        signatures.zip
        ├── John Doe/
        │   ├── sig1.png
        │   ├── sig2.png
        │   └── sig3.png
        ├── Jane Smith/
        │   ├── sig1.png
        │   └── sig2.png
        └── ...
        ```
        """)
        
        uploaded_zip = st.file_uploader("Upload ZIP file with new signatures", type=['zip'])
        
        if uploaded_zip is not None:
            st.success(f"✅ ZIP file uploaded: {uploaded_zip.name}")
            
            # Extract and preview
            with st.expander("👁️ Preview uploaded data"):
                try:
                    with zipfile.ZipFile(uploaded_zip, 'r') as zip_ref:
                        file_list = zip_ref.namelist()
                        
                        # Count folders and files
                        folders = set([f.split('/')[0] for f in file_list if '/' in f])
                        
                        st.info(f"**Found {len(folders)} people/folders**")
                        
                        # Show folder structure
                        folder_stats = {}
                        for folder in folders:
                            files = [f for f in file_list if f.startswith(folder + '/') and not f.endswith('/')]
                            folder_stats[folder] = len(files)
                        
                        df_preview = pd.DataFrame(list(folder_stats.items()), 
                                                 columns=['Person', 'Number of Images'])
                        st.dataframe(df_preview, use_container_width=True)
                        
                except Exception as e:
                    st.error(f"Error reading ZIP file: {str(e)}")
            
            st.markdown("---")
            
            # Retrain configuration
            st.markdown("### ⚙️ Retrain Configuration")
            
            col1, col2 = st.columns(2)
            
            with col1:
                epochs = st.slider("Training Epochs", 10, 100, 30)
                batch_size = st.selectbox("Batch Size", [8, 16, 32], index=1)
            
            with col2:
                learning_rate = st.select_slider(
                    "Learning Rate",
                    options=[1e-5, 5e-5, 1e-4, 5e-4, 1e-3],
                    value=1e-3,
                    format_func=lambda x: f"{x:.0e}"
                )
                use_augmentation = st.checkbox("Use Data Augmentation", value=True)
            
            st.markdown("---")
            
            # Retrain button
            if st.button("🚀 Start Retraining", type="primary"):
                st.warning("🚧 **Retraining feature is a placeholder.** In production, this would:")
                st.markdown("""
                1. Extract and preprocess uploaded data
                2. Merge with existing training data
                3. Retrain the model with configured parameters
                4. Save new model weights
                5. Update label mappings
                
                **This requires:**
                - Google Colab or cloud GPU
                - Original training pipeline
                - Proper data validation
                
                **Recommendation:** Use the Tahap 2 notebook in Google Colab for retraining with new data.
                """)
                
                # Placeholder progress
                with st.spinner("Simulating retrain process..."):
                    import time
                    progress_bar = st.progress(0)
                    for i in range(100):
                        time.sleep(0.02)
                        progress_bar.progress(i + 1)
                    
                    st.success("✅ Retraining simulation completed! (This is a demo)")
    
    # ==================== PAGE: ABOUT ====================
    elif page == "ℹ️ About":
        st.markdown("## ℹ️ About This System")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("""
            ### 🖊️ BSI Signature Detection System
            
            Sistem pengenalan tanda tangan berbasis AI menggunakan Deep Learning untuk mengidentifikasi
            pemilik tanda tangan dari 34 nasabah BSI.
            
            #### 🎯 Features
            - **Single & Batch Upload**: Upload satu atau banyak gambar sekaligus
            - **Top-3 Predictions**: Menampilkan 3 prediksi teratas dengan confidence score
            - **Real-time Processing**: Prediksi instan dengan model yang sudah trained
            - **Prediction History**: Track semua prediksi yang pernah dilakukan
            - **Statistics Dashboard**: Visualisasi performa dan distribusi prediksi
            - **Export Results**: Download hasil dalam format CSV
            - **Retrain Capability**: Upload data baru dan retrain model
            
            #### 🏗️ Architecture
            - **Base Model**: MobileNetV2 (Transfer Learning from ImageNet)
            - **Input Size**: 224 x 224 x 3 (RGB)
            - **Output**: 34 classes (signatures)
            - **Framework**: TensorFlow 2.x / Keras
            
            #### 📊 Performance Metrics
            - **Test Accuracy**: 76.47%
            - **Top-3 Accuracy**: 92.65%
            - **Validation Accuracy**: 97.06%
            
            #### 🔧 Technical Stack
            - **Backend**: TensorFlow, Keras, NumPy
            - **Frontend**: Streamlit
            - **Visualization**: Plotly, Matplotlib
            - **Preprocessing**: PIL, OpenCV
            
            #### 📖 How It Works
            1. **Upload**: User mengupload gambar tanda tangan
            2. **Preprocessing**: Image di-resize dan dinormalisasi
            3. **Prediction**: Model MobileNetV2 melakukan inference
            4. **Results**: Menampilkan Top-3 predictions dengan confidence scores
            
            #### 🚀 Future Improvements
            - [ ] Real-time signature verification
            - [ ] Signature forgery detection
            - [ ] Multi-model ensemble
            - [ ] Mobile app integration
            - [ ] Cloud deployment (AWS/GCP)
            """)
        
        with col2:
            st.markdown("### 📞 Contact")
            st.info("""
            **Project Team**
            
            🏢 Bank Syariah Indonesia  
            📧 Email: contact@bsi.id  
            🌐 Website: www.bsi.co.id
            
            ---
            
            **Developer**
            
            💻 AI Development Team  
            📅 2024
            """)
            
            st.markdown("---")
            
            st.markdown("### 📄 Documentation")
            st.markdown("""
            - [User Guide](#)
            - [API Documentation](#)
            - [Model Training Guide](#)
            - [Deployment Guide](#)
            """)
            
            st.markdown("---")
            
            st.markdown("### 🏆 Competition")
            st.success("""
            **BSI Innovation Idea 2024**
            
            AI-powered solution for
            automatic signature recognition
            and verification.
            """)

# Run app
if __name__ == "__main__":
    main()