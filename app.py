"""
🖊️ Sistem Deteksi Tanda Tangan BSI
Aplikasi Streamlit dengan Fitur Lengkap

Fitur:
- Upload Tunggal & Batch
- Top-3 Prediksi dengan Visualisasi Kepercayaan
- Upload Data Baru & Latih Ulang Model
- Riwayat Prediksi & Statistik
- Ekspor Hasil ke CSV
- Antarmuka Profesional
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
import cv2  # untuk preprocessing identik dengan Tahap 1
import tensorflow as tf  # type: ignore

# Suppress Pylance warnings for tensorflow submodules
keras = tf.keras  # type: ignore

# Konfigurasi halaman
st.set_page_config(
    page_title="Deteksi Tanda Tangan BSI",
    page_icon="🖊️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS Kustom untuk branding BSI
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

# ==================== FUNGSI PEMBANTU ====================

@st.cache_resource
def muat_model_dan_mapping(model_path, label_map_path):
    """Memuat model yang sudah dilatih dan pemetaan label"""
    try:
        model = tf.keras.models.load_model(model_path)
        with open(label_map_path, 'r') as f:
            label_map = json.load(f)
            # Konversi kunci string ke int
            label_map = {int(k): v for k, v in label_map.items()}

        # Buat mapping idx → nama
        idx_to_name = {idx: name for idx, name in label_map.items()}

        return model, idx_to_name
    except Exception as e:
        st.error(f"Gagal memuat model: {str(e)}")
        return None, None


def preprocess_tanda_tangan(image, target_size=(224, 224)):
    """
    ⚠️ IDENTIK dengan Tahap 1 preprocess_signature():
    1. RGBA → RGB (latar belakang putih)
    2. Grayscale → threshold (240) → findContours → boundingRect → CROP + margin 10px
    3. thumbnail() → pertahankan rasio → tempel di TENGAH kanvas putih 224x224
    4. Array [0,255] → MobileNetV2 preprocess → [-1,1]
    """
    # Langkah 1: Tangani PNG transparan
    if image.mode == 'RGBA':
        background = Image.new('RGB', image.size, (255, 255, 255))
        background.paste(image, mask=image.split()[3])
        image = background
    elif image.mode != 'RGB':
        image = image.convert('RGB')

    # Langkah 2: Konversi ke array numpy
    img_array = np.array(image)

    # Langkah 3: Grayscale + threshold — SAMA dengan Tahap 1 (threshold=240)
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    _, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)

    # Langkah 4: Temukan kontur → bounding box → CROP — SAMA dengan Tahap 1
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        x, y, w, h = cv2.boundingRect(np.concatenate(contours))
        margin = 10
        x = max(0, x - margin)
        y = max(0, y - margin)
        w = min(img_array.shape[1] - x, w + 2 * margin)
        h = min(img_array.shape[0] - y, h + 2 * margin)
        img_array = img_array[y:y+h, x:x+w]

    # Langkah 5: thumbnail (pertahankan rasio) + tempel di TENGAH — SAMA dengan Tahap 1
    img_pil = Image.fromarray(img_array)
    img_pil.thumbnail(target_size, Image.Resampling.LANCZOS)
    canvas = Image.new('RGB', target_size, (255, 255, 255))
    offset = (
        (target_size[0] - img_pil.size[0]) // 2,
        (target_size[1] - img_pil.size[1]) // 2
    )
    canvas.paste(img_pil, offset)

    # Langkah 6: [0,255] → MobileNetV2 preprocess → [-1,1]
    img_out = np.array(canvas).astype(np.float32)
    img_out = np.expand_dims(img_out, axis=0)
    return tf.keras.applications.mobilenet_v2.preprocess_input(img_out)


def prediksi_top3(model, image, idx_to_name):
    """Prediksi 3 tanda tangan teratas beserta skor kepercayaan"""
    img_preprocessed = preprocess_tanda_tangan(image)
    predictions = model.predict(img_preprocessed, verbose=0)[0]
    top3_idx = np.argsort(predictions)[-3:][::-1]

    hasil = []
    for peringkat, idx in enumerate(top3_idx, 1):
        nama = idx_to_name[idx]
        kepercayaan = float(predictions[idx])
        hasil.append({
            'peringkat': peringkat,
            'nama': nama,
            'kepercayaan': kepercayaan
        })

    return hasil


def buat_grafik_kepercayaan(prediksi):
    """Buat grafik batang horizontal untuk 3 prediksi teratas"""
    nama = [p['nama'] for p in prediksi]
    nilai = [p['kepercayaan'] * 100 for p in prediksi]
    peringkat = [f"#{p['peringkat']}" for p in prediksi]

    warna = ['#10B981', '#3B82F6', '#9CA3AF']

    fig = go.Figure(data=[
        go.Bar(
            y=peringkat,
            x=nilai,
            orientation='h',
            text=[f"{c:.2f}%" for c in nilai],
            textposition='outside',
            marker=dict(color=warna),
            hovertemplate='<b>%{y}</b>: %{customdata}<br>Kepercayaan: %{x:.2f}%<extra></extra>',
            customdata=nama
        )
    ])

    fig.update_layout(
        title="3 Prediksi Teratas",
        xaxis_title="Kepercayaan (%)",
        yaxis_title="Peringkat",
        height=300,
        showlegend=False,
        xaxis=dict(range=[0, 100]),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
    )

    return fig


def simpan_riwayat(prediksi, nama_gambar):
    """Simpan prediksi ke session state"""
    if 'riwayat_prediksi' not in st.session_state:
        st.session_state.riwayat_prediksi = []

    entri = {
        'waktu': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'nama_gambar': nama_gambar,
        'prediksi': prediksi
    }

    st.session_state.riwayat_prediksi.append(entri)


def warna_kepercayaan(kepercayaan):
    """Tentukan kelas warna berdasarkan tingkat kepercayaan"""
    if kepercayaan >= 0.7:
        return "confidence-high"
    elif kepercayaan >= 0.4:
        return "confidence-medium"
    else:
        return "confidence-low"


def ekspor_ke_csv(riwayat):
    """Ekspor riwayat prediksi ke CSV"""
    data = []
    for entri in riwayat:
        for pred in entri['prediksi']:
            data.append({
                'Waktu': entri['waktu'],
                'Gambar': entri['nama_gambar'],
                'Peringkat': pred['peringkat'],
                'Nama': pred['nama'],
                'Kepercayaan': f"{pred['kepercayaan']*100:.2f}%"
            })

    return pd.DataFrame(data)


# ==================== APLIKASI UTAMA ====================

def main():
    # Header
    st.markdown('<h1 class="main-header">🖊️ B Sign Verification System</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Presented to : BSI Decode the Future with Al Innovation Challenge 2026</p>', unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.image("picture/logo.png", use_container_width=True)
        st.markdown("---")

        # Navigasi
        halaman = st.radio(
            "Navigasi",
            ["🏠 Beranda", "📊 Statistik", "🔄 Latih Ulang Model", "ℹ️ Tentang"],
            label_visibility="collapsed"
        )

        st.markdown("---")

        # Info model
        st.markdown("### 📋 Info Model")
        st.info("""
        **Arsitektur:** MobileNetV2  
        **Kelas:** 34 tanda tangan  
        **Akurasi:** 76,47%  
        **Top-3 Akurasi:** 92,65%
        """)

        st.markdown("---")
        st.markdown("**🏆 BSI Innovation Idea 2024**")
        st.markdown("*Didukung oleh TensorFlow & Streamlit*")

    # ==================== HALAMAN: BERANDA & PREDIKSI ====================
    if halaman == "🏠 Beranda":
        st.markdown("## 📤 Unggah Tanda Tangan untuk Prediksi")

        tab1, tab2 = st.tabs(["📄 Unggah Tunggal", "📦 Unggah Batch"])

        # ===== UNGGAH TUNGGAL =====
        with tab1:
            col1, col2 = st.columns([1, 1])

            with col1:
                st.markdown("### Unggah Gambar")
                berkas_diunggah = st.file_uploader(
                    "Pilih gambar tanda tangan",
                    type=['png', 'jpg', 'jpeg'],
                    help="Unggah format PNG, JPG, atau JPEG"
                )

                if berkas_diunggah is not None:
                    image = Image.open(berkas_diunggah)
                    st.image(image, caption="Tanda Tangan yang Diunggah", use_container_width=True)

                    if st.button("🔍 Prediksi Tanda Tangan", key="prediksi_tunggal"):
                        with st.spinner("Menganalisis tanda tangan..."):
                            MODEL_PATH = "signature_model_final.keras"
                            LABEL_MAP_PATH = "label_map.json"

                            if not os.path.exists(MODEL_PATH):
                                st.error(f"❌ File model tidak ditemukan: {MODEL_PATH}")
                                st.info("Letakkan file model di direktori yang sama dengan aplikasi ini.")
                            elif not os.path.exists(LABEL_MAP_PATH):
                                st.error(f"❌ File label map tidak ditemukan: {LABEL_MAP_PATH}")
                            else:
                                model, idx_to_name = muat_model_dan_mapping(MODEL_PATH, LABEL_MAP_PATH)

                                if model is not None:
                                    prediksi = prediksi_top3(model, image, idx_to_name)
                                    simpan_riwayat(prediksi, berkas_diunggah.name)
                                    st.session_state.prediksi_sekarang = prediksi

            with col2:
                st.markdown("### 🎯 Hasil Prediksi")

                if 'prediksi_sekarang' in st.session_state:
                    prediksi = st.session_state.prediksi_sekarang

                    # Sorot prediksi teratas
                    top_pred = prediksi[0]
                    st.success(f"**Paling Mungkin:** {top_pred['nama']}")
                    st.markdown(f"**Kepercayaan:** {top_pred['kepercayaan']*100:.2f}%")

                    st.markdown("---")
                    st.markdown("#### 📊 3 Prediksi Teratas")

                    for pred in prediksi:
                        emoji_peringkat = "🥇" if pred['peringkat'] == 1 else "🥈" if pred['peringkat'] == 2 else "🥉"
                        kelas_warna = warna_kepercayaan(pred['kepercayaan'])

                        with st.container():
                            col_rank, col_name, col_conf = st.columns([1, 3, 2])
                            with col_rank:
                                st.markdown(f"### {emoji_peringkat}")
                            with col_name:
                                st.markdown(f"**{pred['nama']}**")
                            with col_conf:
                                st.markdown(
                                    f"<span class='{kelas_warna}'>{pred['kepercayaan']*100:.2f}%</span>",
                                    unsafe_allow_html=True
                                )
                            st.progress(pred['kepercayaan'])
                            st.markdown("---")

                    # Grafik kepercayaan
                    fig = buat_grafik_kepercayaan(prediksi)
                    st.plotly_chart(fig, use_container_width=True)

                else:
                    st.info("👆 Unggah gambar lalu klik 'Prediksi Tanda Tangan' untuk melihat hasil")

        # ===== UNGGAH BATCH =====
        with tab2:
            st.markdown("### 📦 Unggah Batch")
            st.info("Unggah beberapa gambar tanda tangan sekaligus untuk prediksi massal")

            berkas_batch = st.file_uploader(
                "Pilih beberapa gambar tanda tangan",
                type=['png', 'jpg', 'jpeg'],
                accept_multiple_files=True,
                help="Unggah beberapa file PNG, JPG, atau JPEG"
            )

            if berkas_batch:
                st.success(f"✅ {len(berkas_batch)} gambar diunggah")

                if st.button("🔍 Prediksi Semua", key="prediksi_batch"):
                    MODEL_PATH = "signature_model_final.keras"
                    LABEL_MAP_PATH = "label_map.json"

                    if os.path.exists(MODEL_PATH) and os.path.exists(LABEL_MAP_PATH):
                        model, idx_to_name = muat_model_dan_mapping(MODEL_PATH, LABEL_MAP_PATH)

                        if model is not None:
                            progress_bar = st.progress(0)
                            status_text = st.empty()
                            hasil_batch = []

                            for idx, berkas in enumerate(berkas_batch):
                                progres = (idx + 1) / len(berkas_batch)
                                progress_bar.progress(progres)
                                status_text.text(f"Memproses {idx+1}/{len(berkas_batch)}: {berkas.name}")

                                image = Image.open(berkas)
                                prediksi = prediksi_top3(model, image, idx_to_name)

                                hasil_batch.append({
                                    'nama_file': berkas.name,
                                    'gambar': image,
                                    'prediksi': prediksi
                                })

                                simpan_riwayat(prediksi, berkas.name)

                            progress_bar.empty()
                            status_text.empty()

                            st.success("✅ Prediksi batch selesai!")
                            st.markdown("### 📊 Hasil Batch")

                            # Buat dataframe hasil
                            data_df = []
                            for hasil in hasil_batch:
                                top = hasil['prediksi'][0]
                                data_df.append({
                                    'Nama File': hasil['nama_file'],
                                    'Prediksi Teratas': top['nama'],
                                    'Kepercayaan': f"{top['kepercayaan']*100:.2f}%",
                                    'Peringkat 2': hasil['prediksi'][1]['nama'],
                                    'Peringkat 3': hasil['prediksi'][2]['nama']
                                })

                            df = pd.DataFrame(data_df)
                            st.dataframe(df, use_container_width=True)

                            # Unduh hasil
                            csv = df.to_csv(index=False)
                            st.download_button(
                                label="📥 Unduh Hasil (CSV)",
                                data=csv,
                                file_name=f"prediksi_batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                mime="text/csv"
                            )

                            # Tampilan detail
                            st.markdown("### 🔍 Tampilan Detail")
                            for hasil in hasil_batch:
                                with st.expander(f"📄 {hasil['nama_file']}"):
                                    col1, col2 = st.columns([1, 2])
                                    with col1:
                                        st.image(hasil['gambar'], use_container_width=True)
                                    with col2:
                                        for pred in hasil['prediksi']:
                                            st.markdown(f"**#{pred['peringkat']}: {pred['nama']}** — {pred['kepercayaan']*100:.2f}%")
                    else:
                        st.error("❌ File model atau label map tidak ditemukan")

    # ==================== HALAMAN: STATISTIK & RIWAYAT ====================
    elif halaman == "📊 Statistik":
        st.markdown("## 📊 Statistik & Riwayat Prediksi")

        if 'riwayat_prediksi' not in st.session_state or len(st.session_state.riwayat_prediksi) == 0:
            st.info("📭 Belum ada prediksi. Unggah tanda tangan di halaman Beranda untuk mulai.")
        else:
            riwayat = st.session_state.riwayat_prediksi

            # Ringkasan metrik
            st.markdown("### 📈 Ringkasan")
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("Total Prediksi", len(riwayat))
            with col2:
                nama_unik = set([h['prediksi'][0]['nama'] for h in riwayat])
                st.metric("Tanda Tangan Unik", len(nama_unik))
            with col3:
                rata_kepercayaan = np.mean([h['prediksi'][0]['kepercayaan'] for h in riwayat])
                st.metric("Rata-rata Kepercayaan", f"{rata_kepercayaan*100:.1f}%")
            with col4:
                tinggi = sum([1 for h in riwayat if h['prediksi'][0]['kepercayaan'] >= 0.7])
                st.metric("Kepercayaan Tinggi", f"{tinggi}/{len(riwayat)}")

            st.markdown("---")

            # Visualisasi
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("### 📊 Tanda Tangan Paling Sering Diprediksi")
                jumlah_prediksi = {}
                for h in riwayat:
                    nama = h['prediksi'][0]['nama']
                    jumlah_prediksi[nama] = jumlah_prediksi.get(nama, 0) + 1

                df_jumlah = pd.DataFrame(list(jumlah_prediksi.items()), columns=['Nama', 'Jumlah'])
                df_jumlah = df_jumlah.sort_values('Jumlah', ascending=False).head(10)

                fig = px.bar(df_jumlah, x='Nama', y='Jumlah',
                             title="10 Tanda Tangan Paling Sering",
                             color='Jumlah',
                             color_continuous_scale='Blues')
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                st.markdown("### 📈 Distribusi Kepercayaan")
                nilai_kepercayaan = [h['prediksi'][0]['kepercayaan'] * 100 for h in riwayat]

                fig = go.Figure(data=[go.Histogram(
                    x=nilai_kepercayaan, nbinsx=20, marker_color='#3B82F6'
                )])
                fig.update_layout(
                    title="Distribusi Kepercayaan Prediksi Teratas",
                    xaxis_title="Kepercayaan (%)",
                    yaxis_title="Frekuensi",
                    showlegend=False
                )
                st.plotly_chart(fig, use_container_width=True)

            # Tabel riwayat
            st.markdown("### 📜 Riwayat Prediksi")

            data_riwayat = []
            for h in riwayat:
                data_riwayat.append({
                    'Waktu': h['waktu'],
                    'Gambar': h['nama_gambar'],
                    'Prediksi Teratas': h['prediksi'][0]['nama'],
                    'Kepercayaan': f"{h['prediksi'][0]['kepercayaan']*100:.2f}%",
                    'Peringkat 2': h['prediksi'][1]['nama'],
                    'Peringkat 3': h['prediksi'][2]['nama']
                })

            df_riwayat = pd.DataFrame(data_riwayat)
            st.dataframe(df_riwayat, use_container_width=True)

            # Opsi ekspor
            col1, col2, col3 = st.columns(3)

            with col1:
                csv = ekspor_ke_csv(riwayat).to_csv(index=False)
                st.download_button(
                    label="📥 Ekspor ke CSV",
                    data=csv,
                    file_name=f"riwayat_prediksi_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                )
            with col2:
                if st.button("🗑️ Hapus Riwayat"):
                    st.session_state.riwayat_prediksi = []
                    st.rerun()
            with col3:
                st.info(f"Total data: {len(riwayat)}")

    # ==================== HALAMAN: LATIH ULANG MODEL ====================
    elif halaman == "🔄 Latih Ulang Model":
        st.markdown("## 🔄 Unggah Data Baru & Latih Ulang Model")

        st.warning("⚠️ **Fitur Lanjutan:** Ini akan melatih ulang model dengan data baru. Pastikan Anda memiliki sumber daya komputasi yang memadai.")

        st.markdown("### 📤 Unggah Data Pelatihan Baru")
        st.info("""
        **Petunjuk:**
        1. Siapkan file ZIP berisi folder untuk setiap orang
        2. Setiap folder berisi gambar tanda tangan orang tersebut
        3. Nama folder = Nama orang
        4. Minimal 3–4 gambar per orang direkomendasikan

        **Contoh struktur:**
        ```
        tanda_tangan.zip
        ├── Budi Santoso/
        │   ├── ttd1.png
        │   ├── ttd2.png
        │   └── ttd3.png
        ├── Siti Rahayu/
        │   ├── ttd1.png
        │   └── ttd2.png
        └── ...
        ```
        """)

        zip_diunggah = st.file_uploader("Unggah file ZIP dengan tanda tangan baru", type=['zip'])

        if zip_diunggah is not None:
            st.success(f"✅ File ZIP diunggah: {zip_diunggah.name}")

            with st.expander("👁️ Pratinjau data yang diunggah"):
                try:
                    with zipfile.ZipFile(zip_diunggah, 'r') as zip_ref:
                        daftar_file = zip_ref.namelist()
                        folder = set([f.split('/')[0] for f in daftar_file if '/' in f])

                        st.info(f"**Ditemukan {len(folder)} orang/folder**")

                        statistik_folder = {}
                        for fo in folder:
                            files = [f for f in daftar_file if f.startswith(fo + '/') and not f.endswith('/')]
                            statistik_folder[fo] = len(files)

                        df_pratinjau = pd.DataFrame(
                            list(statistik_folder.items()),
                            columns=['Nama', 'Jumlah Gambar']
                        )
                        st.dataframe(df_pratinjau, use_container_width=True)

                except Exception as e:
                    st.error(f"Gagal membaca file ZIP: {str(e)}")

            st.markdown("---")

            # Konfigurasi pelatihan ulang
            st.markdown("### ⚙️ Konfigurasi Pelatihan Ulang")

            col1, col2 = st.columns(2)

            with col1:
                epochs = st.slider("Jumlah Epoch", 10, 100, 30)
                batch_size = st.selectbox("Ukuran Batch", [8, 16, 32], index=1)
            with col2:
                learning_rate = st.select_slider(
                    "Learning Rate",
                    options=[1e-5, 5e-5, 1e-4, 5e-4, 1e-3],
                    value=1e-3,
                    format_func=lambda x: f"{x:.0e}"
                )
                augmentasi = st.checkbox("Gunakan Augmentasi Data", value=True)

            st.markdown("---")

            if st.button("🚀 Mulai Pelatihan Ulang", type="primary"):
                st.warning("🚧 **Fitur ini bersifat simulasi.** Dalam produksi, proses ini akan:")
                st.markdown("""
                1. Mengekstrak dan memproses data yang diunggah
                2. Menggabungkan dengan data pelatihan sebelumnya
                3. Melatih ulang model dengan parameter yang dikonfigurasi
                4. Menyimpan bobot model baru
                5. Memperbarui pemetaan label

                **Dibutuhkan:**
                - Google Colab atau GPU cloud
                - Pipeline pelatihan asli (Notebook Tahap 2)
                - Validasi data yang tepat

                **Rekomendasi:** Gunakan notebook Tahap 2 di Google Colab untuk melatih ulang dengan data baru.
                """)

                with st.spinner("Mensimulasikan proses pelatihan ulang..."):
                    import time
                    progress_bar = st.progress(0)
                    for i in range(100):
                        time.sleep(0.02)
                        progress_bar.progress(i + 1)

                    st.success("✅ Simulasi pelatihan ulang selesai! (Ini hanya demo)")

    # ==================== HALAMAN: TENTANG ====================
    elif halaman == "ℹ️ Tentang":
        st.markdown("## ℹ️ Tentang Sistem Ini")

        col1, col2 = st.columns([2, 1])

        with col1:
            st.markdown("""
            ### 🖊️ Sistem Deteksi Tanda Tangan BSI

            Sistem pengenalan tanda tangan berbasis AI menggunakan Deep Learning untuk mengidentifikasi
            pemilik tanda tangan dari 34 nasabah BSI.

            #### 🎯 Fitur Utama
            - **Upload Tunggal & Batch**: Unggah satu atau banyak gambar sekaligus
            - **3 Prediksi Teratas**: Menampilkan 3 kandidat tanda tangan dengan skor kepercayaan
            - **Pemrosesan Real-time**: Prediksi instan menggunakan model terlatih
            - **Riwayat Prediksi**: Lacak semua prediksi yang pernah dilakukan
            - **Dasbor Statistik**: Visualisasi performa dan distribusi prediksi
            - **Ekspor Hasil**: Unduh hasil dalam format CSV
            - **Latih Ulang Model**: Unggah data baru dan latih ulang model

            #### 🏗️ Arsitektur Model
            - **Model Dasar**: MobileNetV2 (Transfer Learning dari ImageNet)
            - **Ukuran Input**: 224 × 224 × 3 (RGB)
            - **Output**: 34 kelas (tanda tangan)
            - **Framework**: TensorFlow 2.x / Keras

            #### 📊 Metrik Performa
            - **Akurasi Uji**: 76,47%
            - **Top-3 Akurasi**: 92,65%
            - **Akurasi Validasi**: 97,06%

            #### 🔧 Tumpukan Teknologi
            - **Backend**: TensorFlow, Keras, NumPy, OpenCV
            - **Frontend**: Streamlit
            - **Visualisasi**: Plotly
            - **Preprocessing**: PIL, OpenCV

            #### 📖 Cara Kerja
            1. **Unggah**: Pengguna mengunggah gambar tanda tangan
            2. **Preprocessing**: Gambar dicrop, diubah ukuran, dan dinormalisasi (identik Tahap 1)
            3. **Prediksi**: Model MobileNetV2 melakukan inferensi
            4. **Hasil**: Menampilkan 3 prediksi teratas dengan skor kepercayaan

            #### 🚀 Rencana Pengembangan
            - [ ] Verifikasi tanda tangan secara real-time
            - [ ] Deteksi pemalsuan tanda tangan
            - [ ] Ensemble multi-model
            - [ ] Integrasi aplikasi mobile
            - [ ] Deployment cloud (AWS/GCP)
            """)

        with col2:
            st.markdown("### 📞 Kontak")
            st.info("""
            **Tim Proyek**

            🏢 Bank Syariah Indonesia  
            📧 Email: contact@bsi.id  
            🌐 Website: www.bsi.co.id

            ---

            **Pengembang**

            💻 Tim Pengembangan AI  
            📅 2024
            """)

            st.markdown("---")
            st.markdown("### 📄 Dokumentasi")
            st.markdown("""
            - [Panduan Pengguna](#)
            - [Dokumentasi API](#)
            - [Panduan Pelatihan Model](#)
            - [Panduan Deployment](#)
            """)

            st.markdown("---")
            st.markdown("### 🏆 Kompetisi")
            st.success("""
            **BSI Innovation Idea 2024**

            Solusi berbasis AI untuk pengenalan
            dan verifikasi tanda tangan otomatis.
            """)


# Jalankan aplikasi
if __name__ == "__main__":
    main()