#!/bin/bash

# BSI Signature Detection - Quick Start Script
# This script will setup and run the Streamlit app

echo "🖊️  BSI Signature Detection System"
echo "===================================="
echo ""

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed!"
    echo "Please install Python 3.10 or higher"
    exit 1
fi

echo "✅ Python found: $(python3 --version)"
echo ""

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
    echo "✅ Virtual environment created"
    echo ""
fi

# Activate virtual environment
echo "🔄 Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "📦 Installing dependencies..."
pip install -q --upgrade pip
pip install -q -r requirements.txt

echo "✅ Dependencies installed"
echo ""

# Check if model file exists
if [ ! -f "signature_model_final.keras" ]; then
    echo "⚠️  WARNING: Model file not found!"
    echo ""
    echo "Please ensure 'signature_model_final.keras' is in the current directory"
    echo "You can:"
    echo "  1. Copy it from Google Drive"
    echo "  2. Download it using the notebook"
    echo "  3. Use Google Drive download in app (see DEPLOYMENT_GUIDE.md)"
    echo ""
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Check if label_map.json exists
if [ ! -f "label_map.json" ]; then
    echo "⚠️  WARNING: label_map.json not found!"
    echo "Please copy it from Google Drive"
    echo ""
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Run Streamlit app
echo "🚀 Starting Streamlit app..."
echo ""
echo "App will open at: http://localhost:8501"
echo "Press Ctrl+C to stop"
echo ""

streamlit run app.py
