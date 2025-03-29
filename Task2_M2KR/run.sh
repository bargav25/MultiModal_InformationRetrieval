#!/bin/bash

# Exit immediately if any command fails
set -e

echo "📦 Installing Python dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

echo "🚀 Starting image scraping from Wikipedia..."
python src/scrape.py

echo "✅ Image scraping complete."

echo "🚀 Running main pipeline (main.pys)..."
python src/main.py

echo "✅ All tasks completed successfully."