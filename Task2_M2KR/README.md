# 🧠 M2KR: Multimodal Retrieval with Wikipedia + FAISS

This project implements an end-to-end **visual retrieval pipeline** that takes **query images** and retrieves the most relevant **Wikipedia articles** by matching them against images scraped or extracted from Wikipedia pages. It uses embeddings from ColQwen2, FAISS for efficient similarity search, and supports both image scraping from live Wikipedia pages and image extraction from screenshots.

---

## 🚀 Features

- 🔍 Scrape images from Wikipedia articles based on query filenames (Alternatively you can choose to extract images from the Wikipedia screenshots using traditional OpenCV techniques: check `extract_images.py`)
- 🖼️ Generate dense embeddings using a transformer-based vision model (`ColQwen2`)
- ⚡ Index passage images using FAISS (`IndexFlatL2`)
- 🎯 Retrieve top-k most relevant images per query
- 🧰 Modular code: easily extendable to other datasets or models

---

## 📁 Project Structure

```
m2kr/
├── src/
│   ├── model.py               # Embedding model using ColQwen2
│   ├── utils.py               # Image resizing, top-k utilities
│   ├── scrape.py              # Wikipedia image scraping logic
│   ├── main.py                # End-to-end pipeline: index + query
│   ├── config.py              # Config
├── data/
│   ├── passage_images/        # Indexed passage images
│   ├── query_images/          # Query images
│   ├── m2kr_data.parquet      # Query metadata
│   ├── faiss_index/           # Saved FAISS index
├── requirements.txt
├── run.sh                     # Shell script to run scrape + main
├── README.md
```

---
## Hardware Used

Multiple A100 or L40S NVIDIA GPUs with around 48 GB GPU RAM

## 📥 Dataset Setup (Before Running)

The folders `data/passage_images/` and `data/query_images/` initially contain only a few sample images.

To run the full pipeline, download the official dataset from Hugging Face:

📦 [M2KR-Challenge Dataset on Hugging Face](https://huggingface.co/datasets/Jingbiao/M2KR-Challenge/tree/main)

Once downloaded and placed correctly, you can proceed to run:

## 🔧 Usage

Run the entire pipeline:

```bash
python -m venv venv        
source venv/bin/activate    
bash run.sh
```

This will:
1. Install requirements
2. Scrape images from Wikipedia based on filenames in `data/passage_images/`
3. Build a FAISS index from those images
4. Embed and query the images from `data/query_images/`
5. Output the top-5 retrieved results per query to `data/final_results.csv`

---

## 📌 Requirements

- Python 3.10

---

## 📄 License

MIT License

---

## 🤝 Contributions

Pull requests are welcome. If you find a bug or want to improve something, feel free to open an issue or submit a PR.

---

## 👨‍💻 Author

**Bargav Jagatha**  
[github.com/bargav25](https://github.com/bargav25)

**Abhishek Varshney**  
[github.com/avarshn](https://github.com/avarshn)
