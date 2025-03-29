
# MIR Challenge Track 1

This repository combines solutions of two separate tasks of the MIR Challenge Track 1.

- **Task 1: MMDocIR – Multi-Modal Retrieval for Long Documents**
- **Task 2: M2KR – Multimodal Retrieval with Wikipedia**

## 📁 Project Structure

```
MIRChallenge_2025/
├── Task1_MMDocIR/
│   ├── data_task1/
│   │   ├── MMDocIR_gt_remove.jsonl  # JSONL file with QuestionID, Questions, Doc Name and other info
│   ├── misc/
│   │   ├── retrieval_workflow.png # Code Workflow
│   ├── src/
│   │   ├── task1_ColQwen2.py      # End-to-end pipeline: embedding + querying + retrieval
│   │   ├── runs.sh                # Shell script to run task1_ColQwen2.py on whole MMDocIR evaluation dataset
│   ├── LICENSE
│   ├── README.md
│   ├── requirements.txt           # Python Packages with their versions used
│
├── Task2_M2KR/
│   ├── src/
│   │   ├── model.py               # Embedding model using ColQwen2
│   │   ├── utils.py               # Image resizing, top-k utilities
│   │   ├── scrape.py              # Wikipedia image scraping logic
│   │   ├── main.py                # End-to-end pipeline: index + query
│   │   ├── config.py              # Config
│   ├── data/
│   │   ├── passage_images/        # Indexed passage images
│   │   ├── query_images/          # Query images
│   │   ├── m2kr_data.parquet      # Query metadata
│   │   ├── faiss_index/           # Saved FAISS index
│   ├── requirements.txt
│   ├── run.sh                     # Shell script to run scrape + main
│   ├── README.md
│
├── .gitignore
├── LICENSE
├── README.md
```

## Task 1: MMDocIR – Multi-Modal Retrieval for Long Documents
This task evaluates the ability of retrieval systems to identify visually-rich information within documents. The MMDocIR evaluation set includes **313 long documents** with an average of **65.1 pages**, categorized across diverse domains.

### Objective
**For a given text query → Retrieve the Relevant Document Page**: Identify the most relevant pages within a document in response to a user query. The retrieval scope for each query is restricted to all pages in the given document.

### Environment Setup
```
conda create -n rag_env python=3.11
conda activate rag_env
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

### How to Run
```
cd Task1_MMDocIR/src
mkdir model_cache
cd src
./run.sh
```

### Hardware Used
Multiple A100 or L40S NVIDIA GPUs with around 48 GB GPU RAM

---

## Task 2: M2KR – Multimodal Retrieval with Wikipedia + FAISS
This task implements a visual retrieval pipeline that takes query images and retrieves the most relevant Wikipedia articles by matching them against images scraped or extracted from Wikipedia pages.

### Environment Setup
```
python -m venv venv
source venv/bin/activate
```

### How to Run
```
cd Task2_M2KR
bash run.sh
```

### Hardware Used
Multiple A100 or L40S NVIDIA GPUs with around 48 GB GPU RAM

---

## 📄 License
MIT License

---

## 🤝 Contributions
Pull requests are welcome. If you find a bug or want to improve something, feel free to open an issue or submit a PR.

---

## 👨‍💻 Authors
- **Bargav Jagatha**  
  [github.com/bargav25](https://github.com/bargav25)
- **Abhishek Varshney**  
  [github.com/avarshn](https://github.com/avarshn)
