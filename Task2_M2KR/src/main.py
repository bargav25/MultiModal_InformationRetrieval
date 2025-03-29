import os
import gc
import torch
import faiss
import numpy as np
import pandas as pd
from tqdm import tqdm

from config import * 
from model import EmbedModel
from utils import resize_image, get_top5_docs

# ----------------------- STEP 1: Prepare Image Paths -----------------------

# Get all subfolders containing passage images
all_img_folders = os.listdir(PASSAGE_SCRAPED_FOLDER)

all_image_paths = []
image_folder_map = []

# Traverse each folder and gather image paths
for img_folder in tqdm(all_img_folders, desc="Processing scraped folders"):
    folder_path = os.path.join(PASSAGE_SCRAPED_FOLDER, img_folder)
    if os.path.isdir(folder_path):
        for img_file in os.listdir(folder_path):
            img_path = os.path.join(folder_path, img_file)
            all_image_paths.append(img_path)
            image_folder_map.append(img_folder)

print("Total Passage Images Found:", len(all_image_paths))

# ----------------------- STEP 2: Generate Embeddings -----------------------

# Load your embedding model
model = EmbedModel(MODEL_NAME)

batch_size = 256
embeddings_list = []

# Embed images in batches
for i in tqdm(range(0, len(all_image_paths), batch_size), desc="Indexing passage images"):
    batch_paths = all_image_paths[i:i + batch_size]
    try:
        batch_embeddings = model.get_image_embeddings(batch_paths).astype(np.float32)
        embeddings_list.append(batch_embeddings)
    except Exception as e:
        print(f"Failed embedding batch {i}-{i+batch_size}: {e}")
    torch.cuda.empty_cache()
    gc.collect()

# Stack all embeddings into a single array
embeddings = np.concatenate(embeddings_list, axis=0)

# ----------------------- STEP 3: Build FAISS Index -----------------------

os.makedirs(OUTPUT_INDEX_FOLDER, exist_ok=True)

embedding_dim = embeddings.shape[1]
index = faiss.IndexFlatL2(embedding_dim)  # Use L2 distance

# Add embeddings to the FAISS index
for i in tqdm(range(0, len(embeddings), batch_size), desc="Building FAISS index"):
    batch = embeddings[i:i + batch_size]
    index.add(batch)

# Save the index
faiss_index_path = os.path.join(OUTPUT_INDEX_FOLDER, "faiss_index.index")
faiss.write_index(index, faiss_index_path)

# Check that the number of indexed vectors matches your mapping
assert len(image_folder_map) == index.ntotal, "Index length mismatch"

print("FAISS Index Built and Saved to:", faiss_index_path)

# ----------------------- STEP 4: Process Query Images -----------------------

# Load query metadata (e.g. DataFrame with image filenames)
query_df = pd.read_parquet(QUERY_FILE)
query_image_names = query_df['img_path'].tolist()

valid_query_image_paths = []
valid_query_indices = []

# Filter out missing query images
for idx, img_name in enumerate(query_image_names):
    img_path = os.path.join(QUERY_IMAGE_FOLDER, img_name)
    if os.path.exists(img_path):
        valid_query_image_paths.append(img_path)
        valid_query_indices.append(idx)
    else:
        print(f"Missing query image: {img_path}")

print("Total Valid Query Images:", len(valid_query_image_paths))

# ----------------------- STEP 5: Run Similarity Search -----------------------

passage_ids = []

for i in tqdm(range(0, len(valid_query_image_paths), batch_size), desc="Processing query images"):
    batch_paths = valid_query_image_paths[i:i+batch_size]
    try:
        batch_embeddings = model.get_image_embeddings(batch_paths).astype(np.float32)
        distances, indices = index.search(batch_embeddings, 20)  # Top-20 search
        batch_top5_docs = get_top5_docs(distances, indices, image_folder_map)
        passage_ids.extend(batch_top5_docs)
    except Exception as e:
        print(f"Error during query batch {i}-{i+batch_size}: {e}")
    torch.cuda.empty_cache()
    gc.collect()

# ----------------------- STEP 6: Save Final Results -----------------------

# Create result DataFrame mapping query indices to top-5 passage folders
res_df = pd.DataFrame({
    'query_id': valid_query_indices,
    'passage_ids': passage_ids
})

res_df.to_csv(FINAL_RESULTS_FILE, index=False)

print(f"Results saved to: {FINAL_RESULTS_FILE}")