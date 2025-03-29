import os
import sys
import argparse
import io
from PIL import Image
from tqdm import tqdm
import json
import pandas as pd
import numpy as np

from datasets import load_dataset

import torch
from torch.utils.data import Dataset, DataLoader
from colpali_engine.models import ColQwen2, ColQwen2Processor
from transformers.utils.import_utils import is_flash_attn_2_available


class EmbedModel:
    def __init__(self, model_name):
        """Initialize embedding model and processor."""
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.embed_model = ColQwen2.from_pretrained(
            self.model_name,
            torch_dtype=torch.bfloat16,
            device_map=self.device,
            attn_implementation="flash_attention_2" if is_flash_attn_2_available() else None,
            cache_dir = "../model_cache/"
        )
        
        self.embed_model.eval()
        self.processor = ColQwen2Processor.from_pretrained(self.model_name)

    
class MultiModalDataset(Dataset):
    def __init__(self, hf_dataset, doc_name_filter):
        """
        Args:
            hf_dataset: Hugging Face dataset object (ds["train"])
            doc_name_filter: Only process images with this `doc_name`
        """
        self.dataset = [entry for entry in hf_dataset if entry["doc_name"] == doc_name_filter]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        passage_id = self.dataset[idx]["passage_id"]
        image_binary = self.dataset[idx]["image_binary"]
        text = self.dataset[idx]["vlm_text"]
        try:
            image = Image.open(io.BytesIO(image_binary)).convert("RGB")
            image = image.resize((596, 842))
        except Exception as e:
            print(f"Error converting image at index {idx}: {e}")
            return passage_id, "ERROR", "ERROR"
        return passage_id, image, text

def collate_fn(batch):
    """
    Custom collate function to ensure DataLoader returns:
    - A list of passage_ids (strings)
    - A list of PIL images
    - A list of OCR text (either from OCR or VLM-based models)
    """
    passage_ids, images, texts = zip(*batch)
    return list(passage_ids), list(images), list(texts)

def process_document_images(ds, doc_name):
    """
    Process images only from the document that matches `doc_name`.
    """
    image_dataset = MultiModalDataset(ds["train"], doc_name)
    image_dataloader = DataLoader(
        image_dataset, 
        batch_size=4, 
        shuffle=False,
        collate_fn=collate_fn
    )

    embedding_store = []
    passage_store = []

    for passage_ids, images, texts in tqdm(image_dataloader, desc="Processing document images"):
        processed_images = processor.process_images(images).to(model.device)
        processed_texts = processor.process_queries(texts).to(model.device)
        with torch.no_grad():
            image_embeddings = model(**processed_images)
            text_embeddings = model(**processed_texts)
            batch_embeddings = torch.concat((image_embeddings, text_embeddings), dim = 1)      # fused embeddings
        embedding_store.extend(list(torch.unbind(batch_embeddings.to("cpu"))))
        passage_store.extend(passage_ids)
    return embedding_store, passage_store

def get_relevant_passage_topk(query_text, embedding_store, passage_store, top_k=5):
    """
    Retrieves the top-k most relevant passages for a given query.
    
    Args:
        query_text (str): Query `question`
        top_k (int): Number of top passages to retrieve.
    
    Returns:
        list: List of passage IDs for the top-k matches.
    """

    batch_queries = processor.process_queries([query_text]).to(model.device)
    with torch.no_grad():
        query_embeddings = model(**batch_queries)
    scores = processor.score_multi_vector(query_embeddings, embedding_store)

    min_k = min(top_k, scores[0].numel())      # To avoid out of range errors if documents has less than top_k pages
    top_k_scores, top_k_indices = scores[0].topk(min_k)   
    top_k_passages = [passage_store[idx] for idx in top_k_indices.tolist()]

    while len(top_k_passages) < top_k:
        top_k_passages.append(top_k_passages[-1])

    return top_k_passages


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--start_idx", type=int, required=True, help="Start index for dataset slicing.")
    parser.add_argument("--end_idx", type=int, required=True, help="End index for dataset slicing.")
    parser.add_argument("--model_checkpoint", type = str, required = True, help = "Hugging Face model checkpoints")
    args = parser.parse_args()


    model_checkpoint = args.model_checkpoint
    vlm_embed_model = EmbedModel(model_name = model_checkpoint)

    model = vlm_embed_model.embed_model
    processor = vlm_embed_model.processor
    model.eval()

    # Load Hugging Face dataset
    ds = load_dataset("MMDocIR/MMDocIR-Challenge", cache_dir="../data_task1")

    # Process JSONL data and update passage_id field with relevant passages
    data_json = []
    with open("../data_task1/MMDocIR_gt_remove.jsonl", 'r', encoding="utf-8") as f:
        for line in f:
            data_json.append(json.loads(line.strip()))

    data_json = data_json[args.start_idx:args.end_idx]

    print(f"Total Questions: {len(data_json)}")

    data_df = pd.DataFrame(data_json)

    doc_list = list(data_df["doc_name"].unique())

    res_passage_ids = []
    res_question_ids = []
    res_questions = []

    for doc_name in doc_list:
        print("*-"*20)
        print(f"Doc Name: {doc_name}")

        embedding_store, passage_store = process_document_images(ds, doc_name)

        if not embedding_store:
            print("No matching document found.")
            exit()

        filtered_df = data_df[data_df["doc_name"] == doc_name]

        for row_idx, row_data in filtered_df.iterrows():

            top_k_matches = get_relevant_passage_topk(row_data["question"], embedding_store, passage_store)

            res_passage_ids.append(top_k_matches)

            res_question_ids.append(row_data['question_id'])
            res_questions.append(row_data['question'])


    # Save the results

    res_df = pd.DataFrame({"question_id": res_question_ids, "question" : res_questions, "passage_id": res_passage_ids })

    result_dir = f"../results_task1/{model_checkpoint}/"

    os.makedirs(result_dir, exist_ok = True)

    res_df.to_csv(f"{result_dir}/task1_{args.start_idx}.csv", index = False)
    print("Saved dataset with passage_ids")