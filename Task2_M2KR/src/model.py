import os
import sys
from colpali_engine.models import ColQwen2, ColQwen2Processor

from transformers.utils.import_utils import is_flash_attn_2_available
from PIL import Image

from PIL import PngImagePlugin
PngImagePlugin.MAX_TEXT_CHUNK = 50 * 1024 * 1024

import torch

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
            cache_dir = "../cache/"
        )
        
        self.embed_model.eval()

        self.processor = ColQwen2Processor.from_pretrained(self.model_name)

    def generate_image_embedding(self, images):
        """Generate embeddings for a batch of images."""
        with torch.no_grad():
            batch_images = self.processor.process_images(images).to(self.embed_model.device)
            image_embeddings = self.embed_model(**batch_images).cpu().float()
        return image_embeddings.numpy().tolist()

    def embed_images_in_batch(self, image_paths, batch_size=4, resize_to = (224, 224)):
        """Embed all images and return embeddings."""
        all_embeddings = []

        for batch_paths in [image_paths[i:i + batch_size] for i in range(0, len(image_paths), batch_size)]:

            images = []
            for img_path in batch_paths:
                tmp_img = Image.open(img_path).resize(resize_to)
                    
                images.append(tmp_img)
            batch_embeddings = self.generate_image_embedding(images)
            all_embeddings.extend(batch_embeddings)

    def get_mean_pooled_image_embeddings_in_batch(self, image_paths, batch_size=4, resize_to = (224, 224)):
        """Embed all images and return embeddings."""
        all_embeddings = []

        for batch_paths in [image_paths[i:i + batch_size] for i in range(0, len(image_paths), batch_size)]:

            images = []
            for img_path in batch_paths:
                tmp_img = Image.open(img_path).resize(resize_to)
                    
                images.append(tmp_img)
            with torch.no_grad():
                batch_images = self.processor.process_images(images).to(self.embed_model.device)
                batch_embeddings = self.embed_model(**batch_images)

            mean_pooled_embedding = torch.mean(batch_embeddings[:, 4:-7, :], axis = 1)
            all_embeddings.extend(mean_pooled_embedding)

        return torch.stack(all_embeddings)
    
    def get_image_embeddings(self, image_paths):
        """Compute image embeddings for a list of image paths using the custom model."""
        embeddings = self.get_mean_pooled_image_embeddings_in_batch(image_paths, batch_size=32)
        embeddings = embeddings.cpu().detach().float().numpy()
        return embeddings
