from PIL import Image, PngImagePlugin
import numpy as np

PngImagePlugin.MAX_TEXT_CHUNK = 50 * 1024 * 1024

def resize_image(img_path, size=(256, 256)):
    """Load and resize image. If load fails, return blank image."""
    try:
        img = Image.open(img_path).resize(size)
    except (IOError, FileNotFoundError, ValueError) as e:
        print(f"Failed to load {img_path}: {e}")
        img = Image.fromarray(np.zeros((size[1], size[0], 3), dtype=np.uint8))
    return img

def get_top5_docs(distances, indices, image_folder_map, pad_value=None):
    """Return top-5 unique document (folder) names for each query."""
    top5_docs = []
    for row in indices:
        docs = []
        seen = set()
        for i in row:
            doc = image_folder_map[i]
            if doc not in seen:
                docs.append(doc)
                seen.add(doc)
        docs += [pad_value] * (5 - len(docs))
        top5_docs.append(docs[:5])
    return top5_docs