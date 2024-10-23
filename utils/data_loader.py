import os
import pandas as pd
from PIL import Image

def load_metadata(metadata_path):
    """ Load metadata from the specified CSV file """
    return pd.read_csv(metadata_path)

def load_image(images_path, img_id):
    """ Load image by its ID """
    img_path = os.path.join(images_path, img_id)
    if os.path.exists(img_path):
        return Image.open(img_path).convert('RGB')
    else:
        raise FileNotFoundError(f"Image file not found: {img_path}")
