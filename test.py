import torch
from transformers import CLIPProcessor, CLIPModel
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image
import pandas as pd
import os

device = "cuda" if torch.cuda.is_available() else "cpu"

# Lấy đường dẫn đến thư mục chứa file .py hiện tại
current_dir = os.path.dirname(os.path.abspath(__file__))

model = CLIPModel.from_pretrained("laion/CLIP-ViT-g-14-laion2B-s12B-b42K", cache_dir=current_dir)
processor = CLIPProcessor.from_pretrained("laion/CLIP-ViT-g-14-laion2B-s12B-b42K", cache_dir=current_dir) 

print(current_dir)