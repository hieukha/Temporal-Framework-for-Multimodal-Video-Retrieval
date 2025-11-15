import torch
from PIL import Image
import requests
import numpy as np
import os
from transformers import AutoProcessor, AutoModel
os.environ['HF_HOME'] = '/workspace/competitions/AIC_2025/SIU_Sayan/Base/Cache'
# Cho phép tải ảnh bị truncated
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

class SigLIP:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = AutoModel.from_pretrained("google/siglip-so400m-patch14-384", cache_dir="/workspace/competitions/AIC_2025/SIU_Sayan/Base/Cache").eval().to(self.device)
        self.processor = AutoProcessor.from_pretrained("google/siglip-so400m-patch14-384")
    def get_image_features(self, image_path : str) -> np.array:
        if (image_path.startswith("http")):
            image = Image.open(requests.get(image_path, stream=True).raw)
        else:
            image = Image.open(image_path)
        inputs = self.processor(images=image, padding="max_length", return_tensors="pt").to(self.device)
        with torch.no_grad(), torch.amp.autocast('cuda'):
            image_features = self.model.get_image_features(**inputs)
            image_features /= image_features.norm(dim=-1, keepdim=True)
        
        # Ensure proper numpy conversion
        result = image_features.cpu().detach().numpy().astype(np.float32)
        return result
    def get_text_features(self, text: str) -> np.array:       
        inputs = self.processor(text=text, padding="max_length", return_tensors="pt").to(self.device)
        with torch.no_grad(), torch.amp.autocast('cuda'):
            text_features = self.model.get_text_features(**inputs)
            text_features /= text_features.norm(dim=-1, keepdim=True)
        
        # Ensure proper numpy conversion
        result = text_features.cpu().detach().numpy().astype(np.float32)
        return result