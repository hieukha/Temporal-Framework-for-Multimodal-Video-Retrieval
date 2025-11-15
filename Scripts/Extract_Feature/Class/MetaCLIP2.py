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

class MetaCLIP2:
    def __init__(self, device=None):
        print("facebook/metaclip-2-worldwide-huge-378")
        # Follow SigLIP style: auto-select device (prefer CUDA), no hard GPU requirement
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        # Load model and processor
        self.model = AutoModel.from_pretrained(
            "facebook/metaclip-2-worldwide-huge-378",
            cache_dir="/workspace/competitions/AIC_2025/SIU_Sayan/Base/Cache"
        ).eval().to(self.device)
        self.processor = AutoProcessor.from_pretrained(
            "facebook/metaclip-2-worldwide-huge-378",
            cache_dir="/workspace/competitions/AIC_2025/SIU_Sayan/Base/Cache"
        )

    def get_image_features(self, image_path: str) -> np.array:
        if (image_path.startswith("http")):
            image = Image.open(requests.get(image_path, stream=True).raw)
        else:
            image = Image.open(image_path)
        
        # Chỉ xử lý image, không cần text input
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)

        with torch.no_grad(), torch.amp.autocast('cuda'):
            # Chỉ truyền image inputs, không truyền text
            image_features = self.model.get_image_features(**inputs)
            image_features /= image_features.norm(dim=-1, keepdim=True)
        
        # Ensure proper numpy conversion
        result = image_features.cpu().detach().numpy().astype(np.float32)
        return result

    def get_text_features(self, text: str) -> np.array:
        inputs = self.processor(text=text, return_tensors="pt").to(self.device)

        with torch.no_grad(), torch.amp.autocast('cuda'):
            # Chỉ truyền text inputs, không truyền image
            text_features = self.model.get_text_features(**inputs)
            text_features /= text_features.norm(dim=-1, keepdim=True)
        
        # Ensure proper numpy conversion
        result = text_features.cpu().detach().numpy().astype(np.float32)
        return result
