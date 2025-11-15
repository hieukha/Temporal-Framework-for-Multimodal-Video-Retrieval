import torch
from urllib.request import urlopen
from PIL import Image
import requests
import numpy as np
import os
from open_clip import create_model_from_pretrained, get_tokenizer 
os.environ['HF_HOME'] = '/workspace/competitions/AIC_2025/SIU_Sayan/Base/Cache'

# Cho phép tải ảnh bị truncated
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


class DFN5B:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.preprocess = create_model_from_pretrained('hf-hub:apple/DFN5B-CLIP-ViT-H-14',cache_dir='/workspace/competitions/AIC_2025/SIU_Sayan/Base/Cache')
        self.tokenizer = get_tokenizer('ViT-H-14')
        self.model = self.model.eval().to(self.device) 

    def get_image_features(self, image_path : str) -> np.array:
        if (image_path.startswith("http")):
            image = Image.open(requests.get(image_path, stream=True).raw)
        else:
            image = Image.open(image_path)
        inputs = self.preprocess(image).unsqueeze(0).to(self.device)
        with torch.no_grad(), torch.amp.autocast('cuda'):
            image_features = self.model.encode_image(inputs)
            image_features /= image_features.norm(dim=-1, keepdim=True)
        
        # Ensure proper numpy conversion
        result = image_features.cpu().detach().numpy().astype(np.float32)
        return result
        
    def get_text_features(self, text: str) -> np.array:       
        inputs = self.tokenizer(text).to(self.device)
        with torch.no_grad(), torch.amp.autocast('cuda'):
            text_features = self.model.encode_text(inputs)
            text_features /= text_features.norm(dim=-1, keepdim=True)
        
        # Ensure proper numpy conversion
        result = text_features.cpu().detach().numpy().astype(np.float32)
        return result
