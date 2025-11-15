import torch
from PIL import Image
import open_clip
import requests
import numpy as np
import os
os.environ['HF_HOME'] = '/workspace/competitions/AIC_2025/SIU_Sayan/Base/Cache'

# Cho phép tải ảnh bị truncated
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

class LAION:
    def __init__(self, device=None):
        print("CLIP-ViT-g-14-laion2B-s34B-b88K")
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        self.model, _, self.preprocess = open_clip.create_model_and_transforms('ViT-g-14', pretrained='laion2b_s34b_b88K')
        self.model.eval()  # model in train mode by default, impacts some models with BatchNorm or stochastic depth active
        self.tokenizer = open_clip.get_tokenizer('ViT-g-14')
        self.model.to(self.device)
    def get_image_features(self, image_path : str) -> np.array:
        if (image_path.startswith("http")):
            image = Image.open(requests.get(image_path, stream=True).raw)
        else:
            image = Image.open(image_path)
        inputs = self.preprocess(image).unsqueeze(0).to(self.device)
        if self.device == "cpu":
            with torch.no_grad():
                image_features = self.model.encode_image(inputs)
                image_features /= image_features.norm(dim=-1, keepdim=True)
        else:
            with torch.no_grad(), torch.amp.autocast('cuda'):
                image_features = self.model.encode_image(inputs)
                image_features /= image_features.norm(dim=-1, keepdim=True)
        
        # Ensure proper numpy conversion
        result = image_features.cpu().detach().numpy().astype(np.float32)
        return result

    def get_text_features(self, text: str) -> np.array:       
        inputs = self.tokenizer(text).to(self.device)
        if self.device == "cpu":
            with torch.no_grad():
                text_features = self.model.encode_text(inputs)
                text_features /= text_features.norm(dim=-1, keepdim=True)
        else:
            with torch.no_grad(), torch.amp.autocast('cuda'):
                text_features = self.model.encode_text(inputs)
                text_features /= text_features.norm(dim=-1, keepdim=True)
        
        # Ensure proper numpy conversion
        result = text_features.cpu().detach().numpy().astype(np.float32)
        return result