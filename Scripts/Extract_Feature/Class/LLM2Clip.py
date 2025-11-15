import os
import torch
from PIL import Image
import requests
import numpy as np
from transformers import AutoTokenizer, AutoModel, AutoConfig, CLIPImageProcessor
from llm2vec import LLM2Vec

os.environ['HF_HOME'] = '/workspace/competitions/AIC_2025/SIU_Sayan/Base/Cache'

class LLM2Clip:
    def __init__(self, device=None):
        print("LLM2Clip - microsoft/LLM2CLIP-Llama-3-8B-Instruct-CC-Finetuned")
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        self.torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        
        # LLM2Clip model configuration
        llm_model_name = "microsoft/LLM2CLIP-Llama-3-8B-Instruct-CC-Finetuned"
        clip_model_name = "microsoft/LLM2CLIP-Openai-L-14-336"
        
        # Load LLM model
        self.config = AutoConfig.from_pretrained(
            llm_model_name, 
            cache_dir="/workspace/competitions/AIC_2025/SIU_Sayan/Base/Cache",
            trust_remote_code=True
        )
        self.llm_model = AutoModel.from_pretrained(
            llm_model_name, 
            cache_dir="/workspace/competitions/AIC_2025/SIU_Sayan/Base/Cache",
            torch_dtype=torch.bfloat16,
            config=self.config, 
            trust_remote_code=True
        ).to(self.device).eval()
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            llm_model_name,
            cache_dir="/workspace/competitions/AIC_2025/SIU_Sayan/Base/Cache"
        )
        
        # CLIP processor for images
        self.processor = CLIPImageProcessor.from_pretrained(
            "openai/clip-vit-large-patch14-336",
            cache_dir="/workspace/competitions/AIC_2025/SIU_Sayan/Base/Cache"
        )
        
        # Set config for LLM2Vec
        self.llm_model.config._name_or_path = "meta-llama/Meta-Llama-3-8B-Instruct"
        
        # Initialize LLM2Vec
        self.l2v = LLM2Vec(
            self.llm_model, 
            self.tokenizer, 
            pooling_mode="mean", 
            max_length=512
        )
        
        # Load CLIP head model
        self.clip_model = AutoModel.from_pretrained(
            clip_model_name, 
            cache_dir="/workspace/competitions/AIC_2025/SIU_Sayan/Base/Cache",
            torch_dtype=torch.bfloat16, 
            trust_remote_code=True
        )
        self.clip_model = self.clip_model.to(self.device).eval()

    def get_image_features(self, image_path: str) -> np.array:
        """Extract image features using LLM2Clip"""
        if image_path.startswith("http"):
            image = Image.open(requests.get(image_path, stream=True).raw).convert("RGB")
        else:
            image = Image.open(image_path).convert("RGB")
        
        # Process image with CLIP processor
        inputs = self.processor(images=image, return_tensors="pt").pixel_values.to(self.device)
        
        with torch.no_grad():
            if self.device == "cuda":
                with torch.amp.autocast('cuda'):
                    # Get image features from CLIP head
                    features = self.clip_model.get_image_features(inputs)
                    features = features / features.norm(dim=-1, keepdim=True)
            else:
                features = self.clip_model.get_image_features(inputs)
                features = features / features.norm(dim=-1, keepdim=True)
        
        # Convert to numpy and ensure consistent shape
        result = features.cpu().detach().numpy().astype(np.float32)
        return result

    def get_text_features(self, text: str) -> np.array:
        """Extract text features using LLM2Clip"""
        # Encode text with LLM2Vec
        text_emb = self.l2v.encode([text], convert_to_tensor=True).to(self.device)
        
        with torch.no_grad():
            if self.device == "cuda":
                with torch.amp.autocast('cuda'):
                    # Get text features from CLIP head
                    text_features = self.clip_model.get_text_features(text_emb)
                    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            else:
                text_features = self.clip_model.get_text_features(text_emb)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        # Convert to numpy
        result = text_features.cpu().detach().numpy().astype(np.float32)
        return result
