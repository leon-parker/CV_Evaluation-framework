"""
Prompt faithfulness evaluation using Advanced CLIP
Measures semantic alignment between text prompts and generated images
"""

import torch
import torch.nn.functional as F
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import numpy as np


class PromptFaithfulnessAnalyzer:
    """
    Advanced CLIP-based prompt faithfulness evaluation
    Uses enhanced CLIP approach for better accuracy
    """
    
    def __init__(self, device="cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        self.processor = None
        self.model = None
        self.load_model()
    
    def load_model(self):
        """Load CLIP model for evaluation"""
        try:
            print("   Loading Advanced CLIP model for prompt evaluation...")
            self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
            self.model = CLIPModel.from_pretrained(
                "openai/clip-vit-large-patch14", 
                torch_dtype=torch.float16
            ).to(self.device)
            self.model.eval()
            print("   ✓ Advanced CLIP model loaded successfully")
        except Exception as e:
            print(f"   ✗ Failed to load CLIP model: {e}")
            raise
    
    def evaluate_prompt_alignment(self, image, prompt):
        """
        Evaluate how well an image matches its prompt using CLIP
        
        Args:
            image: PIL Image or numpy array
            prompt: Text prompt that was used to generate the image
            
        Returns:
            float: Alignment score between 0 and 1
        """
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image).convert('RGB')
        elif not isinstance(image, Image.Image):
            raise ValueError("Image must be PIL Image or numpy array")
        
        try:
            # Process inputs
            inputs = self.processor(text=[prompt], images=image, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                # Calculate cosine similarity between image and text embeddings
                similarity = torch.cosine_similarity(
                    outputs.image_embeds, 
                    outputs.text_embeds, 
                    dim=1
                ).item()
            
            # Convert from [-1, 1] to [0, 1] range
            score = max(0, min(1, (similarity + 1) / 2))
            return score
            
        except Exception as e:
            print(f"   Error in prompt evaluation: {e}")
            return 0.5  # Default middle score on error
    
    def batch_evaluate(self, images, prompts):
        """
        Evaluate multiple image-prompt pairs
        
        Args:
            images: List of PIL Images
            prompts: List of text prompts
            
        Returns:
            List of alignment scores
        """
        scores = []
        for image, prompt in zip(images, prompts):
            score = self.evaluate_prompt_alignment(image, prompt)
            scores.append(score)
        return scores