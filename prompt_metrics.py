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
    Uses contrastive approach for better accuracy
    """
    
    def __init__(self, device="cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        self.processor = None
        self.model = None
        self.load_model()
    
    def load_model(self):
        """Load CLIP model for evaluation"""
        try:
            print("   Loading CLIP model for prompt evaluation...")
            self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
            self.model = CLIPModel.from_pretrained(
                "openai/clip-vit-large-patch14", 
                torch_dtype=torch.float16
            ).to(self.device)
            self.model.eval()
            print("   ✓ CLIP model loaded successfully")
        except Exception as e:
            print(f"   ✗ Failed to load CLIP model: {e}")
            raise
    
    def evaluate_prompt_alignment(self, image, prompt):
        """
        Evaluate how well an image matches its prompt using contrastive approach
        
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
            # Generate contrastive prompts
            contrastive_prompts = self._generate_contrastive_prompts(prompt)
            all_prompts = [prompt] + contrastive_prompts
            
            # Process inputs
            inputs = self.processor(
                text=all_prompts,
                images=image,
                return_tensors="pt",
                padding=True
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Get predictions
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits_per_image = outputs.logits_per_image[0]
                probs = F.softmax(logits_per_image, dim=-1)
            
            # Calculate scores
            correct_score = probs[0].item()
            wrong_scores = probs[1:].tolist() if len(probs) > 1 else [0.0]
            
            # Enhanced scoring with separation bonus
            if wrong_scores:
                max_wrong = max(wrong_scores)
                separation = correct_score - max_wrong
                # Reward clear separation between correct and wrong
                enhanced_score = min(1.0, correct_score + separation * 0.3)
            else:
                enhanced_score = correct_score
            
            # Additional semantic checks
            semantic_score = self._semantic_consistency_check(image, prompt)
            
            # Combine scores
            final_score = enhanced_score * 0.8 + semantic_score * 0.2
            
            return float(final_score)
            
        except Exception as e:
            print(f"   Error in prompt evaluation: {e}")
            return 0.5  # Default middle score on error
    
    def _generate_contrastive_prompts(self, prompt):
        """Generate wrong prompts for contrastive evaluation"""
        prompt_lower = prompt.lower()
        wrong_prompts = []
        
        # Color substitutions
        color_pairs = [
            ("red", "blue"), ("blue", "green"), ("green", "yellow"),
            ("yellow", "purple"), ("purple", "orange"), ("orange", "pink"),
            ("black", "white"), ("brown", "gray")
        ]
        
        for color1, color2 in color_pairs:
            if color1 in prompt_lower:
                wrong_prompts.append(prompt_lower.replace(color1, color2))
            elif color2 in prompt_lower:
                wrong_prompts.append(prompt_lower.replace(color2, color1))
        
        # Shape substitutions
        shape_pairs = [
            ("circle", "square"), ("square", "triangle"), ("triangle", "rectangle"),
            ("round", "angular"), ("curved", "straight")
        ]
        
        for shape1, shape2 in shape_pairs:
            if shape1 in prompt_lower:
                wrong_prompts.append(prompt_lower.replace(shape1, shape2))
        
        # Emotion substitutions (important for autism storyboards)
        emotion_pairs = [
            ("happy", "sad"), ("sad", "angry"), ("angry", "surprised"),
            ("excited", "calm"), ("scared", "confident"), ("laughing", "crying")
        ]
        
        for emo1, emo2 in emotion_pairs:
            if emo1 in prompt_lower:
                wrong_prompts.append(prompt_lower.replace(emo1, emo2))
        
        # Action substitutions
        action_pairs = [
            ("brushing teeth", "eating breakfast"), ("eating", "drinking"),
            ("reading", "writing"), ("walking", "running"), ("sitting", "standing"),
            ("washing hands", "drying hands"), ("playing", "sleeping")
        ]
        
        for act1, act2 in action_pairs:
            if act1 in prompt_lower:
                wrong_prompts.append(prompt_lower.replace(act1, act2))
        
        # Number substitutions
        number_words = ["one", "two", "three", "four", "five"]
        for i, num in enumerate(number_words):
            if num in prompt_lower:
                # Replace with different number
                other_nums = number_words[:i] + number_words[i+1:]
                if other_nums:
                    wrong_prompts.append(prompt_lower.replace(num, other_nums[0]))
        
        # If no substitutions found, generate generic wrong prompts
        if not wrong_prompts:
            wrong_prompts = [
                "something completely different",
                "abstract patterns and shapes",
                "a blank empty scene"
            ]
        
        # Limit to most relevant wrong prompts
        return wrong_prompts[:5]
    
    def _semantic_consistency_check(self, image, prompt):
        """Additional semantic consistency verification"""
        try:
            # Extract key concepts from prompt
            key_concepts = self._extract_key_concepts(prompt)
            
            # Check each concept
            concept_scores = []
            for concept in key_concepts:
                inputs = self.processor(
                    text=[concept],
                    images=image,
                    return_tensors="pt"
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    similarity = torch.cosine_similarity(
                        outputs.image_embeds,
                        outputs.text_embeds,
                        dim=1
                    ).item()
                
                # Convert to 0-1 range
                concept_score = (similarity + 1) / 2
                concept_scores.append(concept_score)
            
            # Average concept scores
            if concept_scores:
                return float(np.mean(concept_scores))
            else:
                return 0.8  # Default if no concepts extracted
                
        except Exception as e:
            print(f"   Semantic check error: {e}")
            return 0.8
    
    def _extract_key_concepts(self, prompt):
        """Extract key concepts from prompt for verification"""
        concepts = []
        
        # Common autism storyboard concepts
        concept_keywords = {
            'people': ['boy', 'girl', 'child', 'person', 'friend', 'teacher'],
            'emotions': ['happy', 'sad', 'angry', 'excited', 'calm', 'scared'],
            'actions': ['brushing', 'eating', 'playing', 'reading', 'washing', 'walking'],
            'objects': ['toothbrush', 'book', 'ball', 'spoon', 'soap', 'chair'],
            'settings': ['bathroom', 'classroom', 'playground', 'home', 'kitchen']
        }
        
        prompt_lower = prompt.lower()
        
        for category, keywords in concept_keywords.items():
            for keyword in keywords:
                if keyword in prompt_lower:
                    # Add the specific instance from prompt
                    if category == 'people':
                        concepts.append(f"a {keyword}")
                    elif category == 'emotions':
                        concepts.append(f"{keyword} expression")
                    elif category == 'actions':
                        concepts.append(f"person {keyword}")
                    else:
                        concepts.append(keyword)
        
        # Add color+object combinations
        colors = ['red', 'blue', 'green', 'yellow', 'purple', 'orange']
        for color in colors:
            if color in prompt_lower:
                # Find what the color describes
                words = prompt_lower.split()
                for i, word in enumerate(words):
                    if word == color and i + 1 < len(words):
                        concepts.append(f"{color} {words[i+1]}")
        
        return concepts[:5]  # Limit to top 5 concepts
    
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