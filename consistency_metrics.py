"""
Character and style consistency evaluation
Measures identity preservation across image sequences
"""

import torch
import numpy as np
import cv2
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from sklearn.metrics.pairwise import cosine_similarity


class ConsistencyAnalyzer:
    """
    Evaluates character and style consistency across image sequences
    Critical for autism storyboards where consistency reduces anxiety
    """
    
    def __init__(self, device="cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        self.processor = None
        self.model = None
        self.face_cascade = None
        self.load_models()
        
        # Storage for sequence analysis
        self.reset_sequence()
    
    def load_models(self):
        """Load CLIP model and face detection"""
        try:
            print("   Loading models for consistency analysis...")
            
            # CLIP for embedding extraction
            self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
            self.model = CLIPModel.from_pretrained(
                "openai/clip-vit-large-patch14",
                torch_dtype=torch.float16
            ).to(self.device)
            
            # Face detection for character focus
            try:
                self.face_cascade = cv2.CascadeClassifier(
                    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
                )
                self.face_detection_available = True
            except:
                self.face_detection_available = False
                print("   ⚠ Face detection not available")
            
            print("   ✓ Consistency models loaded")
            
        except Exception as e:
            print(f"   ✗ Failed to load models: {e}")
            raise
    
    def reset_sequence(self):
        """Reset for new sequence evaluation"""
        self.sequence_images = []
        self.character_embeddings = []
        self.style_embeddings = []
        self.face_regions = []
    
    def add_image_to_sequence(self, image):
        """Add image to sequence for consistency evaluation"""
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image).convert('RGB')
        
        # Store original image
        self.sequence_images.append(image)
        
        # Extract character region
        character_region = self._extract_character_region(image)
        
        # Get embeddings
        char_embedding = self._get_clip_embedding(character_region)
        style_embedding = self._get_clip_embedding(image)
        
        self.character_embeddings.append(char_embedding)
        self.style_embeddings.append(style_embedding)
        
        return len(self.sequence_images)
    
    def evaluate_consistency(self, image1=None, image2=None):
        """
        Evaluate consistency between two images or across sequence
        
        Args:
            image1, image2: Optional specific images to compare
            If not provided, evaluates entire sequence
            
        Returns:
            Dictionary with consistency scores
        """
        if image1 is not None and image2 is not None:
            # Direct comparison mode
            return self._evaluate_pair_consistency(image1, image2)
        else:
            # Sequence mode
            return self._evaluate_sequence_consistency()
    
    def _evaluate_pair_consistency(self, image1, image2):
        """Evaluate consistency between two specific images"""
        # Reset and add images
        self.reset_sequence()
        self.add_image_to_sequence(image1)
        self.add_image_to_sequence(image2)
        
        # Calculate similarities
        char_sim = cosine_similarity(
            self.character_embeddings[0],
            self.character_embeddings[1]
        )[0][0]
        
        style_sim = cosine_similarity(
            self.style_embeddings[0],
            self.style_embeddings[1]
        )[0][0]
        
        # Combined score (character weighted more heavily)
        combined = float(char_sim * 0.7 + style_sim * 0.3)
        
        return {
            'character_consistency': float(char_sim),
            'style_consistency': float(style_sim),
            'combined_consistency': combined,
            'is_consistent': combined > 0.75
        }
    
    def _evaluate_sequence_consistency(self):
        """Evaluate consistency across entire sequence"""
        if len(self.sequence_images) < 2:
            return {'error': 'Need at least 2 images for sequence evaluation'}
        
        # Calculate all pairwise similarities
        char_similarities = []
        style_similarities = []
        
        for i in range(len(self.character_embeddings)):
            for j in range(i + 1, len(self.character_embeddings)):
                # Character similarity
                char_sim = cosine_similarity(
                    self.character_embeddings[i],
                    self.character_embeddings[j]
                )[0][0]
                char_similarities.append(char_sim)
                
                # Style similarity
                style_sim = cosine_similarity(
                    self.style_embeddings[i],
                    self.style_embeddings[j]
                )[0][0]
                style_similarities.append(style_sim)
        
        # Calculate statistics
        char_mean = float(np.mean(char_similarities))
        char_std = float(np.std(char_similarities))
        style_mean = float(np.mean(style_similarities))
        style_std = float(np.std(style_similarities))
        
        # Combined score
        combined = char_mean * 0.7 + style_mean * 0.3
        
        # Drift detection (consistency degradation over sequence)
        drift_score = self._calculate_drift()
        
        return {
            'character_consistency': char_mean,
            'character_std': char_std,
            'style_consistency': style_mean,
            'style_std': style_std,
            'combined_consistency': float(combined),
            'drift_score': drift_score,
            'num_images': len(self.sequence_images),
            'is_consistent': combined > 0.75 and char_std < 0.1
        }
    
    def _extract_character_region(self, image):
        """Extract character-focused region from image"""
        if not self.face_detection_available:
            # Fallback: use center region
            width, height = image.size
            left = width // 4
            top = height // 4
            right = 3 * width // 4
            bottom = 3 * height // 4
            return image.crop((left, top, right, bottom))
        
        # Convert to OpenCV format
        cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=3,
            minSize=(30, 30)
        )
        
        if len(faces) > 0:
            # Use largest face
            largest_face = max(faces, key=lambda x: x[2] * x[3])
            x, y, w, h = largest_face
            
            # Expand region to include upper body
            expansion = 1.8
            center_x = x + w // 2
            center_y = y + h // 2
            
            new_w = int(w * expansion)
            new_h = int(h * expansion * 1.5)
            
            new_x = max(0, center_x - new_w // 2)
            new_y = max(0, center_y - new_h // 3)
            new_x2 = min(cv_image.shape[1], new_x + new_w)
            new_y2 = min(cv_image.shape[0], new_y + new_h)
            
            character_region = cv_image[new_y:new_y2, new_x:new_x2]
            return Image.fromarray(cv2.cvtColor(character_region, cv2.COLOR_BGR2RGB))
        
        # No face found - use center region
        return self._extract_center_region(image)
    
    def _extract_center_region(self, image):
        """Extract center region as fallback"""
        width, height = image.size
        size = min(width, height) // 2
        left = (width - size) // 2
        top = (height - size) // 2
        return image.crop((left, top, left + size, top + size))
    
    def _get_clip_embedding(self, image):
        """Get CLIP embedding for image region"""
        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            embedding = self.model.get_image_features(**inputs)
        
        return embedding.cpu().numpy()
    
    def _calculate_drift(self):
        """Calculate consistency drift across sequence"""
        if len(self.character_embeddings) < 3:
            return 0.0
        
        # Compare each image to the first (reference)
        reference_embedding = self.character_embeddings[0]
        
        similarities = []
        for i in range(1, len(self.character_embeddings)):
            sim = cosine_similarity(
                reference_embedding,
                self.character_embeddings[i]
            )[0][0]
            similarities.append(sim)
        
        # Calculate drift as decrease in similarity over time
        if len(similarities) > 1:
            # Fit linear regression to detect trend
            x = np.arange(len(similarities))
            slope = np.polyfit(x, similarities, 1)[0]
            # Negative slope indicates drift
            drift = -slope if slope < 0 else 0.0
            return float(min(1.0, drift * 10))  # Scale to 0-1
        
        return 0.0
    
    def get_consistency_report(self):
        """Generate detailed consistency report"""
        if len(self.sequence_images) < 2:
            return "No sequence loaded for analysis"
        
        results = self._evaluate_sequence_consistency()
        
        report = [
            "CHARACTER CONSISTENCY ANALYSIS",
            "=" * 40,
            f"Images in sequence: {results['num_images']}",
            f"Character consistency: {results['character_consistency']:.3f} (±{results['character_std']:.3f})",
            f"Style consistency: {results['style_consistency']:.3f} (±{results['style_std']:.3f})",
            f"Combined score: {results['combined_consistency']:.3f}",
            f"Drift score: {results['drift_score']:.3f}",
            "",
            "Assessment: " + (
                "Excellent - Very consistent" if results['combined_consistency'] > 0.85
                else "Good - Acceptably consistent" if results['combined_consistency'] > 0.75
                else "Fair - Some inconsistency" if results['combined_consistency'] > 0.65
                else "Poor - Significant inconsistency"
            )
        ]
        
        return "\n".join(report)