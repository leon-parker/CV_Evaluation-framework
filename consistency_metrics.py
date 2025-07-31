"""
Character and style consistency evaluation
Enhanced with smart character region extraction and comprehensive sequence analysis
Measures identity preservation across image sequences
"""

import torch
import numpy as np
import cv2
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import json
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


class ConsistencyAnalyzer:
    """
    Enhanced evaluator for character and style consistency across image sequences
    Critical for autism storyboards where consistency reduces anxiety
    
    Features:
    - Character region extraction with face detection
    - Separate character and style embeddings
    - Sequence-wide consistency analysis
    - Drift detection over time
    - Dataset comparison capabilities
    """
    
    def __init__(self, device="cuda" if torch.cuda.is_available() else "cpu", 
                 clip_model_name="openai/clip-vit-large-patch14"):
        self.device = device
        self.clip_model_name = clip_model_name
        self.processor = None
        self.model = None
        self.face_cascade = None
        self.load_models()
        
        # Storage for sequence analysis
        self.reset_sequence()
    
    def load_models(self):
        """Load CLIP model and face detection with enhanced error handling"""
        try:
            print("   🎭 Loading Character & Style Consistency Evaluator...")
            
            # CLIP for embedding extraction
            self.processor = CLIPProcessor.from_pretrained(self.clip_model_name)
            self.model = CLIPModel.from_pretrained(
                self.clip_model_name,
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
                print("   ⚠️ Face detection not available - using center regions for character consistency")
            
            print("   ✅ Character & Style Consistency Evaluator ready")
            
        except Exception as e:
            print(f"   ✗ Failed to load consistency models: {e}")
            raise
    
    def reset_sequence(self):
        """Reset for new sequence evaluation"""
        self.image_sequence = []                # Original images
        self.full_image_embeddings = []         # For style consistency  
        self.character_embeddings = []          # For character consistency
        self.character_reference = None         # First character embedding
        self.style_reference = None             # First style embedding
    
    def extract_character_region(self, image):
        """
        Enhanced character region extraction with face detection
        
        Args:
            image: PIL Image
            
        Returns:
            PIL Image: Character-focused region
        """
        if not self.face_detection_available:
            # Fallback to center region
            return self._extract_center_region(image)
        
        # Convert to OpenCV format
        cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        
        # Detect faces with optimized parameters
        faces = self.face_cascade.detectMultiScale(
            gray, 
            scaleFactor=1.1, 
            minNeighbors=4,
            minSize=(30, 30),
            maxSize=(300, 300)
        )
        
        if len(faces) > 0:
            # Use largest face and expand region to include character
            largest_face = max(faces, key=lambda x: x[2] * x[3])
            x, y, w, h = largest_face
            
            # Expand region to capture more of character (body, clothing)
            expansion_factor = 1.8  # Expand to ~2x face size
            center_x = x + w // 2
            center_y = y + h // 2
            
            new_w = int(w * expansion_factor)
            new_h = int(h * expansion_factor * 1.5)  # Taller for body
            
            new_x = max(0, center_x - new_w // 2)
            new_y = max(0, center_y - new_h // 3)  # Offset up to include body
            new_x2 = min(cv_image.shape[1], new_x + new_w)
            new_y2 = min(cv_image.shape[0], new_y + new_h)
            
            # Extract character region
            character_region = cv_image[new_y:new_y2, new_x:new_x2]
            return Image.fromarray(cv2.cvtColor(character_region, cv2.COLOR_BGR2RGB))
        
        # Fallback: use center region of image (likely contains character)
        return self._extract_center_region(image)
    
    def _extract_center_region(self, image):
        """Extract center region as fallback for character focus"""
        width, height = image.size
        
        # Extract center 3/4 of the image
        left = width // 8
        top = height // 8
        right = 7 * width // 8
        bottom = 7 * height // 8
        
        return image.crop((left, top, right, bottom))
    
    def get_clip_embedding(self, image):
        """
        Get CLIP embedding for an image
        
        Args:
            image: PIL Image
            
        Returns:
            numpy array: CLIP embedding
        """
        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            embedding = self.model.get_image_features(**inputs)
            return embedding.cpu().numpy()
    
    def add_image_to_sequence(self, image):
        """
        Add image to sequence and compute both character and style embeddings
        
        Args:
            image: str (path), PIL Image, or numpy array
            
        Returns:
            dict: Embedding information and frame number
        """
        # Convert to PIL if needed
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image).convert("RGB")
        elif not isinstance(image, Image.Image):
            raise ValueError("Image must be PIL Image, numpy array, or file path")
        
        # Store original image
        self.image_sequence.append(image)
        
        # Get style embedding (full image)
        style_embedding = self.get_clip_embedding(image)
        self.full_image_embeddings.append(style_embedding)
        
        # Get character embedding (character-focused region)
        character_region = self.extract_character_region(image)
        character_embedding = self.get_clip_embedding(character_region)
        self.character_embeddings.append(character_embedding)
        
        # Set reference embeddings if this is the first image
        if len(self.image_sequence) == 1:
            self.character_reference = character_embedding
            self.style_reference = style_embedding
        
        frame_num = len(self.image_sequence)
        
        return {
            "style_embedding": style_embedding,
            "character_embedding": character_embedding,
            "frame_number": frame_num
        }
    
    def evaluate_consistency(self, image1=None, image2=None):
        """
        Evaluate consistency between two images or across entire sequence
        
        Args:
            image1, image2: Optional specific images to compare
            If not provided, evaluates entire sequence
            
        Returns:
            Dictionary with consistency scores and analysis
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
            self.full_image_embeddings[0],
            self.full_image_embeddings[1]
        )[0][0]
        
        # Combined score (character weighted more heavily for autism storyboards)
        combined = float(char_sim * 0.7 + style_sim * 0.3)
        
        return {
            'character_consistency': float(char_sim),
            'style_consistency': float(style_sim),
            'combined_consistency': combined,
            'is_consistent': combined > 0.75,
            'frame_count': 2
        }
    
    def _evaluate_sequence_consistency(self):
        """Enhanced sequence consistency evaluation with comprehensive analysis"""
        if len(self.image_sequence) < 2:
            return {'error': 'Need at least 2 images for sequence evaluation'}
        
        # Character consistency - all pairwise similarities
        character_similarities = []
        for i in range(len(self.character_embeddings)):
            for j in range(i + 1, len(self.character_embeddings)):
                sim = cosine_similarity(
                    self.character_embeddings[i],
                    self.character_embeddings[j]
                )[0][0]
                character_similarities.append(sim)
        
        # Style consistency - all pairwise similarities
        style_similarities = []
        for i in range(len(self.full_image_embeddings)):
            for j in range(i + 1, len(self.full_image_embeddings)):
                sim = cosine_similarity(
                    self.full_image_embeddings[i],
                    self.full_image_embeddings[j]
                )[0][0]
                style_similarities.append(sim)
        
        # Calculate comprehensive statistics
        char_mean = float(np.mean(character_similarities))
        char_std = float(np.std(character_similarities))
        style_mean = float(np.mean(style_similarities))
        style_std = float(np.std(style_similarities))
        
        # Combined score (weight character more for autism applications)
        combined = char_mean * 0.7 + style_mean * 0.3
        
        # Drift detection (consistency degradation over sequence)
        drift_score = self._calculate_drift()
        
        # Sequential consistency (frame-to-frame)
        sequential_consistency = self._calculate_sequential_consistency()
        
        return {
            'character_consistency': char_mean,
            'character_std': char_std,
            'style_consistency': style_mean,
            'style_std': style_std,
            'combined_consistency': float(combined),
            'drift_score': drift_score,
            'sequential_consistency': sequential_consistency,
            'frame_count': len(self.image_sequence),
            'character_similarities': character_similarities,
            'style_similarities': style_similarities,
            'is_consistent': combined > 0.75 and char_std < 0.1
        }
    
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
    
    def _calculate_sequential_consistency(self):
        """Calculate frame-to-frame consistency"""
        if len(self.character_embeddings) < 2:
            return 1.0
        
        sequential_sims = []
        for i in range(len(self.character_embeddings) - 1):
            sim = cosine_similarity(
                self.character_embeddings[i],
                self.character_embeddings[i + 1]
            )[0][0]
            sequential_sims.append(sim)
        
        return float(np.mean(sequential_sims))
    
    def get_consistency_report(self):
        """Generate detailed consistency report"""
        if len(self.image_sequence) < 2:
            return "No sequence loaded for analysis"
        
        results = self._evaluate_sequence_consistency()
        
        report = [
            "CHARACTER CONSISTENCY ANALYSIS",
            "=" * 40,
            f"Images in sequence: {results['frame_count']}",
            f"Character consistency: {results['character_consistency']:.3f} (±{results['character_std']:.3f})",
            f"Style consistency: {results['style_consistency']:.3f} (±{results['style_std']:.3f})",
            f"Combined score: {results['combined_consistency']:.3f}",
            f"Drift score: {results['drift_score']:.3f}",
            f"Sequential consistency: {results['sequential_consistency']:.3f}",
            "",
            "Assessment: " + (
                "Excellent - Very consistent" if results['combined_consistency'] > 0.85
                else "Good - Acceptably consistent" if results['combined_consistency'] > 0.75
                else "Fair - Some inconsistency" if results['combined_consistency'] > 0.65
                else "Poor - Significant inconsistency"
            )
        ]
        
        return "\n".join(report)
    
    def compare_methods(self, sequences_dict, method_names=None):
        """
        Compare consistency across different generation methods
        
        Args:
            sequences_dict: Dict with method names as keys, list of image sequences as values
            method_names: Optional custom names for methods
            
        Returns:
            Dict with comparison results and statistics
        """
        if method_names is None:
            method_names = list(sequences_dict.keys())
        
        results = {}
        
        for method_name, sequences in sequences_dict.items():
            method_results = []
            
            print(f"   Evaluating {method_name} sequences...")
            
            for i, sequence in enumerate(sequences):
                self.reset_sequence()
                
                # Add all images in sequence
                for image in sequence:
                    self.add_image_to_sequence(image)
                
                # Evaluate consistency
                seq_results = self._evaluate_sequence_consistency()
                if 'error' not in seq_results:
                    seq_results['sequence_id'] = f"{method_name}_{i+1}"
                    method_results.append(seq_results)
            
            results[method_name] = method_results
        
        # Calculate comparative statistics
        comparison_stats = self._calculate_comparison_stats(results)
        
        return {
            'method_results': results,
            'comparison_stats': comparison_stats
        }
    
    def _calculate_comparison_stats(self, results):
        """Calculate comparative statistics between methods"""
        stats = {}
        
        for method_name, method_results in results.items():
            if method_results:
                char_scores = [r['character_consistency'] for r in method_results]
                style_scores = [r['style_consistency'] for r in method_results]
                combined_scores = [r['combined_consistency'] for r in method_results]
                
                stats[method_name] = {
                    'character_mean': float(np.mean(char_scores)),
                    'character_std': float(np.std(char_scores)),
                    'style_mean': float(np.mean(style_scores)),
                    'style_std': float(np.std(style_scores)),
                    'combined_mean': float(np.mean(combined_scores)),
                    'combined_std': float(np.std(combined_scores)),
                    'sequence_count': len(method_results)
                }
        
        # Calculate separations between methods
        method_names = list(stats.keys())
        if len(method_names) >= 2:
            separations = {}
            for i in range(len(method_names)):
                for j in range(i + 1, len(method_names)):
                    method1, method2 = method_names[i], method_names[j]
                    
                    char_sep = stats[method1]['character_mean'] - stats[method2]['character_mean']
                    style_sep = stats[method1]['style_mean'] - stats[method2]['style_mean']
                    combined_sep = stats[method1]['combined_mean'] - stats[method2]['combined_mean']
                    
                    separations[f"{method1}_vs_{method2}"] = {
                        'character_separation': float(char_sep),
                        'style_separation': float(style_sep),
                        'combined_separation': float(combined_sep),
                        'separation_percentage': float(combined_sep * 100)
                    }
            
            stats['separations'] = separations
        
        return stats
    
    def plot_consistency_comparison(self, comparison_results, save_path=None):
        """
        Create visualization comparing consistency across methods
        
        Args:
            comparison_results: Results from compare_methods()
            save_path: Optional path to save plot
        """
        method_results = comparison_results['method_results']
        
        if len(method_results) < 2:
            print("Need at least 2 methods for comparison plot")
            return
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        plot_data = []
        labels = []
        colors = ['lightgreen', 'lightcoral', 'lightblue', 'lightyellow']
        
        # Prepare data for plotting
        for i, (method_name, results) in enumerate(method_results.items()):
            if results:  # Only plot if we have results
                plot_data.append([r['character_consistency'] for r in results])
                labels.append(method_name.replace('_', ' ').title())
        
        if not plot_data:
            print("No data to plot")
            return
        
        # Character consistency comparison
        if plot_data:
            bp1 = axes[0].boxplot(plot_data, labels=labels, patch_artist=True)
            for patch, color in zip(bp1['boxes'], colors[:len(plot_data)]):
                patch.set_facecolor(color)
            axes[0].set_title('Character Consistency Scores')
            axes[0].set_ylabel('CLIP Similarity Score')
            axes[0].grid(True, alpha=0.3)
            axes[0].tick_params(axis='x', rotation=45)
        
        # Style consistency comparison
        style_data = []
        for method_name, results in method_results.items():
            if results:
                style_data.append([r['style_consistency'] for r in results])
        
        if style_data:
            bp2 = axes[1].boxplot(style_data, labels=labels, patch_artist=True)
            for patch, color in zip(bp2['boxes'], colors[:len(style_data)]):
                patch.set_facecolor(color)
            axes[1].set_title('Style Consistency Scores')
            axes[1].set_ylabel('CLIP Similarity Score')
            axes[1].grid(True, alpha=0.3)
            axes[1].tick_params(axis='x', rotation=45)
        
        # Combined consistency comparison
        combined_data = []
        for method_name, results in method_results.items():
            if results:
                combined_data.append([r['combined_consistency'] for r in results])
        
        if combined_data:
            bp3 = axes[2].boxplot(combined_data, labels=labels, patch_artist=True)
            for patch, color in zip(bp3['boxes'], colors[:len(combined_data)]):
                patch.set_facecolor(color)
            axes[2].set_title('Combined Consistency Scores')
            axes[2].set_ylabel('Combined Score')
            axes[2].grid(True, alpha=0.3)
            axes[2].tick_params(axis='x', rotation=45)
        
        plt.suptitle('Method Comparison: Character Consistency Analysis', 
                     fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"   📊 Comparison plot saved: {save_path}")
        
        plt.show()
    
    def save_consistency_results(self, results, save_path):
        """
        Save consistency evaluation results to JSON
        
        Args:
            results: Results dictionary from evaluation
            save_path: Path to save JSON file
        """
        # Convert numpy types to native Python types for JSON serialization
        def convert_types(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.float32, np.float64)):
                return float(obj)
            elif isinstance(obj, (np.int32, np.int64)):
                return int(obj)
            elif isinstance(obj, dict):
                return {key: convert_types(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_types(item) for item in obj]
            return obj
        
        clean_results = convert_types(results)
        
        with open(save_path, 'w') as f:
            json.dump(clean_results, f, indent=2)
        
        print(f"   💾 Consistency results saved: {save_path}")


# Convenience functions for easy usage
def evaluate_image_pair(image1, image2, device="cuda"):
    """
    Quick evaluation of consistency between two images
    
    Args:
        image1, image2: Image paths, PIL Images, or numpy arrays
        device: Computing device
        
    Returns:
        Dictionary with consistency scores
    """
    analyzer = ConsistencyAnalyzer(device=device)
    return analyzer.evaluate_consistency(image1, image2)


def evaluate_image_sequence(images, device="cuda"):
    """
    Quick evaluation of consistency across an image sequence
    
    Args:
        images: List of image paths, PIL Images, or numpy arrays
        device: Computing device
        
    Returns:
        Dictionary with sequence consistency analysis
    """
    analyzer = ConsistencyAnalyzer(device=device)
    
    for image in images:
        analyzer.add_image_to_sequence(image)
    
    return analyzer.evaluate_consistency()


if __name__ == "__main__":
    # Example usage
    print("Character & Style Consistency Evaluator")
    print("Example usage:")
    print()
    print("# Evaluate two images")
    print("from consistency_metrics import evaluate_image_pair")
    print("results = evaluate_image_pair('image1.png', 'image2.png')")
    print("print(f'Character consistency: {results[\"character_consistency\"]:.3f}')")
    print()
    print("# Evaluate sequence")
    print("from consistency_metrics import evaluate_image_sequence") 
    print("images = ['frame1.png', 'frame2.png', 'frame3.png']")
    print("results = evaluate_image_sequence(images)")
    print("print(f'Combined consistency: {results[\"combined_consistency\"]:.3f}')")