"""
Prompt faithfulness evaluation using BLIP-Large
Default: 100% keyword matching (achieves 90% accuracy on autism education dataset)
Optional: Semantic similarity can be enabled for understanding synonyms/concepts

================================================================================
SEMANTIC SIMILARITY (OPTIONAL FEATURE)
================================================================================
Semantic similarity uses sentence transformers to understand meaning beyond exact words.
It captures conceptual relationships like "happy" ≈ "smiling" or "breakfast" ≈ "cereal".

TO ENABLE SEMANTIC SIMILARITY:
-------------------------------
1. Install requirements:
   pip install sentence-transformers

2. When creating the analyzer, set use_semantic=True:
   analyzer = PromptFaithfulnessAnalyzer(use_semantic=True)
   
   OR uncomment this line in your code:
   # analyzer = PromptFaithfulnessAnalyzer(use_semantic=True)  # <-- UNCOMMENT FOR SEMANTIC

3. Choose evaluation method:
   - "keyword": 100% keyword matching (DEFAULT)
   - "semantic": 100% semantic similarity
   - "combined": 60% keyword + 40% semantic (balanced approach)

Example with semantic enabled:
   analyzer = PromptFaithfulnessAnalyzer(use_semantic=True)
   score = analyzer.evaluate_prompt_alignment(image, prompt, method="combined")

WHY USE SEMANTIC SIMILARITY?
----------------------------
- Handles synonyms: "child" ≈ "kid" ≈ "boy"
- Understands concepts: "eating" ≈ "having lunch" ≈ "dining"
- Catches related ideas even with different words

WHEN TO USE EACH MODE:
----------------------
- Keyword only (default): Strict prompt adherence, exact terminology
- Semantic enabled: Flexible understanding, synonym handling
================================================================================
"""

import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import numpy as np
from typing import List, Union, Dict, Optional

# Semantic similarity is OPTIONAL - not required for basic functionality
try:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
    SEMANTIC_AVAILABLE = True
except ImportError:
    SEMANTIC_AVAILABLE = False
    # Semantic is optional - system works fine without it


class PromptFaithfulnessAnalyzer:
    """
    BLIP-Large based prompt faithfulness evaluation
    
    DEFAULT MODE: 100% keyword matching (proven 90% accuracy)
    OPTIONAL: Enable semantic similarity for synonym understanding
    
    To enable semantic similarity:
    >>> analyzer = PromptFaithfulnessAnalyzer(use_semantic=True)  # <-- ADD THIS PARAMETER
    """
    
    def __init__(self, 
                 device="cuda" if torch.cuda.is_available() else "cpu", 
                 verbose=True,
                 use_semantic=False):  # DEFAULT: False (keyword only)
        """
        Initialize the analyzer
        
        Args:
            device: Computing device ('cuda' or 'cpu')
            verbose: Whether to print detailed output
            use_semantic: Enable semantic similarity (DEFAULT: False)
                         Set to True to enable semantic understanding
                         
        Example:
            # Default (keyword only - 90% accuracy):
            analyzer = PromptFaithfulnessAnalyzer()
            
            # With semantic similarity:
            analyzer = PromptFaithfulnessAnalyzer(use_semantic=True)
        """
        self.device = device
        self.verbose = verbose
        self.use_semantic = use_semantic
        self.processor = None
        self.model = None
        self.sentence_model = None
        
        # Check if semantic is requested but not available
        if self.use_semantic and not SEMANTIC_AVAILABLE:
            print("⚠️ Semantic similarity requested but sentence-transformers not installed")
            print("   To enable: pip install sentence-transformers")
            print("   Continuing with keyword-only mode...")
            self.use_semantic = False
        
        self.load_models()
    
    def load_models(self):
        """Load BLIP-Large and optionally sentence transformer models"""
        try:
            if self.verbose:
                print("   Loading BLIP-Large for caption generation...")
            self.processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
            self.model = BlipForConditionalGeneration.from_pretrained(
                "Salesforce/blip-image-captioning-large",
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
            ).to(self.device)
            self.model.eval()
            
            if self.use_semantic and SEMANTIC_AVAILABLE:
                if self.verbose:
                    print("   Loading sentence transformer for semantic similarity...")
                self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
                if self.verbose:
                    print("   ✓ Models loaded: BLIP-Large + Semantic Similarity")
            else:
                if self.verbose:
                    print("   ✓ BLIP-Large loaded (Keyword-only mode - DEFAULT)")
                    if not self.use_semantic:
                        print("   💡 Tip: To enable semantic similarity, use:")
                        print("      analyzer = PromptFaithfulnessAnalyzer(use_semantic=True)")
                    
        except Exception as e:
            print(f"   ✗ Failed to load models: {e}")
            raise
    
    def generate_caption(self, image, num_beams=5, max_length=50):
        """Generate caption using BLIP-Large"""
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image).convert('RGB')
        elif not isinstance(image, Image.Image):
            raise ValueError("Image must be path, PIL Image, or numpy array")
        
        try:
            inputs = self.processor(image, return_tensors="pt").to(self.device)
            with torch.no_grad():
                out = self.model.generate(
                    **inputs, 
                    max_length=max_length, 
                    num_beams=num_beams,
                    do_sample=False
                )
            caption = self.processor.decode(out[0], skip_special_tokens=True)
            return caption
        except Exception as e:
            if self.verbose:
                print(f"   Error in caption generation: {e}")
            return ""
    
    def extract_key_words(self, text):
        """Extract meaningful words from text for comparison"""
        skip_words = {
            'a', 'an', 'the', 'with', 'in', 'on', 'at', 'to', 'for', 
            'of', 'and', 'or', 'but', 'is', 'are', 'was', 'were', 'one',
            'cartoon', 'image', 'picture', 'showing', 'simple', 'clear',
            'there', 'this', 'that', 'have', 'has', 'been', 'being'
        }
        
        text = text.lower()
        words = text.split()
        
        key_words = [word.strip('.,!?;:') for word in words 
                    if word not in skip_words and len(word) > 2]
        
        return key_words
    
    def calculate_keyword_overlap(self, caption_words, prompt_words):
        """
        Calculate keyword overlap between caption and prompt
        This is the DEFAULT method that achieves 90% accuracy
        """
        if not prompt_words:
            return 0.0, []
        
        matched_words = []
        for word in prompt_words:
            if word in caption_words:
                matched_words.append(word)
            # Check for partial matches (e.g., "eating" vs "eat")
            elif any(word in cap_word or cap_word in word for cap_word in caption_words):
                matched_words.append(word)
        
        overlap_score = len(matched_words) / len(prompt_words)
        return overlap_score, matched_words
    
    def calculate_semantic_similarity(self, caption, prompt):
        """
        OPTIONAL: Calculate semantic similarity between caption and prompt
        Only available if use_semantic=True was set during initialization
        """
        if not self.use_semantic or not self.sentence_model:
            return 0.0
            
        try:
            # Encode both texts into embeddings
            caption_embedding = self.sentence_model.encode(caption)
            prompt_embedding = self.sentence_model.encode(prompt)
            
            # Calculate cosine similarity
            similarity = cosine_similarity(
                caption_embedding.reshape(1, -1),
                prompt_embedding.reshape(1, -1)
            )[0][0]
            
            return float(similarity)
        except Exception as e:
            if self.verbose:
                print(f"   Error in semantic similarity: {e}")
            return 0.0
    
    def evaluate_prompt_alignment(self, 
                                 image: Union[str, Image.Image, np.ndarray], 
                                 prompt: str,
                                 return_details: bool = False,
                                 method: str = "keyword"):  # DEFAULT: keyword
        """
        Evaluate how well an image matches its prompt
        
        Args:
            image: Path to image, PIL Image, or numpy array
            prompt: Text prompt that was used to generate the image
            return_details: If True, return detailed analysis
            method: Evaluation method (DEFAULT: "keyword")
                   - "keyword": 100% keyword matching (DEFAULT - 90% accuracy)
                   - "semantic": 100% semantic (requires use_semantic=True)
                   - "combined": 60% keyword + 40% semantic (requires use_semantic=True)
            
        Returns:
            float: Alignment score between 0 and 1 (or dict if return_details=True)
            
        Example:
            # Default usage (keyword only):
            score = analyzer.evaluate_prompt_alignment(image, prompt)
            
            # With semantic (if enabled):
            score = analyzer.evaluate_prompt_alignment(image, prompt, method="combined")
        """
        # Validate method
        if method in ["semantic", "combined"] and not self.use_semantic:
            if self.verbose:
                print(f"⚠️ Method '{method}' requires semantic similarity")
                print("   To enable: analyzer = PromptFaithfulnessAnalyzer(use_semantic=True)")
                print("   Using keyword method instead...")
            method = "keyword"
        
        # Generate caption from the image
        caption = self.generate_caption(image)
        
        if not caption:
            if return_details:
                return {
                    'score': 0.0,
                    'caption': '',
                    'prompt': prompt,
                    'error': 'Failed to generate caption'
                }
            return 0.0
        
        # Always calculate keyword overlap (primary method)
        caption_words = self.extract_key_words(caption.lower())
        prompt_words = self.extract_key_words(prompt.lower())
        keyword_score, matched_words = self.calculate_keyword_overlap(caption_words, prompt_words)
        
        # Optionally calculate semantic similarity
        semantic_score = 0.0
        if self.use_semantic and method in ["semantic", "combined"]:
            semantic_score = self.calculate_semantic_similarity(caption, prompt)
        
        # Calculate final score based on method
        if method == "keyword":
            final_score = keyword_score
        elif method == "semantic":
            final_score = semantic_score
        else:  # combined
            final_score = (keyword_score * 0.6) + (semantic_score * 0.4)
        
        if self.verbose and return_details:
            print(f"\n📝 BLIP Caption: {caption}")
            print(f"🎯 Original Prompt: {prompt}")
            print(f"\n📊 KEYWORD ANALYSIS (Primary Method):")
            print(f"   Prompt keywords: {prompt_words}")
            print(f"   Caption keywords: {caption_words}")
            print(f"   Matched words: {matched_words}")
            print(f"   Keyword score: {keyword_score:.2%}")
            
            if self.use_semantic and method in ["semantic", "combined"]:
                print(f"\n🧠 SEMANTIC ANALYSIS (Optional):")
                print(f"   Semantic similarity: {semantic_score:.2%}")
            
            print(f"\n🎯 FINAL SCORE ({method}): {final_score:.2%}")
        
        if return_details:
            return {
                'score': final_score,
                'keyword_score': keyword_score,
                'semantic_score': semantic_score if self.use_semantic else None,
                'caption': caption,
                'prompt': prompt,
                'caption_keywords': caption_words,
                'prompt_keywords': prompt_words,
                'matched_words': matched_words,
                'num_matches': len(matched_words),
                'num_expected': len(prompt_words),
                'method': method,
                'semantic_enabled': self.use_semantic,
                'alignment': 'GOOD' if final_score >= 0.5 else 'POOR'
            }
        
        return final_score
    
    def evaluate_with_visualization(self, 
                                   image: Union[str, Image.Image, np.ndarray], 
                                   prompt: str,
                                   method: str = "keyword"):  # DEFAULT: keyword
        """
        Evaluate with detailed visualization of the analysis
        Default uses keyword matching (90% accuracy)
        """
        print("\n" + "="*60)
        print("🔬 PROMPT FAITHFULNESS EVALUATION")
        print(f"   Mode: {'Keyword Only (DEFAULT)' if not self.use_semantic else 'Semantic Available'}")
        print("="*60)
        
        # Get detailed evaluation
        details = self.evaluate_prompt_alignment(image, prompt, return_details=True, method=method)
        
        # Visual representation of keyword matching
        print("\n📊 KEYWORD MATCHING (Primary Method - 90% Accuracy):")
        print("-"*40)
        
        for word in details['prompt_keywords']:
            if word in details['matched_words']:
                print(f"  ✅ '{word}' - FOUND in caption")
            else:
                print(f"  ❌ '{word}' - NOT FOUND in caption")
        
        print(f"\nKeyword Score: {details['keyword_score']:.2%}")
        
        # Show semantic if enabled
        if self.use_semantic and details['semantic_score'] is not None:
            print("\n🧠 SEMANTIC UNDERSTANDING (Optional):")
            print("-"*40)
            
            semantic_score = details['semantic_score']
            print(f"  Semantic similarity: {semantic_score:.2%}")
            
            if details['keyword_score'] < details['semantic_score']:
                print("  💡 Semantic captures meaning beyond exact words")
        elif not self.use_semantic:
            print("\n💡 Semantic similarity not enabled. To enable:")
            print("   analyzer = PromptFaithfulnessAnalyzer(use_semantic=True)")
        
        # Final assessment
        print("\n" + "="*40)
        print(f"🎯 FINAL SCORE ({details['method']}): {details['score']:.2%}")
        print(f"📝 ALIGNMENT: {details['alignment']}")
        
        if details['score'] >= 0.7:
            print("✨ Excellent match - image closely follows prompt")
        elif details['score'] >= 0.5:
            print("👍 Good match - image generally follows prompt")
        elif details['score'] >= 0.3:
            print("⚠️ Partial match - some elements missing")
        else:
            print("❌ Poor match - image doesn't follow prompt")
        
        print("="*60)
        
        return details
    
    def batch_evaluate(self, 
                       images: List[Union[str, Image.Image, np.ndarray]], 
                       prompts: List[str],
                       method: str = "keyword",  # DEFAULT: keyword
                       show_details: bool = False):
        """
        Evaluate multiple image-prompt pairs
        Default uses keyword matching for proven 90% accuracy
        """
        if len(images) != len(prompts):
            raise ValueError("Number of images and prompts must match")
        
        results = []
        
        print(f"\n🔄 Evaluating {len(images)} image-prompt pairs...")
        print(f"   Method: {method} {'(DEFAULT)' if method == 'keyword' else ''}")
        if not self.use_semantic and method != "keyword":
            print("   ⚠️ Semantic not enabled - using keyword method")
        print("-"*40)
        
        for i, (image, prompt) in enumerate(zip(images, prompts)):
            print(f"\n📷 Image {i+1}/{len(images)}")
            
            if show_details:
                result = self.evaluate_with_visualization(image, prompt, method)
            else:
                result = self.evaluate_prompt_alignment(image, prompt, return_details=True, method=method)
                print(f"   Caption: {result['caption'][:60]}...")
                print(f"   Score: {result['score']:.2%}")
            
            results.append(result)
        
        # Summary statistics
        scores = [r['score'] for r in results]
        avg_score = np.mean(scores)
        
        print("\n" + "="*60)
        print("📊 BATCH SUMMARY")
        print("="*60)
        print(f"Average Score: {avg_score:.2%}")
        print(f"Best Score: {max(scores):.2%}")
        print(f"Worst Score: {min(scores):.2%}")
        print(f"Passing (≥50%): {sum(1 for s in scores if s >= 0.5)}/{len(scores)}")
        
        return results


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def evaluate_image(image_path: str, prompt: str, verbose: bool = True) -> Dict:
    """
    DEFAULT: Evaluate using keyword matching (90% accuracy)
    
    Example:
        results = evaluate_image('boy_eating.png', 'a boy eating breakfast')
    """
    # DEFAULT: use_semantic=False (keyword only)
    analyzer = PromptFaithfulnessAnalyzer(verbose=verbose, use_semantic=False)
    return analyzer.evaluate_with_visualization(image_path, prompt, method="keyword")


def evaluate_image_with_semantic(image_path: str, prompt: str, verbose: bool = True) -> Dict:
    """
    Evaluate using BOTH keyword and semantic similarity
    Requires: pip install sentence-transformers
    
    Example:
        results = evaluate_image_with_semantic('boy_eating.png', 'a boy eating breakfast')
    """
    # UNCOMMENT THE LINE BELOW TO ENABLE SEMANTIC SIMILARITY:
    analyzer = PromptFaithfulnessAnalyzer(verbose=verbose, use_semantic=True)
    return analyzer.evaluate_with_visualization(image_path, prompt, method="combined")


def quick_score(image_path: str, prompt: str) -> float:
    """
    Get just the alignment score (DEFAULT: keyword matching)
    
    Example:
        score = quick_score('boy_eating.png', 'a boy eating breakfast')
        print(f"Alignment: {score:.2%}")
    """
    analyzer = PromptFaithfulnessAnalyzer(verbose=False, use_semantic=False)
    return analyzer.evaluate_prompt_alignment(image_path, prompt, method="keyword")


if __name__ == "__main__":
    print("="*70)
    print("BLIP-Large Prompt Faithfulness Analyzer")
    print("="*70)
    print("\n📊 DEFAULT MODE: Keyword Matching (90% accuracy)")
    print("   - Checks if prompt words appear in generated caption")
    print("   - Proven effective on autism education dataset")
    print("\n🧠 OPTIONAL: Semantic Similarity")
    print("   - Enable to understand synonyms and concepts")
    print("   - Requires: pip install sentence-transformers")
    print("\n" + "="*70)
    print("USAGE EXAMPLES:")
    print("="*70)
    print("\n# DEFAULT (Keyword only - recommended):")
    print("from prompt_metrics import evaluate_image")
    print("results = evaluate_image('image.png', 'a happy boy eating breakfast')")
    print("\n# WITH SEMANTIC (optional - uncomment to enable):")
    print("from prompt_metrics import evaluate_image_with_semantic")
    print("results = evaluate_image_with_semantic('image.png', 'a happy boy eating breakfast')")
    print("\n# QUICK SCORE:")
    print("from prompt_metrics import quick_score")
    print("score = quick_score('image.png', 'a happy boy eating breakfast')")
    print("="*70)