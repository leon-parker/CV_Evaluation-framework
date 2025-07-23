"""
Lightweight Consistency-Only Test
Only loads what's needed for character consistency evaluation
"""

import os
import torch
from diffusers import StableDiffusionXLPipeline
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import numpy as np

class LightweightConsistencyTester:
    """Minimal pipeline for consistency testing only"""
    
    def __init__(self, model_path):
        print("🎯 Loading CONSISTENCY-ONLY Pipeline...")
        print("   Skipping: BLIP, SpaCy, Autism Analyzer, Computer Vision")
        print("   Loading: Diffusion + IP-Adapter + CLIP")
        
        # Load only diffusion pipeline
        print("🔧 Loading SDXL pipeline...")
        self.pipe = StableDiffusionXLPipeline.from_single_file(
            model_path, 
            torch_dtype=torch.float16, 
            use_safetensors=True
        ).to("cuda")
        
        self.pipe.enable_vae_tiling()
        
        # Load IP-Adapter for consistency
        print("🎭 Loading IP-Adapter...")
        try:
            self.pipe.load_ip_adapter(
                "h94/IP-Adapter", 
                subfolder="sdxl_models", 
                weight_name="ip-adapter_sdxl.bin",
                torch_dtype=torch.float16
            )
            self.pipe.set_ip_adapter_scale(0.6)  # High for consistency
            self.ip_adapter_loaded = True
            print("✅ IP-Adapter loaded")
        except Exception as e:
            print(f"⚠️ IP-Adapter failed: {e}")
            self.ip_adapter_loaded = False
        
        # Load minimal CLIP for consistency scoring
        print("🧠 Loading CLIP for consistency scoring...")
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to("cuda")
        
        self.character_reference = None
        self.generated_images = []
        
        print("✅ Lightweight Consistency Pipeline Ready!")
    
    def generate_consistency_sequence(self, prompts):
        """Generate sequence with character consistency focus"""
        print(f"\n🎨 Generating {len(prompts)} images for CONSISTENCY evaluation...")
        
        results = []
        
        for i, prompt in enumerate(prompts):
            print(f"\n📸 Frame {i+1}: {prompt[:50]}...")
            
            # FAST generation settings for quick consistency testing
            if self.ip_adapter_loaded and self.character_reference is not None:
                print("   🎭 Using IP-Adapter with character reference")
                result = self.pipe(
                    prompt=prompt,
                    ip_adapter_image=self.character_reference,
                    negative_prompt="multiple people, crowd, blurry",
                    num_images_per_prompt=1,
                    num_inference_steps=15,  # REDUCED from 30
                    guidance_scale=6.0,      # REDUCED from 7.0
                    height=512,              # REDUCED from 1024
                    width=512                # REDUCED from 1024
                )
            else:
                print("   🎨 First image generation (setting reference)")
                # For first image, create a dummy reference to avoid IP-Adapter errors
                dummy_image = Image.new('RGB', (512, 512), color=(128, 128, 128))
                self.pipe.set_ip_adapter_scale(0.01)  # Very low influence
                result = self.pipe(
                    prompt=prompt,
                    ip_adapter_image=dummy_image,
                    negative_prompt="multiple people, crowd, blurry",
                    num_images_per_prompt=1,
                    num_inference_steps=15,  # REDUCED from 30
                    guidance_scale=6.0,      # REDUCED from 7.0
                    height=512,              # REDUCED from 1024
                    width=512                # REDUCED from 1024
                )
                # Reset to normal scale for subsequent images
                self.pipe.set_ip_adapter_scale(0.6)
            
            image = result.images[0]
            
            # Set first image as character reference
            if i == 0:
                self.character_reference = image
                print("   🎭 Set as character reference for consistency")
            
            # Calculate consistency score
            consistency_score = self.calculate_consistency_score(image, i)
            
            results.append({
                "prompt": prompt,
                "image": image,
                "consistency_score": consistency_score,
                "frame_number": i + 1
            })
            
            # Save image
            filename = f"consistency_frame_{i+1}.png"
            image.save(filename)
            print(f"   ✅ Consistency: {consistency_score:.3f} | Saved: {filename}")
            
            self.generated_images.append(image)
        
        return results
    
    def calculate_consistency_score(self, image, frame_index):
        """Calculate CLIP consistency with previous images"""
        if frame_index == 0:
            return 1.0  # First image is reference
        
        # Get CLIP embeddings
        current_embedding = self.get_clip_embedding(image)
        
        if current_embedding is None:
            return 0.5
        
        # Calculate similarity with all previous images
        similarities = []
        for prev_image in self.generated_images:
            prev_embedding = self.get_clip_embedding(prev_image)
            if prev_embedding is not None:
                similarity = np.dot(current_embedding.flatten(), prev_embedding.flatten()) / (
                    np.linalg.norm(current_embedding) * np.linalg.norm(prev_embedding)
                )
                similarities.append(similarity)
        
        return float(np.mean(similarities)) if similarities else 0.5
    
    def get_clip_embedding(self, image):
        """Get CLIP embedding for image"""
        try:
            inputs = self.clip_processor(images=image, return_tensors="pt").to("cuda")
            with torch.no_grad():
                features = self.clip_model.get_image_features(**inputs)
                return features.cpu().numpy()
        except:
            return None
    
    def analyze_consistency_results(self, results):
        """Analyze overall consistency"""
        consistency_scores = [r["consistency_score"] for r in results[1:]]  # Skip first (reference)
        
        if not consistency_scores:
            return {"grade": "N/A", "average": 1.0}
        
        avg_consistency = np.mean(consistency_scores)
        min_consistency = min(consistency_scores)
        
        if avg_consistency > 0.8:
            grade = "Excellent"
        elif avg_consistency > 0.7:
            grade = "Good"
        elif avg_consistency > 0.6:
            grade = "Moderate"
        else:
            grade = "Poor"
        
        return {
            "grade": grade,
            "average": avg_consistency,
            "minimum": min_consistency,
            "scores": consistency_scores
        }

def find_model():
    """Find model file"""
    candidates = [
        "realcartoonxl_v7.safetensors",
        "../realcartoonxl_v7.safetensors",
        "../models/realcartoonxl_v7.safetensors"
    ]
    
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    return None

def main():
    print("🎯 LIGHTWEIGHT CONSISTENCY TEST")
    print("=" * 40)
    print("Focus: Character consistency across scenes")
    print("Loading: Only what's needed for consistency evaluation")
    print()
    
    # Find model
    model_path = find_model()
    if not model_path:
        print("❌ No model found!")
        return
    
    print(f"✅ Found model: {model_path}")
    
    # Initialize lightweight tester
    tester = LightweightConsistencyTester(model_path)
    
    # Consistency test prompts
    consistency_prompts = [
        "young boy Alex with brown hair, blue shirt, cartoon style, simple background",
        "same boy Alex with brown hair and blue shirt brushing teeth, cartoon style",
        "same boy Alex with brown hair and blue shirt eating breakfast, cartoon style",
        "same boy Alex with brown hair and blue shirt reading a book, cartoon style"
    ]
    
    print(f"\n🎯 CONSISTENCY EVALUATION")
    print(f"Testing: Same character 'Alex' across {len(consistency_prompts)} scenes")
    
    # Generate sequence
    results = tester.generate_consistency_sequence(consistency_prompts)
    
    # Analyze results
    analysis = tester.analyze_consistency_results(results)
    
    # Show results
    print(f"\n🎉 CONSISTENCY TEST COMPLETE!")
    print(f"📊 Overall Consistency: {analysis['grade']}")
    print(f"📊 Average Score: {analysis['average']:.3f}")
    print(f"📊 Minimum Score: {analysis['minimum']:.3f}")
    print(f"📁 Images saved as: consistency_frame_1.png to consistency_frame_4.png")
    
    print(f"\n🔍 FRAME-BY-FRAME CONSISTENCY:")
    for result in results:
        frame = result["frame_number"]
        score = result["consistency_score"]
        status = "📍 REFERENCE" if frame == 1 else f"📊 {score:.3f}"
        print(f"   Frame {frame}: {status}")
    
    print(f"\n🎯 CONSISTENCY EVALUATION SUMMARY:")
    if analysis["average"] > 0.8:
        print("   ✅ EXCELLENT: Very consistent character across all scenes")
    elif analysis["average"] > 0.7:
        print("   ✅ GOOD: Mostly consistent character with minor variations")
    elif analysis["average"] > 0.6:
        print("   ⚠️ MODERATE: Some consistency issues detected")
    else:
        print("   ❌ POOR: Significant character inconsistencies")
    
    if tester.ip_adapter_loaded:
        print("   🎭 IP-Adapter was used for character consistency")
    else:
        print("   ⚠️ IP-Adapter unavailable - consistency may be limited")

if __name__ == "__main__":
    main() 