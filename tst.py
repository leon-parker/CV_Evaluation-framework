#!/usr/bin/env python3
"""
Test the Autism Storyboard Evaluation Framework using RealCartoon XL v7 generated images
Generates test images and evaluates them with the framework
"""

import os
import torch
import gc
from PIL import Image
from diffusers import StableDiffusionXLPipeline, EulerAncestralDiscreteScheduler
from compel import Compel, ReturnedEmbeddingsType
import sys

# Add current directory to path for autism evaluation framework
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Test imports first
print("🔍 Checking autism evaluation framework imports...")
try:
    from autism_evaluator import AutismStoryboardEvaluator
    from cv_metrics import VisualQualityAnalyzer
    from complexity_metrics import AutismComplexityAnalyzer
    from evaluation_config import AUTISM_EVALUATION_WEIGHTS
    print("✅ All framework modules imported successfully!")
    FRAMEWORK_AVAILABLE = True
except ImportError as e:
    print(f"❌ Framework import failed: {e}")
    print("Make sure all framework files are in the same directory as this script:")
    print("- autism_evaluator.py")
    print("- cv_metrics.py") 
    print("- complexity_metrics.py")
    print("- prompt_metrics.py")
    print("- consistency_metrics.py")
    print("- utils.py")
    print("- evaluation_config.py")
    FRAMEWORK_AVAILABLE = False

def find_realcartoon_model():
    """Find the RealCartoon XL v7 model"""
    paths = [
        "../models/realcartoonxl_v7.safetensors",
        "models/realcartoonxl_v7.safetensors", 
        "../models/RealCartoon-XL-v7.safetensors",
        "realcartoonxl_v7.safetensors",
        "RealCartoon-XL-v7.safetensors"
    ]
    
    for path in paths:
        if os.path.exists(path):
            return path
    
    # If not found, ask user
    print("RealCartoon XL v7 model not found in common locations.")
    user_path = input("Enter the full path to realcartoonxl_v7.safetensors: ").strip()
    if os.path.exists(user_path):
        return user_path
    
    return None

class RealCartoonGenerator:
    """Simple generator using RealCartoon XL v7"""
    
    def __init__(self, model_path, device="cuda"):
        self.model_path = model_path
        self.device = device
        self.pipeline = None
        self.compel = None
        
    def setup(self):
        """Setup the pipeline"""
        try:
            print("🚀 Setting up RealCartoon XL v7...")
            
            self.pipeline = StableDiffusionXLPipeline.from_single_file(
                self.model_path,
                torch_dtype=torch.float16,
                use_safetensors=True,
                safety_checker=None,
                requires_safety_checker=False
            ).to(self.device)
            
            self.pipeline.scheduler = EulerAncestralDiscreteScheduler.from_config(
                self.pipeline.scheduler.config
            )
            
            self.pipeline.enable_vae_tiling()
            self.pipeline.enable_model_cpu_offload()
            
            self.compel = Compel(
                tokenizer=[self.pipeline.tokenizer, self.pipeline.tokenizer_2],
                text_encoder=[self.pipeline.text_encoder, self.pipeline.text_encoder_2],
                returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED,
                requires_pooled=[False, True],
                truncate_long_prompts=False
            )
            
            print("✅ Pipeline ready!")
            return True
            
        except Exception as e:
            print(f"❌ Setup failed: {e}")
            return False
    
    def generate(self, prompt, negative_prompt, seed=42):
        """Generate an image"""
        try:
            # Encode prompts
            positive_conditioning, positive_pooled = self.compel(prompt)
            negative_conditioning, negative_pooled = self.compel(negative_prompt)
            
            [positive_conditioning, negative_conditioning] = self.compel.pad_conditioning_tensors_to_same_length(
                [positive_conditioning, negative_conditioning]
            )
            
            result = self.pipeline(
                prompt_embeds=positive_conditioning,
                pooled_prompt_embeds=positive_pooled,
                negative_prompt_embeds=negative_conditioning,
                negative_pooled_prompt_embeds=negative_pooled,
                num_inference_steps=25,
                guidance_scale=8.0,
                height=1024,
                width=1024,
                generator=torch.Generator(self.device).manual_seed(seed)
            )
            
            gc.collect()
            return result.images[0] if result and result.images else None
            
        except Exception as e:
            print(f"Generation failed: {e}")
            return None
    
    def cleanup(self):
        """Clean up resources"""
        if self.pipeline:
            del self.pipeline
        if self.compel:
            del self.compel
        torch.cuda.empty_cache()
        gc.collect()

def generate_test_images():
    """Generate test images using RealCartoon XL v7"""
    
    model_path = find_realcartoon_model()
    if not model_path:
        print("❌ Cannot find RealCartoon XL v7 model!")
        return None
    
    print(f"📁 Using model: {model_path}")
    
    # Create output directory
    test_dir = "autism_framework_test"
    os.makedirs(test_dir, exist_ok=True)
    
    generator = RealCartoonGenerator(model_path)
    if not generator.setup():
        return None
    
    test_cases = []
    
    try:
        # Test Case 1: GOOD - Single character, simple scene
        print("\n🎨 Generating GOOD test image (should score HIGH)...")
        good_prompt = (
            "cartoon illustration of one happy young boy brushing teeth with toothbrush, "
            "simple white bathroom background, clean cartoon style, bright lighting, "
            "minimal objects, clear character"
        )
        good_negative = (
            "multiple people, complex background, realistic photo, cluttered, "
            "too many objects, dark, scary, adults, crowd"
        )
        
        good_image = generator.generate(good_prompt, good_negative, seed=12345)
        if good_image:
            good_path = os.path.join(test_dir, "good_single_character.png")
            good_image.save(good_path)
            test_cases.append({
                "name": "Good Single Character",
                "path": good_path,
                "prompt": good_prompt,
                "expected": "HIGH score (0.8+) - autism appropriate"
            })
            print(f"✅ Saved: {good_path}")
        
        # Test Case 2: BAD - Multiple people, complex scene  
        print("\n🎨 Generating BAD test image (should score LOW)...")
        bad_prompt = (
            "realistic photo of busy classroom with five children and two teachers, "
            "complex detailed background with many books, papers, toys, posters, "
            "crowded scene with lots of activity and objects everywhere"
        )
        bad_negative = (
            "simple, cartoon, clean, minimal, single person, empty background"
        )
        
        bad_image = generator.generate(bad_prompt, bad_negative, seed=67890)
        if bad_image:
            bad_path = os.path.join(test_dir, "bad_complex_scene.png")
            bad_image.save(bad_path)
            test_cases.append({
                "name": "Bad Complex Scene", 
                "path": bad_path,
                "prompt": bad_prompt,
                "expected": "LOW score (0.5-) - too complex for autism"
            })
            print(f"✅ Saved: {bad_path}")
        
        # Test Case 3: MEDIUM - Single character but more complex
        print("\n🎨 Generating MEDIUM test image (should score MEDIUM)...")
        medium_prompt = (
            "cartoon boy eating breakfast at kitchen table, "
            "some kitchen items visible, cartoon style, "
            "one character, moderate detail"
        )
        medium_negative = (
            "multiple people, very complex, realistic photo, cluttered"
        )
        
        medium_image = generator.generate(medium_prompt, medium_negative, seed=11111)
        if medium_image:
            medium_path = os.path.join(test_dir, "medium_breakfast.png")
            medium_image.save(medium_path)
            test_cases.append({
                "name": "Medium Complexity",
                "path": medium_path, 
                "prompt": medium_prompt,
                "expected": "MEDIUM score (0.6-0.8) - acceptable but could improve"
            })
            print(f"✅ Saved: {medium_path}")
        
        return test_cases
        
    finally:
        generator.cleanup()

def test_evaluation_framework(test_cases):
    """Test the autism evaluation framework on generated images"""
    
    if not test_cases:
        print("❌ No test images to evaluate!")
        return
    
    if not FRAMEWORK_AVAILABLE:
        print("❌ Autism evaluation framework not available - cannot test!")
        return
    
    try:
        from autism_evaluator import AutismStoryboardEvaluator
        
        print("\n" + "="*70)
        print("🧩 TESTING AUTISM EVALUATION FRAMEWORK")
        print("="*70)
        
        # Initialize evaluator
        print("\n🚀 Initializing evaluator...")
        evaluator = AutismStoryboardEvaluator(verbose=True)
        
        results = []
        
        # Evaluate each test case
        for i, test_case in enumerate(test_cases, 1):
            print(f"\n📊 EVALUATING {i}/{len(test_cases)}: {test_case['name']}")
            print("-" * 50)
            print(f"Expected: {test_case['expected']}")
            
            # Run evaluation
            result = evaluator.evaluate_single_image(
                image=test_case['path'],
                prompt=test_case['prompt'],
                save_report=True,
                output_dir="evaluation_results"
            )
            
            # Store results
            test_case['result'] = result
            results.append(test_case)
            
            print(f"\n📈 RESULTS:")
            print(f"   Score: {result['combined_score']:.3f}")
            print(f"   Grade: {result['autism_grade']}")
            
            # Show key metrics
            if 'complexity' in result['metrics']:
                complexity = result['metrics']['complexity']
                person_count = complexity['person_count']['count']
                bg_score = complexity['background_simplicity']['score']
                print(f"   People: {person_count}")
                print(f"   Background simplicity: {bg_score:.3f}")
            
            print(f"   Top recommendation: {result['recommendations'][0] if result['recommendations'] else 'None'}")
        
        # Summary
        print("\n" + "="*70)
        print("📋 EVALUATION SUMMARY")
        print("="*70)
        
        print(f"{'Test Case':<20} {'Score':<8} {'Grade':<15} {'Expected'}")
        print("-" * 70)
        
        for test_case in results:
            name = test_case['name'][:19]
            score = test_case['result']['combined_score']
            grade = test_case['result']['autism_grade'][:14]
            expected = test_case['expected'][:30]
            print(f"{name:<20} {score:<8.3f} {grade:<15} {expected}")
        
        # Validation
        print(f"\n🔍 VALIDATION:")
        if len(results) >= 2:
            good_score = results[0]['result']['combined_score']  # Should be highest
            bad_score = results[1]['result']['combined_score']   # Should be lowest
            
            print(f"   Good image ({good_score:.3f}) > Bad image ({bad_score:.3f}): {good_score > bad_score}")
            
            if good_score > bad_score:
                print("   ✅ Framework correctly distinguishes autism-appropriate vs inappropriate!")
            else:
                print("   ⚠️ Framework may need calibration")
        
        print(f"\n📁 Detailed reports saved in: evaluation_results/")
        print("✅ Framework testing complete!")
        
    except ImportError as e:
        print(f"❌ Cannot import evaluation framework: {e}")
        print("Make sure all framework files are in the same directory")
    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()

def main():
    """Main function"""
    print("🧩 AUTISM EVALUATION FRAMEWORK TEST WITH REALCARTOON XL v7")
    print("="*70)
    print("This script will:")
    print("1. Generate test images using RealCartoon XL v7")
    print("2. Evaluate them with the autism framework")
    print("3. Validate the scoring works correctly")
    print("="*70)
    
    # Step 1: Generate test images
    print("\n🎨 STEP 1: GENERATING TEST IMAGES...")
    test_cases = generate_test_images()
    
    if not test_cases:
        print("❌ Failed to generate test images!")
        return
    
    print(f"✅ Generated {len(test_cases)} test images")
    
    # Step 2: Test evaluation framework
    print("\n🧩 STEP 2: TESTING EVALUATION FRAMEWORK...")
    test_evaluation_framework(test_cases)

if __name__ == "__main__":
    main()