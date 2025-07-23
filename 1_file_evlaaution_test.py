"""
Evaluation Framework Test Script
Tests your evaluation metrics without importing your modules
Use this to validate your 3-metric system: Simplicity, Accuracy, Consistency
"""

import numpy as np
from PIL import Image, ImageDraw
import cv2
import torch
from transformers import CLIPProcessor, CLIPModel, BlipProcessor, BlipForConditionalGeneration
import matplotlib.pyplot as plt

def test_evaluation_framework_standalone():
    """
    Test evaluation framework without importing your files
    This isolates each metric and tests the scoring logic
    """
    print("🔍 TESTING EVALUATION FRAMEWORK (STANDALONE)")
    print("=" * 50)
    
    # Load models directly (same as your code does)
    print("📦 Loading models...")
    try:
        clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        
        blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
        blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")
        
        if torch.cuda.is_available():
            clip_model = clip_model.to("cuda")
            blip_model = blip_model.to("cuda")
        
        # Face detection
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        print("✅ Models loaded successfully!")
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        return
    
    # Create test images
    test_cases = create_evaluation_test_cases()
    
    print(f"\n🖼️ Testing {len(test_cases)} different scenarios")
    
    # Test each case
    for i, (case_name, case_data) in enumerate(test_cases.items()):
        print(f"\n{'='*20} TEST CASE {i+1}: {case_name.upper()} {'='*20}")
        
        image = case_data["image"]
        prompt = case_data["prompt"]
        expected = case_data["expected"]
        
        # Save test image for inspection
        image.save(f"test_{i+1}_{case_name}.png")
        print(f"💾 Saved: test_{i+1}_{case_name}.png")
        
        # TEST 1: SIMPLICITY METRIC
        print("\n🧩 Testing SIMPLICITY (Autism-friendly)...")
        simplicity_score = test_simplicity_metric_standalone(image, face_cascade)
        print(f"   Result: {simplicity_score:.3f}")
        print(f"   Expected: {expected['simplicity']} ({'✅ PASS' if check_expectation(simplicity_score, expected['simplicity']) else '❌ FAIL'})")
        
        # TEST 2: ACCURACY METRIC  
        print("\n🎯 Testing ACCURACY (Prompt alignment)...")
        accuracy_score = test_accuracy_metric_standalone(image, prompt, clip_processor, clip_model, blip_processor, blip_model)
        print(f"   Result: {accuracy_score:.3f}")
        print(f"   Expected: {expected['accuracy']} ({'✅ PASS' if check_expectation(accuracy_score, expected['accuracy']) else '❌ FAIL'})")
        
        # TEST 3: CONSISTENCY METRIC
        print("\n🔄 Testing CONSISTENCY (Character consistency)...")
        consistency_score = test_consistency_metric_standalone(image, case_data.get("reference_images", []), clip_processor, clip_model)
        print(f"   Result: {consistency_score:.3f}")
        print(f"   Expected: {expected['consistency']} ({'✅ PASS' if check_expectation(consistency_score, expected['consistency']) else '❌ FAIL'})")
        
        # TEST 4: OVERALL WEIGHTED SCORE
        print("\n📊 Testing OVERALL WEIGHTED SCORING...")
        overall_score = calculate_weighted_autism_score(simplicity_score, accuracy_score, consistency_score)
        print(f"   Weighted Score: {overall_score:.3f}")
        print(f"   Formula: {simplicity_score:.3f}×0.364 + {accuracy_score:.3f}×0.333 + {consistency_score:.3f}×0.303")
        print(f"   Expected: {expected['overall']} ({'✅ PASS' if check_expectation(overall_score, expected['overall']) else '❌ FAIL'})")
        
        # SUMMARY
        print(f"\n📋 SUMMARY FOR {case_name}:")
        print(f"   Simplicity: {simplicity_score:.3f} | Accuracy: {accuracy_score:.3f} | Consistency: {consistency_score:.3f}")
        print(f"   Overall: {overall_score:.3f}")
    
    print(f"\n🎉 EVALUATION TESTING COMPLETE!")
    print("Check the saved test images and compare results with expectations.")

def create_evaluation_test_cases():
    """Create specific test cases to validate each metric"""
    test_cases = {}
    
    # TEST CASE 1: PERFECT AUTISM IMAGE
    # Should score HIGH on simplicity, MEDIUM on accuracy (test image), HIGH on consistency (first image)
    perfect_img = Image.new('RGB', (512, 512), color=(250, 250, 250))  # Clean white background
    draw = ImageDraw.Draw(perfect_img)
    # Simple cartoon character - single person
    draw.ellipse([200, 150, 300, 250], fill=(255, 220, 180), outline=(0, 0, 0), width=3)  # Head
    draw.rectangle([225, 240, 275, 350], fill=(100, 150, 200), outline=(0, 0, 0), width=3)  # Body
    draw.ellipse([220, 170, 235, 185], fill=(0, 0, 0))  # Left eye
    draw.ellipse([265, 170, 280, 185], fill=(0, 0, 0))  # Right eye
    
    test_cases["perfect_autism"] = {
        "image": perfect_img,
        "prompt": "a young boy standing, simple background, cartoon style",
        "expected": {
            "simplicity": "high",    # Clean background, single person, simple colors
            "accuracy": "medium",    # Test image won't perfectly match prompt  
            "consistency": "high",   # First image = perfect consistency
            "overall": "high"        # Should score well overall
        }
    }
    
    # TEST CASE 2: COMPLEX BUSY IMAGE  
    # Should score LOW on simplicity, LOW on accuracy, MEDIUM on consistency
    complex_img = Image.new('RGB', (512, 512), color=(100, 100, 100))
    draw = ImageDraw.Draw(complex_img)
    
    # Add lots of visual clutter
    np.random.seed(42)  # Consistent randomness
    for _ in range(50):
        x1, y1 = np.random.randint(0, 400, 2)
        x2, y2 = x1 + np.random.randint(20, 100), y1 + np.random.randint(20, 100)
        color = tuple(np.random.randint(0, 255, 3))
        draw.rectangle([x1, y1, x2, y2], fill=color, outline=(0, 0, 0))
    
    # Add multiple "people" (circles)
    for i, x in enumerate([100, 250, 400]):
        draw.ellipse([x-30, 150, x+30, 210], fill=(255, 200, 200), outline=(0, 0, 0), width=2)
    
    test_cases["complex_busy"] = {
        "image": complex_img,
        "prompt": "a simple boy reading a book",
        "expected": {
            "simplicity": "low",     # Busy background, multiple people, many colors
            "accuracy": "low",       # Doesn't match simple prompt
            "consistency": "medium", # Will compare to first image
            "overall": "low"         # Should score poorly
        }
    }
    
    # TEST CASE 3: DARK IMAGE
    # Should score MEDIUM on simplicity (simple but dark), LOW on accuracy, MEDIUM on consistency
    dark_img = Image.new('RGB', (512, 512), color=(20, 20, 20))  # Very dark
    draw = ImageDraw.Draw(dark_img)
    # Simple but barely visible character
    draw.ellipse([200, 150, 300, 250], fill=(40, 40, 40), outline=(60, 60, 60))
    
    test_cases["dark_image"] = {
        "image": dark_img,
        "prompt": "bright happy boy in sunshine",
        "expected": {
            "simplicity": "medium",  # Simple but darkness is an issue
            "accuracy": "low",       # Doesn't match "bright" prompt
            "consistency": "medium", # Simple comparison
            "overall": "low"         # Dark images should score poorly
        }
    }
    
    # TEST CASE 4: HIGH CONTRAST GOOD IMAGE
    # Should score HIGH on simplicity, MEDIUM on accuracy, HIGH on consistency
    contrast_img = Image.new('RGB', (512, 512), color=(255, 255, 255))  # White background
    draw = ImageDraw.Draw(contrast_img)
    # High contrast simple character
    draw.ellipse([200, 150, 300, 250], fill=(100, 100, 100), outline=(0, 0, 0), width=4)  # Head
    draw.rectangle([225, 240, 275, 350], fill=(200, 50, 50), outline=(0, 0, 0), width=4)   # Body
    draw.ellipse([220, 170, 235, 185], fill=(0, 0, 0))  # Left eye
    draw.ellipse([265, 170, 280, 185], fill=(0, 0, 0))  # Right eye
    
    test_cases["high_contrast"] = {
        "image": contrast_img,
        "prompt": "boy with clear features, cartoon style",
        "expected": {
            "simplicity": "high",    # Clean, single person, good contrast
            "accuracy": "medium",    # Reasonable match to prompt
            "consistency": "high",   # Clear features should be consistent
            "overall": "high"        # Should score well
        }
    }
    
    return test_cases

def test_simplicity_metric_standalone(image, face_cascade):
    """Test the autism simplicity metric (replicate your AutismFriendlyImageAnalyzer logic)"""
    try:
        cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # 1. Person count (25% weight in your system)
        person_count = count_people_simple(cv_image, face_cascade)
        person_score = 1.0 if person_count <= 2 else max(0.0, 1.0 - (person_count - 2) * 0.3)
        
        # 2. Background simplicity (20% weight)
        bg_score = analyze_background_complexity(cv_image)
        
        # 3. Color simplicity (15% weight)  
        color_score = analyze_color_complexity(cv_image)
        
        # 4. Character clarity (20% weight)
        clarity_score = analyze_character_clarity_simple(cv_image)
        
        # 5. Sensory friendliness (15% weight)
        sensory_score = analyze_sensory_simple(cv_image)
        
        # 6. Focus clarity (5% weight)
        focus_score = analyze_focus_simple(cv_image)
        
        # Weighted combination (matching your weights)
        overall_simplicity = (
            person_score * 0.25 +
            bg_score * 0.20 +  
            clarity_score * 0.20 +
            color_score * 0.15 +
            sensory_score * 0.15 +
            focus_score * 0.05
        )
        
        print(f"      Person count: {person_count} (score: {person_score:.3f})")
        print(f"      Background: {bg_score:.3f}")
        print(f"      Colors: {color_score:.3f}")
        print(f"      Clarity: {clarity_score:.3f}")
        
        return overall_simplicity
        
    except Exception as e:
        print(f"      ⚠️ Simplicity test error: {e}")
        return 0.5

def test_accuracy_metric_standalone(image, prompt, clip_processor, clip_model, blip_processor, blip_model):
    """Test the accuracy metric (replicate your TIFA-style scoring)"""
    try:
        # CLIP similarity (main component)
        inputs = clip_processor(text=[prompt], images=image, return_tensors="pt", padding=True)
        if torch.cuda.is_available():
            inputs = {k: v.to("cuda") for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = clip_model(**inputs)
            clip_similarity = torch.cosine_similarity(
                outputs.image_embeds, outputs.text_embeds, dim=1
            ).item()
        
        # BLIP reverse captioning
        blip_inputs = blip_processor(image, return_tensors="pt")
        if torch.cuda.is_available():
            blip_inputs = {k: v.to("cuda") for k, v in blip_inputs.items()}
        
        with torch.no_grad():
            generated_ids = blip_model.generate(**blip_inputs, max_length=50)
        
        caption = blip_processor.decode(generated_ids[0], skip_special_tokens=True)
        
        # Simple caption similarity (placeholder for full semantic analysis)
        caption_words = set(caption.lower().split())
        prompt_words = set(prompt.lower().split())
        caption_similarity = len(caption_words & prompt_words) / len(prompt_words | caption_words) if prompt_words else 0
        
        # Combined accuracy (simplified version of your TIFA)
        accuracy = clip_similarity * 0.7 + caption_similarity * 0.3
        
        print(f"      CLIP similarity: {clip_similarity:.3f}")
        print(f"      Generated caption: '{caption}'")
        print(f"      Caption similarity: {caption_similarity:.3f}")
        
        return accuracy
        
    except Exception as e:
        print(f"      ⚠️ Accuracy test error: {e}")
        return 0.5

def test_consistency_metric_standalone(image, reference_images, clip_processor, clip_model):
    """Test the consistency metric (replicate your ConsistencyManager logic)"""
    try:
        if not reference_images:
            return 1.0  # First image is always consistent
        
        # Get current image embedding
        inputs = clip_processor(images=image, return_tensors="pt")
        if torch.cuda.is_available():
            inputs = {k: v.to("cuda") for k, v in inputs.items()}
        
        with torch.no_grad():
            current_embedding = clip_model.get_image_features(**inputs).cpu().numpy()
        
        # Compare with reference images
        similarities = []
        for ref_img in reference_images:
            ref_inputs = clip_processor(images=ref_img, return_tensors="pt")
            if torch.cuda.is_available():
                ref_inputs = {k: v.to("cuda") for k, v in ref_inputs.items()}
            
            with torch.no_grad():
                ref_embedding = clip_model.get_image_features(**ref_inputs).cpu().numpy()
            
            similarity = np.dot(current_embedding.flatten(), ref_embedding.flatten()) / (
                np.linalg.norm(current_embedding) * np.linalg.norm(ref_embedding)
            )
            similarities.append(similarity)
        
        consistency = np.mean(similarities) if similarities else 1.0
        
        print(f"      Compared to {len(reference_images)} reference images")
        print(f"      Similarities: {[f'{s:.3f}' for s in similarities]}")
        
        return float(max(0.0, min(1.0, consistency)))
        
    except Exception as e:
        print(f"      ⚠️ Consistency test error: {e}")
        return 0.5

def calculate_weighted_autism_score(simplicity, accuracy, consistency):
    """Calculate the weighted autism suitability score (your exact formula)"""
    return (
        simplicity * 0.364 +    # 36.4%
        accuracy * 0.333 +      # 33.3%
        consistency * 0.303     # 30.3%
    )

def check_expectation(score, expected):
    """Check if score matches expectation (high/medium/low)"""
    if expected == "high":
        return score >= 0.7
    elif expected == "medium":
        return 0.4 <= score < 0.7
    elif expected == "low":
        return score < 0.4
    return True

# Helper functions for simplicity analysis
def count_people_simple(cv_image, face_cascade):
    """Simple person counting"""
    gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    return len(faces)

def analyze_background_complexity(cv_image):
    """Simple background complexity analysis"""
    gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 30, 100)
    edge_density = np.sum(edges > 0) / edges.size
    return max(0.0, 1.0 - edge_density * 10)  # Less edges = simpler

def analyze_color_complexity(cv_image):
    """Simple color complexity analysis"""
    rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
    unique_colors = len(np.unique(rgb_image.reshape(-1, 3), axis=0))
    return max(0.0, 1.0 - (unique_colors - 100) / 1000)  # Fewer unique colors = simpler

def analyze_character_clarity_simple(cv_image):
    """Simple character clarity analysis"""
    gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    edge_density = np.sum(edges > 0) / edges.size
    return min(1.0, edge_density * 5)  # More strong edges = clearer

def analyze_sensory_simple(cv_image):
    """Simple sensory analysis"""
    gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
    contrast_std = np.std(gray) / 255.0
    return max(0.0, 1.0 - abs(contrast_std - 0.3) * 2)  # Moderate contrast is good

def analyze_focus_simple(cv_image):
    """Simple focus analysis"""
    gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    center_region = gray[h//4:3*h//4, w//4:3*w//4]
    center_std = np.std(center_region)
    border_std = np.std(gray) - center_std
    return min(1.0, center_std / (border_std + 1))  # More detail in center = better focus

if __name__ == "__main__":
    test_evaluation_framework_standalone()