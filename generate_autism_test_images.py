#!/usr/bin/env python3

"""
SDXL Cartoon v7 Test Image Generator for Autism Framework Evaluation
Auto-generated on 2025-07-23 16:13:54

This script generates 25 test images across different autism-suitability categories:
- EXCELLENT: 6 prompts (should score 0.8+)
- GOOD: 7 prompts (should score 0.6-0.8)  
- MODERATE: 6 prompts (should score 0.4-0.6)
- POOR: 6 prompts (should score 0.0-0.4)
"""

import os
import torch
from diffusers import DiffusionPipeline
import json
from datetime import datetime

class AutismTestImageGenerator:
    def __init__(self):
        self.output_dir = "autism_evaluation_images"
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Load SDXL Cartoon v7 model
        print("Loading SDXL Cartoon v7 model...")
        self.pipe = DiffusionPipeline.from_pretrained(
            "artificialguybr/cartoon-xl-v7",
            torch_dtype=torch.float16,
            use_safetensors=True
        )
        
        if torch.cuda.is_available():
            self.pipe = self.pipe.to("cuda")
            print("Using GPU acceleration")
        else:
            print("Using CPU (will be slower)")
        
        # Generation settings optimized for consistent evaluation
        self.gen_settings = {
            "num_inference_steps": 25,
            "guidance_scale": 7.5,
            "width": 512,
            "height": 512,
            "generator": torch.manual_seed(42)  # Fixed seed for reproducibility
        }
        
        # Test prompts with expected outcomes
        self.test_prompts = [
            {
                        "category": "EXCELLENT",
                        "name": "01_perfect_single_character",
                        "prompt": "single happy child character, simple cartoon style, clean white background, bright friendly colors, clear facial features, standing pose",
                        "negative": "multiple people, crowd, busy background, complex details, realistic style",
                        "expected_people": 1,
                        "expected_bg": "simple",
                        "notes": "Perfect single character scenario"
            },
            {
                        "category": "EXCELLENT",
                        "name": "02_clean_portrait",
                        "prompt": "cartoon portrait of smiling boy, solid blue background, simple art style, clear lines, minimal colors",
                        "negative": "multiple characters, detailed background, complex scene, many colors",
                        "expected_people": 1,
                        "expected_bg": "solid",
                        "notes": "Clean portrait with solid background"
            },
            {
                        "category": "EXCELLENT",
                        "name": "03_simple_two_friends",
                        "prompt": "two cartoon children friends standing together, simple playground background, bright colors, clear character design, minimal details",
                        "negative": "crowd, busy scene, complex background, many characters, realistic style",
                        "expected_people": 2,
                        "expected_bg": "simple",
                        "notes": "Ideal two-character scenario"
            },
            {
                        "category": "EXCELLENT",
                        "name": "04_minimalist_character",
                        "prompt": "cute cartoon animal character, minimalist style, pastel colors, empty background, simple shapes",
                        "negative": "complex details, busy background, multiple animals, realistic rendering",
                        "expected_people": 1,
                        "expected_bg": "minimal",
                        "notes": "Minimalist character design"
            },
            {
                        "category": "EXCELLENT",
                        "name": "05_storybook_single",
                        "prompt": "single cartoon princess character, storybook illustration style, gentle colors, simple castle background, clear design",
                        "negative": "multiple princesses, complex castle, busy scene, many details",
                        "expected_people": 1,
                        "expected_bg": "simple",
                        "notes": "Storybook style single character"
            },
            {
                        "category": "GOOD",
                        "name": "06_two_chars_moderate_bg",
                        "prompt": "two cartoon students in classroom, simple desk and board background, bright educational scene, clear character focus",
                        "negative": "many students, crowded classroom, complex details, realistic style",
                        "expected_people": 2,
                        "expected_bg": "moderate",
                        "notes": "Two characters with moderate background"
            },
            {
                        "category": "GOOD",
                        "name": "07_single_with_objects",
                        "prompt": "cartoon chef character cooking, simple kitchen background with few utensils, colorful but organized scene",
                        "negative": "multiple chefs, cluttered kitchen, many objects, complex scene",
                        "expected_people": 1,
                        "expected_bg": "moderate",
                        "notes": "Single character with some background objects"
            },
            {
                        "category": "GOOD",
                        "name": "08_family_portrait",
                        "prompt": "cartoon family of two parents, simple living room background, warm colors, clear character design",
                        "negative": "large family, many people, complex room, busy details",
                        "expected_people": 2,
                        "expected_bg": "simple",
                        "notes": "Simple family portrait"
            },
            {
                        "category": "GOOD",
                        "name": "09_playground_scene",
                        "prompt": "two cartoon children on swing set, simple playground background, bright day, clear focus on characters",
                        "negative": "many children, busy playground, complex equipment, crowded scene",
                        "expected_people": 2,
                        "expected_bg": "moderate",
                        "notes": "Moderate playground scene"
            },
            {
                        "category": "GOOD",
                        "name": "10_pet_and_owner",
                        "prompt": "cartoon child with pet dog, simple park background, friendly scene, clear character focus",
                        "negative": "multiple pets, many people, busy park, complex scene",
                        "expected_people": 1,
                        "expected_bg": "simple",
                        "notes": "Child with pet scenario"
            },
            {
                        "category": "MODERATE",
                        "name": "11_three_characters",
                        "prompt": "three cartoon friends playing together, simple outdoor background, colorful but clear scene",
                        "negative": "large group, many people, complex background, busy scene",
                        "expected_people": 3,
                        "expected_bg": "simple",
                        "notes": "Three characters - testing person count threshold"
            },
            {
                        "category": "MODERATE",
                        "name": "12_busy_background_single",
                        "prompt": "single cartoon character in busy market scene, many colorful stalls and objects, detailed background",
                        "negative": "multiple main characters, extremely crowded, chaotic scene",
                        "expected_people": 1,
                        "expected_bg": "busy",
                        "notes": "Single character but busy background"
            },
            {
                        "category": "MODERATE",
                        "name": "13_classroom_group",
                        "prompt": "cartoon teacher with three students, detailed classroom with books and posters, educational scene",
                        "negative": "huge class, extremely busy, too many details",
                        "expected_people": 4,
                        "expected_bg": "busy",
                        "notes": "Small group in detailed setting"
            },
            {
                        "category": "MODERATE",
                        "name": "14_party_scene_small",
                        "prompt": "three cartoon children at birthday party, decorations and cake, colorful party background",
                        "negative": "large party, many guests, extremely busy, chaotic",
                        "expected_people": 3,
                        "expected_bg": "busy",
                        "notes": "Small party scene"
            },
            {
                        "category": "MODERATE",
                        "name": "15_sports_team_small",
                        "prompt": "four cartoon kids playing soccer, sports field background, action scene with clear character focus",
                        "negative": "full team, large crowd, stadium, complex scene",
                        "expected_people": 4,
                        "expected_bg": "moderate",
                        "notes": "Small sports group"
            },
            {
                        "category": "POOR",
                        "name": "16_large_crowd",
                        "prompt": "cartoon school assembly with many students and teachers, crowded auditorium, busy scene with lots of characters",
                        "negative": "empty scene, single character, simple background",
                        "expected_people": "8+",
                        "expected_bg": "chaotic",
                        "notes": "Large crowd scene - should score very low"
            },
            {
                        "category": "POOR",
                        "name": "17_carnival_chaos",
                        "prompt": "busy cartoon carnival with many people, rides, games, colorful chaos, lots of activity and detail",
                        "negative": "empty carnival, few people, simple scene",
                        "expected_people": "10+",
                        "expected_bg": "chaotic",
                        "notes": "Chaotic carnival scene"
            },
            {
                        "category": "POOR",
                        "name": "18_city_street_busy",
                        "prompt": "crowded cartoon city street with many people walking, cars, buildings, busy urban scene, lots of details",
                        "negative": "empty street, few people, simple scene",
                        "expected_people": "15+",
                        "expected_bg": "chaotic",
                        "notes": "Busy urban scene"
            },
            {
                        "category": "POOR",
                        "name": "19_concert_audience",
                        "prompt": "cartoon concert with large audience, many people cheering, stage with performers, crowded venue",
                        "negative": "empty venue, few people, simple scene",
                        "expected_people": "20+",
                        "expected_bg": "chaotic",
                        "notes": "Concert crowd scene"
            },
            {
                        "category": "POOR",
                        "name": "20_abstract_confusion",
                        "prompt": "abstract cartoon scene with unclear subjects, many overlapping shapes and colors, confusing composition, no clear focus",
                        "negative": "clear characters, simple scene, obvious subjects",
                        "expected_people": "unclear",
                        "expected_bg": "abstract",
                        "notes": "Abstract/confusing scene"
            },
            {
                        "category": "GOOD",
                        "name": "21_high_contrast",
                        "prompt": "single cartoon character in black and white style, high contrast, clear silhouette, minimal colors",
                        "negative": "many colors, complex scene, multiple characters",
                        "expected_people": 1,
                        "expected_bg": "simple",
                        "notes": "Testing high contrast detection"
            },
            {
                        "category": "EXCELLENT",
                        "name": "22_pastel_gentle",
                        "prompt": "cartoon character in soft pastel colors, gentle lighting, dreamy background, very soft and calming scene",
                        "negative": "bright harsh colors, busy scene, complex details",
                        "expected_people": 1,
                        "expected_bg": "simple",
                        "notes": "Testing gentle pastel colors"
            },
            {
                        "category": "MODERATE",
                        "name": "23_action_scene",
                        "prompt": "two cartoon superheroes in action pose, dynamic background with motion effects, colorful but clear characters",
                        "negative": "many superheroes, extremely busy, chaotic action",
                        "expected_people": 2,
                        "expected_bg": "moderate",
                        "notes": "Testing action/movement detection"
            },
            {
                        "category": "POOR",
                        "name": "24_tiny_characters",
                        "prompt": "many tiny cartoon characters scattered across large landscape, wide view, characters very small and hard to distinguish",
                        "negative": "large clear characters, simple scene, few people",
                        "expected_people": "many",
                        "expected_bg": "complex",
                        "notes": "Testing small character detection"
            },
            {
                        "category": "GOOD",
                        "name": "25_vehicle_scene",
                        "prompt": "cartoon child driving toy car, simple road background, clear character focus with vehicle element",
                        "negative": "many cars, busy traffic, complex scene, multiple drivers",
                        "expected_people": 1,
                        "expected_bg": "simple",
                        "notes": "Testing vehicle/object interaction"
            }
]
    
    def generate_all_test_images(self):
        """Generate all test images"""
        print(f"🚀 Generating {len(self.test_prompts)} test images for autism framework evaluation...")
        print("=" * 80)
        
        results = []
        
        for i, prompt_data in enumerate(self.test_prompts, 1):
            category = prompt_data["category"]
            name = prompt_data["name"]
            prompt = prompt_data["prompt"]
            negative = prompt_data["negative"]
            
            print(f"\n[{i:02d}/{len(self.test_prompts):02d}] {category}: {name}")
            print(f"📝 Prompt: {prompt[:80]}...")
            print(f"🚫 Negative: {negative[:60]}...")
            
            try:
                # Generate image
                image = self.pipe(
                    prompt=prompt,
                    negative_prompt=negative,
                    **self.gen_settings
                ).images[0]
                
                # Save image
                filename = f"{name}.png"
                image_path = os.path.join(self.output_dir, filename)
                image.save(image_path)
                
                results.append({
                    "index": i,
                    "category": category,
                    "name": name,
                    "prompt": prompt,
                    "negative": negative,
                    "expected_people": prompt_data["expected_people"],
                    "expected_bg": prompt_data["expected_bg"],
                    "notes": prompt_data["notes"],
                    "image_path": image_path,
                    "status": "success"
                })
                
                print(f"✅ Generated: {filename}")
                
            except Exception as e:
                print(f"❌ Failed: {e}")
                results.append({
                    "index": i,
                    "category": category,
                    "name": name,
                    "status": "failed",
                    "error": str(e)
                })
        
        # Save generation manifest
        self.save_generation_manifest(results)
        
        # Print summary
        successful = [r for r in results if r["status"] == "success"]
        failed = [r for r in results if r["status"] == "failed"]
        
        print(f"\n🎉 GENERATION COMPLETE")
        print("=" * 50)
        print(f"✅ Successful: {len(successful)}")
        print(f"❌ Failed: {len(failed)}")
        
        # Summary by category
        for category in ["EXCELLENT", "GOOD", "MODERATE", "POOR"]:
            count = len([r for r in successful if r["category"] == category])
            print(f"📊 {category}: {count} images")
        
        print(f"\n📁 Images saved to: {self.output_dir}")
        print(f"📋 Manifest saved to: {self.output_dir}/generation_manifest.json")
        
        return results
    
    def save_generation_manifest(self, results):
        """Save detailed manifest of generated images"""
        manifest = {
            "generation_date": datetime.now().isoformat(),
            "total_prompts": len(self.test_prompts),
            "successful_generations": len([r for r in results if r["status"] == "success"]),
            "failed_generations": len([r for r in results if r["status"] == "failed"]),
            "model": "artificialguybr/cartoon-xl-v7",
            "settings": self.gen_settings,
            "categories": {
                "EXCELLENT": {
                    "description": "Should score 0.8+ - ideal for autism storyboards",
                    "count": len([r for r in results if r.get("category") == "EXCELLENT" and r["status"] == "success"])
                },
                "GOOD": {
                    "description": "Should score 0.6-0.8 - suitable with minor issues", 
                    "count": len([r for r in results if r.get("category") == "GOOD" and r["status"] == "success"])
                },
                "MODERATE": {
                    "description": "Should score 0.4-0.6 - needs improvements",
                    "count": len([r for r in results if r.get("category") == "MODERATE" and r["status"] == "success"])
                },
                "POOR": {
                    "description": "Should score 0.0-0.4 - unsuitable for autism use",
                    "count": len([r for r in results if r.get("category") == "POOR" and r["status"] == "success"])
                }
            },
            "results": results
        }
        
        manifest_path = os.path.join(self.output_dir, "generation_manifest.json")
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)
    
    def create_evaluation_script(self):
        """Create script to evaluate generated images with the autism framework"""
        
        eval_script = f"""#!/usr/bin/env python3

# Import your existing autism analyzer
from scoring import RefinedAutismAnalyzer
import os
import json
from PIL import Image
import numpy as np

def evaluate_test_images():
    """Evaluate all generated test images with the autism framework"""
    
    # Load generation manifest
    with open('{self.output_dir}/generation_manifest.json', 'r') as f:
        manifest = json.load(f)
    
    # Initialize analyzer
    analyzer = RefinedAutismAnalyzer()
    if not analyzer.available:
        print("❌ Autism analyzer not available")
        return
    
    print(f"🔬 Evaluating {manifest['successful_generations']} generated test images...")
    print("=" * 80)
    
    evaluation_results = []
    
    for result in manifest['results']:
        if result['status'] != 'success':
            continue
            
        name = result['name']
        category = result['category']
        image_path = result['image_path']
        expected_people = result['expected_people']
        
        print(f"\n🔍 Evaluating: {name} ({category})")
        
        # Load and analyze image
        image = Image.open(image_path).convert('RGB')
        analysis = analyzer.analyze_autism_suitability(image)
        
        if 'error' not in analysis:
            autism_score = analysis['autism_suitability']
            person_count = analysis['person_count']['count']
            bg_score = analysis['background_simplicity']['score']
            color_score = analysis['color_appropriateness']['score']
            grade = analysis['autism_grade']
            
            print(f"  🧩 Autism Score: {autism_score:.3f} ({grade.split('(')[0].strip()})")
            print(f"  👥 People Detected: {person_count} (expected: {expected_people})")
            print(f"  🎨 Background: {bg_score:.3f}")
            print(f"  🌈 Colors: {color_score:.3f}")
            
            # Check if results match expectations
            category_ranges = {
                "EXCELLENT": (0.8, 1.0),
                "GOOD": (0.6, 0.8),
                "MODERATE": (0.4, 0.6),
                "POOR": (0.0, 0.4)
            }
            
            expected_min, expected_max = category_ranges[category]
            matches_expectation = expected_min <= autism_score <= expected_max
            
            print(f"  ✅ Expectation: {matches_expectation} ({expected_min}-{expected_max})")
            
            evaluation_results.append({
                "name": name,
                "category": category,
                "autism_score": autism_score,
                "person_count": person_count,
                "expected_people": expected_people,
                "bg_score": bg_score,
                "color_score": color_score,
                "grade": grade,
                "matches_expectation": matches_expectation,
                "analysis": analysis
            })
        else:
            print(f"  ❌ Analysis failed: {analysis['error']}")
    
    # Generate evaluation report
    generate_evaluation_report(evaluation_results)
    
    return evaluation_results

def generate_evaluation_report(results):
    """Generate comprehensive evaluation report"""
    
    print(f"\n📊 AUTISM FRAMEWORK EVALUATION REPORT")
    print("=" * 80)
    
    # Overall accuracy
    total_evaluated = len(results)
    correct_predictions = len([r for r in results if r['matches_expectation']])
    accuracy = correct_predictions / total_evaluated if total_evaluated > 0 else 0
    
    print(f"🎯 Overall Accuracy: {accuracy:.2%} ({correct_predictions}/{total_evaluated})")
    
    # Category breakdown
    for category in ["EXCELLENT", "GOOD", "MODERATE", "POOR"]:
        category_results = [r for r in results if r['category'] == category]
        if category_results:
            category_correct = len([r for r in category_results if r['matches_expectation']])
            category_accuracy = category_correct / len(category_results)
            avg_score = np.mean([r['autism_score'] for r in category_results])
            
            print(f"\n📈 {category} Category:")
            print(f"   Accuracy: {category_accuracy:.2%} ({category_correct}/{len(category_results)})")
            print(f"   Avg Score: {avg_score:.3f}")
            
            # Show best and worst performers
            category_results.sort(key=lambda x: x['autism_score'], reverse=True)
            print(f"   Best: {category_results[0]['name']} ({category_results[0]['autism_score']:.3f})")
            print(f"   Worst: {category_results[-1]['name']} ({category_results[-1]['autism_score']:.3f})")
    
    # Person counting accuracy
    person_count_correct = 0
    for r in results:
        expected = r['expected_people']
        detected = r['person_count']
        
        if isinstance(expected, int):
            if detected == expected:
                person_count_correct += 1
        elif expected == "unclear":
            # For unclear cases, any reasonable detection is acceptable
            person_count_correct += 1
        elif "+" in str(expected):
            # For "8+" cases, accept if detected >= threshold
            threshold = int(expected.replace("+", ""))
            if detected >= threshold:
                person_count_correct += 1
    
    person_accuracy = person_count_correct / total_evaluated if total_evaluated > 0 else 0
    print(f"\n👥 Person Counting Accuracy: {person_accuracy:.2%} ({person_count_correct}/{total_evaluated})")
    
    # Save detailed results
    with open('{self.output_dir}/evaluation_results.json', 'w') as f:
        json.dump({
            "evaluation_date": "{datetime.now().isoformat()}",
            "overall_accuracy": accuracy,
            "person_counting_accuracy": person_accuracy,
            "total_evaluated": total_evaluated,
            "results": results
        }, f, indent=2)
    
    print(f"\n💾 Detailed results saved to: {self.output_dir}/evaluation_results.json")

if __name__ == "__main__":
    evaluate_test_images()
"""
        
        eval_script_path = "evaluate_autism_framework.py"
        with open(eval_script_path, 'w') as f:
            f.write(eval_script)
        
        print(f"📋 Evaluation script created: {eval_script_path}")

if __name__ == "__main__":
    generator = AutismTestImageGenerator()
    results = generator.generate_all_test_images()
    generator.create_evaluation_script()
    
    print(f"\n🎯 Next steps:")
    print(f"1. Run this script to generate test images")
    print(f"2. Run 'python evaluate_autism_framework.py' to evaluate with your framework")
    print(f"3. Review results in {generator.output_dir}/evaluation_results.json")
