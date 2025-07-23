#!/usr/bin/env python3
"""
Storyboard Sequence Validation Test
Tests true consistency across storyboard sequences with and without IP-Adapter
Validates that evaluation framework correctly measures character drift
"""

import os
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import time
from diffusers import StableDiffusionXLPipeline
import matplotlib.pyplot as plt

# Import your evaluation modules
from consistency_manager import ConsistencyManager
from quality_evaluator import QualityEvaluator
from image_complexity_analyser import AutismFriendlyImageAnalyzer

class SequenceAwareEvaluator:
    """Sequence-aware evaluator that properly handles first vs subsequent images"""
    
    def __init__(self):
        print("🔄 Loading Sequence-Aware Evaluator...")
        
        self.consistency_manager = ConsistencyManager()
        self.quality_evaluator = QualityEvaluator()
        self.autism_analyzer = AutismFriendlyImageAnalyzer()
        
        # Track sequence state
        self.sequence_position = 0
        self.sequence_scores = []
        
        print("✅ Sequence-Aware Evaluator ready")
    
    def reset_sequence(self):
        """Reset for new storyboard sequence"""
        self.sequence_position = 0
        self.sequence_scores = []
        self.consistency_manager.reset_memory()
        print("🔄 Sequence reset - starting new storyboard")
    
    def evaluate_sequence_image(self, image, prompt, is_reference_set=False):
        """Evaluate image with sequence-aware scoring"""
        self.sequence_position += 1
        is_first_image = (self.sequence_position == 1)
        
        print(f"\n📊 EVALUATING SEQUENCE IMAGE {self.sequence_position}")
        print(f"   Prompt: {prompt[:60]}...")
        print(f"   Is first image: {is_first_image}")
        print(f"   IP-Adapter reference set: {is_reference_set}")
        
        # Core evaluations
        results = {}
        
        # 1. Accuracy (TIFA) - always evaluated
        tifa_result = self.quality_evaluator.calculate_tifa_score(image, prompt)
        results["accuracy"] = tifa_result["score"]
        results["tifa_details"] = tifa_result["details"]
        
        # 2. Simplicity (Autism-friendly) - always evaluated  
        if self.autism_analyzer and self.autism_analyzer.available:
            autism_result = self.autism_analyzer.analyze_autism_suitability(image)
            results["simplicity"] = autism_result["autism_suitability"]
            results["autism_details"] = autism_result
        else:
            results["simplicity"] = 0.5
            results["autism_details"] = {"error": "Autism analyzer not available"}
        
        # 3. Consistency - ONLY for images 2+
        if is_first_image:
            # First image: no consistency score possible
            results["consistency"] = None
            results["consistency_note"] = "Reference image - no consistency measurement"
            
            # Store as reference for future consistency
            self.consistency_manager.store_selected_image(image, prompt, results["accuracy"], 0)
            
        else:
            # Subsequent images: measure consistency against previous images
            image_embedding = self.consistency_manager.get_image_embedding(image)
            
            if image_embedding is not None:
                # Consistency with previous images
                consistency_score = self.consistency_manager.calculate_consistency_score(image_embedding)
                results["consistency"] = consistency_score
                results["consistency_note"] = f"Measured against {self.sequence_position-1} previous images"
                
                # Store this image for future consistency measurements
                self.consistency_manager.store_selected_image(image, prompt, results["accuracy"], self.sequence_position-1)
            else:
                results["consistency"] = 0.0
                results["consistency_note"] = "Consistency measurement failed"
        
        # 4. Calculate sequence-aware overall score
        overall_score = self.calculate_sequence_aware_score(results, is_first_image)
        results["overall_score"] = overall_score
        results["sequence_position"] = self.sequence_position
        
        # Log results
        print(f"   🎯 Accuracy (TIFA): {results['accuracy']:.3f}")
        print(f"   🧩 Simplicity (Autism): {results['simplicity']:.3f}")
        
        if results["consistency"] is not None:
            print(f"   🔄 Consistency: {results['consistency']:.3f}")
        else:
            print(f"   🔄 Consistency: {results['consistency_note']}")
        
        print(f"   📊 Overall Score: {overall_score:.3f}")
        
        # Store for sequence analysis
        self.sequence_scores.append(results)
        
        return results
    
    def calculate_sequence_aware_score(self, results, is_first_image):
        """Calculate overall score with sequence-aware weighting"""
        accuracy = results["accuracy"]
        simplicity = results["simplicity"]
        consistency = results["consistency"]
        
        if is_first_image:
            # First image: accuracy + simplicity only
            weights = {"accuracy": 0.6, "simplicity": 0.4}
            overall_score = (accuracy * weights["accuracy"] + 
                           simplicity * weights["simplicity"])
            
        else:
            # Subsequent images: accuracy + simplicity + consistency
            weights = {"accuracy": 0.4, "simplicity": 0.4, "consistency": 0.2}
            consistency_value = consistency if consistency is not None else 0.5
            overall_score = (accuracy * weights["accuracy"] + 
                           simplicity * weights["simplicity"] + 
                           consistency_value * weights["consistency"])
        
        return overall_score
    
    def get_sequence_report(self):
        """Generate comprehensive sequence analysis report"""
        if not self.sequence_scores:
            return {"error": "No images evaluated in sequence"}
        
        # Extract metrics across sequence
        accuracy_scores = [s["accuracy"] for s in self.sequence_scores]
        simplicity_scores = [s["simplicity"] for s in self.sequence_scores]
        consistency_scores = [s["consistency"] for s in self.sequence_scores if s["consistency"] is not None]
        overall_scores = [s["overall_score"] for s in self.sequence_scores]
        
        # Calculate sequence statistics
        report = {
            "sequence_length": len(self.sequence_scores),
            "accuracy": {
                "scores": accuracy_scores,
                "average": np.mean(accuracy_scores),
                "trend": "stable" if np.std(accuracy_scores) < 0.1 else "variable"
            },
            "simplicity": {
                "scores": simplicity_scores,
                "average": np.mean(simplicity_scores),
                "trend": "stable" if np.std(simplicity_scores) < 0.1 else "variable"
            },
            "consistency": {
                "scores": consistency_scores,
                "average": np.mean(consistency_scores) if consistency_scores else None,
                "trend": self.analyze_consistency_trend(consistency_scores)
            },
            "overall": {
                "scores": overall_scores,
                "average": np.mean(overall_scores),
                "trend": self.analyze_overall_trend(overall_scores)
            }
        }
        
        return report
    
    def analyze_consistency_trend(self, consistency_scores):
        """Analyze if consistency is improving, degrading, or stable"""
        if len(consistency_scores) < 2:
            return "insufficient_data"
        
        # Linear trend analysis
        x = np.arange(len(consistency_scores))
        slope = np.polyfit(x, consistency_scores, 1)[0]
        
        if slope > 0.05:
            return "improving"
        elif slope < -0.05:
            return "degrading"
        else:
            return "stable"
    
    def analyze_overall_trend(self, overall_scores):
        """Analyze overall score trend across sequence"""
        if len(overall_scores) < 2:
            return "insufficient_data"
        
        x = np.arange(len(overall_scores))
        slope = np.polyfit(x, overall_scores, 1)[0]
        
        if slope > 0.03:
            return "improving"
        elif slope < -0.03:
            return "degrading"
        else:
            return "stable"


class StoryboardSequenceTest:
    """Main test class for validating storyboard sequence evaluation"""
    
    def __init__(self, model_path):
        print("🎬 Loading Storyboard Sequence Test...")
        
        self.model_path = model_path
        self.pipe = None
        self.evaluator = SequenceAwareEvaluator()
        
        # Load pipeline
        self._load_pipeline()
        
        print("✅ Storyboard Sequence Test ready")
    
    def _load_pipeline(self):
        """Load SDXL pipeline"""
        print(f"🎨 Loading SDXL pipeline from: {self.model_path}")
        
        try:
            self.pipe = StableDiffusionXLPipeline.from_single_file(
                self.model_path,
                torch_dtype=torch.float16,
                use_safetensors=True,
                variant="fp16"
            ).to("cuda")
            
            self.pipe.enable_vae_tiling()
            self.pipe.enable_model_cpu_offload()
            
            # Try to load IP-Adapter
            try:
                self.pipe.load_ip_adapter(
                    "h94/IP-Adapter",
                    subfolder="sdxl_models", 
                    weight_name="ip-adapter_sdxl.bin",
                    torch_dtype=torch.float16
                )
                self.ip_adapter_available = True
                print("✅ IP-Adapter loaded successfully")
            except Exception as e:
                print(f"⚠️ IP-Adapter not available: {e}")
                self.ip_adapter_available = False
            
            print("✅ SDXL pipeline loaded")
            
        except Exception as e:
            print(f"❌ Failed to load pipeline: {e}")
            self.pipe = None
    
    def create_test_scenarios(self):
        """Create test scenarios for sequence validation"""
        
        scenarios = [
            {
                "name": "Simple Boy Story",
                "character": "Alex (brown hair, blue shirt)",
                "prompts": [
                    "young boy Alex with brown hair wearing blue shirt, cartoon style, simple clean background",
                    "same boy Alex with brown hair and blue shirt brushing teeth in bathroom, cartoon style, clean background",
                    "same boy Alex with brown hair and blue shirt eating cereal at kitchen table, cartoon style, simple background",
                    "same boy Alex with brown hair and blue shirt reading book in bedroom, cartoon style, clean background"
                ],
                "expected": "High consistency with IP-Adapter, degrading without"
            },
            {
                "name": "Simple Girl Story", 
                "character": "Emma (blonde hair, red dress)",
                "prompts": [
                    "young girl Emma with blonde hair wearing red dress, cartoon style, simple clean background",
                    "same girl Emma with blonde hair and red dress playing with toy, cartoon style, clean background", 
                    "same girl Emma with blonde hair and red dress smiling happily, cartoon style, simple background",
                    "same girl Emma with blonde hair and red dress waving goodbye, cartoon style, clean background"
                ],
                "expected": "High consistency with IP-Adapter, degrading without"
            }
        ]
        
        return scenarios
    
    def run_sequence_test(self, scenario, use_ip_adapter=True, output_dir="sequence_validation"):
        """Run a complete sequence test"""
        print(f"\n🎬 RUNNING SEQUENCE TEST: {scenario['name']}")
        print(f"   Character: {scenario['character']}")
        print(f"   IP-Adapter: {'✅ Enabled' if use_ip_adapter else '❌ Disabled'}")
        print(f"   Prompts: {len(scenario['prompts'])}")
        
        if not self.pipe:
            print("❌ Pipeline not available")
            return None
        
        # Create output directory
        test_name = f"{scenario['name'].replace(' ', '_').lower()}_{'with_ip' if use_ip_adapter else 'without_ip'}"
        test_dir = os.path.join(output_dir, test_name)
        os.makedirs(test_dir, exist_ok=True)
        
        # Reset evaluator for new sequence
        self.evaluator.reset_sequence()
        
        # Generation settings
        gen_settings = {
            "height": 1024,
            "width": 1024,
            "num_inference_steps": 25,
            "guidance_scale": 7.0,
            "num_images_per_prompt": 1
        }
        
        sequence_results = []
        character_reference = None
        
        # Generate and evaluate each image in sequence
        for i, prompt in enumerate(scenario["prompts"]):
            is_first_image = (i == 0)
            
            print(f"\n📸 Generating Image {i+1}/{len(scenario['prompts'])}")
            print(f"   Prompt: {prompt[:70]}...")
            
            # Generate image
            start_time = time.time()
            
            try:
                if use_ip_adapter and self.ip_adapter_available and character_reference is not None:
                    # Use IP-Adapter with character reference
                    self.pipe.set_ip_adapter_scale(0.6)
                    result = self.pipe(
                        prompt=prompt,
                        ip_adapter_image=character_reference,
                        negative_prompt="multiple people, crowd, blurry, low quality",
                        **gen_settings
                    )
                else:
                    # Generate without IP-Adapter
                    if use_ip_adapter and self.ip_adapter_available:
                        self.pipe.set_ip_adapter_scale(0.0)  # Disable IP-Adapter
                    
                    result = self.pipe(
                        prompt=prompt,
                        negative_prompt="multiple people, crowd, blurry, low quality", 
                        **gen_settings
                    )
                
                image = result.images[0]
                gen_time = time.time() - start_time
                
                # Set first image as character reference for IP-Adapter
                if is_first_image and use_ip_adapter and self.ip_adapter_available:
                    character_reference = image
                    print(f"   🎭 Set as character reference for IP-Adapter")
                
                # Save image
                filename = f"frame_{i+1:02d}.png"
                image_path = os.path.join(test_dir, filename)
                image.save(image_path)
                
                # Evaluate with sequence-aware scoring
                eval_result = self.evaluator.evaluate_sequence_image(
                    image, 
                    prompt, 
                    is_reference_set=(character_reference is not None)
                )
                
                # Store results
                frame_result = {
                    "frame_number": i + 1,
                    "prompt": prompt,
                    "image": image,
                    "image_path": image_path,
                    "generation_time": gen_time,
                    "evaluation": eval_result,
                    "is_first_image": is_first_image
                }
                
                sequence_results.append(frame_result)
                
                print(f"   ✅ Generated and evaluated in {gen_time:.1f}s")
                
            except Exception as e:
                print(f"   ❌ Generation failed: {e}")
                continue
        
        # Get sequence analysis report
        sequence_report = self.evaluator.get_sequence_report()
        
        # Save annotated sequence
        self._save_annotated_sequence(sequence_results, test_dir, use_ip_adapter)
        
        # Generate sequence report
        self._save_sequence_report(sequence_results, sequence_report, test_dir, scenario, use_ip_adapter)
        
        return {
            "scenario": scenario,
            "use_ip_adapter": use_ip_adapter,
            "results": sequence_results,
            "sequence_report": sequence_report,
            "output_dir": test_dir
        }
    
    def _save_annotated_sequence(self, sequence_results, output_dir, use_ip_adapter):
        """Save sequence with evaluation annotations"""
        for result in sequence_results:
            frame_num = result["frame_number"]
            image = result["image"]
            evaluation = result["evaluation"]
            
            # Create annotated version
            annotated = image.copy()
            draw = ImageDraw.Draw(annotated)
            
            # Try to load font
            try:
                font_large = ImageFont.truetype("arial.ttf", 28)
                font_medium = ImageFont.truetype("arial.ttf", 20)
                font_small = ImageFont.truetype("arial.ttf", 16)
            except:
                font_large = font_medium = font_small = ImageFont.load_default()
            
            # Create overlay
            overlay_height = 160
            overlay = Image.new('RGBA', (image.width, overlay_height), (0, 0, 0, 200))
            annotated.paste(overlay, (0, image.height - overlay_height), overlay)
            
            # Draw annotations
            y = image.height - overlay_height + 10
            
            # Frame info
            ip_status = "WITH IP-ADAPTER" if use_ip_adapter else "WITHOUT IP-ADAPTER"
            title = f"FRAME {frame_num} - {ip_status}"
            draw.text((15, y), title, fill=(255, 255, 255), font=font_large)
            y += 35
            
            # Scores
            accuracy = evaluation["accuracy"]
            simplicity = evaluation["simplicity"] 
            consistency = evaluation["consistency"]
            overall = evaluation["overall_score"]
            
            # Accuracy and simplicity (always present)
            scores_text = f"🎯 Accuracy: {accuracy:.3f}  |  🧩 Simplicity: {simplicity:.3f}"
            draw.text((15, y), scores_text, fill=(255, 255, 255), font=font_medium)
            y += 28
            
            # Consistency (only for frames 2+)
            if consistency is not None:
                consistency_color = (0, 255, 0) if consistency > 0.7 else (255, 255, 0) if consistency > 0.5 else (255, 0, 0)
                consistency_text = f"🔄 Consistency: {consistency:.3f}"
            else:
                consistency_color = (128, 128, 128)
                consistency_text = f"🔄 Reference Frame"
            
            draw.text((15, y), consistency_text, fill=consistency_color, font=font_medium)
            
            # Overall score
            overall_color = (0, 255, 0) if overall > 0.7 else (255, 255, 0) if overall > 0.5 else (255, 0, 0)
            overall_text = f"📊 Overall: {overall:.3f}"
            draw.text((350, y), overall_text, fill=overall_color, font=font_medium)
            y += 25
            
            # Sequence position note
            position_text = f"Position {frame_num} in sequence"
            draw.text((15, y), position_text, fill=(180, 180, 180), font=font_small)
            
            # Save annotated image
            annotated_filename = f"frame_{frame_num:02d}_annotated.png"
            annotated_path = os.path.join(output_dir, annotated_filename)
            annotated.save(annotated_path)
    
    def _save_sequence_report(self, sequence_results, sequence_report, output_dir, scenario, use_ip_adapter):
        """Save detailed sequence analysis report"""
        report_path = os.path.join(output_dir, "sequence_report.txt")
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("STORYBOARD SEQUENCE VALIDATION REPORT\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"Scenario: {scenario['name']}\n")
            f.write(f"Character: {scenario['character']}\n") 
            f.write(f"IP-Adapter: {'Enabled' if use_ip_adapter else 'Disabled'}\n")
            f.write(f"Sequence Length: {len(sequence_results)} frames\n")
            f.write(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Frame-by-frame results
            f.write("FRAME-BY-FRAME ANALYSIS:\n")
            f.write("-" * 30 + "\n\n")
            
            for result in sequence_results:
                frame_num = result["frame_number"]
                evaluation = result["evaluation"]
                
                f.write(f"FRAME {frame_num}:\n")
                f.write(f"Prompt: {result['prompt']}\n")
                f.write(f"Generation Time: {result['generation_time']:.1f}s\n")
                f.write(f"Sequence Position: {evaluation['sequence_position']}\n")
                f.write(f"Is First Image: {result['is_first_image']}\n\n")
                
                f.write("EVALUATION SCORES:\n")
                f.write(f"  Accuracy (TIFA): {evaluation['accuracy']:.3f}\n")
                f.write(f"  Simplicity (Autism): {evaluation['simplicity']:.3f}\n")
                
                if evaluation['consistency'] is not None:
                    f.write(f"  Consistency (CLIP): {evaluation['consistency']:.3f}\n")
                else:
                    f.write(f"  Consistency: {evaluation['consistency_note']}\n")
                
                f.write(f"  Overall Score: {evaluation['overall_score']:.3f}\n\n")
                f.write("-" * 30 + "\n\n")
            
            # Sequence-level analysis
            f.write("SEQUENCE-LEVEL ANALYSIS:\n")
            f.write("-" * 25 + "\n\n")
            
            if sequence_report.get("error"):
                f.write(f"Error: {sequence_report['error']}\n")
            else:
                acc_data = sequence_report["accuracy"]
                simp_data = sequence_report["simplicity"] 
                cons_data = sequence_report["consistency"]
                overall_data = sequence_report["overall"]
                
                f.write(f"ACCURACY ACROSS SEQUENCE:\n")
                f.write(f"  Average: {acc_data['average']:.3f}\n")
                f.write(f"  Trend: {acc_data['trend']}\n")
                f.write(f"  Scores: {[f'{s:.3f}' for s in acc_data['scores']]}\n\n")
                
                f.write(f"SIMPLICITY ACROSS SEQUENCE:\n")
                f.write(f"  Average: {simp_data['average']:.3f}\n")
                f.write(f"  Trend: {simp_data['trend']}\n")
                f.write(f"  Scores: {[f'{s:.3f}' for s in simp_data['scores']]}\n\n")
                
                f.write(f"CONSISTENCY ACROSS SEQUENCE:\n")
                if cons_data['average'] is not None:
                    f.write(f"  Average: {cons_data['average']:.3f}\n")
                    f.write(f"  Trend: {cons_data['trend']}\n")
                    f.write(f"  Scores: {[f'{s:.3f}' for s in cons_data['scores']]}\n")
                else:
                    f.write(f"  No consistency data (single image or measurement failed)\n")
                f.write("\n")
                
                f.write(f"OVERALL SEQUENCE PERFORMANCE:\n")
                f.write(f"  Average: {overall_data['average']:.3f}\n")
                f.write(f"  Trend: {overall_data['trend']}\n")
                f.write(f"  Scores: {[f'{s:.3f}' for s in overall_data['scores']]}\n\n")
            
            # Expected vs actual
            f.write("VALIDATION:\n")
            f.write("-" * 10 + "\n")
            f.write(f"Expected: {scenario['expected']}\n")
            f.write("Actual: ")
            
            if sequence_report.get("consistency", {}).get("average"):
                avg_cons = sequence_report["consistency"]["average"]
                trend = sequence_report["consistency"]["trend"]
                
                if use_ip_adapter:
                    if avg_cons > 0.7:
                        f.write("✅ High consistency maintained with IP-Adapter\n")
                    else:
                        f.write("⚠️ Lower than expected consistency with IP-Adapter\n")
                else:
                    if trend == "degrading":
                        f.write("✅ Expected consistency degradation without IP-Adapter\n")
                    else:
                        f.write("⚠️ Unexpected consistency pattern without IP-Adapter\n")
            else:
                f.write("❓ Insufficient data for validation\n")
        
        print(f"📄 Sequence report saved: {report_path}")
    
    def run_comparative_test(self, scenario, output_dir="sequence_validation"):
        """Run both with and without IP-Adapter for comparison"""
        print(f"\n🔄 COMPARATIVE TEST: {scenario['name']}")
        print("Testing same scenario with and without IP-Adapter...")
        
        # Test with IP-Adapter
        print(f"\n=== WITH IP-ADAPTER ===")
        result_with_ip = self.run_sequence_test(scenario, use_ip_adapter=True, output_dir=output_dir)
        
        # Test without IP-Adapter
        print(f"\n=== WITHOUT IP-ADAPTER ===")
        result_without_ip = self.run_sequence_test(scenario, use_ip_adapter=False, output_dir=output_dir)
        
        # Compare results
        comparison = self._compare_results(result_with_ip, result_without_ip, output_dir, scenario)
        
        return {
            "scenario": scenario,
            "with_ip_adapter": result_with_ip,
            "without_ip_adapter": result_without_ip,
            "comparison": comparison
        }
    
    def _compare_results(self, result_with_ip, result_without_ip, output_dir, scenario):
        """Compare results between with and without IP-Adapter"""
        print(f"\n📊 COMPARING RESULTS...")
        
        if not result_with_ip or not result_without_ip:
            print("❌ Cannot compare - one or both tests failed")
            return {"error": "Comparison failed"}
        
        # Extract consistency scores (excluding first frame)
        with_ip_consistency = []
        without_ip_consistency = []
        
        for result in result_with_ip["results"][1:]:  # Skip first frame
            if result["evaluation"]["consistency"] is not None:
                with_ip_consistency.append(result["evaluation"]["consistency"])
        
        for result in result_without_ip["results"][1:]:  # Skip first frame
            if result["evaluation"]["consistency"] is not None:
                without_ip_consistency.append(result["evaluation"]["consistency"])
        
        comparison = {
            "scenario_name": scenario["name"],
            "frames_compared": min(len(with_ip_consistency), len(without_ip_consistency)),
            "with_ip_adapter": {
                "avg_consistency": np.mean(with_ip_consistency) if with_ip_consistency else None,
                "consistency_trend": result_with_ip["sequence_report"]["consistency"]["trend"],
                "scores": with_ip_consistency
            },
            "without_ip_adapter": {
                "avg_consistency": np.mean(without_ip_consistency) if without_ip_consistency else None,
                "consistency_trend": result_without_ip["sequence_report"]["consistency"]["trend"],
                "scores": without_ip_consistency
            }
        }
        
        # Calculate improvement
        if comparison["with_ip_adapter"]["avg_consistency"] and comparison["without_ip_adapter"]["avg_consistency"]:
            improvement = comparison["with_ip_adapter"]["avg_consistency"] - comparison["without_ip_adapter"]["avg_consistency"]
            comparison["ip_adapter_improvement"] = improvement
            comparison["improvement_percentage"] = (improvement / comparison["without_ip_adapter"]["avg_consistency"]) * 100
        else:
            comparison["ip_adapter_improvement"] = None
            comparison["improvement_percentage"] = None
        
        # Save comparison report
        self._save_comparison_report(comparison, output_dir, scenario)
        
        # Print summary
        print(f"📊 COMPARISON SUMMARY:")
        if comparison["ip_adapter_improvement"]:
            print(f"   With IP-Adapter: {comparison['with_ip_adapter']['avg_consistency']:.3f}")
            print(f"   Without IP-Adapter: {comparison['without_ip_adapter']['avg_consistency']:.3f}")
            print(f"   Improvement: {comparison['ip_adapter_improvement']:+.3f} ({comparison['improvement_percentage']:+.1f}%)")
            
            if comparison["ip_adapter_improvement"] > 0.1:
                print("   ✅ IP-Adapter provides significant consistency improvement")
            elif comparison["ip_adapter_improvement"] > 0.05:
                print("   👍 IP-Adapter provides moderate consistency improvement")
            else:
                print("   ⚠️ IP-Adapter provides minimal consistency improvement")
        else:
            print("   ❓ Could not calculate improvement (insufficient data)")
        
        return comparison
    
    def _save_comparison_report(self, comparison, output_dir, scenario):
        """Save comparison report"""
        report_path = os.path.join(output_dir, f"{scenario['name'].replace(' ', '_').lower()}_comparison.txt")
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("IP-ADAPTER EFFECTIVENESS COMPARISON\n")
            f.write("=" * 40 + "\n\n")
            
            f.write(f"Scenario: {comparison['scenario_name']}\n")
            f.write(f"Frames Compared: {comparison['frames_compared']}\n")
            f.write(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("CONSISTENCY COMPARISON:\n")
            f.write("-" * 25 + "\n\n")
            
            with_ip = comparison["with_ip_adapter"]
            without_ip = comparison["without_ip_adapter"]
            
            f.write("WITH IP-ADAPTER:\n")
            f.write(f"  Average Consistency: {with_ip['avg_consistency']:.3f}\n")
            f.write(f"  Consistency Trend: {with_ip['consistency_trend']}\n")
            f.write(f"  Frame Scores: {[f'{s:.3f}' for s in with_ip['scores']]}\n\n")
            
            f.write("WITHOUT IP-ADAPTER:\n")
            f.write(f"  Average Consistency: {without_ip['avg_consistency']:.3f}\n")
            f.write(f"  Consistency Trend: {without_ip['consistency_trend']}\n")
            f.write(f"  Frame Scores: {[f'{s:.3f}' for s in without_ip['scores']]}\n\n")
            
            f.write("IP-ADAPTER EFFECTIVENESS:\n")
            f.write("-" * 25 + "\n")
            
            if comparison["ip_adapter_improvement"] is not None:
                f.write(f"Consistency Improvement: {comparison['ip_adapter_improvement']:+.3f}\n")
                f.write(f"Percentage Improvement: {comparison['improvement_percentage']:+.1f}%\n\n")
                
                if comparison["ip_adapter_improvement"] > 0.1:
                    f.write("✅ CONCLUSION: IP-Adapter provides significant consistency benefits\n")
                elif comparison["ip_adapter_improvement"] > 0.05:
                    f.write("👍 CONCLUSION: IP-Adapter provides moderate consistency benefits\n")
                elif comparison["ip_adapter_improvement"] > 0:
                    f.write("⚠️ CONCLUSION: IP-Adapter provides minimal consistency benefits\n")
                else:
                    f.write("❌ CONCLUSION: IP-Adapter does not improve consistency\n")
            else:
                f.write("❓ CONCLUSION: Could not measure IP-Adapter effectiveness\n")
        
        print(f"📄 Comparison report saved: {report_path}")


def find_model_path():
    """Find the model file"""
    candidates = [
        "realcartoonxl_v7.safetensors",
        "../realcartoonxl_v7.safetensors",
        "../models/realcartoonxl_v7.safetensors",
        "models/realcartoonxl_v7.safetensors"
    ]
    
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    return None


def main():
    """Main function for sequence validation test"""
    print("🎬 STORYBOARD SEQUENCE VALIDATION TEST")
    print("=" * 60)
    print("Testing evaluation framework with true storyboard sequences")
    print("Validating IP-Adapter impact on character consistency")
    
    # Find model
    model_path = find_model_path()
    if not model_path:
        print("❌ RealCartoonXL v7 model not found!")
        return
    
    print(f"✅ Found model: {model_path}")
    
    # Initialize test
    test = StoryboardSequenceTest(model_path)
    
    if not test.pipe:
        print("❌ Failed to load pipeline")
        return
    
    # Create test scenarios
    scenarios = test.create_test_scenarios()
    
    print(f"\n📋 TEST SCENARIOS:")
    for i, scenario in enumerate(scenarios, 1):
        print(f"   {i}. {scenario['name']} - {scenario['character']}")
        print(f"      Frames: {len(scenario['prompts'])}")
        print(f"      Expected: {scenario['expected']}")
    
    # Run comparative tests
    all_results = []
    
    for scenario in scenarios:
        print(f"\n" + "="*70)
        try:
            result = test.run_comparative_test(scenario)
            all_results.append(result)
            
        except Exception as e:
            print(f"❌ Test failed for {scenario['name']}: {e}")
            import traceback
            traceback.print_exc()
    
    # Final summary
    print(f"\n🎉 SEQUENCE VALIDATION COMPLETE!")
    print(f"📁 Results saved to: sequence_validation/")
    print(f"📊 Scenarios tested: {len(all_results)}")
    
    print(f"\n💡 KEY VALIDATION POINTS:")
    print("   • Does consistency degrade without IP-Adapter?")
    print("   • Does IP-Adapter maintain character consistency?")
    print("   • Are first images properly handled (no consistency score)?")
    print("   • Do subsequent images show meaningful consistency scores?")
    print("   • Does the evaluation framework detect the difference?")
    
    print(f"\n📄 Check individual reports in sequence_validation/ for detailed analysis")


if __name__ == "__main__":
    main()