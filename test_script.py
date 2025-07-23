"""
Comprehensive Deterministic Storyboard Test Script
Tests the complete pipeline with fixed seeds, IP-Adapter consistency, and baked VAE compatibility
Based on the provided deterministic generation and cartoon pipeline systems
"""

import os
import time
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import json
from pathlib import Path
import traceback

# Import the modular components
CartoonPipelineWithIPAdapter = None
generate_enhanced_test_report = None
run_baked_vae_diagnostics = None

MODULES_AVAILABLE = False
MISSING_MODULES = []

try:
    from cartoon_pipeline import CartoonPipelineWithIPAdapter
    try:
        from cartoon_pipeline import generate_enhanced_test_report, run_baked_vae_diagnostics
    except ImportError:
        def generate_enhanced_test_report(*args, **kwargs):
            print("📋 Enhanced test report function not available")
        def run_baked_vae_diagnostics(*args, **kwargs):
            print("🔬 Baked VAE diagnostics function not available")
    
    print("✅ cartoon_pipeline imported successfully")
    MODULES_AVAILABLE = True
except ImportError as e:
    print(f"❌ Failed to import cartoon_pipeline: {e}")
    MISSING_MODULES.append("cartoon_pipeline.py")

try:
    from consistency_manager import ConsistencyManager
    print("✅ consistency_manager imported successfully")
except ImportError as e:
    print(f"❌ Failed to import consistency_manager: {e}")
    MISSING_MODULES.append("consistency_manager.py")

try:
    from quality_evaluator import QualityEvaluator
    print("✅ quality_evaluator imported successfully")
except ImportError as e:
    print(f"❌ Failed to import quality_evaluator: {e}")
    MISSING_MODULES.append("quality_evaluator.py")

try:
    from ip_adapter_manager import IPAdapterManager
    print("✅ ip_adapter_manager imported successfully")
except ImportError as e:
    print(f"❌ Failed to import ip_adapter_manager: {e}")
    MISSING_MODULES.append("ip_adapter_manager.py")

try:
    from caption_analyzer import CaptionConsistencyAnalyzer
    print("✅ caption_analyzer imported successfully")
except ImportError as e:
    print(f"❌ Failed to import caption_analyzer: {e}")
    MISSING_MODULES.append("caption_analyzer.py")

try:
    from prompt_improver import PromptImprover
    print("✅ prompt_improver imported successfully")
except ImportError as e:
    print(f"❌ Failed to import prompt_improver: {e}")
    MISSING_MODULES.append("prompt_improver.py")

if CartoonPipelineWithIPAdapter is not None:
    MODULES_AVAILABLE = True
else:
    MODULES_AVAILABLE = False

if MISSING_MODULES:
    print(f"\n⚠️ Missing modules: {', '.join(MISSING_MODULES)}")
    print("📁 Current directory should contain all your custom modules")
else:
    print("✅ All modules imported successfully!")


class DeterministicStoryboardTester:
    """Comprehensive tester for deterministic storyboard generation"""
    
    def __init__(self):
        print("🧪 Initializing Comprehensive Deterministic Storyboard Tester")
        print("=" * 60)
        
        if not MODULES_AVAILABLE:
            print("❌ Cannot initialize - CartoonPipelineWithIPAdapter not available")
            print(f"📁 Missing modules: {', '.join(MISSING_MODULES)}")
            self.available = False
            return
        
        self.available = True
        self.setup_paths()
        self.test_configs = self.get_test_configurations()
        self.results = {}
        
    def setup_paths(self):
        """Auto-detect model and IP-Adapter paths"""
        print("🔍 Auto-detecting file paths...")
        
        current_dir = Path(__file__).parent
        parent_dir = current_dir.parent
        
        model_candidates = [
            current_dir / "realcartoonxl_v7.safetensors",
            current_dir / "models" / "realcartoonxl_v7.safetensors",
            parent_dir / "models" / "realcartoonxl_v7.safetensors",
            parent_dir / "realcartoonxl_v7.safetensors",
        ]
        
        self.model_path = None
        for candidate in model_candidates:
            if candidate.exists():
                self.model_path = str(candidate)
                print(f"✅ Found model: {self.model_path}")
                break
        
        if not self.model_path:
            for path in [current_dir, parent_dir]:
                for model_file in path.rglob("*realcartoon*.safetensors"):
                    self.model_path = str(model_file)
                    print(f"✅ Found model (recursive): {self.model_path}")
                    break
                if self.model_path:
                    break
        
        ip_adapter_candidates = [
            current_dir / "ip-adapter_sdxl.bin",
            parent_dir / "ip-adapter_sdxl.bin",
            current_dir / "models" / "ip-adapter_sdxl.bin",
        ]
        
        self.ip_adapter_path = None
        for candidate in ip_adapter_candidates:
            if candidate.exists():
                self.ip_adapter_path = str(candidate)
                print(f"✅ Found IP-Adapter: {self.ip_adapter_path}")
                break
        
        if not self.ip_adapter_path:
            for path in [current_dir, parent_dir]:
                for ip_file in path.rglob("*ip-adapter*.bin"):
                    self.ip_adapter_path = str(ip_file)
                    print(f"✅ Found IP-Adapter (recursive): {self.ip_adapter_path}")
                    break
                if self.ip_adapter_path:
                    break
        
        print(f"📁 Model found: {bool(self.model_path)}")
        print(f"🎭 IP-Adapter found: {bool(self.ip_adapter_path)}")
        
        if not self.model_path:
            print("❌ No model file found!")
            print("   Please ensure realcartoonxl_v7.safetensors is available")
        
        if not self.ip_adapter_path:
            print("⚠️ No IP-Adapter found - will test without IP-Adapter")
            print("   Download from: https://huggingface.co/h94/IP-Adapter/resolve/main/sdxl_models/ip-adapter_sdxl.bin")
    
    def get_test_configurations(self):
        """Define different test configurations"""
        return {
            "simplicity_evaluation": {
                "name": "Simplicity Evaluation",
                "description": "Tests simple, clear cartoon generation with minimal complexity",
                "prompts": [
                    "simple cartoon boy, clean art style, minimal details",
                    "same boy sitting, simple background, clean cartoon style", 
                    "same boy standing, basic scene, simple cartoon art",
                    "same boy smiling, plain background, clean simple style"
                ],
                "config": {
                    "generation": {
                        "height": 1024,
                        "width": 1024,
                        "num_inference_steps": 20,
                        "guidance_scale": 6.0,
                        "num_images_per_prompt": 3
                    },
                    "selection": {
                        "use_consistency": False,
                        "use_ip_adapter": True,
                        "quality_weight": 1.0,
                        "consistency_weight": 0.0,
                        "ip_adapter_weight": 0.0
                    },
                    "ip_adapter": {
                        "character_weight": 0.1,
                        "update_reference_from_best": False,
                        "fallback_to_clip": True
                    },
                    "baked_vae": {
                        "use_conservative_optimizations": True,
                        "disable_autocast_on_failure": True,
                        "force_float32_fallback": True
                    }
                }
            },
            
            "accuracy_evaluation": {
                "name": "Accuracy Evaluation",
                "description": "Tests accuracy of prompt-to-image generation and semantic alignment",
                "prompts": [
                    "6-year-old boy with brown hair wearing blue pajamas with white stars",
                    "same boy brushing teeth with white toothbrush in bathroom mirror",
                    "same boy putting on white school shirt and navy blue shorts",
                    "same boy eating cereal with spoon at wooden kitchen table",
                    "same boy carrying red backpack standing by front door"
                ],
                "config": {
                    "generation": {
                        "height": 1024,
                        "width": 1024,
                        "num_inference_steps": 30,
                        "guidance_scale": 8.0,
                        "num_images_per_prompt": 4
                    },
                    "selection": {
                        "use_consistency": True,
                        "use_ip_adapter": True,
                        "quality_weight": 0.7,
                        "consistency_weight": 0.2,
                        "ip_adapter_weight": 0.1
                    },
                    "ip_adapter": {
                        "character_weight": 0.3,
                        "update_reference_from_best": True,
                        "fallback_to_clip": True
                    },
                    "baked_vae": {
                        "use_conservative_optimizations": True,
                        "disable_autocast_on_failure": True,
                        "force_float32_fallback": True
                    }
                }
            },
            
            "consistency_evaluation": {
                "name": "Consistency Evaluation", 
                "description": "Tests character consistency across multiple scenes with IP-Adapter",
                "prompts": [
                    "6-year-old boy Alex, brown hair, brown eyes, blue pajamas with white stars, cartoon style",
                    "same boy Alex in blue star pajamas brushing teeth, cartoon style",
                    "same boy Alex changing into school uniform, white shirt navy shorts, cartoon style",
                    "same boy Alex in school uniform eating breakfast, cartoon style"
                ],
                "config": {
                    "generation": {
                        "height": 1024,
                        "width": 1024,
                        "num_inference_steps": 30,
                        "guidance_scale": 7.0,
                        "num_images_per_prompt": 4
                    },
                    "selection": {
                        "use_consistency": True,
                        "use_ip_adapter": True,
                        "quality_weight": 0.1,
                        "consistency_weight": 0.4,
                        "ip_adapter_weight": 0.5
                    },
                    "ip_adapter": {
                        "character_weight": 0.6,
                        "update_reference_from_best": True,
                        "fallback_to_clip": True
                    },
                    "baked_vae": {
                        "use_conservative_optimizations": True,
                        "disable_autocast_on_failure": True,
                        "force_float32_fallback": True
                    }
                }
            }
        }
    
    def run_comprehensive_test(self, test_name=None):
        """Run comprehensive test suite"""
        print("\n🚀 STARTING COMPREHENSIVE DETERMINISTIC STORYBOARD TEST")
        print("=" * 60)
        
        if not self.available:
            print("❌ Cannot run tests - tester not available due to missing modules!")
            return False
        
        if not self.model_path:
            print("❌ Cannot run tests - no model file found!")
            return False
        
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        main_output_dir = f"deterministic_storyboard_test_{timestamp}"
        os.makedirs(main_output_dir, exist_ok=True)
        
        tests_to_run = [test_name] if test_name and test_name in self.test_configs else list(self.test_configs.keys())
        
        overall_start_time = time.time()
        test_results = {}
        
        for test_config_name in tests_to_run:
            print(f"\n📋 RUNNING TEST: {self.test_configs[test_config_name]['name']}")
            print("-" * 50)
            
            test_output_dir = os.path.join(main_output_dir, test_config_name)
            os.makedirs(test_output_dir, exist_ok=True)
            
            result = self.run_single_test(test_config_name, test_output_dir)
            test_results[test_config_name] = result
            
            if result["success"]:
                consistency_grade = result.get("consistency_report", {}).get("consistency_grade", "Unknown")
                ip_usage = result.get("ip_adapter_usage_rate", 0)
                print(f"✅ {test_config_name}: {consistency_grade} consistency, {ip_usage:.0f}% IP-Adapter usage")
            else:
                print(f"❌ {test_config_name}: Test failed - {result.get('error', 'Unknown error')}")
        
        total_time = time.time() - overall_start_time
        
        self.generate_comprehensive_report(test_results, main_output_dir, total_time)
        
        successful_tests = sum(1 for r in test_results.values() if r["success"])
        total_tests = len(test_results)
        
        print(f"\n🏁 COMPREHENSIVE TEST COMPLETE!")
        print(f"📊 Results: {successful_tests}/{total_tests} tests passed")
        print(f"⏱️ Total time: {total_time/60:.1f} minutes")
        print(f"📁 Results saved to: {os.path.abspath(main_output_dir)}")
        
        return successful_tests == total_tests
    
    def run_single_test(self, test_config_name, output_dir):
        """Run a single test configuration"""
        config = self.test_configs[test_config_name]
        
        print(f"🎯 {config['name']}")
        print(f"📝 {config['description']}")
        print(f"🖼️ Testing {len(config['prompts'])} prompts")
        
        try:
            print("🔧 Initializing pipeline...")
            
            working_config = config["config"].copy()
            
            if "ip_adapter" not in working_config:
                working_config["ip_adapter"] = {
                    "character_weight": 0.3,
                    "update_reference_from_best": True,
                    "fallback_to_clip": True
                }
            
            if "baked_vae" not in working_config:
                working_config["baked_vae"] = {
                    "use_conservative_optimizations": True,
                    "disable_autocast_on_failure": True,
                    "force_float32_fallback": True
                }
            
            pipeline = CartoonPipelineWithIPAdapter(
                model_path=self.model_path,
                ip_adapter_path=self.ip_adapter_path,
                config=working_config
            )
            
            if not pipeline.available:
                return {
                    "success": False,
                    "error": "Pipeline initialization failed",
                    "test_config": test_config_name
                }
            
            pipeline_status = pipeline.get_pipeline_status()
            
            start_time = time.time()
            
            print("🎨 Generating deterministic storyboard...")
            sequence_result = pipeline.generate_storyboard_sequence(
                prompts_list=config["prompts"],
                character_reference_image=None,
                iterations_per_prompt=1,
                images_per_iteration=config["config"]["generation"]["num_images_per_prompt"]
            )
            
            generation_time = time.time() - start_time
            
            if not sequence_result:
                return {
                    "success": False,
                    "error": "Storyboard generation failed",
                    "test_config": test_config_name
                }
            
            print("💾 Saving generated images...")
            self.save_test_images(sequence_result, output_dir, test_config_name)
            
            self.create_test_storyboard(sequence_result, output_dir, config["name"])
            
            analysis_results = self.analyze_test_results(sequence_result, config)
            
            generate_enhanced_test_report(sequence_result, output_dir, generation_time, pipeline_status)
            
            self.save_test_metadata(config, sequence_result, analysis_results, output_dir)
            
            print(f"✅ Test completed in {generation_time:.1f}s")
            
            return {
                "success": True,
                "test_config": test_config_name,
                "generation_time": generation_time,
                "sequence_result": sequence_result,
                "analysis_results": analysis_results,
                "pipeline_status": pipeline_status,
                "frames_generated": len(sequence_result.get("sequence_results", [])),
                "consistency_report": sequence_result.get("consistency_report", {}),
                "ip_adapter_usage_rate": (sequence_result.get("ip_adapter_used_count", 0) / 
                                        max(sequence_result.get("total_frames", 1), 1)) * 100
            }
            
        except Exception as e:
            print(f"❌ Test failed with error: {e}")
            traceback.print_exc()
            return {
                "success": False,
                "error": str(e),
                "test_config": test_config_name,
                "traceback": traceback.format_exc()
            }
    
    def save_test_images(self, sequence_result, output_dir, test_name):
        """Save individual test images"""
        sequence_data = sequence_result.get("sequence_results", [])
        
        for i, frame_data in enumerate(sequence_data):
            if frame_data.get("best_image"):
                filename = f"{test_name}_frame_{i+1:02d}.png"
                filepath = os.path.join(output_dir, filename)
                frame_data["best_image"].save(filepath, quality=95, optimize=True)
        
        print(f"   💾 Saved {len(sequence_data)} individual frames")
    
    def create_test_storyboard(self, sequence_result, output_dir, test_name):
        """Create a composite storyboard image"""
        sequence_data = sequence_result.get("sequence_results", [])
        
        if not sequence_data:
            return
        
        frame_size = 400
        padding = 20
        title_height = 60
        cols = min(3, len(sequence_data))
        rows = (len(sequence_data) + cols - 1) // cols
        
        canvas_width = cols * frame_size + (cols + 1) * padding
        canvas_height = rows * frame_size + (rows + 1) * padding + title_height
        
        canvas = Image.new('RGB', (canvas_width, canvas_height), color='white')
        draw = ImageDraw.Draw(canvas)
        
        try:
            title_font = ImageFont.truetype("arial.ttf", 24)
            label_font = ImageFont.truetype("arial.ttf", 16)
        except:
            title_font = ImageFont.load_default()
            label_font = ImageFont.load_default()
        
        title = f"{test_name} - Deterministic Storyboard"
        title_bbox = draw.textbbox((0, 0), title, font=title_font)
        title_width = title_bbox[2] - title_bbox[0]
        title_x = (canvas_width - title_width) // 2
        draw.text((title_x, 20), title, fill='black', font=title_font)
        
        for i, frame_data in enumerate(sequence_data):
            if not frame_data.get("best_image"):
                continue
                
            row = i // cols
            col = i % cols
            
            x = padding + col * (frame_size + padding)
            y = title_height + padding + row * (frame_size + padding)
            
            frame_image = frame_data["best_image"].resize((frame_size, frame_size), Image.Resampling.LANCZOS)
            canvas.paste(frame_image, (x, y))
            
            label = f"Frame {i+1}"
            score = frame_data.get("final_score", 0)
            ip_used = "IP" if frame_data.get("used_ip_adapter", False) else "CLIP"
            
            label_text = f"{label} (Score: {score:.2f}, {ip_used})"
            label_y = y + frame_size + 5
            draw.text((x, label_y), label_text, fill='black', font=label_font)
        
        storyboard_path = os.path.join(output_dir, f"{test_name}_storyboard.png")
        canvas.save(storyboard_path, quality=95, optimize=True)
        
        print(f"   🖼️ Storyboard saved: {storyboard_path}")
        return storyboard_path
    
    def analyze_test_results(self, sequence_result, config):
        """Analyze test results for quality and consistency"""
        sequence_data = sequence_result.get("sequence_results", [])
        consistency_report = sequence_result.get("consistency_report", {})
        
        analysis = {
            "frame_count": len(sequence_data),
            "average_score": 0.0,
            "score_range": [1.0, 0.0],
            "ip_adapter_usage": 0,
            "consistency_grade": "Unknown",
            "issues_detected": [],
            "recommendations": []
        }
        
        if not sequence_data:
            analysis["issues_detected"].append("No frames generated")
            return analysis
        
        scores = [frame.get("final_score", 0) for frame in sequence_data]
        analysis["average_score"] = np.mean(scores)
        analysis["score_range"] = [min(scores), max(scores)]
        
        ip_usage_count = sum(1 for frame in sequence_data if frame.get("used_ip_adapter", False))
        analysis["ip_adapter_usage"] = ip_usage_count / len(sequence_data) * 100
        
        if consistency_report.get("available", False):
            analysis["consistency_grade"] = consistency_report.get("consistency_grade", "Unknown")
            analysis["consistency_score"] = consistency_report.get("average_consistency", 0)
        
        if analysis["average_score"] < 0.6:
            analysis["issues_detected"].append("Low average quality scores")
            analysis["recommendations"].append("Consider adjusting generation parameters")
        
        if analysis["ip_adapter_usage"] < 50:
            analysis["issues_detected"].append("Low IP-Adapter usage")
            analysis["recommendations"].append("Check IP-Adapter installation and configuration")
        
        if consistency_report.get("available") and consistency_report.get("average_consistency", 0) < 0.7:
            analysis["issues_detected"].append("Low visual consistency")
            analysis["recommendations"].append("Increase IP-Adapter weight or improve prompts")
        
        score_std = np.std(scores)
        if score_std > 0.2:
            analysis["issues_detected"].append("High score variance between frames")
            analysis["recommendations"].append("Check for prompt inconsistencies")
        
        return analysis
    
    def save_test_metadata(self, config, sequence_result, analysis_results, output_dir):
        """Save test metadata and results as JSON"""
        metadata = {
            "test_config": config,
            "analysis_results": analysis_results,
            "sequence_summary": {
                "total_frames": sequence_result.get("total_frames", 0),
                "successful_generations": sequence_result.get("successful_generations", 0),
                "ip_adapter_used_count": sequence_result.get("ip_adapter_used_count", 0),
                "character_reference_established": sequence_result.get("character_reference_established", False)
            },
            "consistency_report": sequence_result.get("consistency_report", {}),
            "ip_adapter_status": sequence_result.get("ip_adapter_status", {}),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        metadata_path = os.path.join(output_dir, "test_metadata.json")
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        print(f"   📄 Metadata saved: {metadata_path}")
    
    def generate_comprehensive_report(self, test_results, output_dir, total_time):
        """Generate comprehensive test report across all configurations"""
        report_path = os.path.join(output_dir, "comprehensive_test_report.txt")
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("COMPREHENSIVE DETERMINISTIC STORYBOARD TEST REPORT\n")
            f.write("=" * 60 + "\n\n")
            
            f.write(f"Test Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total Test Time: {total_time/60:.1f} minutes\n")
            f.write(f"Model Path: {self.model_path}\n")
            f.write(f"IP-Adapter Path: {self.ip_adapter_path or 'Not found'}\n\n")
            
            successful_tests = sum(1 for r in test_results.values() if r["success"])
            total_tests = len(test_results)
            
            f.write("OVERALL SUMMARY:\n")
            f.write("-" * 15 + "\n")
            f.write(f"Tests Passed: {successful_tests}/{total_tests}\n")
            f.write(f"Success Rate: {successful_tests/total_tests*100:.1f}%\n\n")
            
            for test_name, result in test_results.items():
                f.write(f"TEST: {test_name.upper()}\n")
                f.write("-" * (len(test_name) + 6) + "\n")
                
                if result["success"]:
                    f.write("✅ STATUS: PASSED\n")
                    f.write(f"Generation Time: {result['generation_time']:.1f}s\n")
                    f.write(f"Frames Generated: {result['frames_generated']}\n")
                    f.write(f"IP-Adapter Usage: {result['ip_adapter_usage_rate']:.1f}%\n")
                    
                    consistency = result["consistency_report"]
                    if consistency.get("available"):
                        f.write(f"Consistency Grade: {consistency.get('consistency_grade', 'Unknown')}\n")
                        f.write(f"Consistency Score: {consistency.get('average_consistency', 0):.3f}\n")
                    
                    analysis = result["analysis_results"]
                    f.write(f"Average Quality Score: {analysis['average_score']:.3f}\n")
                    f.write(f"Score Range: {analysis['score_range'][0]:.3f} - {analysis['score_range'][1]:.3f}\n")
                    
                    if analysis["issues_detected"]:
                        f.write("Issues Detected:\n")
                        for issue in analysis["issues_detected"]:
                            f.write(f"  - {issue}\n")
                    
                    if analysis["recommendations"]:
                        f.write("Recommendations:\n")
                        for rec in analysis["recommendations"]:
                            f.write(f"  - {rec}\n")
                else:
                    f.write("❌ STATUS: FAILED\n")
                    f.write(f"Error: {result.get('error', 'Unknown error')}\n")
                
                f.write("\n")
            
            f.write("OVERALL RECOMMENDATIONS:\n")
            f.write("-" * 25 + "\n")
            
            all_analysis = [r["analysis_results"] for r in test_results.values() if r["success"]]
            
            if all_analysis:
                avg_ip_usage = np.mean([a["ip_adapter_usage"] for a in all_analysis])
                avg_quality = np.mean([a["average_score"] for a in all_analysis])
                
                if avg_ip_usage < 70:
                    f.write("⚠️ IP-Adapter usage is consistently low across tests\n")
                    f.write("   - Verify IP-Adapter installation and file paths\n")
                    f.write("   - Check GPU memory and compatibility\n\n")
                
                if avg_quality < 0.65:
                    f.write("⚠️ Quality scores are consistently low\n")
                    f.write("   - Consider tuning generation parameters\n")
                    f.write("   - Review prompt engineering\n\n")
                
                f.write("✅ For best results:\n")
                f.write("   - Use IP-Adapter weights between 0.2-0.6\n")
                f.write("   - Enable deterministic seeds for reproducibility\n")
                f.write("   - Monitor for baked VAE compatibility issues\n")
                f.write("   - Test with different guidance scales (5.0-8.0)\n")
            
            else:
                f.write("❌ No successful tests to analyze\n")
                f.write("   - Check model file existence and format\n")
                f.write("   - Verify GPU availability and memory\n")
                f.write("   - Install required dependencies\n")
        
        print(f"📊 Comprehensive report saved: {report_path}")
    
    def run_quick_test(self):
        """Run a quick test with minimal configuration"""
        print("🚀 QUICK DETERMINISTIC TEST")
        print("=" * 30)
        
        if not self.available:
            print("❌ Tester not available - missing modules!")
            return False
        
        if not self.model_path:
            print("❌ No model found for quick test!")
            return False
        
        quick_prompts = [
            "young boy character, cartoon style, simple design",
            "same boy smiling, cartoon style, simple scene"
        ]
        
        quick_config = {
            "generation": {
                "height": 512,
                "width": 512,
                "num_inference_steps": 20,
                "guidance_scale": 7.0,
                "num_images_per_prompt": 2
            },
            "selection": {
                "use_consistency": True,
                "use_ip_adapter": True,
                "quality_weight": 0.5,
                "consistency_weight": 0.3,
                "ip_adapter_weight": 0.2
            },
            "ip_adapter": {
                "character_weight": 0.3,
                "update_reference_from_best": True,
                "fallback_to_clip": True
            },
            "baked_vae": {
                "use_conservative_optimizations": True,
                "disable_autocast_on_failure": True,
                "force_float32_fallback": True
            }
        }
        
        try:
            print("🔧 Initializing pipeline...")
            pipeline = CartoonPipelineWithIPAdapter(
                model_path=self.model_path,
                ip_adapter_path=self.ip_adapter_path,
                config=quick_config
            )
            
            if not pipeline.available:
                print("❌ Pipeline not available!")
                return False
            
            print("🎨 Generating quick test...")
            start_time = time.time()
            
            sequence_result = pipeline.generate_storyboard_sequence(
                prompts_list=quick_prompts,
                character_reference_image=None,
                iterations_per_prompt=1,
                images_per_iteration=2
            )
            
            test_time = time.time() - start_time
            
            if sequence_result and sequence_result.get("sequence_results"):
                frames = len(sequence_result["sequence_results"])
                ip_usage = sequence_result.get("ip_adapter_used_count", 0)
                
                os.makedirs("quick_test_output", exist_ok=True)
                for i, frame_data in enumerate(sequence_result["sequence_results"]):
                    if frame_data.get("best_image"):
                        frame_data["best_image"].save(f"quick_test_output/quick_frame_{i+1}.png")
                
                print(f"✅ Quick test successful!")
                print(f"   Generated: {frames} frames in {test_time:.1f}s")
                print(f"   IP-Adapter used: {ip_usage}/{frames} times")
                print(f"   Images saved to: quick_test_output/")
                
                return True
            else:
                print("❌ Quick test failed - no images generated")
                return False
                
        except Exception as e:
            print(f"❌ Quick test error: {e}")
            return False

    def run_basic_diffusion_test(self):
        """Run a basic test using only the diffusion pipeline without custom modules"""
        print("🔧 RUNNING BASIC DIFFUSION TEST")
        print("=" * 40)
        
        if not self.model_path:
            print("❌ No model found!")
            return False
        
        try:
            from diffusers import StableDiffusionXLPipeline
            
            print("🔧 Loading basic SDXL pipeline...")
            pipe = StableDiffusionXLPipeline.from_single_file(
                self.model_path,
                torch_dtype=torch.float16,
                use_safetensors=True
            ).to("cuda")
            
            pipe.enable_vae_tiling()
            pipe.enable_model_cpu_offload()
            
            print("✅ Pipeline loaded successfully!")
            
            test_prompts = [
                "young boy character, cartoon style, simple clean art",
                "same boy smiling, cartoon style, simple background",
                "same boy waving, cartoon style, friendly expression"
            ]
            
            output_dir = "basic_diffusion_test_output"
            os.makedirs(output_dir, exist_ok=True)
            
            print(f"🎨 Generating {len(test_prompts)} test images...")
            
            for i, prompt in enumerate(test_prompts):
                print(f"   Generating image {i+1}: {prompt[:50]}...")
                
                start_time = time.time()
                
                result = pipe(
                    prompt=prompt,
                    negative_prompt="blurry, low quality, distorted",
                    num_images_per_prompt=1,
                    num_inference_steps=25,
                    guidance_scale=7.5,
                    height=1024,
                    width=1024
                )
                
                gen_time = time.time() - start_time
                
                filename = f"{output_dir}/basic_test_{i+1:02d}.png"
                result.images[0].save(filename)
                
                print(f"   ✅ Saved: {filename} ({gen_time:.1f}s)")
            
            print(f"\n🎉 Basic test complete! Images saved to: {output_dir}")
            return True
            
        except Exception as e:
            print(f"❌ Basic test failed: {e}")
            traceback.print_exc()
            return False


def main():
    """Main test function"""
    print("🎨 COMPREHENSIVE DETERMINISTIC STORYBOARD TEST SUITE")
    print("Testing IP-Adapter consistency, baked VAE compatibility, and fixed seeds")
    print("=" * 70)
    
    tester = DeterministicStoryboardTester()
    
    if not tester.available:
        print("\n❌ Cannot proceed - required modules not available!")
        print("📋 Please ensure these files are in the same directory:")
        for module in MISSING_MODULES:
            print(f"   - {module}")
        return
    
    if not tester.model_path:
        print("\n❌ Cannot proceed - no model file found!")
        print("📋 Please ensure realcartoonxl_v7.safetensors is available")
        print("   Common locations:")
        print("   - Same directory as this script")
        print("   - ./models/ subdirectory")
        print("   - Parent directory")
        return
    
    print(f"\n📋 Available Evaluations:")
    test_configs = tester.test_configs
    for i, (key, config) in enumerate(test_configs.items(), 1):
        print(f"   {i}. {config['name']}")
        print(f"      {config['description']}")
    
    print(f"   {len(test_configs)+1}. Run ALL evaluations (comprehensive)")
    print(f"   {len(test_configs)+2}. Quick test (fast verification)")
    print(f"   {len(test_configs)+3}. Basic diffusion test (no custom modules)")
    
    try:
        choice = input(f"\nSelect evaluation (1-{len(test_configs)+3}): ").strip()
        choice_num = int(choice)
        
        if choice_num == len(test_configs) + 1:
            print("\n🚀 Running ALL comprehensive evaluations...")
            success = tester.run_comprehensive_test()
            if success:
                print("🎉 All evaluations passed!")
            else:
                print("⚠️ Some evaluations failed - check reports for details")
        
        elif choice_num == len(test_configs) + 2:
            tester.run_quick_test()
        
        elif choice_num == len(test_configs) + 3:
            tester.run_basic_diffusion_test()
        
        elif 1 <= choice_num <= len(test_configs):
            test_names = list(test_configs.keys())
            selected_test = test_names[choice_num - 1]
            print(f"\n🚀 Running {test_configs[selected_test]['name']}...")
            tester.run_comprehensive_test(selected_test)
        
        else:
            print("❌ Invalid choice")
    
    except (ValueError, KeyboardInterrupt):
        print("\n⚠️ Evaluation cancelled")
    
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    main()