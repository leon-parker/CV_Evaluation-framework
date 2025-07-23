"""
Metric Evaluator Module
Import this into your test scripts to get focused evaluation options
Usage: from metric_evaluator import MetricEvaluator
"""

import os
import numpy as np
from PIL import Image
import cv2
import time

# Import your existing components
try:
    from image_complexity_analyser import AutismFriendlyImageAnalyzer
    from consistency_manager import ConsistencyManager
    from quality_evaluator import QualityEvaluator
    from caption_analyzer import CaptionConsistencyAnalyzer
except ImportError as e:
    print(f"⚠️ Import warning: {e}")


class MetricEvaluator:
    """
    Focused metric evaluation for images
    Use in test scripts to evaluate specific aspects
    """
    
    def __init__(self, verbose=True):
        self.verbose = verbose
        
        if self.verbose:
            print("🎯 Loading Metric Evaluator...")
        
        # Initialize available evaluators
        self.evaluators = {}
        
        # Simplicity evaluator
        try:
            self.autism_analyzer = AutismFriendlyImageAnalyzer()
            if self.autism_analyzer.available:
                self.evaluators['simplicity'] = True
                if self.verbose:
                    print("✅ Simplicity evaluator ready")
        except:
            self.autism_analyzer = None
            self.evaluators['simplicity'] = False
        
        # Consistency evaluator
        try:
            self.consistency_manager = ConsistencyManager()
            if self.consistency_manager.available:
                self.evaluators['consistency'] = True
                if self.verbose:
                    print("✅ Consistency evaluator ready")
        except:
            self.consistency_manager = None
            self.evaluators['consistency'] = False
        
        # Accuracy evaluator
        try:
            self.quality_evaluator = QualityEvaluator()
            if self.quality_evaluator.available:
                self.evaluators['accuracy'] = True
                if self.verbose:
                    print("✅ Accuracy evaluator ready")
        except:
            self.quality_evaluator = None
            self.evaluators['accuracy'] = False
        
        # Prompt improvement evaluator
        try:
            from prompt_improver import PromptImprover
            if self.consistency_manager and self.quality_evaluator:
                self.prompt_improver = PromptImprover(self.consistency_manager, self.quality_evaluator)
                self.evaluators['prompt_improvement'] = True
                if self.verbose:
                    print("✅ Prompt improvement evaluator ready")
            else:
                self.prompt_improver = None
                self.evaluators['prompt_improvement'] = False
        except:
            self.prompt_improver = None
            self.evaluators['prompt_improvement'] = False
        
        # Caption analyzer
        try:
            self.caption_analyzer = CaptionConsistencyAnalyzer()
        except:
            self.caption_analyzer = None
        
        # Track reference images for consistency
        self.reference_set = False
    
    def get_available_metrics(self):
        """Return list of available evaluation metrics"""
        return [metric for metric, available in self.evaluators.items() if available]
    
    def show_metric_options(self):
        """Display available evaluation options"""
        print("\n🎯 EVALUATION OPTIONS:")
        print("="*40)
        
        if self.evaluators.get('simplicity'):
            print("1️⃣  SIMPLICITY - Autism-friendly design")
            print("    • Person count, background, colors, clarity")
        
        if self.evaluators.get('consistency'):
            print("2️⃣  CONSISTENCY - Character consistency")
            print("    • Visual similarity, character traits")
        
        if self.evaluators.get('accuracy'):
            print("3️⃣  ACCURACY - Prompt alignment & quality")
            print("    • TIFA score, artifacts, semantic match")
        
        if self.evaluators.get('prompt_improvement'):
            print("4️⃣  PROMPT IMPROVEMENT - Enhance prompts")
            print("    • Analyze prompt-image alignment")
            print("    • Generate improved prompts")
            print("    • Consistency-based improvements")
        
        print("5️⃣  PROGRESSIVE - Auto-improve through iterations")
        print("    • Generate → Analyze → Improve → Repeat")
        print("    • Stop when quality thresholds met")
        print("    • Uses autism + consistency + accuracy scoring")
        
        print("6️⃣  SELF-IMPROVE TIFA - Focus on accuracy only")
        print("    • Iteratively improve TIFA scores")
        print("    • Optimize prompt-image alignment")
        
        print("7️⃣  SELF-IMPROVE SIMPLICITY - Focus on autism-friendly design")
        print("    • Iteratively improve autism suitability")
        print("    • Optimize for person count, colors, clarity")
        
        print("8️⃣  SELF-IMPROVE CONSISTENCY - Focus on character consistency")
        print("    • Iteratively improve visual consistency")
        print("    • Optimize character trait matching")
        
        print("9️⃣  ALL - Run all available evaluations")
        print("0️⃣  EXIT")
        print("="*40)
    
    def get_metric_choice(self):
        """Get user's metric choice interactively"""
        self.show_metric_options()
        
        while True:
            choice = input("Choose evaluation metric (0-9): ").strip()
            
            if choice == "0":
                return "exit"
            elif choice == "1" and self.evaluators.get('simplicity'):
                return "simplicity"
            elif choice == "2" and self.evaluators.get('consistency'):
                return "consistency"
            elif choice == "3" and self.evaluators.get('accuracy'):
                return "accuracy"
            elif choice == "4" and self.evaluators.get('prompt_improvement'):
                return "prompt_improvement"
            elif choice == "5":
                return "progressive"
            elif choice == "6":
                return "self_improve_tifa"
            elif choice == "7":
                return "self_improve_simplicity"
            elif choice == "8":
                return "self_improve_consistency"
            elif choice == "9":
                return "all"
            else:
                print("❌ Invalid choice. Please try again.")
    
    def evaluate_simplicity(self, image, return_summary=True):
        """
        Evaluate image for autism-friendly simplicity
        
        Args:
            image: PIL Image or path to image
            return_summary: If True, return condensed summary
        
        Returns:
            dict: Simplicity analysis results
        """
        if not self.evaluators.get('simplicity'):
            return {"error": "Simplicity evaluator not available"}
        
        # Load image if path provided
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")
        
        analysis = self.autism_analyzer.analyze_autism_suitability(image)
        
        if return_summary:
            return {
                "metric": "simplicity",
                "overall_score": analysis['autism_suitability'],
                "grade": analysis['autism_grade'],
                "person_count": analysis['person_count']['count'],
                "person_compliant": analysis['person_count']['is_compliant'],
                "background_score": analysis['background_simplicity']['score'],
                "color_score": analysis['color_appropriateness']['score'],
                "clarity_score": analysis['character_clarity']['score'],
                "sensory_score": analysis['sensory_friendliness']['score'],
                "focus_score": analysis['focus_clarity']['score'],
                "top_recommendations": analysis['recommendations'][:3],
                "full_analysis": analysis
            }
        
        return analysis
    
    def evaluate_consistency(self, images, prompts=None, return_summary=True):
        """
        Evaluate consistency across multiple images
        
        Args:
            images: List of PIL Images or image paths
            prompts: Optional list of prompts for each image
            return_summary: If True, return condensed summary
        
        Returns:
            dict: Consistency analysis results
        """
        if not self.evaluators.get('consistency'):
            return {"error": "Consistency evaluator not available"}
        
        if len(images) < 2:
            return {"error": "Need at least 2 images for consistency evaluation"}
        
        # Load images if paths provided
        loaded_images = []
        for img in images:
            if isinstance(img, str):
                loaded_images.append(Image.open(img).convert("RGB"))
            else:
                loaded_images.append(img)
        
        if prompts is None:
            prompts = [f"Image {i+1}" for i in range(len(loaded_images))]
        
        results = []
        consistency_scores = []
        style_scores = []
        
        # Process each image
        for i, (image, prompt) in enumerate(zip(loaded_images, prompts)):
            # Generate caption
            if self.quality_evaluator:
                caption = self.quality_evaluator.generate_caption(image)
            else:
                caption = "No caption available"
            
            # Get embeddings and calculate consistency
            image_embedding = self.consistency_manager.get_image_embedding(image)
            
            if i > 0:
                consistency_score = self.consistency_manager.calculate_consistency_score(image_embedding)
                style_score = self.consistency_manager.calculate_style_consistency(image_embedding)
            else:
                consistency_score = 1.0  # Reference image
                style_score = 1.0
                # Set character reference for caption analysis
                if self.caption_analyzer:
                    self.caption_analyzer.update_character_profile(caption, is_first_image=True)
            
            # Analyze caption consistency
            caption_issues = {"severity": "low"}
            if self.caption_analyzer and i > 0:
                caption_issues = self.caption_analyzer.analyze_consistency_and_artifacts(caption)
            
            # Store in consistency manager
            self.consistency_manager.store_selected_image(image, prompt, 0.8, i)
            
            results.append({
                "image_index": i,
                "prompt": prompt,
                "caption": caption,
                "consistency_score": consistency_score,
                "style_score": style_score,
                "caption_issues": caption_issues
            })
            
            if i > 0:  # Skip reference image for averages
                consistency_scores.append(consistency_score)
                style_scores.append(style_score)
        
        if return_summary:
            avg_consistency = np.mean(consistency_scores) if consistency_scores else 1.0
            avg_style = np.mean(style_scores) if style_scores else 1.0
            
            # Get overall consistency report
            consistency_report = self.consistency_manager.get_consistency_report()
            
            return {
                "metric": "consistency",
                "overall_consistency": avg_consistency,
                "overall_style": avg_style,
                "consistency_grade": consistency_report.get('consistency_grade', 'Unknown'),
                "image_count": len(loaded_images),
                "consistency_range": [min(consistency_scores), max(consistency_scores)] if consistency_scores else [1.0, 1.0],
                "detailed_results": results,
                "manager_report": consistency_report
            }
        
    def self_improve_tifa(self, initial_prompt, pipeline=None, return_summary=True):
        """
        Self-improvement algorithm focused ONLY on TIFA/accuracy scores
        
        Args:
            initial_prompt: Starting prompt
            pipeline: Generation pipeline (if available)
            return_summary: If True, return condensed summary
        
        Returns:
            dict: TIFA-focused improvement results
        """
        if not pipeline:
            return {
                "error": "Self-improvement requires a generation pipeline",
                "suggestion": "This evaluation works best when integrated with your cartoon_pipeline.py"
            }
        
        if not self.evaluators.get('accuracy'):
            return {"error": "TIFA evaluator not available"}
        
        print(f"\n🎯 TIFA SELF-IMPROVEMENT STARTING")
        print(f"Initial prompt: {initial_prompt}")
        print("="*60)
        
        # Configuration - focused on TIFA only
        max_iterations = 4
        tifa_threshold = 0.8  # Higher threshold for TIFA-only
        
        results = {
            "initial_prompt": initial_prompt,
            "iterations": [],
            "final_result": None,
            "improvement_achieved": False,
            "total_iterations": 0,
            "focus_metric": "tifa"
        }
        
        current_prompt = initial_prompt
        best_tifa_score = 0.0
        
        for iteration in range(max_iterations):
            print(f"\n🔄 TIFA ITERATION {iteration + 1}/{max_iterations}")
            print(f"Current prompt: {current_prompt}")
            
            # Generate images
            generation_result = pipeline.generate_with_selection(
                prompt=current_prompt,
                num_images=4  # More images for better TIFA selection
            )
            
            if not generation_result:
                print(f"❌ Generation failed in iteration {iteration + 1}")
                break
            
            best_image = generation_result["best_image"]
            
            # Evaluate TIFA score
            tifa_result = self.quality_evaluator.calculate_tifa_score(best_image, current_prompt)
            tifa_score = tifa_result["score"]
            
            iteration_eval = {
                "iteration": iteration + 1,
                "prompt": current_prompt,
                "tifa_score": tifa_score,
                "tifa_details": tifa_result["details"],
                "met_threshold": False,
                "image": best_image
            }
            
            print(f"   🎯 TIFA Score: {tifa_score:.3f}")
            print(f"   🔗 CLIP Score: {tifa_result['details'].get('clip_score', 0):.3f}")
            print(f"   🧠 Semantic Score: {tifa_result['details'].get('semantic_score', 0):.3f}")
            
            # Check if threshold met
            if tifa_score >= tifa_threshold:
                iteration_eval["met_threshold"] = True
                print(f"   ✅ TIFA threshold met! Score: {tifa_score:.3f} ≥ {tifa_threshold}")
                results["final_result"] = generation_result
                results["improvement_achieved"] = True
                results["iterations"].append(iteration_eval)
                break
            else:
                print(f"   ⚠️ TIFA threshold not met: {tifa_score:.3f} < {tifa_threshold}")
            
            # Update best score
            if tifa_score > best_tifa_score:
                best_tifa_score = tifa_score
                results["final_result"] = generation_result
            
            # Generate improved prompt focused on TIFA issues
            if iteration < max_iterations - 1:
                current_prompt = self._enhance_prompt_for_tifa(
                    current_prompt, 
                    tifa_result, 
                    best_image
                )
            
            results["iterations"].append(iteration_eval)
        
        results["total_iterations"] = len(results["iterations"])
        
        # Summary
        if results["improvement_achieved"]:
            print(f"\n🎉 SUCCESS! Achieved TIFA target in {results['total_iterations']} iterations")
        else:
            print(f"\n⚠️ Reached max iterations. Best TIFA score: {best_tifa_score:.3f}")
        
        if return_summary:
            return self._summarize_tifa_results(results)
        
        return results
    
    def self_improve_simplicity(self, initial_prompt, pipeline=None, return_summary=True):
        """
        Self-improvement algorithm focused ONLY on autism-friendly simplicity
        
        Args:
            initial_prompt: Starting prompt
            pipeline: Generation pipeline (if available)
            return_summary: If True, return condensed summary
        
        Returns:
            dict: Simplicity-focused improvement results
        """
        if not pipeline:
            return {
                "error": "Self-improvement requires a generation pipeline",
                "suggestion": "This evaluation works best when integrated with your cartoon_pipeline.py"
            }
        
        if not self.evaluators.get('simplicity'):
            return {"error": "Simplicity evaluator not available"}
        
        print(f"\n🧩 SIMPLICITY SELF-IMPROVEMENT STARTING")
        print(f"Initial prompt: {initial_prompt}")
        print("="*60)
        
        # Configuration - focused on autism suitability only
        max_iterations = 4
        autism_threshold = 0.75  # High threshold for simplicity-only
        
        results = {
            "initial_prompt": initial_prompt,
            "iterations": [],
            "final_result": None,
            "improvement_achieved": False,
            "total_iterations": 0,
            "focus_metric": "simplicity"
        }
        
        current_prompt = initial_prompt
        best_autism_score = 0.0
        
        for iteration in range(max_iterations):
            print(f"\n🔄 SIMPLICITY ITERATION {iteration + 1}/{max_iterations}")
            print(f"Current prompt: {current_prompt}")
            
            # Generate images
            generation_result = pipeline.generate_with_selection(
                prompt=current_prompt,
                num_images=4
            )
            
            if not generation_result:
                print(f"❌ Generation failed in iteration {iteration + 1}")
                break
            
            best_image = generation_result["best_image"]
            
            # Evaluate simplicity
            simplicity_result = self.evaluate_simplicity(best_image)
            autism_score = simplicity_result["overall_score"]
            
            iteration_eval = {
                "iteration": iteration + 1,
                "prompt": current_prompt,
                "autism_score": autism_score,
                "person_count": simplicity_result["person_count"],
                "person_compliant": simplicity_result["person_compliant"],
                "background_score": simplicity_result["background_score"],
                "color_score": simplicity_result["color_score"],
                "met_threshold": False,
                "image": best_image,
                "full_analysis": simplicity_result["full_analysis"]
            }
            
            print(f"   🧩 Autism Score: {autism_score:.3f}")
            print(f"   👥 Person Count: {simplicity_result['person_count']} ({'✅' if simplicity_result['person_compliant'] else '❌'})")
            print(f"   🎨 Background: {simplicity_result['background_score']:.3f}")
            print(f"   🌈 Colors: {simplicity_result['color_score']:.3f}")
            
            # Check if threshold met
            if autism_score >= autism_threshold and simplicity_result["person_compliant"]:
                iteration_eval["met_threshold"] = True
                print(f"   ✅ Simplicity threshold met! Score: {autism_score:.3f} ≥ {autism_threshold}")
                results["final_result"] = generation_result
                results["improvement_achieved"] = True
                results["iterations"].append(iteration_eval)
                break
            else:
                print(f"   ⚠️ Simplicity threshold not met: {autism_score:.3f} < {autism_threshold}")
            
            # Update best score
            if autism_score > best_autism_score:
                best_autism_score = autism_score
                results["final_result"] = generation_result
            
            # Generate improved prompt focused on simplicity issues
            if iteration < max_iterations - 1:
                current_prompt = self._enhance_prompt_for_simplicity(
                    current_prompt, 
                    simplicity_result
                )
            
            results["iterations"].append(iteration_eval)
        
        results["total_iterations"] = len(results["iterations"])
        
        # Summary
        if results["improvement_achieved"]:
            print(f"\n🎉 SUCCESS! Achieved simplicity target in {results['total_iterations']} iterations")
        else:
            print(f"\n⚠️ Reached max iterations. Best autism score: {best_autism_score:.3f}")
        
        if return_summary:
            return self._summarize_simplicity_results(results)
        
        return results
    
    def self_improve_consistency(self, initial_prompt, pipeline=None, return_summary=True):
        """
        Self-improvement algorithm focused ONLY on character consistency
        
        Args:
            initial_prompt: Starting prompt
            pipeline: Generation pipeline (if available)
            return_summary: If True, return condensed summary
        
        Returns:
            dict: Consistency-focused improvement results
        """
        if not pipeline:
            return {
                "error": "Self-improvement requires a generation pipeline",
                "suggestion": "This evaluation works best when integrated with your cartoon_pipeline.py"
            }
        
        if not self.evaluators.get('consistency'):
            return {"error": "Consistency evaluator not available"}
        
        print(f"\n🔄 CONSISTENCY SELF-IMPROVEMENT STARTING")
        print(f"Initial prompt: {initial_prompt}")
        print("="*60)
        
        # Configuration - focused on consistency only
        max_iterations = 4
        consistency_threshold = 0.8
        
        results = {
            "initial_prompt": initial_prompt,
            "iterations": [],
            "final_result": None,
            "improvement_achieved": False,
            "total_iterations": 0,
            "focus_metric": "consistency",
            "reference_image": None
        }
        
        current_prompt = initial_prompt
        best_consistency_score = 0.0
        
        # Generate reference image first
        print(f"\n📌 Generating reference image...")
        ref_result = pipeline.generate_with_selection(prompt=current_prompt, num_images=3)
        if not ref_result:
            return {"error": "Failed to generate reference image"}
        
        reference_image = ref_result["best_image"]
        results["reference_image"] = reference_image
        
        # Store reference in consistency manager
        self.consistency_manager.store_selected_image(reference_image, current_prompt, 0.8, 0)
        print(f"   ✅ Reference image established")
        
        for iteration in range(max_iterations):
            print(f"\n🔄 CONSISTENCY ITERATION {iteration + 1}/{max_iterations}")
            print(f"Current prompt: {current_prompt}")
            
            # Generate images
            generation_result = pipeline.generate_with_selection(
                prompt=current_prompt,
                num_images=4
            )
            
            if not generation_result:
                print(f"❌ Generation failed in iteration {iteration + 1}")
                break
            
            best_image = generation_result["best_image"]
            
            # Evaluate consistency with reference
            image_embedding = self.consistency_manager.get_image_embedding(best_image)
            consistency_score = self.consistency_manager.calculate_consistency_score(image_embedding)
            style_consistency = self.consistency_manager.calculate_style_consistency(image_embedding)
            
            # Caption consistency analysis
            caption_issues = {"severity": "low"}
            if self.caption_analyzer and self.quality_evaluator:
                caption = self.quality_evaluator.generate_caption(best_image)
                caption_issues = self.caption_analyzer.analyze_consistency_and_artifacts(caption)
            
            iteration_eval = {
                "iteration": iteration + 1,
                "prompt": current_prompt,
                "consistency_score": consistency_score,
                "style_consistency": style_consistency,
                "caption_issues": caption_issues,
                "met_threshold": False,
                "image": best_image
            }
            
            print(f"   🔄 Consistency Score: {consistency_score:.3f}")
            print(f"   🎨 Style Consistency: {style_consistency:.3f}")
            print(f"   📝 Caption Issues: {caption_issues['severity']}")
            
            # Check if threshold met
            if consistency_score >= consistency_threshold and caption_issues["severity"] == "low":
                iteration_eval["met_threshold"] = True
                print(f"   ✅ Consistency threshold met! Score: {consistency_score:.3f} ≥ {consistency_threshold}")
                results["final_result"] = generation_result
                results["improvement_achieved"] = True
                results["iterations"].append(iteration_eval)
                break
            else:
                print(f"   ⚠️ Consistency threshold not met: {consistency_score:.3f} < {consistency_threshold}")
            
            # Update best score
            if consistency_score > best_consistency_score:
                best_consistency_score = consistency_score
                results["final_result"] = generation_result
            
            # Generate improved prompt focused on consistency issues
            if iteration < max_iterations - 1:
                current_prompt = self._enhance_prompt_for_consistency(
                    current_prompt, 
                    caption_issues,
                    consistency_score
                )
            
            results["iterations"].append(iteration_eval)
        
        results["total_iterations"] = len(results["iterations"])
        
        # Summary
        if results["improvement_achieved"]:
            print(f"\n🎉 SUCCESS! Achieved consistency target in {results['total_iterations']} iterations")
        else:
            print(f"\n⚠️ Reached max iterations. Best consistency score: {best_consistency_score:.3f}")
        
        if return_summary:
            return self._summarize_consistency_results(results)
        
        return results
    
    def evaluate_accuracy(self, image, prompt, return_summary=True):
        """
        Evaluate image accuracy against prompt
        
        Args:
            image: PIL Image or path to image
            prompt: Text prompt used to generate the image
            return_summary: If True, return condensed summary
        
        Returns:
            dict: Accuracy analysis results
        """
        if not self.evaluators.get('accuracy'):
            return {"error": "Accuracy evaluator not available"}
        
        # Load image if path provided
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")
        
        tifa_result = self.quality_evaluator.calculate_tifa_score(image, prompt)
        
        if return_summary:
            details = tifa_result.get('details', {})
            artifacts = details.get('artifacts_detected', {})
            
            return {
                "metric": "accuracy",
                "overall_score": tifa_result['score'],
                "clip_score": details.get('clip_score', 0.5),
                "semantic_score": details.get('semantic_score', 0.5),
                "visual_quality": details.get('visual_quality_score', 0.5),
                "artifact_severity": artifacts.get('severity', 'unknown'),
                "caption_issues": artifacts.get('caption_issues', []),
                "visual_issues": artifacts.get('visual_issues', []),
                "generated_caption": details.get('generated_caption', ''),
                "quality_grade": self._get_quality_grade(tifa_result['score']),
                "full_result": tifa_result
            }
        
        return tifa_result
    
    def evaluate_prompt_improvement(self, image, prompt, return_summary=True):
        """
        Evaluate prompt-image alignment and generate improvements
        
        Args:
            image: PIL Image or path to image
            prompt: Original prompt used
            return_summary: If True, return condensed summary
        
        Returns:
            dict: Prompt improvement analysis and suggestions
        """
        if not self.evaluators.get('prompt_improvement'):
            return {"error": "Prompt improvement evaluator not available"}
        
        # Load image if path provided
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")
        
        # Analyze current prompt-image alignment
        is_first_image = len(self.consistency_manager.selected_images_history) == 0
        analysis = self.prompt_improver.analyze_prompt_image_alignment(
            image, prompt, is_first_image
        )
        
        # Generate improved prompts
        improved = self.prompt_improver.improve_prompts(
            prompt, analysis, ""
        )
        
        if return_summary:
            return {
                "metric": "prompt_improvement",
                "original_prompt": prompt,
                "improved_positive": improved["positive"],
                "improved_negative": improved["negative"],
                "clip_similarity": analysis["clip_similarity"],
                "alignment_quality": analysis["alignment_quality"],
                "generated_caption": analysis["caption"],
                "consistency_issues_found": improved["consistency_issues_found"],
                "improvements_applied": improved["improvements"],
                "consistency_severity": analysis["consistency_issues"]["severity"],
                "full_analysis": analysis,
                "full_improvements": improved
            }
        
        return {
            "analysis": analysis,
            "improvements": improved
        }
    
    def evaluate_progressive_iteration(self, initial_prompt, pipeline=None, return_summary=True):
        """
        Progressive iteration: Generate → Analyze → Improve → Repeat
        
        Args:
            initial_prompt: Starting prompt
            pipeline: Generation pipeline (if available)
            return_summary: If True, return condensed summary
        
        Returns:
            dict: Progressive iteration results with all iterations
        """
        if not pipeline:
            return {
                "error": "Progressive iteration requires a generation pipeline",
                "suggestion": "This evaluation works best when integrated with your cartoon_pipeline.py"
            }
        
        print(f"\n🔄 PROGRESSIVE ITERATION STARTING")
        print(f"Initial prompt: {initial_prompt}")
        print("="*60)
        
        # Configuration for progressive improvement
        max_iterations = 3
        quality_threshold = 0.7
        autism_threshold = 0.6
        
        results = {
            "initial_prompt": initial_prompt,
            "iterations": [],
            "final_result": None,
            "improvement_achieved": False,
            "total_iterations": 0
        }
        
        current_prompt = initial_prompt
        best_overall_score = 0.0
        
        for iteration in range(max_iterations):
            print(f"\n🔄 ITERATION {iteration + 1}/{max_iterations}")
            print(f"Current prompt: {current_prompt}")
            
            # Use pipeline's progressive improvement method
            iteration_result = pipeline.generate_with_progressive_improvement(
                prompt=current_prompt,
                num_images=3,
                quality_threshold=quality_threshold,
                max_iterations=1,  # Single iteration per call
                autism_threshold=autism_threshold
            )
            
            if not iteration_result:
                print(f"❌ Generation failed in iteration {iteration + 1}")
                break
            
            # Extract scores
            final_score = iteration_result["best_score"]
            autism_score = iteration_result.get("autism_score", 0.5)
            used_ip_adapter = iteration_result.get("used_ip_adapter", False)
            
            # Evaluate with all metrics
            iteration_eval = {
                "iteration": iteration + 1,
                "prompt": current_prompt,
                "final_prompt": iteration_result.get("final_prompt", current_prompt),
                "overall_score": final_score,
                "autism_score": autism_score,
                "used_ip_adapter": used_ip_adapter,
                "met_thresholds": False
            }
            
            # Run individual evaluations on best image
            if iteration_result["best_image"]:
                # Simplicity evaluation
                if self.evaluators.get('simplicity'):
                    simplicity_result = self.evaluate_simplicity(iteration_result["best_image"])
                    iteration_eval["simplicity"] = simplicity_result
                
                # Accuracy evaluation  
                if self.evaluators.get('accuracy'):
                    accuracy_result = self.evaluate_accuracy(iteration_result["best_image"], current_prompt)
                    iteration_eval["accuracy"] = accuracy_result
            
            print(f"   📊 Overall Score: {final_score:.3f}")
            print(f"   🧩 Autism Score: {autism_score:.3f}")
            print(f"   🎭 Used IP-Adapter: {'Yes' if used_ip_adapter else 'No'}")
            
            # Check if thresholds met
            meets_quality = final_score >= quality_threshold
            meets_autism = autism_score >= autism_threshold
            
            if meets_quality and meets_autism:
                iteration_eval["met_thresholds"] = True
                print(f"   ✅ Thresholds met! Quality: {final_score:.3f} ≥ {quality_threshold}, Autism: {autism_score:.3f} ≥ {autism_threshold}")
                results["final_result"] = iteration_result
                results["improvement_achieved"] = True
                results["iterations"].append(iteration_eval)
                break
            else:
                print(f"   ⚠️ Thresholds not met. Quality: {final_score:.3f}, Autism: {autism_score:.3f}")
            
            # Update best score
            if final_score > best_overall_score:
                best_overall_score = final_score
                results["final_result"] = iteration_result
            
            # Prepare next iteration
            if iteration < max_iterations - 1:
                current_prompt = iteration_result.get("final_prompt", current_prompt)
                if current_prompt == initial_prompt:
                    # Manual enhancement if no improvement suggested
                    current_prompt = self._enhance_prompt_for_next_iteration(current_prompt, iteration_eval)
            
            results["iterations"].append(iteration_eval)
        
        results["total_iterations"] = len(results["iterations"])
        
        # Summary
        if results["improvement_achieved"]:
            print(f"\n🎉 SUCCESS! Achieved target quality in {results['total_iterations']} iterations")
        else:
            print(f"\n⚠️ Reached max iterations. Best score: {best_overall_score:.3f}")
        
        if return_summary:
            return self._summarize_progressive_results(results)
        
        return results
    
    def _enhance_prompt_for_next_iteration(self, prompt, current_eval):
        """Manually enhance prompt if automatic improvement didn't work"""
        enhancements = []
        
        # Add autism-specific improvements
        if current_eval.get("autism_score", 0) < 0.6:
            enhancements.extend(["single person", "simple background", "clear character"])
        
        # Add quality improvements
        if current_eval.get("overall_score", 0) < 0.7:
            enhancements.extend(["detailed", "high quality", "well-defined"])
        
        if enhancements:
            enhanced = prompt + ", " + ", ".join(enhancements)
            print(f"   🔧 Manual enhancement applied: {', '.join(enhancements)}")
            return enhanced
        
    def _enhance_prompt_for_tifa(self, prompt, tifa_result, image):
        """Enhance prompt specifically for TIFA improvement"""
        details = tifa_result.get("details", {})
        clip_score = details.get("clip_score", 0.5)
        semantic_score = details.get("semantic_score", 0.5)
        artifacts = details.get("artifacts_detected", {})
        
        enhancements = []
        
        # Low CLIP score improvements
        if clip_score < 0.6:
            enhancements.extend(["detailed", "accurate representation", "clear depiction"])
            print(f"   🔧 Added CLIP improvements: low score {clip_score:.3f}")
        
        # Low semantic score improvements  
        if semantic_score < 0.6:
            enhancements.extend(["specific", "well-defined", "precise"])
            print(f"   🔧 Added semantic improvements: low score {semantic_score:.3f}")
        
        # Artifact-based improvements
        if artifacts.get("severity") == "high":
            enhancements.extend(["high quality", "clean", "professional"])
            print(f"   🔧 Added quality improvements: artifacts detected")
        
        # Visual quality improvements
        visual_quality = details.get("visual_quality_score", 1.0)
        if visual_quality < 0.7:
            enhancements.extend(["sharp", "well-lit", "clear image"])
            print(f"   🔧 Added visual improvements: low quality {visual_quality:.3f}")
        
        if enhancements:
            enhanced = prompt + ", " + ", ".join(enhancements)
            print(f"   ✨ TIFA enhancement applied: {', '.join(enhancements)}")
            return enhanced
        
        return prompt
    
    def _enhance_prompt_for_simplicity(self, prompt, simplicity_result):
        """Enhance prompt specifically for autism-friendly simplicity"""
        enhancements = []
        analysis = simplicity_result["full_analysis"]
        
        # Person count issues
        if not simplicity_result["person_compliant"]:
            enhancements.extend(["single person", "one character only", "solo"])
            print(f"   🔧 Added person count fix: {simplicity_result['person_count']} people detected")
        
        # Background simplicity
        if simplicity_result["background_score"] < 0.6:
            enhancements.extend(["simple background", "clean background", "minimal background"])
            print(f"   🔧 Added background simplicity: score {simplicity_result['background_score']:.3f}")
        
        # Color appropriateness
        if simplicity_result["color_score"] < 0.6:
            enhancements.extend(["soft colors", "muted colors", "gentle color palette"])
            print(f"   🔧 Added color improvements: score {simplicity_result['color_score']:.3f}")
        
        # Character clarity
        if simplicity_result["clarity_score"] < 0.6:
            enhancements.extend(["clear character", "well-defined character", "sharp character"])
            print(f"   🔧 Added clarity improvements: score {simplicity_result['clarity_score']:.3f}")
        
        # Sensory friendliness
        if simplicity_result["sensory_score"] < 0.6:
            enhancements.extend(["calm scene", "peaceful", "non-overwhelming"])
            print(f"   🔧 Added sensory improvements: score {simplicity_result['sensory_score']:.3f}")
        
        # General autism-friendly terms
        autism_terms = ["autism-friendly", "educational style", "storybook illustration"]
        enhancements.extend(autism_terms[:1])  # Add one general term
        
        if enhancements:
            enhanced = prompt + ", " + ", ".join(enhancements)
            print(f"   ✨ Simplicity enhancement applied: {', '.join(enhancements)}")
            return enhanced
        
        return prompt
    
    def _enhance_prompt_for_consistency(self, prompt, caption_issues, consistency_score):
        """Enhance prompt specifically for character consistency"""
        enhancements = []
        
        # Character inconsistency fixes
        if caption_issues.get("character_inconsistencies"):
            enhancements.extend(["same character", "consistent character", "matching appearance"])
            print(f"   🔧 Added character consistency fixes")
        
        # Multiple people issues
        if caption_issues.get("scene_issues"):
            enhancements.extend(["single character", "one person only", "solo character"])
            print(f"   🔧 Added scene consistency fixes")
        
        # Low consistency score
        if consistency_score < 0.6:
            enhancements.extend(["consistent design", "same person", "identical character"])
            print(f"   🔧 Added consistency boost: score {consistency_score:.3f}")
        
        # Hair color enforcement (if established)
        if hasattr(self.caption_analyzer, 'character_traits'):
            hair_color = self.caption_analyzer.character_traits.get("hair_color")
            if hair_color:
                enhancements.append(f"{hair_color} hair")
                print(f"   🔧 Enforced hair color: {hair_color}")
        
        # Style consistency
        enhancements.extend(["same art style", "consistent style", "matching visual style"])
        
        if enhancements:
            enhanced = prompt + ", " + ", ".join(enhancements)
            print(f"   ✨ Consistency enhancement applied: {', '.join(enhancements)}")
            return enhanced
        
        return prompt
    
    def _summarize_tifa_results(self, results):
        """Create summary of TIFA-focused results"""
        if not results["iterations"]:
            return {"error": "No iterations completed"}
        
        first_iter = results["iterations"][0]
        last_iter = results["iterations"][-1]
        
        tifa_improvement = last_iter["tifa_score"] - first_iter["tifa_score"]
        
        return {
            "metric": "self_improve_tifa",
            "success": results["improvement_achieved"],
            "total_iterations": results["total_iterations"],
            "initial_prompt": results["initial_prompt"],
            "final_prompt": last_iter.get("prompt", ""),
            "tifa_improvement": tifa_improvement,
            "final_tifa_score": last_iter["tifa_score"],
            "met_threshold": last_iter.get("met_threshold", False),
            "iterations_detail": results["iterations"]
        }
    
    def _summarize_simplicity_results(self, results):
        """Create summary of simplicity-focused results"""
        if not results["iterations"]:
            return {"error": "No iterations completed"}
        
        first_iter = results["iterations"][0]
        last_iter = results["iterations"][-1]
        
        autism_improvement = last_iter["autism_score"] - first_iter["autism_score"]
        
        return {
            "metric": "self_improve_simplicity",
            "success": results["improvement_achieved"],
            "total_iterations": results["total_iterations"],
            "initial_prompt": results["initial_prompt"],
            "final_prompt": last_iter.get("prompt", ""),
            "autism_improvement": autism_improvement,
            "final_autism_score": last_iter["autism_score"],
            "final_person_compliant": last_iter["person_compliant"],
            "met_threshold": last_iter.get("met_threshold", False),
            "iterations_detail": results["iterations"]
        }
    
    def _summarize_consistency_results(self, results):
        """Create summary of consistency-focused results"""
        if not results["iterations"]:
            return {"error": "No iterations completed"}
        
        first_iter = results["iterations"][0]
        last_iter = results["iterations"][-1]
        
        consistency_improvement = last_iter["consistency_score"] - first_iter["consistency_score"]
        
        return {
            "metric": "self_improve_consistency",
            "success": results["improvement_achieved"],
            "total_iterations": results["total_iterations"],
            "initial_prompt": results["initial_prompt"],
            "final_prompt": last_iter.get("prompt", ""),
            "consistency_improvement": consistency_improvement,
            "final_consistency_score": last_iter["consistency_score"],
            "final_style_consistency": last_iter["style_consistency"],
            "met_threshold": last_iter.get("met_threshold", False),
            "iterations_detail": results["iterations"]
        }
    
    def _summarize_progressive_results(self, results):
        """Create summary of progressive iteration results"""
        if not results["iterations"]:
            return {"error": "No iterations completed"}
        
        first_iter = results["iterations"][0]
        last_iter = results["iterations"][-1]
        
        score_improvement = last_iter["overall_score"] - first_iter["overall_score"]
        autism_improvement = last_iter["autism_score"] - first_iter["autism_score"]
        
        return {
            "metric": "progressive",
            "success": results["improvement_achieved"],
            "total_iterations": results["total_iterations"],
            "initial_prompt": results["initial_prompt"],
            "final_prompt": last_iter.get("final_prompt", ""),
            "score_improvement": score_improvement,
            "autism_improvement": autism_improvement,
            "final_overall_score": last_iter["overall_score"],
            "final_autism_score": last_iter["autism_score"],
            "met_thresholds": last_iter.get("met_thresholds", False),
            "iterations_detail": results["iterations"]
        }
    
    def evaluate_all(self, image, prompt=None, images_for_consistency=None, prompts_for_consistency=None):
        """
        Evaluate image(s) with all available metrics
        
        Args:
            image: Primary image (PIL Image or path)
            prompt: Prompt for accuracy evaluation
            images_for_consistency: List of images for consistency check
            prompts_for_consistency: List of prompts for consistency
        
        Returns:
            dict: Combined results from all evaluations
        """
        results = {"evaluated_metrics": []}
        
        # Simplicity evaluation
        if self.evaluators.get('simplicity'):
            results["simplicity"] = self.evaluate_simplicity(image)
            results["evaluated_metrics"].append("simplicity")
        
        # Accuracy evaluation (if prompt provided)
        if self.evaluators.get('accuracy') and prompt:
            results["accuracy"] = self.evaluate_accuracy(image, prompt)
            results["evaluated_metrics"].append("accuracy")
        
        # Consistency evaluation (if multiple images provided)
        if self.evaluators.get('consistency') and images_for_consistency:
            all_images = [image] + list(images_for_consistency)
            all_prompts = None
            if prompts_for_consistency:
                all_prompts = [prompt or "Primary image"] + list(prompts_for_consistency)
            
            results["consistency"] = self.evaluate_consistency(all_images, all_prompts)
            results["evaluated_metrics"].append("consistency")
        
        return results
    
    def quick_evaluate(self, image, prompt=None, metric_choice=None):
        """
        Quick evaluation with user choice
        
        Args:
            image: PIL Image or path to image
            prompt: Optional prompt for accuracy evaluation
            metric_choice: Optional pre-selected metric ('simplicity', 'consistency', 'accuracy', 'all', 'exit')
        
        Returns:
            dict: Evaluation results
        """
        # Get metric choice if not provided
        if metric_choice is None:
            metric_choice = self.get_metric_choice()
        
        if metric_choice == "exit":
            return {"message": "Evaluation cancelled"}
        
        # Load image if path
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")
        
        if metric_choice == "simplicity":
            return self.evaluate_simplicity(image)
        
        elif metric_choice == "accuracy":
            if not prompt:
                prompt = input("Enter the prompt used to generate this image: ").strip()
            return self.evaluate_accuracy(image, prompt)
        
        elif metric_choice == "prompt_improvement":
            if not prompt:
                prompt = input("Enter the original prompt used for this image: ").strip()
            return self.evaluate_prompt_improvement(image, prompt)
        
        elif metric_choice == "progressive":
            if not prompt:
                prompt = input("Enter the initial prompt for progressive iteration: ").strip()
            print("⚠️ Progressive iteration requires integration with generation pipeline")
            print("💡 Use this in your test script with: evaluator.evaluate_progressive_iteration(prompt, pipeline)")
            return {"message": "Progressive iteration requires pipeline integration"}
        
        elif metric_choice == "self_improve_tifa":
            if not prompt:
                prompt = input("Enter the initial prompt for TIFA self-improvement: ").strip()
            print("⚠️ TIFA self-improvement requires integration with generation pipeline")
            print("💡 Use this in your test script with: evaluator.self_improve_tifa(prompt, pipeline)")
            return {"message": "TIFA self-improvement requires pipeline integration"}
        
        elif metric_choice == "self_improve_simplicity":
            if not prompt:
                prompt = input("Enter the initial prompt for simplicity self-improvement: ").strip()
            print("⚠️ Simplicity self-improvement requires integration with generation pipeline")
            print("💡 Use this in your test script with: evaluator.self_improve_simplicity(prompt, pipeline)")
            return {"message": "Simplicity self-improvement requires pipeline integration"}
        
        elif metric_choice == "self_improve_consistency":
            if not prompt:
                prompt = input("Enter the initial prompt for consistency self-improvement: ").strip()
            print("⚠️ Consistency self-improvement requires integration with generation pipeline")
            print("💡 Use this in your test script with: evaluator.self_improve_consistency(prompt, pipeline)")
            return {"message": "Consistency self-improvement requires pipeline integration"}
        
        elif metric_choice == "consistency":
            print("Consistency evaluation requires multiple images.")
            return {"error": "Consistency needs multiple images - use evaluate_consistency() directly"}
        
        elif metric_choice == "all":
            if not prompt:
                prompt = input("Enter the prompt for accuracy evaluation (optional): ").strip()
                prompt = prompt if prompt else None
            return self.evaluate_all(image, prompt)
        
        return {"error": "Invalid metric choice"}
    
    def _get_quality_grade(self, score):
        """Convert score to quality grade"""
        if score >= 0.8:
            return "Excellent"
        elif score >= 0.7:
            return "Good"
        elif score >= 0.6:
            return "Moderate"
        else:
            return "Poor"
    
    def print_results(self, results):
        """Pretty print evaluation results"""
        if "error" in results:
            print(f"❌ {results['error']}")
            return
        
        metric = results.get("metric", "unknown")
        
        if metric == "simplicity":
            print(f"\n🧩 SIMPLICITY EVALUATION")
            print(f"Overall Score: {results['overall_score']:.3f}")
            print(f"Grade: {results['grade']}")
            print(f"👥 Person Count: {results['person_count']} ({'✅' if results['person_compliant'] else '❌'})")
            print(f"🎨 Background: {results['background_score']:.3f}")
            print(f"🌈 Colors: {results['color_score']:.3f}")
            print(f"👤 Clarity: {results['clarity_score']:.3f}")
            print(f"🧩 Sensory: {results['sensory_score']:.3f}")
            print(f"🎯 Focus: {results['focus_score']:.3f}")
            
            if results['top_recommendations']:
                print("💡 Top Recommendations:")
                for i, rec in enumerate(results['top_recommendations'], 1):
                    print(f"   {i}. {rec}")
        
        elif metric == "consistency":
            print(f"\n🔄 CONSISTENCY EVALUATION")
            print(f"Overall Consistency: {results['overall_consistency']:.3f}")
            print(f"Style Consistency: {results['overall_style']:.3f}")
            print(f"Grade: {results['consistency_grade']}")
            print(f"Images Analyzed: {results['image_count']}")
        
        elif metric == "accuracy":
            print(f"\n🎯 ACCURACY EVALUATION")
            print(f"Overall Score: {results['overall_score']:.3f}")
            print(f"Quality Grade: {results['quality_grade']}")
            print(f"🔗 CLIP Score: {results['clip_score']:.3f}")
            print(f"🧠 Semantic: {results['semantic_score']:.3f}")
            print(f"👁️ Visual Quality: {results['visual_quality']:.3f}")
            print(f"🔍 Artifacts: {results['artifact_severity']}")
            
            if results['caption_issues']:
                print("⚠️ Caption Issues:", ", ".join(results['caption_issues']))
            if results['visual_issues']:
                print("👁️ Visual Issues:", ", ".join(results['visual_issues']))
        
        elif metric == "prompt_improvement":
            print(f"\n📝 PROMPT IMPROVEMENT EVALUATION")
            print(f"CLIP Similarity: {results['clip_similarity']:.3f}")
            print(f"Alignment Quality: {results['alignment_quality']}")
            print(f"Generated Caption: {results['generated_caption']}")
            print(f"Consistency Issues: {results['consistency_severity']}")
            
            print(f"\n🔧 ORIGINAL PROMPT:")
            print(f"   {results['original_prompt']}")
            
            print(f"\n✨ IMPROVED POSITIVE:")
            print(f"   {results['improved_positive']}")
            
            print(f"\n❌ IMPROVED NEGATIVE:")
            print(f"   {results['improved_negative']}")
            
            if results['improvements_applied']:
                print("💡 Improvements Applied:")
                for i, improvement in enumerate(results['improvements_applied'][:3], 1):
                    print(f"   {i}. {improvement}")
        
        elif metric == "progressive":
            print(f"\n🔄 PROGRESSIVE ITERATION RESULTS")
            print(f"Success: {'✅ Yes' if results['success'] else '❌ No'}")
            print(f"Total Iterations: {results['total_iterations']}")
            print(f"Score Improvement: {results['score_improvement']:+.3f}")
            print(f"Autism Improvement: {results['autism_improvement']:+.3f}")
            print(f"Final Overall Score: {results['final_overall_score']:.3f}")
            print(f"Final Autism Score: {results['final_autism_score']:.3f}")
            print(f"Met Thresholds: {'✅' if results['met_thresholds'] else '❌'}")
            
            print(f"\n🔧 PROMPT EVOLUTION:")
            print(f"   Initial: {results['initial_prompt']}")
            print(f"   Final:   {results['final_prompt']}")
            
            if len(results['iterations_detail']) > 1:
                print(f"\n📈 ITERATION BREAKDOWN:")
                for i, iter_detail in enumerate(results['iterations_detail'], 1):
                    print(f"   {i}. Score: {iter_detail['overall_score']:.3f}, Autism: {iter_detail['autism_score']:.3f}")
        
        elif metric == "self_improve_tifa":
            print(f"\n🎯 TIFA SELF-IMPROVEMENT RESULTS")
            print(f"Success: {'✅ Yes' if results['success'] else '❌ No'}")
            print(f"Total Iterations: {results['total_iterations']}")
            print(f"TIFA Improvement: {results['tifa_improvement']:+.3f}")
            print(f"Final TIFA Score: {results['final_tifa_score']:.3f}")
            print(f"Met Threshold: {'✅' if results['met_threshold'] else '❌'}")
            
            print(f"\n🔧 PROMPT EVOLUTION:")
            print(f"   Initial: {results['initial_prompt']}")
            print(f"   Final:   {results['final_prompt']}")
        
        elif metric == "self_improve_simplicity":
            print(f"\n🧩 SIMPLICITY SELF-IMPROVEMENT RESULTS")
            print(f"Success: {'✅ Yes' if results['success'] else '❌ No'}")
            print(f"Total Iterations: {results['total_iterations']}")
            print(f"Autism Improvement: {results['autism_improvement']:+.3f}")
            print(f"Final Autism Score: {results['final_autism_score']:.3f}")
            print(f"Person Compliant: {'✅' if results['final_person_compliant'] else '❌'}")
            print(f"Met Threshold: {'✅' if results['met_threshold'] else '❌'}")
            
            print(f"\n🔧 PROMPT EVOLUTION:")
            print(f"   Initial: {results['initial_prompt']}")
            print(f"   Final:   {results['final_prompt']}")
        
        elif metric == "self_improve_consistency":
            print(f"\n🔄 CONSISTENCY SELF-IMPROVEMENT RESULTS")
            print(f"Success: {'✅ Yes' if results['success'] else '❌ No'}")
            print(f"Total Iterations: {results['total_iterations']}")
            print(f"Consistency Improvement: {results['consistency_improvement']:+.3f}")
            print(f"Final Consistency Score: {results['final_consistency_score']:.3f}")
            print(f"Final Style Consistency: {results['final_style_consistency']:.3f}")
            print(f"Met Threshold: {'✅' if results['met_threshold'] else '❌'}")
            
            print(f"\n🔧 PROMPT EVOLUTION:")
            print(f"   Initial: {results['initial_prompt']}")
            print(f"   Final:   {results['final_prompt']}")
        
        elif "evaluated_metrics" in results:
            print(f"\n🎯 COMPREHENSIVE EVALUATION")
            print(f"Metrics Evaluated: {', '.join(results['evaluated_metrics'])}")
            
            for metric_name in results['evaluated_metrics']:
                if metric_name in results:
                    print(f"\n--- {metric_name.upper()} ---")
                    self.print_results(results[metric_name])
    
    def reset_consistency_memory(self):
        """Reset consistency tracking for new image sequence"""
        if self.consistency_manager:
            self.consistency_manager.reset_memory()
        if self.caption_analyzer:
            self.caption_analyzer = CaptionConsistencyAnalyzer()
        self.reference_set = False
        if self.verbose:
            print("🔄 Consistency memory reset")


# Convenience functions for easy importing
def evaluate_simplicity(image, verbose=True):
    """Quick simplicity evaluation"""
    evaluator = MetricEvaluator(verbose=verbose)
    return evaluator.evaluate_simplicity(image)

def evaluate_consistency(images, prompts=None, verbose=True):
    """Quick consistency evaluation"""
    evaluator = MetricEvaluator(verbose=verbose)
    return evaluator.evaluate_consistency(images, prompts)

def evaluate_accuracy(image, prompt, verbose=True):
    """Quick accuracy evaluation"""
    evaluator = MetricEvaluator(verbose=verbose)
    return evaluator.evaluate_accuracy(image, prompt)

def self_improve_tifa(initial_prompt, pipeline, verbose=True):
    """Quick TIFA self-improvement"""
    evaluator = MetricEvaluator(verbose=verbose)
    return evaluator.self_improve_tifa(initial_prompt, pipeline)

def self_improve_simplicity(initial_prompt, pipeline, verbose=True):
    """Quick simplicity self-improvement"""
    evaluator = MetricEvaluator(verbose=verbose)
    return evaluator.self_improve_simplicity(initial_prompt, pipeline)

def self_improve_consistency(initial_prompt, pipeline, verbose=True):
    """Quick consistency self-improvement"""
    evaluator = MetricEvaluator(verbose=verbose)
    return evaluator.self_improve_consistency(initial_prompt, pipeline)

def quick_evaluate_with_choice(image, prompt=None):
    """Quick evaluation with user choice"""
    evaluator = MetricEvaluator(verbose=True)
    results = evaluator.quick_evaluate(image, prompt)
    evaluator.print_results(results)
    return results


# Example usage for test scripts
if __name__ == "__main__":
    print("🎯 Metric Evaluator Module")
    print("This module is designed to be imported into test scripts")
    print("\nExample usage:")
    print("  from metric_evaluator import MetricEvaluator, quick_evaluate_with_choice")
    print("  results = quick_evaluate_with_choice('path/to/image.jpg', 'your prompt')")
    print("\nProgressive iteration example:")
    print("  from metric_evaluator import evaluate_progressive_iteration")
    print("  results = evaluate_progressive_iteration('boy reading book', your_pipeline)")