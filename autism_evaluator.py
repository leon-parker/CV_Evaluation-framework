#!/usr/bin/env python3
"""
Autism Storyboard Evaluation Framework - Main Module
Integrates all evaluation metrics for comprehensive assessment
"""

import os
import json
import numpy as np
from pathlib import Path
from PIL import Image
from typing import Union, Dict, List, Optional, Tuple
import torch
import warnings
warnings.filterwarnings('ignore')

# Import all evaluation modules
from cv_metrics import VisualQualityAnalyzer
from prompt_metrics import PromptFaithfulnessAnalyzer
from consistency_metrics import ConsistencyAnalyzer
from complexity_metrics import AutismComplexityAnalyzer
from evaluation_config import AUTISM_EVALUATION_WEIGHTS, METRIC_THRESHOLDS
from utils import ImageUtils, ReportGenerator, VisualizationTools


class AutismStoryboardEvaluator:
    """
    Comprehensive evaluation system for autism-appropriate storyboard images
    
    This evaluator integrates multiple assessment criteria:
    1. Visual Quality - Technical image quality and artifacts
    2. Prompt Faithfulness - Semantic alignment with text description
    3. Character Consistency - Identity preservation across sequences
    4. Autism Complexity - Specialized metrics for autism education
    
    The combined score reflects overall suitability for autism education use.
    """
    
    def __init__(self, 
                 device: str = "cuda" if torch.cuda.is_available() else "cpu",
                 verbose: bool = True):
        """
        Initialize the autism storyboard evaluator
        
        Args:
            device: Computing device ('cuda' or 'cpu')
            verbose: Whether to print progress messages
        """
        self.device = device
        self.verbose = verbose
        self.evaluators = {}
        
        if self.verbose:
            print("🚀 Initializing Autism Storyboard Evaluator")
            print("=" * 60)
            print(f"Device: {self.device}")
        
        self._load_evaluators()
        
        if self.verbose:
            print("\n✅ Autism Storyboard Evaluator ready!")
            print("=" * 60)
    
    def _load_evaluators(self):
        """Load all evaluation modules"""
        
        # Visual Quality
        if self.verbose:
            print("\n📊 Loading Visual Quality Analyzer...")
        try:
            self.evaluators['visual_quality'] = VisualQualityAnalyzer()
            if self.verbose:
                print("   ✓ Visual quality analyzer loaded")
        except Exception as e:
            print(f"   ✗ Failed to load visual quality: {e}")
            self.evaluators['visual_quality'] = None
        
        # Prompt Faithfulness
        if self.verbose:
            print("\n🎯 Loading Prompt Faithfulness Analyzer...")
        try:
            self.evaluators['prompt_faithfulness'] = PromptFaithfulnessAnalyzer(self.device)
            if self.verbose:
                print("   ✓ Prompt faithfulness analyzer loaded")
        except Exception as e:
            print(f"   ✗ Failed to load prompt faithfulness: {e}")
            self.evaluators['prompt_faithfulness'] = None
        
        # Character Consistency
        if self.verbose:
            print("\n🎭 Loading Character Consistency Analyzer...")
        try:
            self.evaluators['consistency'] = ConsistencyAnalyzer(self.device)
            if self.verbose:
                print("   ✓ Consistency analyzer loaded")
        except Exception as e:
            print(f"   ✗ Failed to load consistency: {e}")
            self.evaluators['consistency'] = None
        
        # Autism Complexity
        if self.verbose:
            print("\n🧩 Loading Autism Complexity Analyzer...")
        try:
            self.evaluators['complexity'] = AutismComplexityAnalyzer()
            if self.verbose:
                print("   ✓ Complexity analyzer loaded")
        except Exception as e:
            print(f"   ✗ Failed to load complexity: {e}")
            self.evaluators['complexity'] = None
    
    def evaluate_single_image(self,
                            image: Union[str, Path, Image.Image, np.ndarray],
                            prompt: str,
                            reference_image: Optional[Union[str, Path, Image.Image]] = None,
                            save_report: bool = True,
                            output_dir: str = "evaluation_results") -> Dict:
        """
        Comprehensive evaluation of a single image
        
        Args:
            image: Image to evaluate (path, PIL Image, or numpy array)
            prompt: Text prompt that was used to generate the image
            reference_image: Optional reference for consistency checking
            save_report: Whether to save detailed reports
            output_dir: Directory for saving results
            
        Returns:
            Dictionary containing:
            - combined_score: Overall autism appropriateness (0-1)
            - autism_grade: Letter grade assessment
            - scores: Individual metric scores
            - metrics: Detailed metric data
            - recommendations: Specific improvement suggestions
        """
        
        if self.verbose:
            print(f"\n🔍 Evaluating single image")
            print(f"   Prompt: {prompt[:60]}...")
        
        # Load and prepare image
        image = ImageUtils.load_image(image)
        image_name = "evaluated_image"
        
        # Initialize results
        results = {
            'image_name': image_name,
            'prompt': prompt,
            'metrics': {},
            'scores': {},
            'combined_score': 0.0,
            'autism_grade': 'F',
            'recommendations': []
        }
        
        # 1. Visual Quality Assessment
        if self.evaluators.get('visual_quality'):
            if self.verbose:
                print("\n   📊 Analyzing visual quality...")
            try:
                vq_results = self.evaluators['visual_quality'].analyze_image_quality(image)
                results['metrics']['visual_quality'] = vq_results
                results['scores']['visual_quality'] = vq_results['quality_score']
                if self.verbose:
                    print(f"      Quality score: {vq_results['quality_score']:.3f}")
            except Exception as e:
                print(f"      ✗ Visual quality failed: {e}")
                results['scores']['visual_quality'] = 0.7  # Default
        
        # 2. Prompt Faithfulness Assessment
        if self.evaluators.get('prompt_faithfulness'):
            if self.verbose:
                print("\n   🎯 Analyzing prompt faithfulness...")
            try:
                pf_score = self.evaluators['prompt_faithfulness'].evaluate_prompt_alignment(
                    image, prompt
                )
                results['metrics']['prompt_faithfulness'] = {'score': pf_score}
                results['scores']['prompt_faithfulness'] = pf_score
                if self.verbose:
                    print(f"      Faithfulness score: {pf_score:.3f}")
            except Exception as e:
                print(f"      ✗ Prompt faithfulness failed: {e}")
                results['scores']['prompt_faithfulness'] = 0.7
        
        # 3. Autism Complexity Analysis
        if self.evaluators.get('complexity'):
            if self.verbose:
                print("\n   🧩 Analyzing autism-specific complexity...")
            try:
                complexity_results = self.evaluators['complexity'].analyze_complexity(image)
                results['metrics']['complexity'] = complexity_results
                
                # Extract individual scores
                results['scores']['person_count'] = complexity_results['person_count']['score']
                results['scores']['background_simplicity'] = complexity_results['background_simplicity']['score']
                results['scores']['color_appropriateness'] = complexity_results['color_appropriateness']['score']
                results['scores']['character_clarity'] = complexity_results['character_clarity']['score']
                results['scores']['sensory_friendliness'] = complexity_results['sensory_friendliness']['score']
                
                if self.verbose:
                    print(f"      Autism suitability: {complexity_results['autism_suitability']:.3f}")
                    print(f"      Person count: {complexity_results['person_count']['count']}")
            except Exception as e:
                print(f"      ✗ Complexity analysis failed: {e}")
        
        # 4. Character Consistency (if reference provided)
        if reference_image and self.evaluators.get('consistency'):
            if self.verbose:
                print("\n   🎭 Analyzing character consistency...")
            try:
                consistency_results = self.evaluators['consistency'].evaluate_consistency(
                    reference_image, image
                )
                results['metrics']['consistency'] = consistency_results
                results['scores']['character_consistency'] = consistency_results['character_consistency']
                results['scores']['style_consistency'] = consistency_results['style_consistency']
                
                if self.verbose:
                    print(f"      Character consistency: {consistency_results['character_consistency']:.3f}")
            except Exception as e:
                print(f"      ✗ Consistency analysis failed: {e}")
        
        # Calculate combined score
        results['combined_score'] = self._calculate_combined_score(results['scores'])
        results['autism_grade'] = self._get_autism_grade(results['combined_score'])
        results['recommendations'] = self._generate_recommendations(results)
        
        if self.verbose:
            print(f"\n   🏆 Overall Score: {results['combined_score']:.3f}")
            print(f"   📊 Grade: {results['autism_grade']}")
        
        # Save reports if requested
        if save_report:
            self._save_evaluation_results(results, image, output_dir)
        
        return results
    
    def evaluate_sequence(self,
                         images: List[Union[str, Path, Image.Image]],
                         prompts: List[str],
                         sequence_name: str = "storyboard",
                         save_report: bool = True,
                         output_dir: str = "evaluation_results") -> Dict:
        """
        Evaluate a sequence of images as a storyboard
        
        Args:
            images: List of images in the sequence
            prompts: Corresponding prompts for each image
            sequence_name: Name for this storyboard
            save_report: Whether to save reports
            output_dir: Output directory
            
        Returns:
            Dictionary with sequence evaluation results
        """
        
        if self.verbose:
            print(f"\n📚 Evaluating storyboard sequence: {sequence_name}")
            print(f"   Number of frames: {len(images)}")
        
        sequence_results = {
            'sequence_name': sequence_name,
            'num_images': len(images),
            'image_results': [],
            'sequence_metrics': {},
            'overall_score': 0.0,
            'overall_grade': 'F',
            'recommendations': []
        }
        
        # Evaluate each image
        for i, (img, prompt) in enumerate(zip(images, prompts)):
            if self.verbose:
                print(f"\n   --- Frame {i+1}/{len(images)} ---")
            
            # Use previous image as reference for consistency
            ref_image = images[i-1] if i > 0 else None
            
            img_results = self.evaluate_single_image(
                img, prompt, ref_image, save_report=False
            )
            img_results['frame_number'] = i + 1
            sequence_results['image_results'].append(img_results)
        
        # Analyze sequence-wide consistency
        if self.evaluators.get('consistency') and len(images) > 1:
            if self.verbose:
                print("\n   🎭 Analyzing sequence-wide consistency...")
            
            self.evaluators['consistency'].reset_sequence()
            for img in images:
                self.evaluators['consistency'].add_image_to_sequence(img)
            
            seq_consistency = self.evaluators['consistency'].evaluate_consistency()
            sequence_results['sequence_metrics']['consistency'] = seq_consistency
        
        # Calculate overall scores
        all_scores = [r['combined_score'] for r in sequence_results['image_results']]
        sequence_results['overall_score'] = float(np.mean(all_scores))
        sequence_results['overall_grade'] = self._get_autism_grade(sequence_results['overall_score'])
        
        # Generate sequence-level recommendations
        sequence_results['recommendations'] = self._generate_sequence_recommendations(sequence_results)
        
        if self.verbose:
            print(f"\n   🏆 Overall Sequence Score: {sequence_results['overall_score']:.3f}")
            print(f"   📊 Grade: {sequence_results['overall_grade']}")
        
        # Save reports
        if save_report:
            self._save_sequence_results(sequence_results, output_dir)
        
        return sequence_results
    
    def _calculate_combined_score(self, scores: Dict[str, float]) -> float:
        """Calculate weighted combined score"""
        
        weights = AUTISM_EVALUATION_WEIGHTS
        weighted_sum = 0.0
        total_weight = 0.0
        
        # Map scores to weights
        score_mapping = {
            'visual_quality': 'visual_quality',
            'prompt_faithfulness': 'prompt_faithfulness',
            'person_count': 'person_count',
            'background_simplicity': 'background_simplicity',
            'color_appropriateness': 'color_appropriateness',
            'character_clarity': 'character_clarity',
            'sensory_friendliness': 'sensory_friendliness'
        }
        
        for score_key, weight_key in score_mapping.items():
            if score_key in scores and weight_key in weights:
                weighted_sum += scores[score_key] * weights[weight_key]
                total_weight += weights[weight_key]
        
        # Handle consistency if available
        if 'character_consistency' in scores and 'style_consistency' in scores:
            consistency_score = (scores['character_consistency'] * 0.7 + 
                               scores['style_consistency'] * 0.3)
            weighted_sum += consistency_score * weights.get('consistency', 0.1)
            total_weight += weights.get('consistency', 0.1)
        
        if total_weight == 0:
            return 0.5
        
        combined = weighted_sum / total_weight
        
        # Apply critical penalties
        if scores.get('person_count', 1.0) < 0.5:  # Too many people
            combined *= 0.8  # 20% penalty
        
        return float(max(0.0, min(1.0, combined)))
    
    def _get_autism_grade(self, score: float) -> str:
        """Convert numeric score to letter grade"""
        if score >= 0.90:
            return "A+ (Excellent for autism education)"
        elif score >= 0.85:
            return "A (Very suitable)"
        elif score >= 0.80:
            return "B+ (Good for autism education)"
        elif score >= 0.75:
            return "B (Suitable with minor improvements)"
        elif score >= 0.70:
            return "C+ (Acceptable with improvements)"
        elif score >= 0.65:
            return "C (Needs improvements)"
        elif score >= 0.60:
            return "D+ (Significant issues)"
        elif score >= 0.55:
            return "D (Many issues)"
        else:
            return "F (Not suitable for autism education)"
    
    def _generate_recommendations(self, results: Dict) -> List[str]:
        """Generate specific improvement recommendations"""
        recommendations = []
        scores = results.get('scores', {})
        metrics = results.get('metrics', {})
        
        # Check each metric against thresholds
        if scores.get('visual_quality', 1.0) < METRIC_THRESHOLDS['visual_quality']:
            recommendations.append("📸 Improve image quality: reduce blur/noise/artifacts")
        
        if scores.get('prompt_faithfulness', 1.0) < METRIC_THRESHOLDS['prompt_faithfulness']:
            recommendations.append("🎯 Improve prompt accuracy: ensure image matches description")
        
        # Autism-specific recommendations
        if 'complexity' in metrics:
            complexity = metrics['complexity']
            
            # Person count
            person_count = complexity['person_count']['count']
            if person_count > 2:
                recommendations.append(f"👥 CRITICAL: Reduce to 1-2 people (currently {person_count})")
            elif person_count == 0:
                recommendations.append("👤 Add clear main character to scene")
            
            # Background
            if complexity['background_simplicity']['score'] < METRIC_THRESHOLDS['background_simplicity']:
                recommendations.append("🎨 Simplify background: remove distracting elements")
            
            # Colors
            if complexity['color_appropriateness']['dominant_colors'] > 6:
                recommendations.append("🌈 Reduce color palette to 4-6 main colors")
            
            # Clarity
            if complexity['character_clarity']['score'] < METRIC_THRESHOLDS['character_clarity']:
                recommendations.append("✏️ Improve character definition: clearer outlines")
        
        # Overall assessment
        if results['combined_score'] >= 0.85:
            recommendations.insert(0, "🌟 Excellent for autism education! Minor refinements only.")
        elif results['combined_score'] >= 0.70:
            recommendations.insert(0, "✅ Good foundation - apply recommendations for excellence")
        else:
            recommendations.insert(0, "⚠️ Significant improvements needed for autism suitability")
        
        return recommendations
    
    def _generate_sequence_recommendations(self, sequence_results: Dict) -> List[str]:
        """Generate recommendations for entire sequence"""
        recommendations = []
        
        # Check consistency across sequence
        if 'consistency' in sequence_results['sequence_metrics']:
            consistency = sequence_results['sequence_metrics']['consistency']
            if consistency['combined_consistency'] < 0.75:
                recommendations.append("🎭 Improve character consistency across frames")
            if consistency['drift_score'] > 0.2:
                recommendations.append("📉 Reduce visual drift between frames")
        
        # Check for common issues across frames
        all_scores = sequence_results['image_results']
        
        # Average person count
        avg_people = np.mean([r['metrics']['complexity']['person_count']['count'] 
                            for r in all_scores if 'complexity' in r['metrics']])
        if avg_people > 2:
            recommendations.append(f"👥 Reduce character count across sequence (avg: {avg_people:.1f})")
        
        # Consistency in quality
        quality_scores = [r['combined_score'] for r in all_scores]
        quality_std = np.std(quality_scores)
        if quality_std > 0.15:
            recommendations.append("📊 Improve consistency of quality across frames")
        
        return recommendations
    
    def _save_evaluation_results(self, results: Dict, image: Image.Image, output_dir: str):
        """Save evaluation results and visualizations"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate unique filename
        timestamp = Path(output_dir) / f"eval_{np.random.randint(1000, 9999)}"
        
        # Save text report
        text_report = ReportGenerator.generate_text_report(results, f"{timestamp}_report.txt")
        if self.verbose:
            print(f"\n   📄 Report saved: {timestamp}_report.txt")
        
        # Save JSON data
        ReportGenerator.generate_json_report(results, f"{timestamp}_data.json")
        
        # Create visual dashboard
        try:
            fig = VisualizationTools.create_evaluation_dashboard(
                results, image, f"{timestamp}_dashboard.png"
            )
            if self.verbose:
                print(f"   📊 Dashboard saved: {timestamp}_dashboard.png")
        except Exception as e:
            print(f"   ⚠️ Could not create dashboard: {e}")
    
    def _save_sequence_results(self, sequence_results: Dict, output_dir: str):
        """Save sequence evaluation results"""
        os.makedirs(output_dir, exist_ok=True)
        
        seq_name = sequence_results['sequence_name'].replace(' ', '_')
        base_path = Path(output_dir) / f"seq_{seq_name}"
        
        # Save comprehensive report
        with open(f"{base_path}_report.txt", 'w') as f:
            f.write(f"STORYBOARD SEQUENCE EVALUATION: {sequence_results['sequence_name']}\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Overall Score: {sequence_results['overall_score']:.3f}\n")
            f.write(f"Grade: {sequence_results['overall_grade']}\n")
            f.write(f"Number of frames: {sequence_results['num_images']}\n\n")
            
            f.write("FRAME-BY-FRAME RESULTS:\n")
            f.write("-" * 40 + "\n")
            for result in sequence_results['image_results']:
                f.write(f"\nFrame {result['frame_number']}:\n")
                f.write(f"  Score: {result['combined_score']:.3f}\n")
                f.write(f"  Grade: {result['autism_grade']}\n")
            
            f.write("\n\nRECOMMENDATIONS:\n")
            for rec in sequence_results['recommendations']:
                f.write(f"• {rec}\n")
        
        # Save JSON data
        ReportGenerator.generate_json_report(sequence_results, f"{base_path}_data.json")
        
        if self.verbose:
            print(f"\n   💾 Sequence results saved to {output_dir}")


# Convenience function for quick evaluation
def evaluate_image(image_path: str, prompt: str, verbose: bool = True) -> Dict:
    """
    Quick evaluation of a single image
    
    Args:
        image_path: Path to image file
        prompt: Text prompt used to generate the image
        verbose: Whether to print progress
        
    Returns:
        Evaluation results dictionary
    """
    evaluator = AutismStoryboardEvaluator(verbose=verbose)
    return evaluator.evaluate_single_image(image_path, prompt)


if __name__ == "__main__":
    # Example usage
    print("Autism Storyboard Evaluation Framework")
    print("Example usage:")
    print()
    print("from autism_evaluator import evaluate_image")
    print("results = evaluate_image('image.png', 'a boy brushing teeth')")
    print("print(f'Score: {results[\"combined_score\"]:.3f}')")
    print("print(f'Grade: {results[\"autism_grade\"]}')")