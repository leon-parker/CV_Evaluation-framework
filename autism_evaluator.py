#!/usr/bin/env python3
"""
Autism Storyboard Evaluation Framework - Main Module
Integrates all evaluation metrics for comprehensive assessment
UPDATED: Proper two-level hierarchy with normalized weights from dissertation
Level 1: Categories (Simplicity 36.36%, Accuracy 33.33%, Consistency 30.30%)
Level 2: Sub-metrics within each category
ENHANCED: Added overall autism appropriateness assessment for batch evaluation
"""

import os
import json
import numpy as np
from pathlib import Path
from PIL import Image
from typing import Union, Dict, List, Optional, Tuple
import torch
import warnings
from datetime import datetime
warnings.filterwarnings('ignore')

# Import all evaluation modules
from cv_metrics import VisualQualityAnalyzer
from prompt_metrics import PromptFaithfulnessAnalyzer
from consistency_metrics import ConsistencyAnalyzer
from complexity_metrics import AutismComplexityAnalyzer
from evaluation_config import (
    CATEGORY_WEIGHTS,
    SIMPLICITY_WEIGHTS,
    ACCURACY_WEIGHTS,
    CONSISTENCY_WEIGHTS,
    METRIC_THRESHOLDS
)
from utils import ImageUtils, ReportGenerator, VisualizationTools


class AutismStoryboardEvaluator:
    """
    Comprehensive evaluation system for autism-appropriate storyboard images
    
    Uses two-level scoring hierarchy:
    Level 1 - Main Categories (normalized from dissertation's 36/33/30):
    1. Simplicity (36.36%) - Autism-specific complexity metrics
    2. Accuracy (33.33%) - Visual Quality + Prompt Faithfulness
    3. Consistency (30.30%) - Character/style preservation across sequences
    
    Level 2 - Sub-metrics within each category with their own weights
    
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
        self.last_category_scores = {}  # Store category scores for reporting
        self.evaluation_history = []  # Store all evaluations for overall assessment
        
        if self.verbose:
            print("🚀 Initializing Autism Storyboard Evaluator")
            print("=" * 60)
            print(f"Device: {self.device}")
            print("📊 Two-Level Hierarchy (normalized from 36/33/30):")
            print("   Level 1: Simplicity (36.36%), Accuracy (33.33%), Consistency (30.30%)")
            print("   Level 2: Sub-metrics within each category")
        
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
            - category_scores: Breakdown by main categories
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
            'category_scores': {},
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
                results['scores']['focus_clarity'] = complexity_results.get('focus_clarity', {}).get('score', 0.5)
                
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
        
        # Calculate combined score using two-level hierarchy
        results['combined_score'] = self._calculate_combined_score(results['scores'])
        results['category_scores'] = self.last_category_scores.copy()  # Store category scores
        results['autism_grade'] = self._get_autism_grade(results['combined_score'])
        results['recommendations'] = self._generate_recommendations(results)
        
        # Add to evaluation history for overall assessment
        self.evaluation_history.append(results)
        
        if self.verbose:
            print(f"\n   🏆 Overall Score: {results['combined_score']:.3f}")
            print(f"   📊 Grade: {results['autism_grade']}")
        
        # Save reports if requested
        if save_report:
            self._save_evaluation_results(results, image, output_dir)
        
        return results
    
    def evaluate_batch(self,
                      images: List[Union[str, Path, Image.Image]],
                      prompts: List[str],
                      save_individual_reports: bool = False,
                      save_overall_report: bool = True,
                      output_dir: str = "evaluation_results") -> Dict:
        """
        Evaluate multiple images and provide overall autism appropriateness assessment
        
        Args:
            images: List of images to evaluate
            prompts: Corresponding prompts for each image
            save_individual_reports: Whether to save reports for each image
            save_overall_report: Whether to save overall assessment report
            output_dir: Directory for saving results
            
        Returns:
            Dictionary with individual results and overall assessment
        """
        if len(images) != len(prompts):
            raise ValueError("Number of images and prompts must match")
        
        if self.verbose:
            print(f"\n📚 Batch Evaluation: {len(images)} images")
            print("=" * 60)
        
        # Clear evaluation history for fresh batch
        self.evaluation_history = []
        
        # Evaluate each image
        batch_results = {
            'num_images': len(images),
            'individual_results': [],
            'overall_assessment': {}
        }
        
        for i, (img, prompt) in enumerate(zip(images, prompts)):
            if self.verbose:
                print(f"\n--- Image {i+1}/{len(images)} ---")
            
            result = self.evaluate_single_image(
                img, prompt, 
                save_report=save_individual_reports,
                output_dir=output_dir
            )
            batch_results['individual_results'].append(result)
        
        # Generate overall assessment
        batch_results['overall_assessment'] = self.generate_overall_assessment(
            batch_results['individual_results']
        )
        
        # Display overall assessment
        if self.verbose:
            self._print_overall_assessment(batch_results['overall_assessment'])
        
        # Save overall report if requested
        if save_overall_report:
            self._save_overall_assessment(batch_results, output_dir)
        
        return batch_results
    
    def generate_overall_assessment(self, evaluation_results: List[Dict]) -> Dict:
        """
        Generate an overall autism appropriateness assessment for multiple evaluated images
        
        Args:
            evaluation_results: List of evaluation result dictionaries
            
        Returns:
            Dictionary with overall assessment metrics
        """
        if not evaluation_results:
            return {'error': 'No evaluation results provided'}
        
        # Extract scores
        scores = [r.get('combined_score', 0) for r in evaluation_results]
        
        # Calculate statistics
        num_images = len(scores)
        avg_score = float(np.mean(scores))
        min_score = float(np.min(scores))
        max_score = float(np.max(scores))
        std_score = float(np.std(scores))
        
        # Calculate category averages
        category_averages = {
            'simplicity': [],
            'accuracy': [],
            'consistency': []
        }
        
        for result in evaluation_results:
            cat_scores = result.get('category_scores', {})
            for category in category_averages:
                if category in cat_scores and cat_scores[category] is not None:
                    category_averages[category].append(cat_scores[category])
        
        for category in category_averages:
            if category_averages[category]:
                category_averages[category] = float(np.mean(category_averages[category]))
            else:
                category_averages[category] = None
        
        # Categorize images by quality tier
        excellent_count = sum(1 for s in scores if s >= 0.90)
        good_count = sum(1 for s in scores if 0.80 <= s < 0.90)
        acceptable_count = sum(1 for s in scores if 0.60 <= s < 0.80)
        poor_count = sum(1 for s in scores if s < 0.60)
        
        # Autism appropriate threshold (0.7)
        appropriate_count = sum(1 for s in scores if s >= 0.70)
        appropriate_percentage = (appropriate_count / num_images * 100) if num_images > 0 else 0
        
        # Determine overall grade
        overall_grade = self._get_autism_grade(avg_score)
        
        # Determine overall assessment message
        if avg_score >= 0.85:
            assessment_message = "EXCELLENT: Highly suitable for autism education"
        elif avg_score >= 0.70:
            assessment_message = "GOOD: Suitable for autism education with minor improvements"
        elif avg_score >= 0.60:
            assessment_message = "ACCEPTABLE: Needs improvements for optimal autism suitability"
        else:
            assessment_message = "POOR: Significant improvements needed for autism education"
        
        # Identify common issues
        common_issues = self._identify_common_issues(evaluation_results)
        
        return {
            'num_images': num_images,
            'average_score': avg_score,
            'min_score': min_score,
            'max_score': max_score,
            'std_deviation': std_score,
            'overall_grade': overall_grade,
            'assessment_message': assessment_message,
            'category_averages': category_averages,
            'quality_distribution': {
                'excellent': excellent_count,
                'good': good_count,
                'acceptable': acceptable_count,
                'poor': poor_count
            },
            'quality_percentages': {
                'excellent': (excellent_count / num_images * 100) if num_images > 0 else 0,
                'good': (good_count / num_images * 100) if num_images > 0 else 0,
                'acceptable': (acceptable_count / num_images * 100) if num_images > 0 else 0,
                'poor': (poor_count / num_images * 100) if num_images > 0 else 0
            },
            'autism_appropriate': {
                'count': appropriate_count,
                'percentage': appropriate_percentage
            },
            'common_issues': common_issues
        }
    
    def _identify_common_issues(self, evaluation_results: List[Dict]) -> List[str]:
        """Identify common issues across evaluated images"""
        issues = []
        
        # Count issue frequencies
        too_many_people = 0
        complex_background = 0
        poor_clarity = 0
        too_many_colors = 0
        low_quality = 0
        poor_prompt = 0
        
        for result in evaluation_results:
            metrics = result.get('metrics', {})
            scores = result.get('scores', {})
            
            # Check complexity issues
            if 'complexity' in metrics:
                complexity = metrics['complexity']
                if complexity['person_count']['count'] > 2:
                    too_many_people += 1
                if complexity['background_simplicity']['score'] < 0.6:
                    complex_background += 1
                if complexity['color_appropriateness']['dominant_colors'] > 6:
                    too_many_colors += 1
                if complexity['character_clarity']['score'] < 0.6:
                    poor_clarity += 1
            
            # Check quality issues
            if scores.get('visual_quality', 1.0) < 0.7:
                low_quality += 1
            if scores.get('prompt_faithfulness', 1.0) < 0.7:
                poor_prompt += 1
        
        # Add issues if they affect >30% of images
        threshold = len(evaluation_results) * 0.3
        
        if too_many_people > threshold:
            issues.append(f"👥 Too many people in {too_many_people}/{len(evaluation_results)} images")
        if complex_background > threshold:
            issues.append(f"🎨 Complex backgrounds in {complex_background}/{len(evaluation_results)} images")
        if too_many_colors > threshold:
            issues.append(f"🌈 Excessive colors in {too_many_colors}/{len(evaluation_results)} images")
        if poor_clarity > threshold:
            issues.append(f"✏️ Poor character clarity in {poor_clarity}/{len(evaluation_results)} images")
        if low_quality > threshold:
            issues.append(f"📸 Visual quality issues in {low_quality}/{len(evaluation_results)} images")
        if poor_prompt > threshold:
            issues.append(f"🎯 Prompt alignment issues in {poor_prompt}/{len(evaluation_results)} images")
        
        if not issues:
            issues.append("✅ No common issues detected")
        
        return issues
    
    def _print_overall_assessment(self, assessment: Dict):
        """Print formatted overall assessment"""
        print("\n" + "=" * 70)
        print("📊 OVERALL AUTISM APPROPRIATENESS ASSESSMENT")
        print("=" * 70)
        
        print(f"Total Images Evaluated: {assessment['num_images']}")
        print(f"Average Autism Score: {assessment['average_score']:.3f}")
        print(f"Score Range: {assessment['min_score']:.3f} - {assessment['max_score']:.3f}")
        print(f"Standard Deviation: {assessment['std_deviation']:.3f}")
        print(f"Overall Grade: {assessment['overall_grade']}")
        print(f"\n{assessment['assessment_message']}")
        
        # Category breakdown
        if assessment['category_averages']:
            print("\nCategory Averages:")
            if assessment['category_averages']['simplicity'] is not None:
                print(f"  - Simplicity (36.36%): {assessment['category_averages']['simplicity']:.3f}")
            if assessment['category_averages']['accuracy'] is not None:
                print(f"  - Accuracy (33.33%): {assessment['category_averages']['accuracy']:.3f}")
            if assessment['category_averages']['consistency'] is not None:
                print(f"  - Consistency (30.30%): {assessment['category_averages']['consistency']:.3f}")
        
        # Quality distribution
        print("\nQuality Distribution:")
        dist = assessment['quality_distribution']
        pct = assessment['quality_percentages']
        print(f"  - Excellent (0.9+):     {dist['excellent']} images ({pct['excellent']:.0f}%)")
        print(f"  - Good (0.8-0.9):       {dist['good']} images ({pct['good']:.0f}%)")
        print(f"  - Acceptable (0.6-0.8): {dist['acceptable']} images ({pct['acceptable']:.0f}%)")
        print(f"  - Poor (<0.6):          {dist['poor']} images ({pct['poor']:.0f}%)")
        
        # Autism appropriateness
        print(f"\n✅ Autism-Appropriate (≥0.7): {assessment['autism_appropriate']['count']}/{assessment['num_images']} ({assessment['autism_appropriate']['percentage']:.0f}%)")
        
        # Common issues
        if assessment['common_issues']:
            print("\nCommon Issues Detected:")
            for issue in assessment['common_issues']:
                print(f"  {issue}")
        
        print("=" * 70)
    
    def _save_overall_assessment(self, batch_results: Dict, output_dir: str):
        """Save overall assessment report"""
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = Path(output_dir) / f"overall_assessment_{timestamp}.txt"
        json_path = Path(output_dir) / f"overall_assessment_{timestamp}.json"
        
        # Generate text report
        assessment = batch_results['overall_assessment']
        report = []
        report.append("OVERALL AUTISM APPROPRIATENESS ASSESSMENT")
        report.append("=" * 70)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        report.append(f"Total Images Evaluated: {assessment['num_images']}")
        report.append(f"Average Autism Score: {assessment['average_score']:.3f}")
        report.append(f"Score Range: {assessment['min_score']:.3f} - {assessment['max_score']:.3f}")
        report.append(f"Standard Deviation: {assessment['std_deviation']:.3f}")
        report.append(f"Overall Grade: {assessment['overall_grade']}")
        report.append("")
        report.append(f"ASSESSMENT: {assessment['assessment_message']}")
        report.append("")
        
        # Add detailed breakdown
        report.append("CATEGORY AVERAGES:")
        if assessment['category_averages']['simplicity'] is not None:
            report.append(f"  Simplicity (36.36%): {assessment['category_averages']['simplicity']:.3f}")
        if assessment['category_averages']['accuracy'] is not None:
            report.append(f"  Accuracy (33.33%): {assessment['category_averages']['accuracy']:.3f}")
        if assessment['category_averages']['consistency'] is not None:
            report.append(f"  Consistency (30.30%): {assessment['category_averages']['consistency']:.3f}")
        report.append("")
        
        report.append("QUALITY DISTRIBUTION:")
        dist = assessment['quality_distribution']
        pct = assessment['quality_percentages']
        report.append(f"  Excellent (0.9+):     {dist['excellent']} images ({pct['excellent']:.0f}%)")
        report.append(f"  Good (0.8-0.9):       {dist['good']} images ({pct['good']:.0f}%)")
        report.append(f"  Acceptable (0.6-0.8): {dist['acceptable']} images ({pct['acceptable']:.0f}%)")
        report.append(f"  Poor (<0.6):          {dist['poor']} images ({pct['poor']:.0f}%)")
        report.append("")
        
        report.append(f"AUTISM APPROPRIATENESS:")
        report.append(f"  Appropriate (≥0.7): {assessment['autism_appropriate']['count']}/{assessment['num_images']} ({assessment['autism_appropriate']['percentage']:.0f}%)")
        report.append("")
        
        if assessment['common_issues']:
            report.append("COMMON ISSUES:")
            for issue in assessment['common_issues']:
                report.append(f"  {issue}")
            report.append("")
        
        # Individual image summary
        report.append("INDIVIDUAL IMAGE SCORES:")
        report.append("-" * 40)
        for i, result in enumerate(batch_results['individual_results'], 1):
            report.append(f"Image {i}: {result['combined_score']:.3f} - {result['autism_grade']}")
        
        # Save text report
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report))
        
        # Save JSON data
        ReportGenerator.generate_json_report(batch_results, str(json_path))
        
        if self.verbose:
            print(f"\n📄 Overall assessment saved to: {report_path}")
            print(f"📊 Data saved to: {json_path}")
    
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
            'overall_category_scores': {},
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
        
        # Calculate average category scores
        for category in ['simplicity', 'accuracy', 'consistency']:
            category_scores = [r['category_scores'].get(category, 0) 
                             for r in sequence_results['image_results'] 
                             if category in r.get('category_scores', {})]
            if category_scores:
                sequence_results['overall_category_scores'][category] = float(np.mean(category_scores))
        
        sequence_results['overall_grade'] = self._get_autism_grade(sequence_results['overall_score'])
        
        # Generate sequence-level recommendations
        sequence_results['recommendations'] = self._generate_sequence_recommendations(sequence_results)
        
        # Add overall assessment
        sequence_results['overall_assessment'] = self.generate_overall_assessment(
            sequence_results['image_results']
        )
        
        if self.verbose:
            print(f"\n   🏆 Overall Sequence Score: {sequence_results['overall_score']:.3f}")
            print(f"   📊 Grade: {sequence_results['overall_grade']}")
            self._print_overall_assessment(sequence_results['overall_assessment'])
        
        # Save reports
        if save_report:
            self._save_sequence_results(sequence_results, output_dir)
        
        return sequence_results
    
    def _calculate_combined_score(self, scores: Dict[str, float]) -> float:
        """
        Calculate weighted combined score using two-level hierarchy:
        1. Calculate category scores (simplicity, accuracy, consistency)
        2. Weight categories for final score using normalized weights
        """
        
        # Initialize category scores
        category_scores = {
            'simplicity': 0.0,
            'accuracy': 0.0,
            'consistency': None  # Only calculated if consistency data exists
        }
        
        # ============================================================================
        # SIMPLICITY CATEGORY (36.36% of total when normalized)
        # ============================================================================
        simplicity_score = 0.0
        simplicity_weight_total = 0.0
        
        simplicity_metrics = {
            'person_count': scores.get('person_count', 0.0),
            'background_simplicity': scores.get('background_simplicity', 0.0),
            'color_appropriateness': scores.get('color_appropriateness', 0.0),
            'character_clarity': scores.get('character_clarity', 0.0),
            'sensory_friendliness': scores.get('sensory_friendliness', 0.0),
            'focus_clarity': scores.get('focus_clarity', 0.0)
        }
        
        for metric, score in simplicity_metrics.items():
            if metric in SIMPLICITY_WEIGHTS:
                weight = SIMPLICITY_WEIGHTS[metric]
                simplicity_score += score * weight
                simplicity_weight_total += weight
        
        if simplicity_weight_total > 0:
            category_scores['simplicity'] = simplicity_score / simplicity_weight_total
        else:
            category_scores['simplicity'] = 0.5  # Default if no simplicity metrics
        
        # ============================================================================
        # ACCURACY CATEGORY (33.33% of total when normalized)
        # ============================================================================
        accuracy_score = 0.0
        accuracy_weight_total = 0.0
        
        accuracy_metrics = {
            'prompt_faithfulness': scores.get('prompt_faithfulness', 0.0),
            'visual_quality': scores.get('visual_quality', 0.0)
        }
        
        for metric, score in accuracy_metrics.items():
            if metric in ACCURACY_WEIGHTS:
                weight = ACCURACY_WEIGHTS[metric]
                accuracy_score += score * weight
                accuracy_weight_total += weight
        
        if accuracy_weight_total > 0:
            category_scores['accuracy'] = accuracy_score / accuracy_weight_total
        else:
            category_scores['accuracy'] = 0.5  # Default if no accuracy metrics
        
        # ============================================================================
        # CONSISTENCY CATEGORY (30.30% of total when normalized, only if available)
        # ============================================================================
        if 'character_consistency' in scores or 'style_consistency' in scores:
            consistency_score = 0.0
            consistency_weight_total = 0.0
            
            consistency_metrics = {
                'character_consistency': scores.get('character_consistency', 0.0),
                'style_consistency': scores.get('style_consistency', 0.0)
            }
            
            for metric, score in consistency_metrics.items():
                if metric in CONSISTENCY_WEIGHTS:
                    weight = CONSISTENCY_WEIGHTS[metric]
                    consistency_score += score * weight
                    consistency_weight_total += weight
            
            if consistency_weight_total > 0:
                category_scores['consistency'] = consistency_score / consistency_weight_total
            else:
                category_scores['consistency'] = 0.5
        
        # ============================================================================
        # CALCULATE FINAL COMBINED SCORE
        # ============================================================================
        combined_score = 0.0
        
        # If consistency is not applicable, redistribute its weight proportionally
        if category_scores['consistency'] is None:
            # Without consistency: redistribute 30.30% proportionally between 36.36% and 33.33%
            # Original ratio: 36.36:33.33 = 1.09:1
            # New weights: 36.36/(36.36+33.33) = 0.522, 33.33/(36.36+33.33) = 0.478
            adjusted_weights = {
                'simplicity': 0.522,  # 36.36% / (36.36% + 33.33%) 
                'accuracy': 0.478     # 33.33% / (36.36% + 33.33%)
            }
            
            combined_score = (
                category_scores['simplicity'] * adjusted_weights['simplicity'] +
                category_scores['accuracy'] * adjusted_weights['accuracy']
            )
        else:
            # All three categories available - use normalized weights
            combined_score = (
                category_scores['simplicity'] * CATEGORY_WEIGHTS['simplicity'] +
                category_scores['accuracy'] * CATEGORY_WEIGHTS['accuracy'] +
                category_scores['consistency'] * CATEGORY_WEIGHTS['consistency']
            )
        
        # ============================================================================
        # APPLY CRITICAL PENALTIES
        # ============================================================================
        
        # Critical penalty for too many people (affects final score directly)
        person_count_score = scores.get('person_count', 1.0)
        if person_count_score < 0.5:  # Too many people (score < 0.5 means 3+ people)
            combined_score *= 0.8  # 20% penalty
            if self.verbose:
                print(f"   ⚠️ Applied 20% penalty for excessive person count")
        
        # Store category scores for reporting
        self.last_category_scores = category_scores
        
        # Ensure score is in valid range
        combined_score = float(max(0.0, min(1.0, combined_score)))
        
        if self.verbose:
            print(f"\n   📊 Category Scores:")
            print(f"      Simplicity: {category_scores['simplicity']:.3f} (36.36% weight)")
            print(f"      Accuracy: {category_scores['accuracy']:.3f} (33.33% weight)")
            if category_scores['consistency'] is not None:
                print(f"      Consistency: {category_scores['consistency']:.3f} (30.30% weight)")
            else:
                print(f"      Consistency: N/A (weights redistributed)")
            print(f"      Final Combined: {combined_score:.3f}")
        
        return combined_score
    
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
        """Generate specific improvement recommendations based on category scores"""
        recommendations = []
        scores = results.get('scores', {})
        metrics = results.get('metrics', {})
        category_scores = results.get('category_scores', {})
        
        # Check each category
        
        # SIMPLICITY issues
        if category_scores.get('simplicity', 1.0) < 0.7:
            if 'complexity' in metrics:
                complexity = metrics['complexity']
                
                # Person count (most critical)
                person_count = complexity['person_count']['count']
                if person_count > 2:
                    recommendations.append(f"👥 CRITICAL: Reduce to 1-2 people (currently {person_count})")
                elif person_count == 0:
                    recommendations.append("👤 Add clear main character to scene")
                
                # Background
                if scores.get('background_simplicity', 1.0) < METRIC_THRESHOLDS['background_simplicity']:
                    recommendations.append("🎨 Simplify background: remove distracting elements")
                
                # Colors
                if complexity['color_appropriateness']['dominant_colors'] > 6:
                    recommendations.append("🌈 Reduce color palette to 4-6 main colors")
                
                # Clarity
                if scores.get('character_clarity', 1.0) < METRIC_THRESHOLDS['character_clarity']:
                    recommendations.append("✏️ Improve character definition: clearer outlines")
                
                # Sensory
                if scores.get('sensory_friendliness', 1.0) < METRIC_THRESHOLDS['sensory_friendliness']:
                    recommendations.append("⚡ Reduce visual complexity to avoid sensory overload")
        
        # ACCURACY issues
        if category_scores.get('accuracy', 1.0) < 0.7:
            if scores.get('visual_quality', 1.0) < METRIC_THRESHOLDS['visual_quality']:
                recommendations.append("📸 Improve image quality: reduce blur/noise/artifacts")
            
            if scores.get('prompt_faithfulness', 1.0) < METRIC_THRESHOLDS['prompt_faithfulness']:
                recommendations.append("🎯 Improve prompt accuracy: ensure image matches description")
        
        # CONSISTENCY issues (if applicable)
        if category_scores.get('consistency') is not None and category_scores['consistency'] < 0.7:
            recommendations.append("🎭 Improve character/style consistency")
        
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
            if consistency.get('combined_consistency', 0) < 0.75:
                recommendations.append("🎭 Improve character consistency across frames")
            if consistency.get('drift_score', 0) > 0.2:
                recommendations.append("📉 Reduce visual drift between frames")
        
        # Check category performance
        overall_categories = sequence_results.get('overall_category_scores', {})
        
        if overall_categories.get('simplicity', 1.0) < 0.7:
            recommendations.append("🧩 Simplify visuals across sequence (reduce complexity)")
        
        if overall_categories.get('accuracy', 1.0) < 0.7:
            recommendations.append("🎯 Improve accuracy (prompt faithfulness + quality) across sequence")
        
        if overall_categories.get('consistency', 1.0) < 0.7:
            recommendations.append("🎭 Strengthen character/style consistency between frames")
        
        # Check for common issues across frames
        all_scores = sequence_results['image_results']
        
        # Average person count
        person_counts = []
        for r in all_scores:
            if 'complexity' in r.get('metrics', {}):
                person_counts.append(r['metrics']['complexity']['person_count']['count'])
        
        if person_counts:
            avg_people = np.mean(person_counts)
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
        
        # Generate enhanced text report
        text_report = self._generate_enhanced_text_report(results)
        with open(f"{timestamp}_report.txt", 'w', encoding='utf-8') as f:
            f.write(text_report)
        
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
    
    def _generate_enhanced_text_report(self, results: Dict) -> str:
        """Generate text report with category scores"""
        report = []
        report.append("AUTISM STORYBOARD EVALUATION REPORT")
        report.append("=" * 60)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # Overall scores
        report.append("OVERALL ASSESSMENT")
        report.append("-" * 30)
        report.append(f"Combined Autism Score: {results['combined_score']:.3f}")
        report.append(f"Grade: {results['autism_grade']}")
        report.append(f"Image: {results.get('image_name', 'N/A')}")
        report.append(f"Prompt: {results.get('prompt', 'N/A')}")
        report.append("")
        
        # Category scores
        if 'category_scores' in results:
            report.append("CATEGORY SCORES (Two-Level Hierarchy)")
            report.append("-" * 30)
            cat_scores = results['category_scores']
            report.append(f"Simplicity (36.36% weight): {cat_scores.get('simplicity', 0):.3f}")
            report.append(f"Accuracy (33.33% weight): {cat_scores.get('accuracy', 0):.3f}")
            if cat_scores.get('consistency') is not None:
                report.append(f"Consistency (30.30% weight): {cat_scores.get('consistency', 0):.3f}")
            else:
                report.append("Consistency: N/A (single image - weights redistributed)")
            report.append("")
        
        # Individual metrics grouped by category
        report.append("DETAILED METRICS BY CATEGORY")
        report.append("-" * 30)
        
        scores = results.get('scores', {})
        
        # Simplicity metrics
        report.append("\n📊 SIMPLICITY METRICS (36.36% of total):")
        simplicity_metrics = [
            ('person_count', 'Person Count', 0.40),
            ('background_simplicity', 'Background Simplicity', 0.25),
            ('color_appropriateness', 'Color Appropriateness', 0.15),
            ('character_clarity', 'Character Clarity', 0.10),
            ('sensory_friendliness', 'Sensory Friendliness', 0.07),
            ('focus_clarity', 'Focus Clarity', 0.03)
        ]
        for metric_key, metric_name, weight in simplicity_metrics:
            if metric_key in scores:
                report.append(f"  {metric_name}: {scores[metric_key]:.3f} (weight: {weight:.0%})")
        
        # Accuracy metrics
        report.append("\n🎯 ACCURACY METRICS (33.33% of total):")
        accuracy_metrics = [
            ('prompt_faithfulness', 'Prompt Faithfulness', 0.60),
            ('visual_quality', 'Visual Quality', 0.40)
        ]
        for metric_key, metric_name, weight in accuracy_metrics:
            if metric_key in scores:
                report.append(f"  {metric_name}: {scores[metric_key]:.3f} (weight: {weight:.0%})")
        
        # Consistency metrics (if available)
        if 'character_consistency' in scores or 'style_consistency' in scores:
            report.append("\n🎭 CONSISTENCY METRICS (30.30% of total):")
            consistency_metrics = [
                ('character_consistency', 'Character Consistency', 0.70),
                ('style_consistency', 'Style Consistency', 0.30)
            ]
            for metric_key, metric_name, weight in consistency_metrics:
                if metric_key in scores:
                    report.append(f"  {metric_name}: {scores[metric_key]:.3f} (weight: {weight:.0%})")
        
        report.append("")
        
        # Autism-specific analysis
        if 'metrics' in results and 'complexity' in results['metrics']:
            complexity = results['metrics']['complexity']
            
            report.append("AUTISM-SPECIFIC ANALYSIS")
            report.append("-" * 30)
            
            # Person count
            person_data = complexity.get('person_count', {})
            report.append(f"Person Count: {person_data.get('count', 'N/A')} "
                         f"({'✓ Compliant' if person_data.get('is_compliant') else '✗ Non-compliant'})")
            
            # Background
            bg_data = complexity.get('background_simplicity', {})
            report.append(f"Background: {'Simple' if bg_data.get('is_simple') else 'Complex'} "
                         f"(score: {bg_data.get('score', 0):.3f})")
            
            # Colors
            color_data = complexity.get('color_appropriateness', {})
            report.append(f"Dominant Colors: {color_data.get('dominant_colors', 'N/A')}")
            report.append(f"Color Score: {color_data.get('score', 0):.3f}")
            
            report.append("")
        
        # Recommendations
        report.append("RECOMMENDATIONS")
        report.append("-" * 30)
        for i, rec in enumerate(results.get('recommendations', []), 1):
            report.append(f"{i}. {rec}")
        
        return '\n'.join(report)
    
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
            
            # Category breakdown with normalized weights
            if sequence_results.get('overall_category_scores'):
                f.write("OVERALL CATEGORY SCORES (normalized from 36/33/30):\n")
                f.write("-" * 40 + "\n")
                weight_map = {
                    'simplicity': '36.36%',
                    'accuracy': '33.33%',
                    'consistency': '30.30%'
                }
                for category, score in sequence_results['overall_category_scores'].items():
                    weight = weight_map.get(category, '0%')
                    f.write(f"{category.title()}: {score:.3f} ({weight} weight)\n")
                f.write("\n")
            
            f.write("FRAME-BY-FRAME RESULTS:\n")
            f.write("-" * 40 + "\n")
            for result in sequence_results['image_results']:
                f.write(f"\nFrame {result['frame_number']}:\n")
                f.write(f"  Score: {result['combined_score']:.3f}\n")
                f.write(f"  Grade: {result['autism_grade']}\n")
                if result.get('category_scores'):
                    f.write("  Categories:\n")
                    for cat, score in result['category_scores'].items():
                        if score is not None:
                            f.write(f"    {cat.title()}: {score:.3f}\n")
            
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
        Evaluation results dictionary with two-level hierarchy scoring
    """
    evaluator = AutismStoryboardEvaluator(verbose=verbose)
    return evaluator.evaluate_single_image(image_path, prompt)


if __name__ == "__main__":
    # Example usage
    print("Autism Storyboard Evaluation Framework")
    print("Two-Level Hierarchy Scoring System (normalized from dissertation's 36/33/30):")
    print("")
    print("LEVEL 1 - Main Categories:")
    print("  • Simplicity: 36.36% (autism-specific complexity)")
    print("  • Accuracy: 33.33% (prompt faithfulness + visual quality)")
    print("  • Consistency: 30.30% (character/style preservation)")
    print("")
    print("LEVEL 2 - Sub-metrics within each category")
    print("")
    print