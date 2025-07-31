"""
Autism Storyboard Evaluation Framework
A modular system for evaluating AI-generated images for autism education

Author: Leon Parker
Institution: Newcastle University
"""

from .autism_evaluator import AutismStoryboardEvaluator
from .evaluation_config import AUTISM_EVALUATION_WEIGHTS, METRIC_THRESHOLDS

__version__ = "1.0.0"
__author__ = "Leon Parker"
__all__ = [
    "AutismStoryboardEvaluator",
    "AUTISM_EVALUATION_WEIGHTS", 
    "METRIC_THRESHOLDS"
]

# Package-level docstring for documentation
"""
This framework provides comprehensive evaluation of AI-generated storyboard images
for autism education, integrating multiple metrics:

1. Visual Quality: Artifact detection and technical quality assessment
2. Prompt Faithfulness: Semantic alignment between text and image
3. Character Consistency: Identity preservation across sequences
4. Autism Complexity: Specialized metrics for autism-appropriate design

Usage:
    from autism_evaluation import AutismStoryboardEvaluator
    
    evaluator = AutismStoryboardEvaluator()
    results = evaluator.evaluate_single_image(image_path, prompt)
"""