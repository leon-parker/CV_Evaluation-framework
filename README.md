Autism Storyboard Evaluation Framework
A comprehensive evaluation system for assessing AI-generated images for autism education, developed as part of MSc Computer Science dissertation at Newcastle University.
Overview
This framework provides automated evaluation of storyboard images across three critical categories for autism education:

Simplicity (36%): Autism-specific complexity metrics including person count, background clarity, and sensory load
Accuracy (33%): Prompt faithfulness and visual quality assessment
Consistency (30%): Character and style preservation across sequences

Key Features

🎯 BLIP-Large Integration: 90% accuracy in prompt faithfulness evaluation
🧩 Smart Consensus Person Counting: Multi-method approach for accurate person detection
📊 Comprehensive Reporting: Visual dashboards with detailed metrics and recommendations
🔄 Sequence Evaluation: CLIP-based consistency analysis across multi-frame storyboards
🏆 Autism-Informed Scoring: Weighted metrics based on special education expert feedback

Installation
bash# Clone the repository
git clone https://github.com/leon-parker/CV_Evaluation-framework
cd CV_Evaluation-framework

# Install dependencies
pip install -r requirements.txt
Quick Start
pythonfrom autism_evaluator import AutismStoryboardEvaluator

# Initialize evaluator
evaluator = AutismStoryboardEvaluator(verbose=True)

# Evaluate an image
results = evaluator.evaluate_single_image(
    image="path/to/image.png",
    prompt="one cartoon boy brushing teeth, simple background"
)

print(f"Combined Score: {results['combined_score']:.3f}")
print(f"Grade: {results['autism_grade']}")
Evaluation Framework
Three-Category Scoring System
1. Simplicity (36% weight) - Most Critical for Autism

Person Count (40%): Maximum 2 people, ideal 1 person
Background Simplicity (30%): Minimal visual clutter
Color Appropriateness (15%): 4-6 main colors ideal
Character Clarity (10%): Clear outlines and definition
Sensory Friendliness (5%): Avoiding overstimulation

2. Accuracy (33% weight) - Semantic and Technical Quality

Prompt Faithfulness (50%): BLIP-Large caption matching
Visual Quality (50%): Random Forest artifact detection

3. Consistency (30% weight) - For Multi-Frame Sequences

Character Consistency (70%): CLIP-based identity preservation
Style Consistency (30%): Visual style stability

Grading Scale

A+ (≥0.90): Excellent for autism education
A (≥0.85): Very suitable
B+ (≥0.80): Good for autism education
B (≥0.75): Suitable with minor improvements
C+ (≥0.70): Acceptable with improvements
C (≥0.65): Needs improvements
D+ (≥0.60): Significant issues
D (≥0.55): Many issues
F (<0.55): Not suitable for autism education

Key Components
Core Modules

autism_evaluator.py: Main evaluation orchestrator
complexity_metrics.py: Smart consensus person counting and autism-specific analysis
prompt_metrics.py: BLIP-Large integration for prompt faithfulness
consistency_metrics.py: CLIP-based character and style consistency
cv_metrics.py: Random Forest visual quality assessment
evaluation_config.py: Configurable weights and thresholds

Performance Metrics

BLIP-Large Accuracy: 90% on autism education test set
Visual Simplicity Classifier: 97% binary accuracy (100% sensitivity)
Artifact Detection: 100% precision and recall
Character Consistency: +16.8% improvement with IP-Adapter (Cohen's d = 2.01)
Educator Validation: Perfect ratings (5.0/5.0) for top storyboards

Requirements

Python 3.8+
PyTorch 2.0+
CUDA-capable GPU recommended
8GB+ RAM

Key Dependencies
torch>=2.0.0
transformers>=4.30.0
scikit-learn>=1.3.0
opencv-python>=4.8.0
Pillow>=10.0.0
numpy>=1.24.0
matplotlib>=3.7.0
Usage Examples
Evaluate Image Sequence
pythonimages = ["frame1.png", "frame2.png", "frame3.png"]
prompts = ["boy waking up", "boy brushing teeth", "boy eating breakfast"]

sequence_results = evaluator.evaluate_sequence(
    images=images,
    prompts=prompts,
    sequence_name="morning_routine"
)
Evaluate with Reference Image
pythonresults = evaluator.evaluate_single_image(
    image="generated_image.png",
    prompt="cartoon boy playing",
    reference_image="character_reference.png"
)
Output
The framework generates:

Detailed text reports with metrics and recommendations
JSON data files with complete evaluation results
Visual dashboards showing score breakdowns

Contact
Leon Parker
Email: l.parker5@newcastle.ac.uk
GitHub: leon-parker