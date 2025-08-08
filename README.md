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

# Or install as package
pip install -e .
Quick Start
Evaluate a Single Image
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
print("\nTop Recommendations:")
for rec in results['recommendations'][:3]:
    print(f"  • {rec}")
Evaluate a Storyboard Sequence
python# Evaluate multiple frames as a sequence
images = ["frame1.png", "frame2.png", "frame3.png", "frame4.png"]
prompts = [
    "cartoon boy waking up in bed, simple bedroom",
    "same boy brushing teeth in bathroom", 
    "same boy eating breakfast at table",
    "same boy putting on school uniform"
]

sequence_results = evaluator.evaluate_sequence(
    images=images,
    prompts=prompts,
    sequence_name="morning_routine",
    save_report=True,
    output_dir="evaluation_results/sequences"
)

print(f"Overall Score: {sequence_results['overall_score']:.3f}")
print(f"Grade: {sequence_results['overall_grade']}")
Evaluate with Character Reference
python# Maintain character consistency using reference image
results = evaluator.evaluate_single_image(
    image="generated_images/alex_playing.png",
    prompt="cartoon boy Alex playing with red ball, simple playground",
    reference_image="characters/alex_reference.png"
)

if 'consistency' in results['metrics']:
    print(f"Character Consistency: {results['metrics']['consistency']['character_consistency']:.3f}")
    print(f"Style Consistency: {results['metrics']['consistency']['style_consistency']:.3f}")
Evaluation Framework
Three-Category Scoring System
The framework evaluates images across three main categories, each with specific sub-metrics:
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
Enhanced Modules
AutismComplexityAnalyzer

Smart Consensus Person Counting: Combines face detection, CLIP analysis, shape detection, and skin region analysis
Cartoon-Optimized Background Analysis: Calibrated thresholds for cartoon-style images
Multi-Scale Frequency Analysis: Detects visual complexity patterns

VisualQualityAnalyzer

Random Forest Classifier: 100% accuracy on artifact detection
Statistical Feature Extraction: Laplacian variance, MAD noise, dynamic range
ML-Based Quality Scoring: Trained on balanced clean/flawed datasets

PromptFaithfulnessAnalyzer

BLIP-Large Integration: 90% true accuracy on autism education prompts
Category Coverage: Emotions, objects, actions, clothing, food
Fast Inference: 0.4 min per batch vs 20.8 min for LLaVA

ConsistencyAnalyzer

CLIP-Based Embeddings: Face, body, and global style analysis
Drift Detection: Tracks consistency degradation over sequences
IP-Adapter Validation: 16.8% improvement over baseline (p < .001)

Output Files
The framework generates comprehensive evaluation outputs:
evaluation_results/
├── single_image/
│   ├── eval_[id]_report.txt      # Detailed text report
│   ├── eval_[id]_data.json       # Complete metrics data
│   └── eval_[id]_dashboard.png   # Visual evaluation dashboard
└── sequences/
    ├── seq_[name]_report.txt     # Sequence evaluation report
    └── seq_[name]_data.json      # Frame-by-frame metrics
Architecture
CV_Evaluation-framework/
├── autism_evaluator.py         # Main evaluation orchestrator
├── complexity_metrics.py       # Autism-specific analysis with smart consensus
├── prompt_metrics.py           # BLIP-Large prompt faithfulness (90% accuracy)
├── consistency_metrics.py      # CLIP-based character/style consistency
├── cv_metrics.py              # Visual quality with Random Forest
├── evaluation_config.py       # Weights and thresholds (3-category system)
├── utils.py                   # Visualization and reporting tools
├── run_evaluation.py          # Example usage and demonstrations
├── requirements.txt           # Dependencies
└── setup.py                  # Package installation
Requirements

Python: 3.8+
PyTorch: 2.0+ with CUDA support recommended
Hardware: 8GB+ RAM, GPU with 4GB+ VRAM preferred
Storage: ~5GB for models and cache

Key Dependencies
torch>=2.0.0
transformers>=4.30.0
scikit-learn>=1.3.0
opencv-python>=4.8.0
Pillow>=10.0.0
numpy>=1.24.0
matplotlib>=3.7.0
seaborn>=0.12.0
Performance Metrics
Quantitative Results

BLIP-Large Accuracy: 90% on autism education test set
Visual Simplicity Classifier: 97% binary accuracy (100% sensitivity)
Artifact Detection: 100% precision and recall
Character Consistency: +16.8% improvement with IP-Adapter (Cohen's d = 2.01)

Educator Validation

Perfect ratings (5.0/5.0) for top storyboards across all criteria
Strong agreement (M = 4.80) on importance of visual simplicity
High willingness (100%) to adopt AI-generated visuals with quality assurance

Citation
If you use this framework in your research, please cite:
bibtex@mastersthesis{parker2025autism,
  title={Investigating How AI Can Support the Creation of Storyboards for Autism Education},
  author={Parker, Leon},
  year={2025},
  school={Newcastle University},
  type={MSc Computer Science Dissertation}
}
Advanced Usage
Batch Evaluation
python# Evaluate multiple images efficiently
test_cases = [
    {"image": "img1.png", "prompt": "girl reading book"},
    {"image": "img2.png", "prompt": "boy washing hands"},
    {"image": "img3.png", "prompt": "children playing"}
]

results = []
for test_case in test_cases:
    result = evaluator.evaluate_single_image(**test_case, save_report=False)
    results.append({
        "image": test_case["image"],
        "score": result["combined_score"],
        "grade": result["autism_grade"]
    })

# Print summary
avg_score = sum(r["score"] for r in results) / len(results)
print(f"Average Score: {avg_score:.3f}")
Custom Thresholds
pythonfrom evaluation_config import METRIC_THRESHOLDS

# Adjust thresholds for specific use cases
METRIC_THRESHOLDS['person_count'] = 0.9  # Stricter person limit
METRIC_THRESHOLDS['background_simplicity'] = 0.7  # Higher simplicity requirement
Troubleshooting
Common Issues

CUDA Out of Memory: Reduce batch size or use CPU fallback
Model Download Failed: Check internet connection and Hugging Face access
Low Scores: Review recommendations and ensure cartoon-style inputs
Import Errors: Verify all dependencies installed correctly

Future Enhancements

 Integration with Stable Diffusion 3.5 for generation
 Real-time educator feedback loop
 Multilingual prompt support
 Adaptive complexity based on learner profiles
 Web interface for non-technical users

License
This project is licensed under the MIT License - see LICENSE file for details.
Acknowledgments

Newcastle University School of Computing
Dr. Alaa Alahmadi (Supervisor)
Special education professionals who provided feedback
Hugging Face, OpenAI CLIP, and Salesforce BLIP communities

Contact
Leon Parker
Email: l.parker5@newcastle.ac.uk
GitHub: leon-parker
Project: CV_Evaluation-framework