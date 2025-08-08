Autism Storyboard Evaluation Framework

A comprehensive evaluation system for assessing AI-generated images for autism education, developed as part of MSc Computer Science dissertation at Newcastle University.

## Overview

This framework provides automated evaluation of storyboard images across multiple criteria critical for autism education:

- **Visual Quality**: Technical image quality and artifact detection
- **Prompt Faithfulness**: Semantic alignment between text and generated images  
- **Character Consistency**: Identity preservation across image sequences
- **Autism Complexity**: Specialized metrics for autism-appropriate design

## Key Features

- 🎯 **Comprehensive Evaluation**: Integrates BLIP-Large for prompt faithfulness (90% accuracy)
- 🧩 **Autism-Specific Metrics**: Evaluates person count, background simplicity, sensory friendliness
- 📊 **Detailed Reporting**: Generates visual dashboards and detailed text/JSON reports
- 🔄 **Sequence Support**: Evaluates consistency across multi-frame storyboards
- 🏆 **Actionable Feedback**: Provides specific recommendations for improvement

## Installation

```bash
# Clone the repository
git clone https://github.com/leon-parker/CV_Evaluation-framework
cd CV_Evaluation-framework

# Install dependencies
pip install -r requirements.txt

# Or install as package
pip install -e .

# Optional: For semantic similarity in prompt evaluation
pip install sentence-transformers
Quick Start
Evaluate a Single Image
pythonfrom autism_evaluator import AutismStoryboardEvaluator

# Initialize evaluator
evaluator = AutismStoryboardEvaluator()

# Evaluate an image
results = evaluator.evaluate_single_image(
    image="path/to/image.png",
    prompt="one cartoon boy brushing teeth, simple background"
)

print(f"Autism Suitability Score: {results['combined_score']:.3f}")
print(f"Grade: {results['autism_grade']}")
print("Recommendations:")
for rec in results['recommendations']:
    print(f"  • {rec}")
Evaluate a Storyboard Sequence
python# Evaluate multiple frames as a sequence
images = ["frame1.png", "frame2.png", "frame3.png"]
prompts = [
    "boy waking up in bed",
    "boy brushing teeth", 
    "boy eating breakfast"
]

sequence_results = evaluator.evaluate_sequence(
    images=images,
    prompts=prompts,
    sequence_name="morning_routine"
)
Evaluation Metrics
1. Visual Quality (15% weight)

Sharpness/blur detection using Laplacian variance
Noise analysis via Median Absolute Deviation
Exposure and contrast evaluation
Random Forest-based artifact detection

2. Prompt Faithfulness (20% weight)

BLIP-Large caption generation (90% accuracy)
Keyword matching (default mode)
Optional semantic similarity for synonym understanding
Transparent evaluation showing what BLIP "sees"

3. Autism-Specific Complexity (65% weight)

Person Count (25%): Maximum 2 people recommended
Background Simplicity (15%): Minimal visual clutter
Color Appropriateness (10%): 4-6 main colors ideal
Character Clarity (8%): Clear outlines and definition
Sensory Friendliness (7%): Avoiding overstimulation

4. Character Consistency (10% weight when applicable)

CLIP-based embedding comparison
Character identity preservation (70% weight)
Style consistency across frames (30% weight)
Drift detection in sequences

Output Files
The framework generates comprehensive evaluation outputs:
evaluation_results/
├── eval_[id]_report.txt      # Human-readable report
├── eval_[id]_data.json       # Complete evaluation data
└── eval_[id]_dashboard.png   # Visual evaluation dashboard
Architecture
CV_Evaluation-framework/
├── autism_evaluator.py         # Main evaluation orchestrator
├── cv_metrics.py               # Visual quality analysis with Random Forest
├── prompt_metrics.py           # BLIP-Large prompt faithfulness (90% accuracy)
├── consistency_metrics.py      # CLIP-based character/style consistency
├── complexity_metrics.py       # Autism-specific analysis with smart consensus
├── evaluation_config.py        # Weights and thresholds
├── utils.py                    # Utilities and visualization
├── run_evaluation.py           # Example usage and demos
├── setup.py                    # Package setup
└── requirements.txt            # Dependencies
Prompt Faithfulness Methods
The framework uses BLIP-Large for prompt evaluation with two modes:
Default: Keyword Matching (100%)

Proven 90% accuracy on autism education dataset
Checks if prompt words appear in generated caption
Fast and reliable

Optional: Semantic Similarity

Enable with use_semantic=True in PromptFaithfulnessAnalyzer
Understands synonyms and concepts
Requires pip install sentence-transformers

Requirements

Python 3.8+
PyTorch 2.0+
CUDA-capable GPU recommended (CPU fallback available)
8GB+ RAM
See requirements.txt for full dependencies

Key Dependencies
torch>=2.0.0
transformers>=4.30.0
scikit-learn>=1.3.0
opencv-python>=4.8.0
Pillow>=10.0.0
numpy>=1.24.0
matplotlib>=3.7.0
Citation
If you use this framework in your research, please cite:
bibtex@mastersthesis{parker2025autism,
  title={Investigating How AI Can Support the Creation of Storyboards for Autism Education},
  author={Parker, Leon},
  year={2025},
  school={Newcastle University},
  type={MSc Computer Science Dissertation}
}
License
This project is licensed under the MIT License - see LICENSE file for details.
Key Findings

BLIP-Large achieved 90% accuracy for prompt faithfulness evaluation
Visual simplicity classifier achieved 97% binary accuracy
IP-Adapter improved character consistency by 16.8% (p < .001)
System rated perfect by special education professionals for top storyboards

Acknowledgments

Newcastle University School of Computing
Dr. Alaa Alahmadi (Supervisor)
Special education professionals who provided feedback
Open-source communities behind Hugging Face Diffusers, CLIP, and BLIP

Contact
Leon Parker - l.parker5@newcastle.ac.uk
Project Link: https://github.com/leon-parker/CV_Evaluation-framework