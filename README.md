# Autism Storyboard Evaluation Framework

A comprehensive evaluation system for assessing AI-generated images for autism education, developed as part of MSc Computer Science dissertation at Newcastle University.

## Overview

This framework provides automated evaluation of storyboard images across multiple criteria critical for autism education:

- **Visual Quality**: Technical image quality and artifact detection
- **Prompt Faithfulness**: Semantic alignment between text and generated images  
- **Character Consistency**: Identity preservation across image sequences
- **Autism Complexity**: Specialized metrics for autism-appropriate design

## Key Features

- 🎯 **Comprehensive Evaluation**: Integrates multiple state-of-the-art models (CLIP, BLIP-2, etc.)
- 🧩 **Autism-Specific Metrics**: Evaluates person count, background simplicity, sensory friendliness
- 📊 **Detailed Reporting**: Generates visual dashboards and detailed text/JSON reports
- 🔄 **Sequence Support**: Evaluates consistency across multi-frame storyboards
- 🏆 **Actionable Feedback**: Provides specific recommendations for improvement

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/autism-evaluation
cd autism-evaluation

# Install dependencies
pip install -r requirements.txt

# Or install as package
pip install -e .
```

## Quick Start

### Evaluate a Single Image

```python
from autism_evaluator import AutismStoryboardEvaluator

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
```

### Evaluate a Storyboard Sequence

```python
# Evaluate multiple frames as a sequence
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
```

## Evaluation Metrics

### 1. Visual Quality (15% weight)
- Sharpness/blur detection
- Noise levels
- Exposure quality
- Artifact detection

### 2. Prompt Faithfulness (20% weight)
- Advanced CLIP-based semantic alignment
- Contrastive evaluation approach
- Key concept verification

### 3. Autism-Specific Complexity (65% weight)
- **Person Count** (25%): Maximum 2 people recommended
- **Background Simplicity** (15%): Minimal visual clutter
- **Color Appropriateness** (10%): 4-6 main colors ideal
- **Character Clarity** (8%): Clear outlines and definition
- **Sensory Friendliness** (7%): Avoiding overstimulation

### 4. Character Consistency (10% weight when applicable)
- Character identity preservation
- Style consistency across frames
- Drift detection in sequences

## Output Files

The framework generates comprehensive evaluation outputs:

```
evaluation_results/
├── eval_[id]_report.txt      # Human-readable report
├── eval_[id]_data.json       # Complete evaluation data
└── eval_[id]_dashboard.png   # Visual evaluation dashboard
```

## Architecture

```
autism_evaluation/
├── __init__.py                 # Package initialization
├── autism_evaluator.py         # Main evaluation orchestrator
├── cv_metrics.py              # Visual quality analysis
├── prompt_metrics.py          # Prompt faithfulness (CLIP)
├── consistency_metrics.py     # Character/style consistency
├── complexity_metrics.py      # Autism-specific analysis
├── evaluation_config.py       # Weights and thresholds
├── utils.py                   # Utilities and visualization
└── run_evaluation.py          # Example usage
```

## Requirements

- Python 3.8+
- CUDA-capable GPU recommended (CPU fallback available)
- 8GB+ RAM
- See `requirements.txt` for full dependencies

## Citation

If you use this framework in your research, please cite:

```bibtex
@mastersthesis{parker2025autism,
  title={Investigating How AI Can Support the Creation of Storyboards for Autism Education},
  author={Parker, Leon},
  year={2025},
  school={Newcastle University},
  type={MSc Computer Science Dissertation}
}
```

## License

This project is licensed under the MIT License - see LICENSE file for details.

## Acknowledgments

- Newcastle University School of Computing
- Special education professionals who provided feedback
- Open-source communities behind CLIP, SDXL, and Diffusers

## Contact

Leon Parker - l.parker5@newcastle.ac.uk

Project Link: [https://github.com/yourusername/autism-evaluation](https://github.com/yourusername/autism-evaluation)