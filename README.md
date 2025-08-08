# 🧩 Autism Storyboard Evaluation Framework

A comprehensive evaluation system for assessing AI-generated images for autism education, developed as part of MSc Computer Science dissertation at Newcastle University.

## 🎯 Overview

This framework provides automated evaluation of storyboard images specifically designed for autism education contexts. It implements a two-level scoring hierarchy based on educator feedback and autism-specific visual design principles.

### Key Achievements
- **90% accuracy** in prompt faithfulness evaluation using BLIP-Large
- **97% binary accuracy** in visual simplicity classification (100% sensitivity)
- **100% precision and recall** in artifact detection
- **Perfect 5.0/5.0 ratings** from special education professionals for top-scoring storyboards

## 📊 Scoring System

### Two-Level Hierarchy

**Level 1: Main Categories** (normalized from dissertation's 36/33/30 to sum to 100%)
- **Simplicity**: 36.36% of total score
- **Accuracy**: 33.33% of total score  
- **Consistency**: 30.30% of total score (when applicable)

**Level 2: Sub-metrics Within Each Category**

#### 📐 Simplicity (36.36% weight) - Most Critical for Autism
| Metric | Weight | Description |
|--------|--------|-------------|
| Person Count | 40% | Maximum 2 people, ideal 1 person |
| Background Simplicity | 25% | Minimal visual clutter |
| Color Appropriateness | 15% | 4-6 main colors ideal |
| Character Clarity | 10% | Clear outlines and definition |
| Sensory Friendliness | 7% | Avoiding overstimulation |
| Focus Clarity | 3% | Clear focal point |

#### 🎯 Accuracy (33.33% weight) - Semantic and Technical Quality
| Metric | Weight | Description |
|--------|--------|-------------|
| Prompt Faithfulness | 60% | BLIP-Large caption matching |
| Visual Quality | 40% | Random Forest artifact detection |

#### 🔄 Consistency (30.30% weight) - For Multi-Frame Sequences
| Metric | Weight | Description |
|--------|--------|-------------|
| Character Consistency | 70% | CLIP-based identity preservation |
| Style Consistency | 30% | Visual style stability |

*Note: When consistency is not applicable (single images), weights are redistributed proportionally: Simplicity 52.2%, Accuracy 47.8%*

## 🏆 Grading Scale

| Grade | Score Range | Description |
|-------|-------------|-------------|
| **A+** | ≥0.90 | Excellent for autism education |
| **A** | ≥0.85 | Very suitable |
| **B+** | ≥0.80 | Good for autism education |
| **B** | ≥0.75 | Suitable with minor improvements |
| **C+** | ≥0.70 | Acceptable with improvements |
| **C** | ≥0.65 | Needs improvements |
| **D+** | ≥0.60 | Significant issues |
| **D** | ≥0.55 | Many issues |
| **F** | <0.55 | Not suitable for autism education |

## 🚀 Installation

```bash
# Clone the repository
git clone https://github.com/leon-parker/CV_Evaluation-framework
cd CV_Evaluation-framework

# Install dependencies
pip install -r requirements.txt

# Optional: Install for development
pip install -e .
```

### Requirements
- Python 3.8+
- PyTorch 2.0+
- CUDA-capable GPU (recommended)
- 8GB+ RAM

### Key Dependencies
```
torch>=2.0.0
transformers>=4.30.0
scikit-learn>=1.3.0
opencv-python>=4.8.0
Pillow>=10.0.0
numpy>=1.24.0
matplotlib>=3.7.0
```

## 💻 Quick Start

```python
from autism_evaluator import AutismStoryboardEvaluator

# Initialize evaluator
evaluator = AutismStoryboardEvaluator(verbose=True)

# Evaluate a single image
results = evaluator.evaluate_single_image(
    image="path/to/image.png",
    prompt="one cartoon boy brushing teeth, simple background"
)

# Display results
print(f"Combined Score: {results['combined_score']:.3f}")
print(f"Grade: {results['autism_grade']}")
print(f"Category Breakdown:")
for category, score in results['category_scores'].items():
    print(f"  {category.title()}: {score:.3f}")
```

## 📖 Usage Examples

### Evaluate Image Sequence
```python
# Define your storyboard frames
images = ["frame1.png", "frame2.png", "frame3.png"]
prompts = [
    "boy waking up in bed",
    "boy brushing teeth in bathroom",
    "boy eating breakfast at table"
]

# Evaluate the sequence
sequence_results = evaluator.evaluate_sequence(
    images=images,
    prompts=prompts,
    sequence_name="morning_routine"
)

print(f"Overall Sequence Score: {sequence_results['overall_score']:.3f}")
```

### Evaluate with Character Reference
```python
# Use reference image for consistency checking
results = evaluator.evaluate_single_image(
    image="generated_image.png",
    prompt="cartoon boy playing with ball",
    reference_image="character_reference.png"
)

# Check consistency scores if available
if 'consistency' in results['category_scores']:
    print(f"Character Consistency: {results['scores']['character_consistency']:.3f}")
```

### Batch Evaluation
```python
# Evaluate multiple images
test_images = [
    ("img1.png", "boy reading book"),
    ("img2.png", "girl drawing picture"),
    ("img3.png", "children playing together")
]

for image_path, prompt in test_images:
    results = evaluator.evaluate_single_image(image_path, prompt, save_report=False)
    print(f"{image_path}: Score={results['combined_score']:.3f}, Grade={results['autism_grade']}")
```

## 📁 Project Structure

```
CV_Evaluation-framework/
├── autism_evaluator.py        # Main evaluation orchestrator
├── complexity_metrics.py      # Autism-specific complexity analysis
├── prompt_metrics.py          # BLIP-Large prompt faithfulness
├── consistency_metrics.py     # CLIP-based consistency checking
├── cv_metrics.py             # Visual quality assessment
├── evaluation_config.py      # Configuration and weights
├── utils.py                  # Helper functions and visualization
├── requirements.txt          # Package dependencies
├── setup.py                  # Installation setup
└── README.md                # This file
```

## 📈 Key Features

### 🧩 Smart Consensus Person Counting
Multi-method approach combining:
- Face detection (frontal + profile)
- CLIP semantic analysis
- Shape detection
- Skin region analysis

### 🎯 BLIP-Large Integration
- 90% accuracy on autism education test set
- Keyword-based matching (default)
- Optional semantic similarity for synonym understanding

### 📊 Comprehensive Reporting
- Detailed text reports with category breakdowns
- JSON data export for further analysis
- Visual dashboards with score visualization

### 🔄 Sequence Evaluation
- Frame-to-frame consistency tracking
- Drift detection across sequences
- Character identity preservation scoring

## 🔬 Performance Metrics

| Component | Performance | Details |
|-----------|------------|---------|
| **Prompt Faithfulness** | 90% accuracy | BLIP-Large on autism education dataset |
| **Visual Simplicity** | 97% binary accuracy | 100% sensitivity for simple scenes |
| **Artifact Detection** | 100% precision/recall | Random Forest classifier |
| **Person Counting** | 97% accuracy | Smart consensus algorithm |
| **Consistency Improvement** | +16.8% | With IP-Adapter (Cohen's d = 2.01) |

## 📝 Output Files

The framework generates three types of outputs:

1. **Text Reports** (`eval_XXXX_report.txt`)
   - Overall scores and grades
   - Category breakdowns
   - Detailed metrics
   - Recommendations

2. **JSON Data** (`eval_XXXX_data.json`)
   - Complete evaluation results
   - All metric values
   - Machine-readable format

3. **Visual Dashboards** (`eval_XXXX_dashboard.png`)
   - Score visualization
   - Metric breakdown charts
   - Visual quality indicators

## 🎓 Academic Context

This framework was developed as part of:
- **Dissertation**: "Investigating How AI Can Support the Creation of Storyboards for Autism Education"
- **Program**: MSc Computer Science
- **Institution**: Newcastle University
- **Year**: 2024-2025

### Citation
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

## 🤝 Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for:
- Bug fixes
- Performance improvements
- Additional evaluation metrics
- Documentation improvements

## 📬 Contact

**Leon Parker**  
MSc Computer Science Student  
Newcastle University  

- 📧 Email: l.parker5@newcastle.ac.uk
- 💻 GitHub: [leon-parker](https://github.com/leon-parker)
- 🔗 LinkedIn: [Add if available]

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Special thanks to the special education professionals who provided invaluable feedback
- Newcastle University School of Computing for research support
- The autism education community for insights into visual design principles

---

*Developed for improving autism education through AI*