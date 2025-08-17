# 🧩 Autism Storyboard Evaluation Framework

A comprehensive evaluation system for assessing AI-generated images for autism education, developed as part of MSc Computer Science dissertation at Newcastle University.

## 🎯 Overview

This framework provides automated evaluation of storyboard images specifically designed for autism education contexts. It implements a two-level scoring hierarchy based on educator feedback and autism-specific visual design principles.

### Key Features
- **BLIP-Large based** prompt faithfulness evaluation with proven accuracy
- **Multi-method** visual quality assessment using Random Forest classification
- **Character consistency** tracking across image sequences using CLIP
- **Autism-specific** complexity metrics based on educational research
- **Two-level scoring** hierarchy with normalized weights

## 📊 Scoring System

### Two-Level Hierarchy

**Level 1: Main Categories** (normalized from dissertation's 36/33/30 to sum to 100%)
- **Simplicity**: 36.36% of total score (autism-specific complexity metrics)
- **Accuracy**: 33.33% of total score (visual quality + prompt faithfulness)
- **Consistency**: 30.30% of total score (character/style preservation across sequences)

**Level 2: Sub-metrics Within Each Category**

#### 📐 Simplicity (36.36% weight) - Most Critical for Autism
| Metric | Weight | Description |
|--------|--------|-------------|
| Person Count | 40% | Maximum 2 people, ideal 1 person |
| Background Simplicity | 30% | Minimal visual clutter |
| Color Appropriateness | 15% | 4-6 main colors ideal |
| Character Clarity | 8% | Clear outlines and definition |
| Sensory Friendliness | 5% | Avoiding overstimulation |
| Focus Clarity | 2% | Clear focal point |

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

### Prerequisites
- Python 3.8+
- CUDA-capable GPU (recommended for model loading)
- 8GB+ RAM

### Step 1: Clone Repository
```bash
git clone https://github.com/leon-parker/CV_Evaluation-framework
cd CV_Evaluation-framework
```

### Step 2: Install Dependencies
```bash
# Install all required packages
pip install -r requirements.txt

# Optional: Install for development
pip install -e .
```

### Step 3: Verify Installation
```bash
python -c "from autism_evaluator import AutismStoryboardEvaluator; print('✅ Installation successful!')"
```

### Key Dependencies
```
torch>=2.0.0                 # Core ML framework
transformers>=4.30.0         # BLIP-Large and CLIP models
diffusers>=0.21.0           # Required for image generation scripts
compel>=2.0.0               # Required for generation scripts
scikit-learn>=1.3.0         # Random Forest classification
opencv-python>=4.8.0        # Computer vision operations
sentence-transformers>=2.2.0 # Optional: Enhanced semantic similarity
```

## 💻 Quick Start

### Basic Image Evaluation
```python
from autism_evaluator import AutismStoryboardEvaluator

# Initialize evaluator
evaluator = AutismStoryboardEvaluator(verbose=True)

# Evaluate a single image
results = evaluator.evaluate_single_image(
    image="path/to/image.png",
    prompt="boy brushing teeth",  # Keep prompts simple for BLIP evaluation
    save_report=True
)

# Display results
print(f"Combined Score: {results['combined_score']:.3f}")
print(f"Grade: {results['autism_grade']}")
print(f"Category Breakdown:")
for category, score in results['category_scores'].items():
    print(f"  {category.title()}: {score:.3f}")
```

### Batch Evaluation with Overall Assessment
```python
# Evaluate multiple images and get overall assessment
images = ["image1.png", "image2.png", "image3.png"]
prompts = ["boy eating", "girl reading", "child playing"]

batch_results = evaluator.evaluate_batch(
    images=images,
    prompts=prompts,
    save_overall_report=True
)

# View overall assessment
assessment = batch_results['overall_assessment']
print(f"Average Score: {assessment['average_score']:.3f}")
print(f"Autism Appropriate: {assessment['autism_appropriate']['percentage']:.0f}%")
```

## 📖 Usage Examples

### Evaluate Image Sequence with Consistency
```python
# Define your storyboard frames
images = ["frame1.png", "frame2.png", "frame3.png"]
prompts = [
    "boy waking up",
    "boy brushing teeth", 
    "boy eating breakfast"
]

# Evaluate the sequence (includes consistency analysis)
sequence_results = evaluator.evaluate_sequence(
    images=images,
    prompts=prompts,
    sequence_name="morning_routine"
)

print(f"Overall Sequence Score: {sequence_results['overall_score']:.3f}")
print(f"Character Consistency: {sequence_results['overall_category_scores']['consistency']:.3f}")
```

### Evaluate with Character Reference
```python
# Use reference image for consistency checking
results = evaluator.evaluate_single_image(
    image="generated_image.png",
    prompt="boy playing ball",
    reference_image="character_reference.png"  # For consistency analysis
)

# Check consistency scores if available
if 'consistency' in results['category_scores']:
    print(f"Character Consistency: {results['category_scores']['consistency']:.3f}")
```

### Enable Enhanced Semantic Similarity (Optional)
```python
# The framework includes optional semantic similarity for prompt evaluation
# To enable, ensure sentence-transformers is installed:
# pip install sentence-transformers

from prompt_metrics import PromptFaithfulnessAnalyzer

# Enable semantic understanding for synonyms/concepts
analyzer = PromptFaithfulnessAnalyzer(use_semantic=True)
score = analyzer.evaluate_prompt_alignment(
    image="image.png", 
    prompt="happy child eating breakfast",
    method="combined"  # 60% keyword + 40% semantic
)
```

## 📁 Project Structure

```
CV_Evaluation-framework/
├── autism_evaluator.py          # Main evaluation orchestrator
├── complexity_metrics.py        # Autism-specific complexity analysis
├── prompt_metrics.py            # BLIP-Large prompt faithfulness
├── consistency_metrics.py       # CLIP-based consistency checking
├── cv_metrics.py                # Visual quality assessment
├── evaluation_config.py         # Configuration and weights
├── utils.py                     # Helper functions and visualization
├── autism_generator.py          # Multi-generation autism-optimized image creation
├── Test_sequence_script.py      # Advanced test with AI refiner
├── requirements.txt             # Package dependencies
├── setup.py                     # Installation setup
└── README.md                   # This file
```

## 📈 Key Technical Features

### 🧩 Autism-Specific Person Counting
Multi-method approach combining:
- Face detection (frontal + profile cascades)
- CLIP semantic analysis for character understanding
- Shape detection optimized for cartoon styles  
- Skin region analysis with diverse tone ranges
- Smart consensus algorithm for final count

### 🎯 BLIP-Large Prompt Faithfulness
- Keyword-based matching (default) with high accuracy
- Optional semantic similarity for synonym understanding
- Optimized for autism education terminology
- Simple prompt preprocessing for better BLIP alignment

### 📊 Enhanced Visual Quality Assessment
- Random Forest classifier for artifact detection
- Multi-scale sharpness analysis
- Comprehensive noise detection using MAD (Median Absolute Deviation)
- Statistical threshold optimization

### 🔄 CLIP-Based Sequence Consistency
- Character-focused region extraction with face detection
- Separate character and style embedding analysis
- Drift detection across long sequences
- Frame-to-frame consistency scoring

## 📝 Output Files

The framework generates three types of outputs:

1. **Text Reports** (`eval_XXXX_report.txt`)
   - Overall scores and grades with category breakdown
   - Detailed metrics and autism-specific analysis
   - Specific recommendations for improvement

2. **JSON Data** (`eval_XXXX_data.json`)
   - Complete evaluation results in machine-readable format
   - All metric values and intermediate calculations
   - Suitable for further analysis and research

3. **Visual Dashboards** (`eval_XXXX_dashboard.png`)
   - Score visualization with gauge charts
   - Metric breakdown bar charts
   - Visual quality indicators and recommendations

## 🎓 Academic Context

This framework was developed as part of:
- **Dissertation**: "Investigating How AI Can Support the Creation of Storyboards for Autism Education"
- **Program**: MSc Computer Science
- **Institution**: Newcastle University
- **Year**: 2024-2025

### Research Contributions
- Two-level hierarchical scoring system optimized for autism education
- Multi-method person counting algorithm achieving high accuracy
- Integration of educational psychology principles with computer vision
- Comprehensive evaluation framework for AI-generated educational content

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

## 🔧 Advanced Usage

### Custom Model Paths
For image generation scripts, specify model paths:
```python
from autism_generator import AutismStoryboardGenerator

generator = AutismStoryboardGenerator(
    base_model_path="path/to/sdxl_model.safetensors",
    refiner_model_path="path/to/refiner_model.safetensors"  # Optional
)

if generator.setup():
    result = generator.generate_autism_appropriate_image(
        prompt="boy reading book",
        output_dir="generated_images"
    )
```

### Testing the Framework
Run the comprehensive test suite:
```bash
python Test_sequence_script.py  # Generates test images and validates scoring
python simplified_test.py       # Basic functionality test
```


## ⚠️ Important Notes

1. **GPU Memory**: Model loading requires significant GPU memory (8GB+ recommended)
2. **Model Files**: Generation scripts require SDXL model files (not included)
3. **Prompt Simplification**: Keep evaluation prompts simple for optimal BLIP performance
4. **Semantic Features**: Enhanced semantic similarity is optional and requires sentence-transformers

