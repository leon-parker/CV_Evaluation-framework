"""
Configuration settings for autism storyboard evaluation
Defines weights, thresholds, and parameters for all metrics
"""

# Evaluation weights for combined scoring
# These weights were calibrated based on autism education priorities
AUTISM_EVALUATION_WEIGHTS = {
    # Technical quality
    'visual_quality': 0.15,          # Artifact-free rendering
    
    # Semantic accuracy
    'prompt_faithfulness': 0.20,     # Matches intended content
    
    # Autism-specific metrics (total: 0.65)
    'person_count': 0.25,            # CRITICAL: 1-2 people max
    'background_simplicity': 0.15,   # Reduced visual clutter
    'color_appropriateness': 0.10,   # Sensory-friendly palette
    'character_clarity': 0.08,       # Clear character definition
    'sensory_friendliness': 0.07,    # Avoid overstimulation
    
    # Consistency (when applicable)
    'consistency': 0.10,             # Character/style preservation
}

# Minimum acceptable thresholds for each metric
METRIC_THRESHOLDS = {
    'visual_quality': 0.7,           # Good technical quality
    'prompt_faithfulness': 0.75,     # Strong semantic match
    'person_count': 0.8,             # Strict limit on people
    'background_simplicity': 0.65,   # Reasonably clean
    'color_appropriateness': 0.6,    # Appropriate palette
    'character_clarity': 0.6,        # Clear definition
    'sensory_friendliness': 0.6,     # Not overwhelming
    'consistency': 0.7,              # Good preservation
}

# Visual complexity classification thresholds
COMPLEXITY_THRESHOLDS = {
    'simple': {
        'edge_density': 0.05,
        'object_count': 5,
        'color_count': 6,
        'texture_complexity': 0.3
    },
    'moderate': {
        'edge_density': 0.10,
        'object_count': 10,
        'color_count': 10,
        'texture_complexity': 0.5
    },
    'complex': {
        'edge_density': 0.15,
        'object_count': 15,
        'color_count': 15,
        'texture_complexity': 0.7
    }
}

# Model configurations
MODEL_CONFIGS = {
    'clip_model': "openai/clip-vit-large-patch14",
    'device': 'cuda',  # Will fall back to CPU if unavailable
    'batch_size': 4,
    'cache_dir': './model_cache'
}

# Report generation settings
REPORT_SETTINGS = {
    'save_visualizations': True,
    'include_recommendations': True,
    'generate_pdf': False,  # Requires additional dependencies
    'verbose': True
}

# Autism-specific design guidelines
AUTISM_GUIDELINES = {
    'max_people': 2,
    'ideal_people': 1,
    'max_colors': 6,
    'ideal_colors': 4,
    'min_character_size': 0.15,  # Minimum 15% of image height
    'max_background_objects': 5,
    'preferred_style': 'cartoon',
    'avoid_patterns': ['stripes', 'checkerboard', 'spiral'],
    'preferred_emotions': ['happy', 'calm', 'neutral']
}