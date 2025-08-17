"""
Configuration settings for autism storyboard evaluation
Defines weights, thresholds, and parameters for all metrics
Using two-level hierarchy: Categories -> Sub-metrics

WEIGHTS NORMALIZED: Original dissertation weights (36%, 33%, 30% = 99%)
have been normalized to sum to exactly 100% while preserving proportions
"""

# ============================================================================
# TWO-LEVEL SCORING HIERARCHY
# ============================================================================

# Level 1: Main category weights (normalized from 36/33/30 to sum to 1.0)
# Original weights from dissertation: 36%, 33%, 30% (sum = 99%)
# Normalized to maintain proportions while summing to 100%
CATEGORY_WEIGHTS = {
    'simplicity': 0.363636,    # 36/99 = 36.36% - Most critical for autism
    'accuracy': 0.333333,      # 33/99 = 33.33% - Technical and semantic quality
    'consistency': 0.303030    # 30/99 = 30.30% - Character/style preservation
}
# Note: These are the exact normalized values from the dissertation's 36/33/30 split

# Level 2: Sub-metric weights within each category (each category must sum to 1.0)
SIMPLICITY_WEIGHTS = {
    'person_count': 0.40,           # 40% of simplicity - CRITICAL: 1-2 people max
    'background_simplicity': 0.30,  # 30% of simplicity - Reduced visual clutter
    'color_appropriateness': 0.15,  # 15% of simplicity - Sensory-friendly palette
    'character_clarity': 0.08,      # 8% of simplicity - Clear character definition
    'sensory_friendliness': 0.05,   # 5% of simplicity - Avoid overstimulation
    'focus_clarity': 0.02           # 2% of simplicity - Clear focal point
}

ACCURACY_WEIGHTS = {
    'prompt_faithfulness': 0.60,    # 60% of accuracy - Semantic match
    'visual_quality': 0.40          # 40% of accuracy - Technical quality
}

CONSISTENCY_WEIGHTS = {
    'character_consistency': 0.70,  # 70% of consistency - Identity preservation
    'style_consistency': 0.30       # 30% of consistency - Style preservation
}

# Combined structure for easy access
EVALUATION_WEIGHTS = {
    'categories': CATEGORY_WEIGHTS,
    'simplicity': SIMPLICITY_WEIGHTS,
    'accuracy': ACCURACY_WEIGHTS,
    'consistency': CONSISTENCY_WEIGHTS
}

# ============================================================================
# LEGACY FLAT WEIGHTS (for backward compatibility if needed)
# These are the calculated flat weights: category_weight * sub_metric_weight
# Using normalized category weights
# ============================================================================
AUTISM_EVALUATION_WEIGHTS_FLAT = {
    # Simplicity metrics (36.36% total)
    'person_count': 0.1455,          # 0.363636 * 0.40 = 0.1455
    'background_simplicity': 0.1091,  # 0.363636 * 0.30 = 0.1091
    'color_appropriateness': 0.0545,  # 0.363636 * 0.15 = 0.0545
    'character_clarity': 0.0291,      # 0.363636 * 0.08 = 0.0291
    'sensory_friendliness': 0.0182,   # 0.363636 * 0.05 = 0.0182
    'focus_clarity': 0.0073,          # 0.363636 * 0.02 = 0.0073
    
    # Accuracy metrics (33.33% total)
    'prompt_faithfulness': 0.2000,    # 0.333333 * 0.60 = 0.2000
    'visual_quality': 0.1333,         # 0.333333 * 0.40 = 0.1333
    
    # Consistency metrics (30.30% total when applicable)
    'character_consistency': 0.2121,  # 0.303030 * 0.70 = 0.2121
    'style_consistency': 0.0909       # 0.303030 * 0.30 = 0.0909
}

# For backward compatibility
AUTISM_EVALUATION_WEIGHTS = AUTISM_EVALUATION_WEIGHTS_FLAT

# ============================================================================
# THRESHOLDS AND OTHER SETTINGS (unchanged)
# ============================================================================

# Minimum acceptable thresholds for each metric
METRIC_THRESHOLDS = {
    'visual_quality': 0.7,           # Good technical quality
    'prompt_faithfulness': 0.75,     # Strong semantic match
    'person_count': 0.8,             # Strict limit on people
    'background_simplicity': 0.65,   # Reasonably clean
    'color_appropriateness': 0.6,    # Appropriate palette
    'character_clarity': 0.6,        # Clear definition
    'sensory_friendliness': 0.6,     # Not overwhelming
    'focus_clarity': 0.5,            # Has focal point
    'character_consistency': 0.7,    # Good preservation
    'style_consistency': 0.7         # Style maintained
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
    'blip_model': "Salesforce/blip-image-captioning-large",  # Added: specific BLIP model
    'clip_model': "openai/clip-vit-large-patch14",
    'sentence_transformer_model': "all-MiniLM-L6-v2",  # Added: for semantic similarity
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

# Validation function to ensure weights are properly configured
def validate_weights():
    """Validate that all weights sum to 1.0 within their respective groups"""
    issues = []
    
    # Check category weights (allowing for normalized 99% weights)
    category_sum = sum(CATEGORY_WEIGHTS.values())
    if abs(category_sum - 1.0) > 0.001:
        issues.append(f"Category weights sum to {category_sum}, should be 1.0 (normalized from 36/33/30)")
    
    # Check simplicity weights
    simplicity_sum = sum(SIMPLICITY_WEIGHTS.values())
    if abs(simplicity_sum - 1.0) > 0.001:
        issues.append(f"Simplicity weights sum to {simplicity_sum}, should be 1.0")
    
    # Check accuracy weights
    accuracy_sum = sum(ACCURACY_WEIGHTS.values())
    if abs(accuracy_sum - 1.0) > 0.001:
        issues.append(f"Accuracy weights sum to {accuracy_sum}, should be 1.0")
    
    # Check consistency weights
    consistency_sum = sum(CONSISTENCY_WEIGHTS.values())
    if abs(consistency_sum - 1.0) > 0.001:
        issues.append(f"Consistency weights sum to {consistency_sum}, should be 1.0")
    
    if issues:
        print("⚠️ Weight configuration issues:")
        for issue in issues:
            print(f"  - {issue}")
        return False
    
    return True

# Run validation on import
if __name__ == "__main__" or True:  # Always validate on import
    if validate_weights():
        print("✅ Weight configuration validated successfully")
        print(f"   Category weights (normalized from 36/33/30):")
        print(f"   - Simplicity: {CATEGORY_WEIGHTS['simplicity']:.4f} (36.36%)")
        print(f"   - Accuracy: {CATEGORY_WEIGHTS['accuracy']:.4f} (33.33%)")
        print(f"   - Consistency: {CATEGORY_WEIGHTS['consistency']:.4f} (30.30%)")
    else:
        print("❌ Weight configuration has errors")