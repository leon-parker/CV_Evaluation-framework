#!/usr/bin/env python3
"""
Example usage of the Autism Storyboard Evaluation Framework
Shows how to evaluate single images and sequences
"""

from pathlib import Path
from autism_evaluator import AutismStoryboardEvaluator


def evaluate_single_image_example():
    """Example: Evaluate a single generated image"""
    
    print("=" * 60)
    print("EXAMPLE 1: Single Image Evaluation")
    print("=" * 60)
    
    # Initialize evaluator
    evaluator = AutismStoryboardEvaluator(verbose=True)
    
    # Example image and prompt
    image_path = "generated_images/boy_brushing_teeth.png"
    prompt = "one cartoon boy brushing teeth with toothbrush in bathroom, simple background"
    
    # Run evaluation
    results = evaluator.evaluate_single_image(
        image=image_path,
        prompt=prompt,
        save_report=True,
        output_dir="evaluation_results/single_image"
    )
    
    # Print summary
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    print(f"Combined Score: {results['combined_score']:.3f}")
    print(f"Autism Grade: {results['autism_grade']}")
    print("\nTop Recommendations:")
    for i, rec in enumerate(results['recommendations'][:3], 1):
        print(f"{i}. {rec}")
    
    return results


def evaluate_sequence_example():
    """Example: Evaluate a storyboard sequence"""
    
    print("\n" * 2)
    print("=" * 60)
    print("EXAMPLE 2: Storyboard Sequence Evaluation")
    print("=" * 60)
    
    # Initialize evaluator
    evaluator = AutismStoryboardEvaluator(verbose=True)
    
    # Example storyboard: Morning routine
    images = [
        "storyboard/frame1_waking_up.png",
        "storyboard/frame2_brushing_teeth.png",
        "storyboard/frame3_eating_breakfast.png",
        "storyboard/frame4_getting_dressed.png"
    ]
    
    prompts = [
        "cartoon boy waking up in bed, simple bedroom background",
        "same cartoon boy brushing teeth in bathroom, simple background",
        "same cartoon boy eating breakfast at table, simple kitchen",
        "same cartoon boy putting on school uniform, simple background"
    ]
    
    # Run sequence evaluation
    results = evaluator.evaluate_sequence(
        images=images,
        prompts=prompts,
        sequence_name="morning_routine",
        save_report=True,
        output_dir="evaluation_results/sequences"
    )
    
    # Print summary
    print("\n" + "=" * 60)
    print("SEQUENCE EVALUATION SUMMARY")
    print("=" * 60)
    print(f"Overall Score: {results['overall_score']:.3f}")
    print(f"Overall Grade: {results['overall_grade']}")
    print(f"\nFrame Scores:")
    for img_result in results['image_results']:
        print(f"  Frame {img_result['frame_number']}: "
              f"{img_result['combined_score']:.3f} "
              f"({img_result['autism_grade']})")
    
    if 'consistency' in results['sequence_metrics']:
        consistency = results['sequence_metrics']['consistency']
        print(f"\nSequence Consistency: {consistency['combined_consistency']:.3f}")
    
    print("\nSequence Recommendations:")
    for i, rec in enumerate(results['recommendations'], 1):
        print(f"{i}. {rec}")
    
    return results


def evaluate_with_reference_example():
    """Example: Evaluate with character reference for consistency"""
    
    print("\n" * 2)
    print("=" * 60)
    print("EXAMPLE 3: Evaluation with Character Reference")
    print("=" * 60)
    
    # Initialize evaluator
    evaluator = AutismStoryboardEvaluator(verbose=True)
    
    # Reference and generated images
    reference_image = "characters/alex_reference.png"
    generated_image = "generated_images/alex_playing.png"
    prompt = "cartoon boy Alex playing with red ball in playground, simple background"
    
    # Evaluate with reference
    results = evaluator.evaluate_single_image(
        image=generated_image,
        prompt=prompt,
        reference_image=reference_image,
        save_report=True,
        output_dir="evaluation_results/with_reference"
    )
    
    # Print consistency results
    print("\n" + "=" * 60)
    print("CONSISTENCY EVALUATION SUMMARY")
    print("=" * 60)
    print(f"Combined Score: {results['combined_score']:.3f}")
    print(f"Autism Grade: {results['autism_grade']}")
    
    if 'consistency' in results['metrics']:
        consistency = results['metrics']['consistency']
        print(f"\nCharacter Consistency: {consistency['character_consistency']:.3f}")
        print(f"Style Consistency: {consistency['style_consistency']:.3f}")
    
    return results


def batch_evaluation_example():
    """Example: Batch evaluate multiple images"""
    
    print("\n" * 2)
    print("=" * 60)
    print("EXAMPLE 4: Batch Evaluation")
    print("=" * 60)
    
    # Initialize evaluator
    evaluator = AutismStoryboardEvaluator(verbose=False)  # Less verbose for batch
    
    # Multiple test cases
    test_cases = [
        {
            "image": "test_images/good_single_character.png",
            "prompt": "one happy cartoon girl reading book, simple background"
        },
        {
            "image": "test_images/bad_too_many_people.png",
            "prompt": "group of children playing in busy playground"
        },
        {
            "image": "test_images/good_clear_action.png",
            "prompt": "cartoon boy washing hands with soap, simple bathroom"
        },
        {
            "image": "test_images/bad_complex_background.png",
            "prompt": "child in detailed classroom with many objects"
        }
    ]
    
    # Evaluate each
    all_results = []
    for i, test_case in enumerate(test_cases, 1):
        print(f"\nEvaluating image {i}/{len(test_cases)}...")
        
        results = evaluator.evaluate_single_image(
            image=test_case["image"],
            prompt=test_case["prompt"],
            save_report=False  # Don't save individual reports
        )
        
        all_results.append({
            "image": test_case["image"],
            "score": results["combined_score"],
            "grade": results["autism_grade"],
            "main_issue": results["recommendations"][1] if len(results["recommendations"]) > 1 else "None"
        })
    
    # Print summary table
    print("\n" + "=" * 60)
    print("BATCH EVALUATION RESULTS")
    print("=" * 60)
    print(f"{'Image':<30} {'Score':<8} {'Grade':<15} {'Main Issue'}")
    print("-" * 80)
    
    for result in all_results:
        image_name = Path(result["image"]).name
        print(f"{image_name:<30} {result['score']:<8.3f} {result['grade']:<15} {result['main_issue']}")
    
    # Calculate statistics
    scores = [r["score"] for r in all_results]
    print(f"\nAverage Score: {sum(scores)/len(scores):.3f}")
    print(f"Best Score: {max(scores):.3f}")
    print(f"Worst Score: {min(scores):.3f}")
    
    return all_results


def main():
    """Run all examples"""
    
    print("🧩 AUTISM STORYBOARD EVALUATION FRAMEWORK - EXAMPLES")
    print("=" * 60)
    print("This script demonstrates various ways to use the evaluation system")
    print("=" * 60)
    
    # Check if example images exist
    example_images = [
        "generated_images/boy_brushing_teeth.png",
        "storyboard/frame1_waking_up.png"
    ]
    
    images_exist = all(Path(img).exists() for img in example_images)
    
    if not images_exist:
        print("\n⚠️  Example images not found!")
        print("To run these examples, you need to have generated images in:")
        print("  - generated_images/")
        print("  - storyboard/")
        print("  - test_images/")
        print("  - characters/")
        print("\nCreating mock evaluation with placeholder...")
        
        # Create a simple demonstration
        evaluator = AutismStoryboardEvaluator(verbose=True)
        print("\n✅ Evaluator initialized successfully!")
        print("\nYou can now use it with your own images:")
        print("  results = evaluator.evaluate_single_image('your_image.png', 'your prompt')")
        return
    
    # Run examples
    try:
        # Example 1: Single image
        single_results = evaluate_single_image_example()
        
        # Example 2: Sequence
        sequence_results = evaluate_sequence_example()
        
        # Example 3: With reference
        reference_results = evaluate_with_reference_example()
        
        # Example 4: Batch
        batch_results = batch_evaluation_example()
        
        print("\n" * 2)
        print("=" * 60)
        print("✅ ALL EXAMPLES COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print("\nCheck the 'evaluation_results' directory for:")
        print("  - Detailed reports (.txt)")
        print("  - Raw data (.json)")
        print("  - Visual dashboards (.png)")
        
    except Exception as e:
        print(f"\n❌ Error running examples: {e}")
        print("\nMake sure all required models are installed:")
        print("  - transformers")
        print("  - torch")
        print("  - opencv-python")
        print("  - scikit-learn")
        print("  - matplotlib")
        print("  - seaborn")


if __name__ == "__main__":
    main()