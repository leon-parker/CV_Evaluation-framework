#!/usr/bin/env python3
"""
Extended Validation Test - Test Refined Analyzer on New Image Set
Tests robustness across different image types and edge cases
"""

import os
import torch
import numpy as np
from PIL import Image
import cv2
from transformers import CLIPProcessor, CLIPModel
from sklearn.cluster import KMeans
import glob
import time

# Import your refined analyzer
from scoring import RefinedAutismAnalyzer

def generate_test_image_set():
    """Generate diverse test images for validation"""
    print("🎨 Generating diverse test image set...")
    
    # You could generate new images here or use existing ones
    # For now, let's assume we have a new set of test images
    test_scenarios = [
        "single child reading book, simple white background, cartoon style",
        "two children playing together, minimal playground, cartoon style", 
        "child eating breakfast, clean kitchen table, cartoon style",
        "crowded classroom with many students, detailed background",
        "busy playground with multiple children, complex background",
        "child brushing teeth, simple bathroom, cartoon style",
        "family dinner with 5 people, detailed dining room",
        "child sleeping in bed, simple bedroom, cartoon style"
    ]
    
    return test_scenarios

def test_edge_cases():
    """Test specific edge cases that might challenge the system"""
    print("🔬 Testing Edge Cases...")
    
    edge_cases = {
        "no_people": "empty classroom, desks and chairs, no people visible",
        "partial_person": "child partially hidden behind door, cartoon style",
        "reflection": "child looking in mirror, simple bathroom, cartoon style", 
        "silhouette": "child silhouette against window, simple background",
        "cartoon_animals": "cartoon cat and dog playing, simple background",
        "abstract_art": "colorful abstract shapes, no recognizable objects",
        "photo_realistic": "realistic photo of child in classroom",
        "very_simple": "simple stick figure drawing of child"
    }
    
    return edge_cases

def comprehensive_validation_test():
    """Run comprehensive validation on multiple image sets"""
    print("🧪 COMPREHENSIVE VALIDATION TEST")
    print("=" * 60)
    
    # Initialize analyzer
    analyzer = RefinedAutismAnalyzer()
    if not analyzer.available:
        print("❌ Analyzer not available")
        return
    
    # Test multiple image directories
    test_directories = [
        "sdxl_evaluation_output",  # Original test set
        "new_test_images",         # New test set (if exists)
        "edge_case_images",        # Edge cases (if exists) 
        "validation_set"           # Additional validation (if exists)
    ]
    
    all_results = []
    
    for test_dir in test_directories:
        if not os.path.exists(test_dir):
            print(f"⚠️ Directory {test_dir} not found, skipping...")
            continue
            
        print(f"\n📁 TESTING DIRECTORY: {test_dir}")
        print("-" * 40)
        
        # Find all images in directory
        image_patterns = ["*.png", "*.jpg", "*.jpeg"]
        image_files = []
        
        for pattern in image_patterns:
            image_files.extend(glob.glob(os.path.join(test_dir, pattern)))
        
        if not image_files:
            print(f"No images found in {test_dir}")
            continue
            
        directory_results = []
        
        for i, image_path in enumerate(sorted(image_files), 1):
            filename = os.path.basename(image_path)
            print(f"\n🔍 ANALYZING {i}/{len(image_files)}: {filename}")
            
            try:
                # Load and analyze image
                image = Image.open(image_path).convert('RGB')
                result = analyzer.analyze_autism_suitability(image)
                
                if "error" not in result:
                    autism_score = result['autism_suitability']
                    person_count = result['person_count']['count']
                    person_compliant = result['person_count']['is_compliant']
                    bg_score = result['background_simplicity']['score'] 
                    color_score = result['color_appropriateness']['score']
                    grade = result['autism_grade'].split('(')[0].strip()
                    
                    # Quick summary
                    print(f"   📊 Score: {autism_score:.3f} ({grade})")
                    print(f"   👥 People: {person_count} ({'✅' if person_compliant else '❌'})")
                    print(f"   🎨 Background: {bg_score:.3f} | Colors: {color_score:.3f}")
                    
                    # Store results
                    result_data = {
                        "directory": test_dir,
                        "filename": filename,
                        "autism_score": autism_score,
                        "grade": grade,
                        "person_count": person_count,
                        "person_compliant": person_compliant,
                        "bg_score": bg_score,
                        "color_score": color_score,
                        "detection_breakdown": result['person_count']['detection_breakdown'],
                        "recommendations": result['recommendations']
                    }
                    
                    directory_results.append(result_data)
                    all_results.append(result_data)
                    
                    # Flag potential issues
                    if autism_score > 0.8 and person_count > 2:
                        print(f"   ⚠️ POTENTIAL ISSUE: High score despite many people")
                    if autism_score < 0.3 and person_count <= 2:
                        print(f"   ⚠️ POTENTIAL ISSUE: Low score despite appropriate person count")
                        
                else:
                    print(f"   ❌ Analysis failed: {result['error']}")
                    
            except Exception as e:
                print(f"   ❌ Error processing {filename}: {e}")
        
        # Directory summary
        if directory_results:
            scores = [r['autism_score'] for r in directory_results]
            compliant_count = sum(1 for r in directory_results if r['person_compliant'])
            
            print(f"\n📊 {test_dir.upper()} SUMMARY:")
            print(f"   Images analyzed: {len(directory_results)}")
            print(f"   Average score: {np.mean(scores):.3f}")
            print(f"   Score range: {np.min(scores):.3f} - {np.max(scores):.3f}")
            print(f"   Person compliant: {compliant_count}/{len(directory_results)} ({compliant_count/len(directory_results)*100:.1f}%)")
    
    # Overall analysis
    print(f"\n🎯 OVERALL VALIDATION ANALYSIS")
    print("=" * 50)
    
    if all_results:
        analyze_validation_results(all_results)
    else:
        print("❌ No results to analyze")

def analyze_validation_results(results):
    """Analyze validation results for patterns and issues"""
    
    total_images = len(results)
    scores = [r['autism_score'] for r in results]
    
    print(f"📊 OVERALL STATISTICS:")
    print(f"   Total images: {total_images}")
    print(f"   Average score: {np.mean(scores):.3f} ± {np.std(scores):.3f}")
    print(f"   Median score: {np.median(scores):.3f}")
    print(f"   Score range: {np.min(scores):.3f} - {np.max(scores):.3f}")
    
    # Grade distribution
    grades = [r['grade'] for r in results]
    grade_counts = {}
    for grade in grades:
        grade_counts[grade] = grade_counts.get(grade, 0) + 1
    
    print(f"\n📈 GRADE DISTRIBUTION:")
    for grade, count in sorted(grade_counts.items()):
        percentage = count / total_images * 100
        print(f"   {grade}: {count} images ({percentage:.1f}%)")
    
    # Person count analysis
    person_counts = [r['person_count'] for r in results]
    compliant_images = [r for r in results if r['person_compliant']]
    
    print(f"\n👥 PERSON COUNT ANALYSIS:")
    print(f"   Person compliant: {len(compliant_images)}/{total_images} ({len(compliant_images)/total_images*100:.1f}%)")
    
    person_count_dist = {}
    for count in person_counts:
        person_count_dist[count] = person_count_dist.get(count, 0) + 1
    
    for count, freq in sorted(person_count_dist.items()):
        print(f"   {count} people: {freq} images")
    
    # Score vs person count correlation
    print(f"\n🔍 SCORE VS PERSON COUNT:")
    for people in sorted(set(person_counts)):
        people_scores = [r['autism_score'] for r in results if r['person_count'] == people]
        if people_scores:
            avg_score = np.mean(people_scores)
            print(f"   {people} people: {avg_score:.3f} average score ({len(people_scores)} images)")
    
    # Flag potential issues
    print(f"\n⚠️ POTENTIAL ISSUES:")
    issues_found = 0
    
    # High score with many people
    high_score_many_people = [r for r in results if r['autism_score'] > 0.7 and r['person_count'] > 2]
    if high_score_many_people:
        issues_found += len(high_score_many_people)
        print(f"   High score (>0.7) with >2 people: {len(high_score_many_people)} images")
        for r in high_score_many_people[:3]:  # Show first 3
            print(f"     - {r['filename']}: {r['autism_score']:.3f} score, {r['person_count']} people")
    
    # Low score with appropriate person count
    low_score_good_people = [r for r in results if r['autism_score'] < 0.4 and r['person_count'] <= 2]
    if low_score_good_people:
        issues_found += len(low_score_good_people)
        print(f"   Low score (<0.4) with ≤2 people: {len(low_score_good_people)} images")
        for r in low_score_good_people[:3]:  # Show first 3
            print(f"     - {r['filename']}: {r['autism_score']:.3f} score, {r['person_count']} people")
    
    # Inconsistent detection methods
    inconsistent_detection = []
    for r in results:
        detection = r['detection_breakdown']
        face_count = detection.get('face_detection', 0)
        clip_count = detection.get('clip_analysis', 0)
        
        if abs(face_count - clip_count) > 3:  # Large disagreement
            inconsistent_detection.append(r)
    
    if inconsistent_detection:
        issues_found += len(inconsistent_detection)
        print(f"   Large detection disagreement: {len(inconsistent_detection)} images")
    
    if issues_found == 0:
        print(f"   ✅ No significant issues detected!")
    
    # Validation summary
    print(f"\n🎉 VALIDATION SUMMARY:")
    if np.mean(scores) > 0.6:
        print("   ✅ Overall high quality scores")
    else:
        print("   ⚠️ Overall scores lower than expected")
    
    compliant_rate = len(compliant_images) / total_images
    if compliant_rate > 0.6:
        print("   ✅ Good person count compliance rate")
    else:
        print("   ⚠️ Low person count compliance rate")
    
    if issues_found / total_images < 0.1:
        print("   ✅ Low issue rate - system performing well")
    else:
        print("   ⚠️ Higher than expected issue rate")

def quick_robustness_test():
    """Quick test with just existing images to check robustness"""
    print("🚀 QUICK ROBUSTNESS TEST")
    print("=" * 40)
    
    analyzer = RefinedAutismAnalyzer()
    if not analyzer.available:
        print("❌ Analyzer not available")
        return
    
    # Test with original images multiple times to check consistency
    output_dir = "sdxl_evaluation_output"
    if not os.path.exists(output_dir):
        print("❌ No test images found")
        return
    
    image_files = glob.glob(os.path.join(output_dir, "*_clean.png"))
    if not image_files:
        print("❌ No clean images found")
        return
    
    print(f"📁 Testing {len(image_files)} images for consistency...")
    
    # Test each image multiple times
    consistency_results = {}
    
    for image_path in image_files[:3]:  # Test first 3 for speed
        filename = os.path.basename(image_path)
        image = Image.open(image_path).convert('RGB')
        
        scores = []
        for run in range(3):  # Run 3 times
            result = analyzer.analyze_autism_suitability(image)
            if "error" not in result:
                scores.append(result['autism_suitability'])
        
        if scores:
            consistency_results[filename] = {
                "scores": scores,
                "mean": np.mean(scores),
                "std": np.std(scores),
                "consistent": np.std(scores) < 0.01  # Very low variation expected
            }
    
    print("\n🔍 CONSISTENCY CHECK:")
    for filename, data in consistency_results.items():
        consistent = "✅" if data["consistent"] else "⚠️"
        print(f"   {consistent} {filename}: {data['mean']:.3f} ± {data['std']:.4f}")
    
    all_consistent = all(data["consistent"] for data in consistency_results.values())
    if all_consistent:
        print("\n✅ All images show consistent scoring!")
    else:
        print("\n⚠️ Some inconsistency detected - may need investigation")

if __name__ == "__main__":
    print("🧪 EXTENDED VALIDATION TESTING")
    print("=" * 60)
    print("Testing refined autism analyzer on new image sets...")
    print()
    
    # Run quick robustness test first
    quick_robustness_test()
    print("\n" + "="*60 + "\n")
    
    # Run comprehensive validation if more images available
    comprehensive_validation_test()
    
    print("\n🎯 EXTENDED VALIDATION COMPLETE!")
    print("Check results above for any issues or improvements needed.")