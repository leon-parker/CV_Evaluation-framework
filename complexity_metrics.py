#!/usr/bin/env python3
"""
Improved Balanced Autism Analyzer - Better weight distribution for moderate images
Key improvements:
- Person count weight: 35% → 40% (autism priority)
- Background simplicity weight: 35% → 30% (less harsh on moderate)
- Graduated penalties instead of harsh binary cutoffs
- Softer background penalties for moderate complexity
- Reduced person count penalties for 3 people
- Adjusted category thresholds for realistic performance
- Maintains complex image detection accuracy
"""

import os
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import cv2
from transformers import CLIPProcessor, CLIPModel
from sklearn.cluster import KMeans
import time
import warnings
warnings.filterwarnings('ignore')


class AutismComplexityAnalyzer:
    """Improved balanced autism-friendly image analyzer with better weight distribution"""
    
    def __init__(self):
        print("🔧 Loading Improved Balanced Autism-Friendly Analyzer...")
        
        try:
            # Load CLIP for semantic understanding
            self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
            
            if torch.cuda.is_available():
                self.clip_model = self.clip_model.to("cuda")
            
            self.clip_available = True
            
        except Exception as e:
            print(f"⚠️ CLIP not available: {e}")
            self.clip_available = False
        
        # Load face detection cascades
        try:
            self.face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )
            self.profile_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_profileface.xml'
            )
            self.face_detection_available = True
        except:
            self.face_detection_available = False
            print("⚠️ Face detection not available for person counting")
        
        self.available = True
        print("✅ Improved Balanced Autism-Friendly Analyzer ready with better weight distribution")
    
    def analyze_complexity(self, image):
        """
        Comprehensive autism-friendly complexity analysis with enhanced algorithms
        
        Args:
            image: PIL Image or numpy array
            
        Returns:
            Dictionary with all complexity metrics and scores
        """
        return self.analyze_autism_suitability(image)
    
    def analyze_autism_suitability(self, image):
        """IMPROVED: Final autism suitability analysis with better weight distribution"""
        if not self.available:
            return {"error": "Analyzer not available"}
        
        # Convert PIL to cv2 if needed
        if isinstance(image, Image.Image):
            cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        else:
            cv_image = image
        
        print(f"   🔧 Starting improved analysis with better weight distribution...")
        
        # Improved analyses
        results = {
            "person_count": self.refined_person_counting(cv_image),
            "background_simplicity": self.improved_background_analysis(cv_image),
            "color_appropriateness": self.calibrated_color_analysis(cv_image),
            "character_clarity": self.analyze_character_clarity(cv_image),
            "sensory_friendliness": self.analyze_sensory_friendliness(cv_image),
            "focus_clarity": self.analyze_focus_clarity(cv_image)
        }
        
        # Calculate overall autism suitability with improved weights and penalties
        results["autism_suitability"] = self.calculate_improved_autism_score(results)
        results["complexity_category"] = self.get_complexity_category(results["autism_suitability"])
        results["autism_grade"] = self._get_autism_grade(results["autism_suitability"])
        results["recommendations"] = self.generate_improved_recommendations(results)
        
        return results
    
    def refined_person_counting(self, cv_image):
        """Refined person counting with better consistency"""
        height, width = cv_image.shape[:2]
        detection_results = {}
        
        # Method 1: Enhanced face detection
        face_count = self.enhanced_face_detection(cv_image)
        detection_results["face_detection"] = face_count
        
        # Method 2: CLIP semantic analysis
        if self.clip_available:
            clip_count = self.clip_person_analysis(cv_image)
            detection_results["clip_analysis"] = clip_count
        else:
            detection_results["clip_analysis"] = 0
        
        # Method 3: Refined shape detection
        shape_count = self.refined_shape_detection(cv_image)
        detection_results["shape_detection"] = shape_count
        
        # Method 4: Skin region analysis
        skin_count = self.improved_skin_detection(cv_image)
        detection_results["skin_detection"] = skin_count
        
        # Smart consensus algorithm
        final_count = self.smart_consensus_count(detection_results)
        
        print(f"   🔍 Detection breakdown: {detection_results}")
        print(f"   🎯 Smart consensus: {final_count} people")
        
        # Both 1 and 2 characters are equally compliant
        is_compliant = final_count <= 2
        
        if final_count <= 2:
            compliance_level = "Excellent"
        elif final_count == 3:
            compliance_level = "Good" 
        else:
            compliance_level = "Poor"
        
        # IMPROVED: Softer penalty for 3 people, but stricter for crowds
        if final_count <= 2:
            score = 1.0
        elif final_count == 3:
            score = 0.85  # Less harsh penalty
        elif final_count == 4:
            score = 0.70  # Moderate penalty
        elif final_count <= 6:
            score = max(0.4, 1.0 - (final_count - 2) * 0.15)  # Stricter for crowds
        else:
            score = max(0.2, 1.0 - (final_count - 2) * 0.12)  # Heavy penalty for large crowds
        
        return {
            "count": final_count,
            "is_compliant": is_compliant,
            "compliance_level": compliance_level,
            "detection_breakdown": detection_results,
            "score": score,
            "detection_method": "smart_consensus",
            "details": f"Smart consensus: {final_count} people - {'Perfect' if final_count <= 2 else 'Good' if final_count == 3 else 'Too many'} for autism storyboards"
        }
    
    # Alias for backward compatibility
    _analyze_person_count_enhanced = refined_person_counting

    def enhanced_face_detection(self, cv_image):
        """Enhanced face detection with better parameters for cartoons"""
        if not self.face_detection_available:
            return 0
            
        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        
        # Multiple scale face detection for cartoon styles
        frontal_faces = self.face_cascade.detectMultiScale(
            gray, 
            scaleFactor=1.05,  # Smaller scale factor for better detection
            minNeighbors=3,    # Reduced for cartoon faces
            minSize=(25, 25),  # Smaller minimum size
            maxSize=(300, 300) # Reasonable maximum
        )
        
        profile_faces = self.profile_cascade.detectMultiScale(
            gray,
            scaleFactor=1.05,
            minNeighbors=3,
            minSize=(25, 25),
            maxSize=(300, 300)
        )
        
        # Merge and remove overlaps
        all_faces = list(frontal_faces) + list(profile_faces)
        merged_faces = self.merge_overlapping_detections(all_faces, overlap_threshold=0.4)
        
        print(f"   👤 Enhanced face detection: {len(merged_faces)} faces")
        return len(merged_faces)
    
    # Alias for backward compatibility
    _enhanced_face_detection = enhanced_face_detection
    
    def merge_overlapping_detections(self, detections, overlap_threshold=0.4):
        """Merge overlapping detections with improved algorithm"""
        if len(detections) <= 1:
            return detections
        
        # Convert to format suitable for Non-Maximum Suppression
        boxes = []
        for (x, y, w, h) in detections:
            boxes.append([x, y, x + w, y + h])
        
        if not boxes:
            return []
        
        boxes = np.array(boxes)
        
        # Simple NMS implementation
        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        indices = np.argsort(areas)[::-1]  # Sort by area, largest first
        
        keep = []
        while len(indices) > 0:
            current = indices[0]
            keep.append(current)
            
            if len(indices) == 1:
                break
            
            # Calculate IoU with remaining boxes
            current_box = boxes[current]
            remaining_boxes = boxes[indices[1:]]
            
            # Calculate intersection
            x1 = np.maximum(current_box[0], remaining_boxes[:, 0])
            y1 = np.maximum(current_box[1], remaining_boxes[:, 1])
            x2 = np.minimum(current_box[2], remaining_boxes[:, 2])
            y2 = np.minimum(current_box[3], remaining_boxes[:, 3])
            
            intersection = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
            
            # Calculate IoU
            current_area = areas[current]
            remaining_areas = areas[indices[1:]]
            union = current_area + remaining_areas - intersection
            
            iou = intersection / union
            
            # Keep boxes with IoU below threshold
            indices = indices[1:][iou < overlap_threshold]
        
        # Convert back to (x, y, w, h) format
        final_detections = []
        for idx in keep:
            x1, y1, x2, y2 = boxes[idx]
            final_detections.append((int(x1), int(y1), int(x2-x1), int(y2-y1)))
        
        return final_detections
    
    # Alias for backward compatibility
    _merge_overlapping_detections = merge_overlapping_detections
    
    def clip_person_analysis(self, cv_image):
        """Enhanced CLIP analysis with better prompts"""
        if not self.clip_available:
            return 1
            
        try:
            pil_image = Image.fromarray(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB))
            
            # More specific prompts for better accuracy
            count_descriptions = [
                "empty scene with no people",
                "one single person alone",
                "exactly two people together", 
                "three people in a group",
                "four people together",
                "five or more people",
                "a crowd of many people"
            ]
            
            inputs = self.clip_processor(
                text=count_descriptions,
                images=pil_image,
                return_tensors="pt",
                padding=True
            )
            
            if torch.cuda.is_available():
                inputs = {k: v.to("cuda") for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.clip_model(**inputs)
                probs = outputs.logits_per_image.softmax(dim=-1).cpu().numpy()[0]
            
            # Find best match with confidence threshold
            best_idx = np.argmax(probs)
            confidence = probs[best_idx]
            
            # Map to person counts
            count_mapping = [0, 1, 2, 3, 4, 5, 6]
            estimated_count = count_mapping[best_idx]
            
            print(f"   🤖 CLIP analysis: '{count_descriptions[best_idx]}' (conf: {confidence:.3f})")
            
            # Apply confidence weighting
            if confidence < 0.3:
                # Low confidence, return more conservative estimate
                return max(1, estimated_count // 2)
            else:
                return estimated_count
                
        except Exception as e:
            print(f"   ⚠️ CLIP analysis failed: {e}")
            return 1
    
    # Alias for backward compatibility
    _clip_person_analysis = clip_person_analysis

    def refined_shape_detection(self, cv_image):
        """Refined human shape detection optimized for cartoons"""
        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        height, width = gray.shape
        
        # Use multiple edge detection approaches
        # Approach 1: Canny edges
        edges1 = cv2.Canny(gray, 30, 100)
        
        # Approach 2: Adaptive threshold
        edges2 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        edges2 = cv2.bitwise_not(edges2)  # Invert for edge-like appearance
        
        # Combine edges
        combined_edges = cv2.bitwise_or(edges1, edges2)
        
        # Morphological operations to clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        combined_edges = cv2.morphologyEx(combined_edges, cv2.MORPH_CLOSE, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(combined_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Analyze contours for human-like shapes
        person_candidates = 0
        min_area = (height * width) * 0.005  # Minimum 0.5% of image
        max_area = (height * width) * 0.7    # Maximum 70% of image
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if min_area < area < max_area:
                # Get bounding rectangle
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = h / w if w > 0 else 0
                
                # Human-like criteria for cartoons
                if (1.3 <= aspect_ratio <= 3.5 and  # Taller than wide
                    h > height * 0.15 and           # Substantial height
                    w > width * 0.05):              # Reasonable width
                    
                    # Additional check: contour should be roughly centered vertically
                    center_y = y + h // 2
                    if 0.2 * height < center_y < 0.8 * height:
                        person_candidates += 1
        
        print(f"   🏃 Shape detection: {person_candidates} human-like shapes")
        return min(person_candidates, 8)  # Cap at reasonable number
    
    # Alias for backward compatibility
    _refined_shape_detection = refined_shape_detection

    def improved_skin_detection(self, cv_image):
        """Improved skin detection with cartoon-appropriate ranges"""
        # Convert to HSV and LAB for better skin detection
        hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(cv_image, cv2.COLOR_BGR2LAB)
        
        # Multiple skin tone ranges for diverse cartoon characters
        skin_masks = []
        
        # HSV ranges for different skin tones
        hsv_ranges = [
            ([0, 20, 50], [25, 255, 255]),      # Light skin
            ([0, 25, 80], [20, 200, 255]),      # Medium skin  
            ([10, 30, 60], [25, 255, 200]),     # Tan skin
            ([8, 40, 40], [25, 255, 180])       # Darker skin
        ]
        
        for lower, upper in hsv_ranges:
            mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
            skin_masks.append(mask)
        
        # Combine all skin masks
        combined_mask = skin_masks[0]
        for mask in skin_masks[1:]:
            combined_mask = cv2.bitwise_or(combined_mask, mask)
        
        # Clean up the mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
        
        # Find skin regions
        contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        significant_regions = 0
        total_area = cv_image.shape[0] * cv_image.shape[1]
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > total_area * 0.008:  # At least 0.8% of image
                # Check if region is face-like (roughly circular/oval)
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = h / w if w > 0 else 0
                if 0.7 <= aspect_ratio <= 1.8:  # Face-like proportions
                    significant_regions += 1
        
        # Estimate people from skin regions
        estimated_people = min(significant_regions, 8)
        print(f"   🎨 Skin detection: {estimated_people} skin regions")
        return estimated_people
    
    # Alias for backward compatibility
    _improved_skin_detection = improved_skin_detection

    def smart_consensus_count(self, detection_results):
        """Smart consensus algorithm for person counting"""
        face_count = detection_results.get("face_detection", 0)
        clip_count = detection_results.get("clip_analysis", 1)
        shape_count = detection_results.get("shape_detection", 0)
        skin_count = detection_results.get("skin_detection", 0)
        
        # Rule-based consensus with priority weighting
        
        # Rule 1: If CLIP and face detection agree, trust them highly
        if abs(clip_count - face_count) <= 1 and face_count > 0:
            consensus = int((clip_count + face_count) / 2)
            print(f"   ✅ CLIP and face detection agree: {consensus}")
            return consensus
        
        # Rule 2: If face detection finds people, it's usually reliable
        if face_count > 0:
            # Weight face detection heavily, but consider CLIP
            if clip_count > 0:
                consensus = int(0.7 * face_count + 0.3 * clip_count)
            else:
                consensus = face_count
            print(f"   👤 Face detection weighted: {consensus}")
            return min(consensus, 10)
        
        # Rule 3: If no faces detected, rely on CLIP and shape detection
        if face_count == 0:
            if clip_count > 0 and shape_count > 0:
                consensus = int(0.6 * clip_count + 0.4 * shape_count)
            elif clip_count > 0:
                consensus = clip_count
            elif shape_count > 0:
                consensus = shape_count
            else:
                consensus = max(1, skin_count // 2)  # Conservative from skin
            
            print(f"   🤖 No faces, using semantic: {consensus}")
            return min(consensus, 10)
        
        # Rule 4: Fallback weighted average
        weights = [0.4, 0.3, 0.2, 0.1]  # face, clip, shape, skin
        values = [face_count, clip_count, shape_count, skin_count]
        
        weighted_sum = sum(w * v for w, v in zip(weights, values))
        consensus = max(1, round(weighted_sum))
        
        print(f"   📊 Weighted average: {consensus}")
        return min(consensus, 10)
    
    # Alias for backward compatibility
    _smart_consensus_count = smart_consensus_count

    def calculate_improved_autism_score(self, results):
        """OPTIMIZED: Better penalties and person count handling based on test results"""
        # IMPROVED WEIGHTS: Better balance for moderate complexity images
        weights = {
            "person_count": 0.40,           # Increased from 35% (autism priority)
            "background_simplicity": 0.30,  # Decreased from 35% (less harsh on moderate)
            "color_appropriateness": 0.15,  # Unchanged
            "character_clarity": 0.08,      # Unchanged
            "sensory_friendliness": 0.05,   # Unchanged
            "focus_clarity": 0.02           # Unchanged
        }
        
        total_score = 0
        total_weight = 0
        
        for metric, weight in weights.items():
            if metric in results and "score" in results[metric]:
                score = results[metric]["score"]
                total_score += score * weight
                total_weight += weight
        
        if total_weight == 0:
            return 0.5
        
        autism_suitability = total_score / total_weight
        
        # OPTIMIZED: Stronger penalties for complex scenarios
        bg_score = results["background_simplicity"]["score"]
        person_count = results["person_count"]["count"]
        
        # Enhanced background complexity penalties - balanced approach
        if bg_score < 0.20:
            # Extremely complex backgrounds: Severe penalty
            complexity_penalty = 0.28 + (bg_score * 0.5)  # Penalty ranges from 0.28 to 0.38
            autism_suitability *= complexity_penalty
            print(f"   🚨 Severe background penalty applied: {complexity_penalty:.3f}")
        elif bg_score < 0.30:
            # Very complex backgrounds: Heavy penalty  
            complexity_penalty = 0.48 + (bg_score * 0.5)  # Penalty ranges from 0.48 to 0.63
            autism_suitability *= complexity_penalty
            print(f"   🔥 Heavy background penalty applied: {complexity_penalty:.3f}")
        elif bg_score < 0.40:
            # Moderately complex backgrounds: Moderate penalty
            complexity_penalty = 0.72 + (bg_score * 0.28)  # Penalty ranges from 0.72 to 0.83
            autism_suitability *= complexity_penalty
            print(f"   ⚠️ Moderate background penalty applied: {complexity_penalty:.3f}")
        
        # Enhanced person count penalties for crowds - balanced
        if person_count >= 8:
            # Large crowds: Major penalty
            crowd_penalty = max(0.35, 1.0 - (person_count - 7) * 0.09)  # Slightly less harsh
            autism_suitability *= crowd_penalty
            print(f"   👥 Large crowd penalty applied: {crowd_penalty:.3f}")
        elif person_count >= 5:
            # Medium crowds: Moderate penalty
            crowd_penalty = max(0.65, 1.0 - (person_count - 4) * 0.07)  # Slightly less harsh
            autism_suitability *= crowd_penalty
            print(f"   👥 Medium crowd penalty applied: {crowd_penalty:.3f}")
        elif person_count == 0:
            # No people detected
            autism_suitability -= 0.1
        
        return float(min(1.0, max(0.0, autism_suitability)))
    
    # Alias for backward compatibility
    _calculate_autism_score_enhanced = calculate_improved_autism_score

    def get_complexity_category(self, suitability_score):
        """FINAL OPTIMIZATION: Sweet spot threshold for balanced accuracy"""
        if suitability_score >= 0.62:  # Maintains simple classification
            return "Low Complexity (Autism-Friendly)"
        elif suitability_score >= 0.38:  # Sweet spot for optimal moderate/complex balance
            return "Moderate Complexity (Acceptable with Caution)"
        else:
            return "High Complexity (Not Recommended for Autism Use)"
    
    def _get_autism_grade(self, suitability_score):
        """Convert autism suitability score to grade"""
        if suitability_score >= 0.9:
            return "A+ (Excellent for autism storyboards)"
        elif suitability_score >= 0.8:
            return "A (Very suitable for autism storyboards)"
        elif suitability_score >= 0.7:
            return "B+ (Good for autism storyboards)"
        elif suitability_score >= 0.6:
            return "B (Acceptable for autism storyboards)"
        elif suitability_score >= 0.5:
            return "C+ (Needs some improvements)"
        elif suitability_score >= 0.4:
            return "C (Several issues to address)"
        elif suitability_score >= 0.3:
            return "D+ (Many issues for autism use)"
        elif suitability_score >= 0.2:
            return "D (Poor for autism storyboards)"
        else:
            return "F (Unsuitable for autism storyboards)"

    def generate_improved_recommendations(self, results):
        """IMPROVED: Generate recommendations with graduated warnings"""
        recommendations = []
        
        # Person count analysis
        person_data = results["person_count"]
        person_count = person_data["count"]
        
        if person_count == 0:
            recommendations.append("⚠️ No people detected - ensure main character is clearly visible")
        elif person_count == 1:
            recommendations.append("✅ Perfect: Single character ideal for autism storyboards")
        elif person_count == 2:
            recommendations.append("✅ Perfect: Two characters ideal for autism interaction stories")
        elif person_count == 3:
            recommendations.append("👍 Good: Three characters acceptable but monitor complexity")
        elif person_count == 4:
            recommendations.append("⚠️ Moderate: Four characters may be challenging for some users")
        else:
            recommendations.append(f"🚨 Critical: Too many people ({person_count}) - reduce to 1-3 characters")
        
        # IMPROVED: Background analysis with more lenient graduated warnings
        bg_score = results["background_simplicity"]["score"]
        if bg_score >= 0.6:
            recommendations.append("✅ Excellent: Background is clean and autism-friendly")
        elif bg_score >= 0.5:
            recommendations.append("👍 Good: Background mostly appropriate")
        elif bg_score >= 0.35:
            recommendations.append("👌 Acceptable: Background complexity manageable for most autism users")
        elif bg_score >= 0.25:
            recommendations.append("⚠️ Moderate: Background could be simpler for better autism suitability")
        elif bg_score >= 0.15:
            recommendations.append("🚨 Poor: Background busy - reduce visual clutter")
        else:
            recommendations.append("🔥 Critical: Background extremely complex - major simplification required")
        
        # Color analysis
        color_data = results["color_appropriateness"]
        color_count = color_data["dominant_colors"]
        color_score = color_data["score"]
        
        if color_count <= 4 and color_score >= 0.8:
            recommendations.append("✅ Excellent: Color palette perfect for autism use")
        elif color_count <= 6:
            recommendations.append("👍 Good: Color count appropriate for cartoons")
        else:
            recommendations.append(f"🌈 Reduce colors: {color_count} colors may overwhelm - target 4-6 main colors")
        
        # Character clarity
        if results["character_clarity"]["score"] < 0.6:
            recommendations.append("✏️ Improve character definition with clearer outlines")
        
        # Sensory friendliness
        if results["sensory_friendliness"]["score"] < 0.6:
            recommendations.append("⚡ Reduce visual complexity to avoid sensory overload")
        
        # Overall assessment with 3-category system
        overall_score = results["autism_suitability"]
        category = self.get_complexity_category(overall_score)
        recommendations.append(f"📊 Classification: {category}")
        
        # Add final assessment
        if overall_score >= 0.85:
            recommendations.append("🏆 Outstanding for autism education!")
        elif overall_score >= 0.7:
            recommendations.append("✅ Excellent for autism storyboards")
        elif overall_score >= 0.6:
            recommendations.append("👍 Good for autism use with minor tweaks")
        elif overall_score >= 0.45:
            recommendations.append("⚠️ Needs improvements for autism suitability")
        else:
            recommendations.append("🚨 Major changes needed for autism education use")
        
        return recommendations
    
    # Alias for backward compatibility
    _generate_recommendations_enhanced = generate_improved_recommendations

    def improved_background_analysis(self, cv_image):
        """IMPROVED: More lenient background complexity detection"""
        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        height, width = gray.shape
        
        # Method 1: More lenient edge density
        edges = cv2.Canny(gray, 40, 120)
        edge_density = np.sum(edges > 0) / edges.size
        
        # IMPROVED: More lenient penalty for edges
        edge_simplicity = 1.0 - min(1.0, edge_density * 14)  # Reduced from 16
        
        # Method 2: More lenient texture analysis
        texture_complexity = self.calculate_improved_texture(gray)
        texture_simplicity = 1.0 - min(1.0, texture_complexity * 9)  # Reduced from 10
        
        # Method 3: More lenient color complexity
        color_complexity = self.calculate_improved_color_complexity(cv_image)
        color_simplicity = 1.0 - min(1.0, color_complexity * 1.3)  # Reduced from 1.5
        
        # Method 4: More lenient object counting
        object_count = self.improved_object_counting(cv_image)
        object_simplicity = 1.0 - min(1.0, (object_count / 14.0))  # Less strict (was 12.0)
        
        # Method 5: More lenient frequency analysis
        freq_complexity = self.improved_frequency_analysis(gray)
        freq_simplicity = 1.0 - min(1.0, freq_complexity * 2.8)  # Reduced from 3.0
        
        # More balanced weights (unchanged)
        weights = [0.3, 0.25, 0.2, 0.15, 0.1]
        simplicities = [edge_simplicity, texture_simplicity, color_simplicity, object_simplicity, freq_simplicity]
        
        background_simplicity = sum(w * s for w, s in zip(weights, simplicities))
        
        # IMPROVED: More lenient grading with realistic thresholds
        if background_simplicity >= 0.75:
            grade = "Excellent - Very clean background"
        elif background_simplicity >= 0.6:
            grade = "Good - Mostly clean background"
        elif background_simplicity >= 0.45:
            grade = "Acceptable - Manageable background complexity"
        elif background_simplicity >= 0.3:
            grade = "Moderate - Somewhat busy background"
        elif background_simplicity >= 0.15:
            grade = "Poor - Busy/distracting background"
        else:
            grade = "Critical - Extremely complex background"
        
        complexity_breakdown = {
            "edge_simplicity": float(edge_simplicity),
            "texture_simplicity": float(texture_simplicity),
            "color_simplicity": float(color_simplicity),
            "object_simplicity": float(object_simplicity),
            "frequency_simplicity": float(freq_simplicity),
            "edge_density": float(edge_density),
            "object_count": object_count
        }
        
        print(f"   🎨 Improved background analysis: simplicity={background_simplicity:.3f}, objects={object_count}")
        
        return {
            "score": float(background_simplicity),
            "complexity_breakdown": complexity_breakdown,
            "grade": grade,
            "is_simple": background_simplicity > 0.65,
            "edge_density": float(edge_density),
            "object_count": object_count,
            "details": f"Improved analysis: {background_simplicity:.3f} - {grade}"
        }
    
    # Alias for backward compatibility
    _analyze_background_enhanced = improved_background_analysis

    def calculate_improved_texture(self, gray_image):
        """More lenient texture calculation"""
        kernel_size = 7
        kernel = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size ** 2)
        
        mean = cv2.filter2D(gray_image.astype(np.float32), -1, kernel)
        sqr_diff = (gray_image.astype(np.float32) - mean) ** 2
        variance = cv2.filter2D(sqr_diff, -1, kernel)
        
        # More lenient texture complexity evaluation
        avg_variance = np.mean(variance) / (255 ** 2)
        return min(1.0, avg_variance * 3.2)  # Reduced from 3.5
    
    # Alias for backward compatibility
    _calculate_refined_texture = calculate_improved_texture

    def calculate_improved_color_complexity(self, cv_image):
        """More lenient color complexity evaluation"""
        rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        pixels = rgb_image.reshape(-1, 3)
        
        try:
            n_clusters = min(8, max(3, len(np.unique(pixels, axis=0)) // 1000))  # More lenient than 900
            if n_clusters > 1:
                kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                kmeans.fit(pixels[::15])  # Less frequent sampling
                dominant_colors = n_clusters
            else:
                dominant_colors = 1
        except:
            dominant_colors = 4
        
        color_variance = np.var(rgb_image.reshape(-1, 3), axis=0).mean() / (255 ** 2)
        
        # More lenient color complexity calculation
        color_complexity = (dominant_colors / 8.0) * 0.7 + color_variance * 0.3  # More lenient
        
        return min(1.0, color_complexity)
    
    # Alias for backward compatibility
    _calculate_calibrated_color_complexity = calculate_improved_color_complexity

    def improved_object_counting(self, cv_image):
        """More lenient object counting with reasonable parameters"""
        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        
        thresh1 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 3)
        _, thresh2 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        combined = cv2.bitwise_and(thresh1, thresh2)
        
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel)
        combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel, iterations=2)
        
        contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        height, width = gray.shape
        min_area = (height * width) * 0.003  # More lenient minimum
        max_area = (height * width) * 0.6    # Unchanged maximum
        
        significant_objects = 0
        for contour in contours:
            area = cv2.contourArea(contour)
            if min_area < area < max_area:
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = max(w, h) / min(w, h) if min(w, h) > 0 else 0
                
                if aspect_ratio < 12 and w > 2 and h > 2:  # More lenient criteria
                    significant_objects += 1
        
        return min(significant_objects, 30)  # Higher cap
    
    # Alias for backward compatibility
    _refined_object_counting = improved_object_counting

    def improved_frequency_analysis(self, gray_image):
        """More lenient frequency analysis"""
        fft = np.fft.fft2(gray_image)
        fft_shift = np.fft.fftshift(fft)
        magnitude_spectrum = np.abs(fft_shift)
        
        h, w = magnitude_spectrum.shape
        center_h, center_w = h // 2, w // 2
        
        y, x = np.ogrid[:h, :w]
        center_distance = np.sqrt((x - center_w)**2 + (y - center_h)**2)
        
        # More lenient frequency analysis
        mid_freq_mask = (center_distance >= min(h, w) * 0.12) & (center_distance < min(h, w) * 0.38)
        high_freq_mask = center_distance >= min(h, w) * 0.38
        
        total_energy = np.sum(magnitude_spectrum)
        mid_freq_energy = np.sum(magnitude_spectrum[mid_freq_mask])
        high_freq_energy = np.sum(magnitude_spectrum[high_freq_mask])
        
        if total_energy > 0:
            mid_freq_ratio = mid_freq_energy / total_energy
            high_freq_ratio = high_freq_energy / total_energy
            
            # More lenient weighting
            complexity = mid_freq_ratio * 0.6 + high_freq_ratio * 0.4
        else:
            complexity = 0
        
        return min(1.0, complexity * 2.0)  # Reduced from 2.2
    
    # Alias for backward compatibility
    _cartoon_frequency_analysis = improved_frequency_analysis

    def calibrated_color_analysis(self, cv_image):
        """Calibrated color analysis with cartoon-appropriate thresholds"""
        rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        hsv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
        
        # More sophisticated color counting for cartoons
        pixels = rgb_image.reshape(-1, 3)
        
        try:
            # Optimized clustering for cartoon color palettes
            n_clusters = min(10, max(3, len(np.unique(pixels, axis=0)) // 1000))
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            kmeans.fit(pixels[::20])  # Sample for efficiency
            
            # Analyze cluster properties
            cluster_centers = kmeans.cluster_centers_
            labels = kmeans.labels_
            
            # Count significant colors (those that appear frequently)
            unique_labels, counts = np.unique(labels, return_counts=True)
            total_pixels = len(labels)
            
            significant_colors = 0
            for label, count in zip(unique_labels, counts):
                if count / total_pixels > 0.02:  # Color appears in >2% of image
                    significant_colors += 1
            
            dominant_colors = significant_colors
            
        except:
            # Fallback method
            unique_colors = len(np.unique(pixels.view(np.dtype((np.void, pixels.dtype.itemsize * pixels.shape[1])))))
            dominant_colors = min(8, unique_colors // 500)
        
        # Adjusted scoring for cartoon color palettes
        # Cartoons typically use 4-6 main colors effectively
        if dominant_colors <= 4:
            color_count_score = 1.0
        elif dominant_colors <= 6:
            color_count_score = 0.9
        elif dominant_colors <= 8:
            color_count_score = 0.7
        else:
            color_count_score = max(0.2, 1.0 - (dominant_colors - 8) * 0.1)
        
        # Saturation analysis (cartoons can handle higher saturation)
        saturation_values = hsv_image[:,:,1]
        avg_saturation = np.mean(saturation_values) / 255.0
        
        # Adjusted optimal saturation for cartoons
        optimal_saturation = 0.65  # Slightly higher for cartoons
        saturation_score = 1.0 - abs(avg_saturation - optimal_saturation) * 1.5
        saturation_score = max(0.0, min(1.0, saturation_score))
        
        # Brightness analysis
        brightness_values = hsv_image[:,:,2]
        avg_brightness = np.mean(brightness_values) / 255.0
        brightness_variance = np.var(brightness_values) / (255**2)
        
        # Cartoon-appropriate brightness targets
        brightness_score = 1.0 - abs(avg_brightness - 0.72) * 1.3  # Slightly brighter target
        brightness_consistency = 1.0 - min(1.0, brightness_variance * 6)  # More tolerance
        brightness_score = (brightness_score + brightness_consistency) / 2
        brightness_score = max(0.0, min(1.0, brightness_score))
        
        # Weighted combination for cartoon appropriateness
        color_appropriateness = (color_count_score * 0.5 + saturation_score * 0.3 + brightness_score * 0.2)
        
        # Adjusted grading for cartoons
        if color_appropriateness >= 0.85:
            grade = "Excellent - Perfect cartoon color palette"
        elif color_appropriateness >= 0.7:
            grade = "Good - Suitable cartoon colors"
        elif color_appropriateness >= 0.55:
            grade = "Moderate - Acceptable for cartoons"
        else:
            grade = "Poor - Too complex/inappropriate for autism use"
        
        print(f"   🌈 Color analysis: {dominant_colors} colors, score={color_appropriateness:.3f}")
        
        return {
            "score": float(color_appropriateness),
            "dominant_colors": dominant_colors,
            "color_count_score": float(color_count_score),
            "avg_saturation": float(avg_saturation),
            "saturation_score": float(saturation_score),
            "avg_brightness": float(avg_brightness),
            "brightness_score": float(brightness_score),
            "is_appropriate": color_appropriateness > 0.7,
            "grade": grade,
            "details": f"Optimized for cartoon style: {dominant_colors} colors, {grade}"
        }
    
    # Alias for backward compatibility
    _analyze_colors_enhanced = calibrated_color_analysis

    def analyze_character_clarity(self, cv_image):
        """Character clarity analysis"""
        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        
        edges = cv2.Canny(gray, 100, 200)
        strong_edge_density = np.sum(edges > 0) / edges.size
        
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        avg_contrast = np.mean(gradient_magnitude) / 255.0
        
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        shape_quality = 0
        if len(contours) > 0:
            large_contours = [c for c in contours if cv2.contourArea(c) > 500]
            
            if large_contours:
                smoothness_scores = []
                for contour in large_contours[:5]:
                    epsilon = 0.02 * cv2.arcLength(contour, True)
                    approx = cv2.approxPolyDP(contour, epsilon, True)
                    smoothness = len(approx) / len(contour)
                    smoothness_scores.append(smoothness)
                
                shape_quality = np.mean(smoothness_scores) if smoothness_scores else 0
        
        edge_clarity = min(1.0, strong_edge_density * 8)
        contrast_clarity = min(1.0, avg_contrast * 3)
        shape_clarity = min(1.0, shape_quality * 5)
        
        character_clarity = (edge_clarity * 0.4 + contrast_clarity * 0.4 + shape_clarity * 0.2)
        
        if character_clarity >= 0.8:
            grade = "Excellent - Very clear character definition"
        elif character_clarity >= 0.6:
            grade = "Good - Clear character boundaries"
        elif character_clarity >= 0.4:
            grade = "Moderate - Somewhat unclear boundaries"
        else:
            grade = "Poor - Unclear or blurry characters"
        
        return {
            "score": float(character_clarity),
            "edge_clarity": float(edge_clarity),
            "contrast_clarity": float(contrast_clarity),
            "shape_clarity": float(shape_clarity),
            "edge_strength": float(strong_edge_density),
            "contrast": float(avg_contrast),
            "is_clear": character_clarity > 0.6,
            "grade": grade,
            "details": "Higher score = clearer character boundaries and definition"
        }
    
    # Alias for backward compatibility
    _analyze_character_clarity = analyze_character_clarity

    def analyze_sensory_friendliness(self, cv_image):
        """Sensory friendliness analysis"""
        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        
        fft = np.fft.fft2(gray)
        fft_shift = np.fft.fftshift(fft)
        magnitude_spectrum = np.abs(fft_shift)
        
        h, w = magnitude_spectrum.shape
        center_h, center_w = h // 2, w // 2
        y, x = np.ogrid[:h, :w]
        center_distance = np.sqrt((x - center_w)**2 + (y - center_h)**2)
        
        high_freq_mask = center_distance >= min(h, w) * 0.4
        total_energy = np.sum(magnitude_spectrum)
        high_freq_energy = np.sum(magnitude_spectrum[high_freq_mask])
        high_freq_ratio = high_freq_energy / total_energy if total_energy > 0 else 0
        
        kernel = np.ones((5,5), np.float32) / 25
        local_mean = cv2.filter2D(gray.astype(np.float32), -1, kernel)
        intensity_variance = np.var(gray.astype(np.float32) - local_mean) / (255**2)
        
        motion_blur_kernel = np.zeros((9, 9))
        motion_blur_kernel[4, :] = 1/9
        blurred = cv2.filter2D(gray, -1, motion_blur_kernel)
        blur_difference = np.mean(np.abs(gray.astype(float) - blurred.astype(float))) / 255.0
        
        pattern_score = 1.0 - min(1.0, high_freq_ratio * 3)
        intensity_score = 1.0 - min(1.0, intensity_variance * 10)
        motion_score = max(0.0, 1.0 - blur_difference * 2)
        
        sensory_friendliness = (pattern_score * 0.5 + intensity_score * 0.3 + motion_score * 0.2)
        
        if sensory_friendliness >= 0.8:
            grade = "Excellent - Very sensory-friendly"
        elif sensory_friendliness >= 0.6:
            grade = "Good - Sensory-appropriate"
        elif sensory_friendliness >= 0.4:
            grade = "Moderate - Some sensory concerns"
        else:
            grade = "Poor - Potentially overstimulating"
        
        return {
            "score": float(sensory_friendliness),
            "pattern_score": float(pattern_score),
            "intensity_score": float(intensity_score),
            "high_freq_ratio": float(high_freq_ratio),
            "intensity_variance": float(intensity_variance),
            "intensity_variation": float(intensity_variance),
            "is_sensory_friendly": sensory_friendliness > 0.6,
            "grade": grade,
            "details": "Higher score = less overwhelming, more sensory-appropriate"
        }
    
    # Alias for backward compatibility
    _analyze_sensory_load_enhanced = analyze_sensory_friendliness

    def analyze_focus_clarity(self, cv_image):
        """Focus clarity analysis"""
        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        
        h, w = gray.shape
        y, x = np.ogrid[:h, :w]
        center_y, center_x = h // 2, w // 2
        
        center_distance = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        max_distance = np.sqrt(center_x**2 + center_y**2)
        center_weights = np.exp(-center_distance**2 / (2 * (max_distance/3)**2))
        
        edges = cv2.Canny(gray, 100, 200)
        center_edge_density = np.sum(edges * center_weights) / np.sum(center_weights)
        peripheral_edge_density = np.sum(edges * (1 - center_weights)) / np.sum(1 - center_weights)
        
        focus_ratio = center_edge_density / (peripheral_edge_density + 1e-6)
        focus_score = min(1.0, focus_ratio / 2.0)
        
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        center_contrast = np.sum(gradient_magnitude * center_weights) / np.sum(center_weights)
        peripheral_contrast = np.sum(gradient_magnitude * (1 - center_weights)) / np.sum(1 - center_weights)
        contrast_ratio = center_contrast / (peripheral_contrast + 1e-6)
        contrast_focus = min(1.0, contrast_ratio / 2.0)
        
        focus_clarity = (focus_score * 0.6 + contrast_focus * 0.4)
        
        if focus_clarity >= 0.7:
            grade = "Excellent - Clear single focus"
        elif focus_clarity >= 0.5:
            grade = "Good - Reasonably focused"
        elif focus_clarity >= 0.3:
            grade = "Moderate - Somewhat unfocused"
        else:
            grade = "Poor - No clear focal point"
        
        return {
            "score": float(focus_clarity),
            "focus_ratio": float(focus_ratio),
            "contrast_ratio": float(contrast_ratio),
            "center_edge_density": float(center_edge_density),
            "peripheral_edge_density": float(peripheral_edge_density),
            "has_clear_focus": focus_clarity > 0.5,
            "grade": grade,
            "details": "Higher score = clearer focal point, less distraction"
        }
    
    # Alias for backward compatibility
    _analyze_focus_clarity = analyze_focus_clarity