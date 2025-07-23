#!/usr/bin/env python3
"""
Refined Evaluation Framework - Final Accuracy Pass
Fixes background scoring, color thresholds, and detection consistency
"""

import os
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import cv2
from transformers import CLIPProcessor, CLIPModel
from sklearn.cluster import KMeans
import time

class RefinedAutismAnalyzer:
    """Final refined autism-friendly image analyzer with calibrated accuracy"""
    
    def __init__(self):
        print("🔧 Loading Refined Autism-Friendly Analyzer...")
        
        try:
            # Load CLIP for semantic understanding
            self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
            
            if torch.cuda.is_available():
                self.clip_model = self.clip_model.to("cuda")
            
            # Face detection cascades
            self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            self.profile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_profileface.xml')
            
            self.available = True
            print("✅ Refined Autism-Friendly Analyzer ready")
            
        except Exception as e:
            print(f"❌ Failed to load analyzer: {e}")
            self.available = False
    
    def refined_person_counting(self, cv_image):
        """Refined person counting with better consistency"""
        height, width = cv_image.shape[:2]
        detection_results = {}
        
        # Method 1: Enhanced face detection
        face_count = self.enhanced_face_detection(cv_image)
        detection_results["face_detection"] = face_count
        
        # Method 2: CLIP semantic analysis
        clip_count = self.clip_person_analysis(cv_image)
        detection_results["clip_analysis"] = clip_count
        
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
        
        # Determine compliance
        is_compliant = final_count <= 2
        compliance_level = "Excellent" if final_count <= 1 else "Good" if final_count == 2 else "Poor"
        
        return {
            "count": final_count,
            "is_compliant": is_compliant,
            "compliance_level": compliance_level,
            "detection_breakdown": detection_results,
            "score": 1.0 if final_count <= 2 else max(0.0, 1.0 - (final_count - 2) * 0.25),
            "details": f"Smart consensus from multiple detection methods: {final_count} people"
        }
    
    def enhanced_face_detection(self, cv_image):
        """Enhanced face detection with better parameters for cartoons"""
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
    
    def clip_person_analysis(self, cv_image):
        """Enhanced CLIP analysis with better prompts"""
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
    
    def calibrated_background_analysis(self, cv_image):
        """Calibrated background analysis with cartoon-appropriate thresholds"""
        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        height, width = gray.shape
        
        # Method 1: Calibrated edge density
        edges = cv2.Canny(gray, 40, 120)
        edge_density = np.sum(edges > 0) / edges.size
        
        # Adjusted thresholds for cartoon images
        edge_simplicity = 1.0 - min(1.0, edge_density * 15)  # Increased multiplier
        
        # Method 2: Refined texture analysis
        texture_complexity = self.calculate_refined_texture(gray)
        texture_simplicity = 1.0 - min(1.0, texture_complexity * 8)  # Adjusted scaling
        
        # Method 3: Calibrated color complexity
        color_complexity = self.calculate_calibrated_color_complexity(cv_image)
        color_simplicity = 1.0 - min(1.0, color_complexity * 1.2)  # Gentler scaling
        
        # Method 4: Refined object counting
        object_count = self.refined_object_counting(cv_image)
        object_simplicity = 1.0 - min(1.0, (object_count / 15.0))  # Adjusted threshold
        
        # Method 5: Frequency analysis with cartoon focus
        freq_complexity = self.cartoon_frequency_analysis(gray)
        freq_simplicity = 1.0 - min(1.0, freq_complexity * 2.5)  # Adjusted scaling
        
        # Weighted combination optimized for cartoon evaluation
        weights = [0.3, 0.2, 0.2, 0.15, 0.15]  # Prioritize edges and texture
        simplicities = [edge_simplicity, texture_simplicity, color_simplicity, object_simplicity, freq_simplicity]
        
        background_simplicity = sum(w * s for w, s in zip(weights, simplicities))
        
        # Adjusted grading thresholds
        if background_simplicity >= 0.75:
            grade = "Excellent - Very clean background"
        elif background_simplicity >= 0.6:
            grade = "Good - Mostly clean background"
        elif background_simplicity >= 0.45:
            grade = "Moderate - Somewhat busy background"
        else:
            grade = "Poor - Very busy/distracting background"
        
        complexity_breakdown = {
            "edge_simplicity": float(edge_simplicity),
            "texture_simplicity": float(texture_simplicity),
            "color_simplicity": float(color_simplicity),
            "object_simplicity": float(object_simplicity),
            "frequency_simplicity": float(freq_simplicity),
            "edge_density": float(edge_density),
            "object_count": object_count
        }
        
        print(f"   🎨 Background analysis: simplicity={background_simplicity:.3f}, objects={object_count}")
        
        return {
            "score": float(background_simplicity),
            "complexity_breakdown": complexity_breakdown,
            "grade": grade,
            "details": f"Calibrated simplicity: {background_simplicity:.3f} - {grade}"
        }
    
    def calculate_refined_texture(self, gray_image):
        """Refined texture calculation optimized for cartoons"""
        # Use smaller kernel for cartoon-appropriate texture analysis
        kernel_size = 7
        kernel = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size ** 2)
        
        mean = cv2.filter2D(gray_image.astype(np.float32), -1, kernel)
        sqr_diff = (gray_image.astype(np.float32) - mean) ** 2
        variance = cv2.filter2D(sqr_diff, -1, kernel)
        
        # More conservative texture complexity for cartoons
        avg_variance = np.mean(variance) / (255 ** 2)
        return min(1.0, avg_variance * 3)  # Reduced scaling
    
    def calculate_calibrated_color_complexity(self, cv_image):
        """Calibrated color complexity for cartoon images"""
        rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        
        # Use more appropriate clustering for cartoons
        pixels = rgb_image.reshape(-1, 3)
        
        try:
            # Reduced number of clusters for cartoon analysis
            n_clusters = min(6, len(np.unique(pixels, axis=0)))
            if n_clusters > 1:
                kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                kmeans.fit(pixels[::10])  # Sample pixels for efficiency
                dominant_colors = n_clusters
            else:
                dominant_colors = 1
        except:
            dominant_colors = 4
        
        # Color variance with cartoon-appropriate scaling
        color_variance = np.var(rgb_image.reshape(-1, 3), axis=0).mean() / (255 ** 2)
        
        # Adjusted complexity calculation for cartoons
        color_complexity = (dominant_colors / 8.0) * 0.7 + color_variance * 0.3
        
        return min(1.0, color_complexity)
    
    def refined_object_counting(self, cv_image):
        """Refined object counting with cartoon-appropriate parameters"""
        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        
        # Multiple approaches for better object detection
        # Approach 1: Adaptive threshold
        thresh1 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 3)
        
        # Approach 2: Otsu threshold
        _, thresh2 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Combine thresholds
        combined = cv2.bitwise_and(thresh1, thresh2)
        
        # Clean up with morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel)
        combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel, iterations=2)
        
        # Find contours
        contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Count significant objects with cartoon-appropriate criteria
        height, width = gray.shape
        min_area = (height * width) * 0.001  # Smaller minimum for cartoon objects
        max_area = (height * width) * 0.6    # Reasonable maximum
        
        significant_objects = 0
        for contour in contours:
            area = cv2.contourArea(contour)
            if min_area < area < max_area:
                # Additional filters for meaningful objects
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = max(w, h) / min(w, h) if min(w, h) > 0 else 0
                
                # Filter out very elongated shapes (likely artifacts)
                if aspect_ratio < 10 and w > 5 and h > 5:
                    significant_objects += 1
        
        return min(significant_objects, 25)  # Reasonable cap
    
    def cartoon_frequency_analysis(self, gray_image):
        """Frequency analysis optimized for cartoon images"""
        # Apply FFT
        fft = np.fft.fft2(gray_image)
        fft_shift = np.fft.fftshift(fft)
        magnitude_spectrum = np.abs(fft_shift)
        
        h, w = magnitude_spectrum.shape
        center_h, center_w = h // 2, w // 2
        
        y, x = np.ogrid[:h, :w]
        center_distance = np.sqrt((x - center_w)**2 + (y - center_h)**2)
        
        # Adjusted frequency analysis for cartoons
        # Cartoons typically have more mid-frequency content
        mid_freq_mask = (center_distance >= min(h, w) * 0.15) & (center_distance < min(h, w) * 0.4)
        high_freq_mask = center_distance >= min(h, w) * 0.4
        
        total_energy = np.sum(magnitude_spectrum)
        mid_freq_energy = np.sum(magnitude_spectrum[mid_freq_mask])
        high_freq_energy = np.sum(magnitude_spectrum[high_freq_mask])
        
        if total_energy > 0:
            mid_freq_ratio = mid_freq_energy / total_energy
            high_freq_ratio = high_freq_energy / total_energy
            
            # Combine ratios with cartoon-appropriate weighting
            complexity = mid_freq_ratio * 0.6 + high_freq_ratio * 0.4
        else:
            complexity = 0
        
        return min(1.0, complexity * 2)
    
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
            "grade": grade,
            "details": f"Optimized for cartoon style: {dominant_colors} colors, {grade}"
        }
    
    def analyze_autism_suitability(self, image):
        """Final refined autism suitability analysis"""
        if not self.available:
            return {"error": "Analyzer not available"}
        
        # Convert PIL to cv2 if needed
        if isinstance(image, Image.Image):
            cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        else:
            cv_image = image
        
        print(f"   🔧 Starting refined analysis...")
        
        # Refined analyses
        results = {
            "person_count": self.refined_person_counting(cv_image),
            "background_simplicity": self.calibrated_background_analysis(cv_image),
            "color_appropriateness": self.calibrated_color_analysis(cv_image),
            "character_clarity": self.analyze_character_clarity(cv_image),  # Keep original
            "sensory_friendliness": self.analyze_sensory_friendliness(cv_image),  # Keep original
            "focus_clarity": self.analyze_focus_clarity(cv_image)  # Keep original
        }
        
        # Calculate overall autism suitability with refined weights
        results["autism_suitability"] = self.calculate_refined_autism_score(results)
        results["autism_grade"] = self.get_autism_grade(results["autism_suitability"])
        results["recommendations"] = self.generate_refined_recommendations(results)
        
        return results
    
    def calculate_refined_autism_score(self, results):
        """Calculate autism score with refined weighting"""
        # Fine-tuned weights based on autism education priorities
        weights = {
            "person_count": 0.40,           # CRITICAL: max 2 people
            "background_simplicity": 0.25,  # Very important for focus
            "color_appropriateness": 0.15,  # Important for sensory processing
            "character_clarity": 0.10,      # Important for comprehension
            "sensory_friendliness": 0.07,   # Avoid overstimulation
            "focus_clarity": 0.03          # Supporting factor
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
        
        # Apply bonus/penalty modifiers
        person_count = results["person_count"]["count"]
        
        # Bonus for ideal person count
        if person_count == 1:
            autism_suitability += 0.05  # Small bonus for single character
        elif person_count == 0:
            autism_suitability -= 0.1   # Penalty for no people
        
        # Penalty for too many people gets stronger
        if person_count > 2:
            penalty = min(0.3, (person_count - 2) * 0.08)
            autism_suitability -= penalty
        
        return float(min(1.0, max(0.0, autism_suitability)))
    
    def generate_refined_recommendations(self, results):
        """Generate more precise recommendations"""
        recommendations = []
        
        # Person count analysis
        person_data = results["person_count"]
        person_count = person_data["count"]
        
        if person_count == 0:
            recommendations.append("⚠️ No people detected - ensure main character is clearly visible")
        elif person_count == 1:
            recommendations.append("✅ Perfect: Single character ideal for autism storyboards")
        elif person_count == 2:
            recommendations.append("✅ Good: Two characters acceptable for autism education")
        else:
            recommendations.append(f"🚨 CRITICAL: Too many people ({person_count}) - reduce to 1-2 characters maximum")
        
        # Background analysis
        bg_score = results["background_simplicity"]["score"]
        if bg_score >= 0.8:
            recommendations.append("✅ Excellent: Background is clean and autism-friendly")
        elif bg_score >= 0.6:
            recommendations.append("👍 Good: Background mostly appropriate")
        elif bg_score >= 0.45:
            recommendations.append("⚠️ Moderate: Background could be simpler")
        else:
            recommendations.append("🚨 Critical: Background too busy - significantly reduce visual clutter")
        
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
        
        # Overall assessment
        overall_score = results["autism_suitability"]
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
    
    # Include the unchanged analysis methods from previous version
    def analyze_character_clarity(self, cv_image):
        """Character clarity analysis (unchanged from previous version)"""
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
            "grade": grade,
            "details": "Higher score = clearer character boundaries and definition"
        }
    
    def analyze_sensory_friendliness(self, cv_image):
        """Sensory friendliness analysis (unchanged)"""
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
            "grade": grade,
            "details": "Higher score = less overwhelming, more sensory-appropriate"
        }
    
    def analyze_focus_clarity(self, cv_image):
        """Focus clarity analysis (unchanged)"""
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
            "grade": grade,
            "details": "Higher score = clearer focal point, less distraction"
        }
    
    def get_autism_grade(self, suitability_score):
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


def test_refined_accuracy():
    """Test the refined evaluation accuracy"""
    print("🔧 TESTING REFINED EVALUATION ACCURACY")
    print("=" * 50)
    
    # Test with existing generated images
    output_dir = "sdxl_evaluation_output"
    
    if not os.path.exists(output_dir):
        print("❌ No generated images found. Run the SDXL generation test first.")
        return
    
    # Load existing images
    import glob
    image_files = []
    for i in range(1, 7):
        pattern = f"{i:02d}_*_clean.png"
        matches = glob.glob(os.path.join(output_dir, pattern))
        if matches:
            image_files.append(matches[0])
    
    if not image_files:
        print("❌ No clean generated images found")
        return
    
    print(f"📁 Found {len(image_files)} generated images")
    
    # Initialize refined analyzer
    refined_analyzer = RefinedAutismAnalyzer()
    if not refined_analyzer.available:
        print("❌ Refined analyzer not available")
        return
    
    results_summary = []
    
    # Test each image
    for i, image_path in enumerate(image_files, 1):
        filename = os.path.basename(image_path)
        print(f"\n🔧 REFINED ANALYSIS {i}: {filename}")
        print("-" * 45)
        
        # Load image
        image = Image.open(image_path).convert('RGB')
        
        # Analyze with refined method
        result = refined_analyzer.analyze_autism_suitability(image)
        
        if "error" not in result:
            autism_score = result['autism_suitability']
            person_count = result['person_count']['count']
            bg_score = result['background_simplicity']['score']
            color_score = result['color_appropriateness']['score']
            
            print(f"✅ REFINED RESULTS:")
            print(f"   🧩 Autism Score: {autism_score:.3f} ({result['autism_grade'].split('(')[0].strip()})")
            print(f"   👥 People: {person_count} ({'✅' if result['person_count']['is_compliant'] else '❌'})")
            print(f"   🎨 Background: {bg_score:.3f} ({result['background_simplicity']['grade'].split(' - ')[0]})")
            print(f"   🌈 Colors: {color_score:.3f} ({result['color_appropriateness']['dominant_colors']} colors)")
            
            # Show detection breakdown
            detection = result['person_count']['detection_breakdown']
            print(f"   🔍 Person detection: {detection}")
            
            # Show top recommendations
            for rec in result['recommendations'][:2]:
                print(f"   💡 {rec}")
                
            results_summary.append({
                "image": filename,
                "autism_score": autism_score,
                "person_count": person_count,
                "bg_score": bg_score,
                "color_score": color_score,
                "grade": result['autism_grade'].split('(')[0].strip()
            })
        else:
            print(f"❌ Analysis failed: {result['error']}")
    
    # Summary comparison
    print(f"\n📊 REFINED ACCURACY SUMMARY")
    print("=" * 50)
    
    if results_summary:
        # Sort by autism score for ranking
        results_summary.sort(key=lambda x: x['autism_score'], reverse=True)
        
        print("🏆 FINAL RANKING (Best to Worst):")
        for i, result in enumerate(results_summary, 1):
            score = result['autism_score']
            people = result['person_count']
            bg = result['bg_score']
            grade = result['grade']
            print(f"   {i}. {score:.3f} ({grade}) - {people} people, BG:{bg:.3f} - {result['image']}")
        
        # Check expected patterns
        simple_images = [r for r in results_summary if 'simple' in r['image'].lower()]
        complex_images = [r for r in results_summary if any(x in r['image'].lower() for x in ['complex', 'multiple', 'busy'])]
        
        if simple_images and complex_images:
            simple_avg = np.mean([r['autism_score'] for r in simple_images])
            complex_avg = np.mean([r['autism_score'] for r in complex_images])
            
            print(f"\n🎯 VALIDATION:")
            print(f"   Simple images average: {simple_avg:.3f}")
            print(f"   Complex images average: {complex_avg:.3f}")
            print(f"   Difference: {simple_avg - complex_avg:.3f}")
            
            if simple_avg > complex_avg:
                print("   ✅ EXPECTED: Simple > Complex (Good accuracy!)")
            else:
                print("   ⚠️ UNEXPECTED: Complex >= Simple")
    
    print(f"\n🎉 REFINED IMPROVEMENTS COMPLETED!")
    print("🔧 Key refinements made:")
    print("   • Calibrated background thresholds for cartoon style")
    print("   • Smart consensus person counting algorithm")
    print("   • Cartoon-optimized color analysis")
    print("   • Enhanced face detection with NMS")
    print("   • Refined scoring weights and grading")

if __name__ == "__main__":
    test_refined_accuracy()