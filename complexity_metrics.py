"""
Autism-specific complexity analysis
Enhanced with smart consensus person counting and calibrated cartoon analysis
Evaluates visual complexity, sensory appropriateness, and autism design principles
"""

import cv2
import numpy as np
from PIL import Image
from sklearn.cluster import KMeans
import torch
from transformers import CLIPProcessor, CLIPModel
import warnings
warnings.filterwarnings('ignore')


class AutismComplexityAnalyzer:
    """
    Enhanced analyzer for autism-specific requirements with smart consensus:
    - Smart consensus person counting (max 2)
    - Calibrated background simplicity for cartoons
    - Color appropriateness with cartoon-specific thresholds
    - Sensory friendliness analysis
    - Character clarity assessment
    """
    
    def __init__(self):
        print("🔧 Loading Enhanced Autism Complexity Analyzer...")
        
        # Load CLIP for semantic understanding
        try:
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
        
        print("✅ Enhanced Autism Complexity Analyzer ready")
    
    def analyze_complexity(self, image):
        """
        Comprehensive autism-friendly complexity analysis with enhanced algorithms
        
        Args:
            image: PIL Image or numpy array
            
        Returns:
            Dictionary with all complexity metrics and scores
        """
        # Convert to CV format
        if isinstance(image, Image.Image):
            cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        else:
            cv_image = image
        
        print(f"   🔧 Starting enhanced autism complexity analysis...")
        
        # Run all enhanced analyses
        results = {
            'person_count': self._analyze_person_count_enhanced(cv_image),
            'background_simplicity': self._analyze_background_enhanced(cv_image),
            'color_appropriateness': self._analyze_colors_enhanced(cv_image),
            'character_clarity': self._analyze_character_clarity(cv_image),
            'sensory_friendliness': self._analyze_sensory_load_enhanced(cv_image),
            'focus_clarity': self._analyze_focus_clarity(cv_image)
        }
        
        # Calculate overall autism suitability with refined weights
        results['autism_suitability'] = self._calculate_autism_score_enhanced(results)
        results['autism_grade'] = self._get_autism_grade(results['autism_suitability'])
        results['recommendations'] = self._generate_recommendations_enhanced(results)
        
        return results
    
    def _analyze_person_count_enhanced(self, cv_image):
        """Enhanced person counting with smart consensus algorithm"""
        height, width = cv_image.shape[:2]
        detection_results = {}
        
        # Method 1: Enhanced face detection
        face_count = self._enhanced_face_detection(cv_image)
        detection_results["face_detection"] = face_count
        
        # Method 2: CLIP semantic analysis
        if self.clip_available:
            clip_count = self._clip_person_analysis(cv_image)
            detection_results["clip_analysis"] = clip_count
        else:
            detection_results["clip_analysis"] = 0
        
        # Method 3: Refined shape detection
        shape_count = self._refined_shape_detection(cv_image)
        detection_results["shape_detection"] = shape_count
        
        # Method 4: Skin region analysis
        skin_count = self._improved_skin_detection(cv_image)
        detection_results["skin_detection"] = skin_count
        
        # Smart consensus algorithm
        final_count = self._smart_consensus_count(detection_results)
        
        print(f"   🔍 Detection breakdown: {detection_results}")
        print(f"   🎯 Smart consensus: {final_count} people")
        
        # Calculate compliance and scoring
        is_compliant = final_count <= 2
        compliance_level = "Excellent" if final_count <= 2 else "Poor"
        
        # FIXED: Both 1 and 2 people get perfect score
        if final_count <= 2:
            score = 1.0  # Both 1 and 2 people are perfect for autism
        else:
            score = max(0.0, 1.0 - (final_count - 2) * 0.25)
        
        return {
            'count': final_count,
            'score': score,
            'is_compliant': is_compliant,
            'compliance_level': compliance_level,
            'detection_breakdown': detection_results,
            'detection_method': 'smart_consensus'
        }
    
    def _enhanced_face_detection(self, cv_image):
        """Enhanced face detection with better parameters for cartoons"""
        if not self.face_detection_available:
            return 0
            
        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        
        # Multiple scale face detection optimized for cartoon styles
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
        merged_faces = self._merge_overlapping_detections(all_faces, overlap_threshold=0.4)
        
        print(f"   👤 Enhanced face detection: {len(merged_faces)} faces")
        return len(merged_faces)
    
    def _merge_overlapping_detections(self, detections, overlap_threshold=0.4):
        """Improved Non-Maximum Suppression for face detection"""
        if len(detections) <= 1:
            return detections
        
        # Convert to format suitable for NMS
        boxes = []
        for (x, y, w, h) in detections:
            boxes.append([x, y, x + w, y + h])
        
        if not boxes:
            return []
        
        boxes = np.array(boxes)
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
    
    def _clip_person_analysis(self, cv_image):
        """Enhanced CLIP analysis with better prompts for person counting"""
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
    
    def _refined_shape_detection(self, cv_image):
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
    
    def _improved_skin_detection(self, cv_image):
        """Improved skin detection with cartoon-appropriate ranges"""
        # Convert to HSV for better skin detection
        hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
        
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
    
    def _smart_consensus_count(self, detection_results):
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
    
    def _analyze_background_enhanced(self, cv_image):
        """Enhanced background analysis with cartoon-appropriate thresholds"""
        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        height, width = gray.shape
        
        # Method 1: Calibrated edge density
        edges = cv2.Canny(gray, 40, 120)
        edge_density = np.sum(edges > 0) / edges.size
        
        # Adjusted thresholds for cartoon images
        edge_simplicity = 1.0 - min(1.0, edge_density * 15)  # Increased multiplier
        
        # Method 2: Refined texture analysis
        texture_complexity = self._calculate_refined_texture(gray)
        texture_simplicity = 1.0 - min(1.0, texture_complexity * 8)  # Adjusted scaling
        
        # Method 3: Calibrated color complexity
        color_complexity = self._calculate_calibrated_color_complexity(cv_image)
        color_simplicity = 1.0 - min(1.0, color_complexity * 1.2)  # Gentler scaling
        
        # Method 4: Refined object counting
        object_count = self._refined_object_counting(cv_image)
        object_simplicity = 1.0 - min(1.0, (object_count / 15.0))  # Adjusted threshold
        
        # Method 5: Frequency analysis with cartoon focus
        freq_complexity = self._cartoon_frequency_analysis(gray)
        freq_simplicity = 1.0 - min(1.0, freq_complexity * 2.5)  # Adjusted scaling
        
        # Weighted combination optimized for cartoon evaluation
        weights = [0.3, 0.2, 0.2, 0.15, 0.15]  # Prioritize edges and texture
        simplicities = [edge_simplicity, texture_simplicity, color_simplicity, object_simplicity, freq_simplicity]
        
        background_simplicity = sum(w * s for w, s in zip(weights, simplicities))
        
        print(f"   🎨 Background analysis: simplicity={background_simplicity:.3f}, objects={object_count}")
        
        return {
            'score': float(background_simplicity),
            'edge_density': float(edge_density),
            'object_count': object_count,
            'is_simple': background_simplicity > 0.65
        }
    
    def _calculate_refined_texture(self, gray_image):
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
    
    def _calculate_calibrated_color_complexity(self, cv_image):
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
    
    def _refined_object_counting(self, cv_image):
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
    
    def _cartoon_frequency_analysis(self, gray_image):
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
    
    def _analyze_colors_enhanced(self, cv_image):
        """Enhanced color analysis with cartoon-appropriate thresholds"""
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
        
        print(f"   🌈 Color analysis: {dominant_colors} colors, score={color_appropriateness:.3f}")
        
        return {
            'score': float(color_appropriateness),
            'dominant_colors': dominant_colors,
            'avg_saturation': float(avg_saturation),
            'avg_brightness': float(avg_brightness),
            'is_appropriate': color_appropriateness > 0.7
        }
    
    def _analyze_character_clarity(self, cv_image):
        """Analyze how clearly characters are defined"""
        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        
        # Strong edge detection
        edges = cv2.Canny(gray, 100, 200)
        strong_edge_density = np.sum(edges > 0) / edges.size
        
        # Gradient analysis
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        avg_contrast = np.mean(gradient_magnitude) / 255.0
        
        # Contour quality analysis
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        shape_quality = 0
        if len(contours) > 0:
            large_contours = [c for c in contours if cv2.contourArea(c) > 500]
            
            if large_contours:
                smoothness_scores = []
                for contour in large_contours[:5]:  # Top 5 largest
                    epsilon = 0.02 * cv2.arcLength(contour, True)
                    approx = cv2.approxPolyDP(contour, epsilon, True)
                    if len(contour) > 0:
                        smoothness = len(approx) / len(contour)
                        smoothness_scores.append(smoothness)
                
                shape_quality = np.mean(smoothness_scores) if smoothness_scores else 0
        
        # Combine metrics
        edge_clarity = min(1.0, strong_edge_density * 8)
        contrast_clarity = min(1.0, avg_contrast * 3)
        shape_clarity = min(1.0, shape_quality * 5)
        
        character_clarity = (edge_clarity * 0.4 + contrast_clarity * 0.4 + shape_clarity * 0.2)
        
        return {
            'score': float(character_clarity),
            'edge_strength': float(strong_edge_density),
            'contrast': float(avg_contrast),
            'is_clear': character_clarity > 0.6
        }
    
    def _analyze_sensory_load_enhanced(self, cv_image):
        """Enhanced sensory friendliness analysis"""
        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        
        # High frequency content analysis (busy patterns)
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
        
        # Intensity variations
        kernel = np.ones((5,5), np.float32) / 25
        local_mean = cv2.filter2D(gray.astype(np.float32), -1, kernel)
        intensity_variance = np.var(gray.astype(np.float32) - local_mean) / (255**2)
        
        # Motion blur check (could indicate movement/instability)
        motion_blur_kernel = np.zeros((9, 9))
        motion_blur_kernel[4, :] = 1/9
        blurred = cv2.filter2D(gray, -1, motion_blur_kernel)
        blur_difference = np.mean(np.abs(gray.astype(float) - blurred.astype(float))) / 255.0
        
        # Sensory friendliness score
        pattern_score = 1.0 - min(1.0, high_freq_ratio * 3)
        intensity_score = 1.0 - min(1.0, intensity_variance * 10)
        motion_score = max(0.0, 1.0 - blur_difference * 2)
        
        sensory_score = (pattern_score * 0.5 + intensity_score * 0.3 + motion_score * 0.2)
        
        return {
            'score': float(sensory_score),
            'high_freq_ratio': float(high_freq_ratio),
            'intensity_variation': float(intensity_variance),
            'is_sensory_friendly': sensory_score > 0.6
        }
    
    def _analyze_focus_clarity(self, cv_image):
        """Analyze if there's a clear focal point"""
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
        
        # Gradient-based focus analysis
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        center_contrast = np.sum(gradient_magnitude * center_weights) / np.sum(center_weights)
        peripheral_contrast = np.sum(gradient_magnitude * (1 - center_weights)) / np.sum(1 - center_weights)
        contrast_ratio = center_contrast / (peripheral_contrast + 1e-6)
        contrast_focus = min(1.0, contrast_ratio / 2.0)
        
        focus_clarity = (focus_score * 0.6 + contrast_focus * 0.4)
        
        return {
            'score': float(focus_clarity),
            'focus_ratio': float(focus_ratio),
            'has_clear_focus': focus_clarity > 0.5
        }
    
    def _calculate_autism_score_enhanced(self, results):
        """Calculate overall autism suitability with refined weighting"""
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
        
        # FIXED: Bonus for ideal person count (1-2 people both get bonus)
        if person_count == 1 or person_count == 2:
            autism_suitability += 0.05  # Bonus for 1-2 people (both are ideal)
        elif person_count == 0:
            autism_suitability -= 0.1   # Penalty for no people
        
        # Penalty for too many people gets stronger
        if person_count > 2:
            penalty = min(0.3, (person_count - 2) * 0.08)
            autism_suitability -= penalty
        
        return float(min(1.0, max(0.0, autism_suitability)))
    
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
    
    def _generate_recommendations_enhanced(self, results):
        """Generate enhanced recommendations based on analysis"""
        recommendations = []
        
        # Person count analysis (most critical)
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
        
        # Character clarity
        if results["character_clarity"]["score"] < 0.6:
            recommendations.append("✏️ Improve character definition with clearer outlines")
        
        # Sensory friendliness
        if results["sensory_friendliness"]["score"] < 0.6:
            recommendations.append("⚡ Reduce visual complexity to avoid sensory overload")
        
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