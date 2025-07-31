"""
Autism-specific complexity analysis
Evaluates visual complexity, sensory appropriateness, and autism design principles
"""

import cv2
import numpy as np
from PIL import Image
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')


class AutismComplexityAnalyzer:
    """
    Analyzes images for autism-specific requirements:
    - Person count (max 2)
    - Background simplicity
    - Color appropriateness
    - Sensory friendliness
    - Character clarity
    """
    
    def __init__(self):
        self.face_cascade = None
        self.profile_cascade = None
        self.load_detectors()
    
    def load_detectors(self):
        """Load face detection cascades"""
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
            print("   ⚠ Face detection unavailable for person counting")
    
    def analyze_complexity(self, image):
        """
        Comprehensive autism-friendly complexity analysis
        
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
        
        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        
        # Run all analyses
        results = {
            'person_count': self._analyze_person_count(cv_image, gray),
            'background_simplicity': self._analyze_background(cv_image, gray),
            'color_appropriateness': self._analyze_colors(cv_image),
            'character_clarity': self._analyze_character_clarity(gray),
            'sensory_friendliness': self._analyze_sensory_load(cv_image, gray),
            'focus_clarity': self._analyze_focus_clarity(gray)
        }
        
        # Calculate overall autism suitability
        results['autism_suitability'] = self._calculate_autism_score(results)
        results['recommendations'] = self._generate_recommendations(results)
        
        return results
    
    def _analyze_person_count(self, cv_image, gray):
        """Count people in image - critical for autism suitability"""
        height, width = gray.shape
        
        # Multiple detection methods
        face_count = 0
        
        if self.face_detection_available:
            # Frontal faces
            frontal_faces = self.face_cascade.detectMultiScale(
                gray, scaleFactor=1.05, minNeighbors=3, minSize=(25, 25)
            )
            
            # Profile faces
            profile_faces = self.profile_cascade.detectMultiScale(
                gray, scaleFactor=1.05, minNeighbors=3, minSize=(25, 25)
            )
            
            # Merge overlapping detections
            all_faces = list(frontal_faces) + list(profile_faces)
            face_count = len(self._merge_overlapping_detections(all_faces))
        
        # Backup: human shape detection
        shape_count = self._detect_human_shapes(gray)
        
        # Conservative estimate
        person_count = max(face_count, min(face_count + 1, shape_count))
        
        # Score calculation (critical metric)
        if person_count <= 1:
            score = 1.0
        elif person_count == 2:
            score = 0.8
        else:
            score = max(0.0, 1.0 - (person_count - 2) * 0.3)
        
        return {
            'count': person_count,
            'score': score,
            'is_compliant': person_count <= 2,
            'detection_method': 'face_detection' if face_count > 0 else 'shape_detection'
        }
    
    def _analyze_background(self, cv_image, gray):
        """Analyze background complexity"""
        # Edge density
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        
        # Object counting
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Count significant objects
        min_area = gray.shape[0] * gray.shape[1] * 0.001
        object_count = sum(1 for c in contours if cv2.contourArea(c) > min_area)
        
        # Texture analysis
        texture_score = self._calculate_texture_complexity(gray)
        
        # Combined simplicity score
        edge_simplicity = 1.0 - min(1.0, edge_density * 15)
        object_simplicity = 1.0 - min(1.0, object_count / 15)
        texture_simplicity = 1.0 - texture_score
        
        simplicity_score = (edge_simplicity * 0.4 + 
                          object_simplicity * 0.3 + 
                          texture_simplicity * 0.3)
        
        return {
            'score': simplicity_score,
            'edge_density': edge_density,
            'object_count': object_count,
            'is_simple': simplicity_score > 0.65
        }
    
    def _analyze_colors(self, cv_image):
        """Analyze color appropriateness for autism"""
        # Convert to RGB for analysis
        rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        pixels = rgb_image.reshape(-1, 3)
        
        # Color clustering
        n_colors = min(8, len(np.unique(pixels, axis=0)) // 100)
        if n_colors > 1:
            kmeans = KMeans(n_clusters=n_colors, random_state=42, n_init=10)
            kmeans.fit(pixels[::20])  # Sample for speed
            dominant_colors = len(kmeans.cluster_centers_)
        else:
            dominant_colors = 1
        
        # Saturation analysis
        hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
        avg_saturation = np.mean(hsv[:,:,1]) / 255.0
        
        # Brightness analysis
        avg_brightness = np.mean(hsv[:,:,2]) / 255.0
        
        # Scoring
        color_count_score = 1.0 if dominant_colors <= 4 else max(0.2, 1.0 - (dominant_colors - 4) * 0.15)
        saturation_score = 1.0 - abs(avg_saturation - 0.6) * 2  # Optimal around 0.6
        brightness_score = 1.0 - abs(avg_brightness - 0.7) * 2   # Optimal around 0.7
        
        color_score = (color_count_score * 0.5 + 
                      saturation_score * 0.3 + 
                      brightness_score * 0.2)
        
        return {
            'score': color_score,
            'dominant_colors': dominant_colors,
            'avg_saturation': avg_saturation,
            'avg_brightness': avg_brightness,
            'is_appropriate': color_score > 0.7
        }
    
    def _analyze_character_clarity(self, gray):
        """Analyze how clearly characters are defined"""
        # Strong edge detection
        edges = cv2.Canny(gray, 100, 200)
        edge_strength = np.sum(edges > 0) / edges.size
        
        # Contrast analysis
        contrast = np.std(gray) / 255.0
        
        # Contour quality
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Analyze largest contours (likely characters)
            areas = [cv2.contourArea(c) for c in contours]
            large_contours = [c for c, a in zip(contours, areas) if a > np.mean(areas)]
            
            if large_contours:
                # Check contour smoothness
                smoothness_scores = []
                for contour in large_contours[:3]:  # Top 3 largest
                    perimeter = cv2.arcLength(contour, True)
                    if perimeter > 0:
                        # Approximation quality
                        epsilon = 0.02 * perimeter
                        approx = cv2.approxPolyDP(contour, epsilon, True)
                        smoothness = 1.0 - min(1.0, len(approx) / 50)
                        smoothness_scores.append(smoothness)
                
                contour_quality = np.mean(smoothness_scores) if smoothness_scores else 0.5
            else:
                contour_quality = 0.5
        else:
            contour_quality = 0.3
        
        clarity_score = (edge_strength * 10 * 0.4 + 
                        contrast * 2 * 0.3 + 
                        contour_quality * 0.3)
        
        return {
            'score': min(1.0, clarity_score),
            'edge_strength': edge_strength,
            'contrast': contrast,
            'is_clear': clarity_score > 0.6
        }
    
    def _analyze_sensory_load(self, cv_image, gray):
        """Analyze potential sensory overload factors"""
        # High frequency content (busy patterns)
        fft = np.fft.fft2(gray)
        fft_shift = np.fft.fftshift(fft)
        magnitude = np.abs(fft_shift)
        
        h, w = magnitude.shape
        center = (h // 2, w // 2)
        high_freq_mask = np.zeros_like(magnitude, dtype=bool)
        
        y, x = np.ogrid[:h, :w]
        distance = np.sqrt((x - center[1])**2 + (y - center[0])**2)
        high_freq_mask[distance > min(h, w) * 0.3] = True
        
        high_freq_ratio = np.sum(magnitude[high_freq_mask]) / np.sum(magnitude)
        
        # Intensity variations
        local_std = cv2.Laplacian(gray, cv2.CV_64F).var() / 1000
        
        # Motion blur check (could indicate movement/instability)
        motion_score = self._check_motion_blur(gray)
        
        # Sensory friendliness score
        pattern_score = 1.0 - min(1.0, high_freq_ratio * 3)
        intensity_score = 1.0 - min(1.0, local_std)
        stability_score = 1.0 - motion_score
        
        sensory_score = (pattern_score * 0.5 + 
                        intensity_score * 0.3 + 
                        stability_score * 0.2)
        
        return {
            'score': sensory_score,
            'high_freq_ratio': high_freq_ratio,
            'intensity_variation': local_std,
            'is_sensory_friendly': sensory_score > 0.6
        }
    
    def _analyze_focus_clarity(self, gray):
        """Analyze if there's a clear focal point"""
        # Center-weighted edge detection
        h, w = gray.shape
        center_y, center_x = h // 2, w // 2
        
        # Create center weight mask
        y, x = np.ogrid[:h, :w]
        center_distance = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        max_distance = np.sqrt(center_x**2 + center_y**2)
        center_weights = 1.0 - (center_distance / max_distance)
        
        # Edge detection
        edges = cv2.Canny(gray, 50, 150)
        
        # Weighted edge density
        center_edges = np.sum(edges * center_weights) / np.sum(center_weights)
        peripheral_edges = np.sum(edges * (1 - center_weights)) / np.sum(1 - center_weights)
        
        focus_ratio = center_edges / (peripheral_edges + 1e-6)
        focus_score = min(1.0, focus_ratio / 3)
        
        return {
            'score': focus_score,
            'focus_ratio': focus_ratio,
            'has_clear_focus': focus_score > 0.5
        }
    
    def _merge_overlapping_detections(self, detections, overlap_thresh=0.3):
        """Merge overlapping face detections"""
        if len(detections) <= 1:
            return detections
        
        # Convert to boxes format
        boxes = []
        for (x, y, w, h) in detections:
            boxes.append([x, y, x + w, y + h])
        
        boxes = np.array(boxes)
        
        # Simple NMS
        keep = []
        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        indices = np.argsort(areas)[::-1]
        
        while len(indices) > 0:
            current = indices[0]
            keep.append(current)
            
            if len(indices) == 1:
                break
            
            # Calculate IoU
            xx1 = np.maximum(boxes[current, 0], boxes[indices[1:], 0])
            yy1 = np.maximum(boxes[current, 1], boxes[indices[1:], 1])
            xx2 = np.minimum(boxes[current, 2], boxes[indices[1:], 2])
            yy2 = np.minimum(boxes[current, 3], boxes[indices[1:], 3])
            
            w = np.maximum(0, xx2 - xx1)
            h = np.maximum(0, yy2 - yy1)
            
            overlap = (w * h) / areas[indices[1:]]
            indices = indices[1:][overlap < overlap_thresh]
        
        # Convert back to detection format
        merged = []
        for idx in keep:
            x1, y1, x2, y2 = boxes[idx]
            merged.append((x1, y1, x2 - x1, y2 - y1))
        
        return merged
    
    def _detect_human_shapes(self, gray):
        """Detect human-like shapes as backup"""
        edges = cv2.Canny(gray, 30, 100)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        height, width = gray.shape
        min_area = height * width * 0.01
        max_area = height * width * 0.5
        
        human_shapes = 0
        for contour in contours:
            area = cv2.contourArea(contour)
            if min_area < area < max_area:
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = h / w if w > 0 else 0
                
                # Human-like proportions
                if 1.2 < aspect_ratio < 3.5:
                    human_shapes += 1
        
        return min(human_shapes, 5)
    
    def _calculate_texture_complexity(self, gray):
        """Calculate texture complexity score"""
        # Local variance
        kernel_size = 5
        kernel = np.ones((kernel_size, kernel_size)) / (kernel_size ** 2)
        
        mean = cv2.filter2D(gray.astype(float), -1, kernel)
        sqr_mean = cv2.filter2D(gray.astype(float) ** 2, -1, kernel)
        variance = sqr_mean - mean ** 2
        
        avg_variance = np.mean(variance) / (255 ** 2)
        return min(1.0, avg_variance * 5)
    
    def _check_motion_blur(self, gray):
        """Check for motion blur indicators"""
        # Detect directional blur using different kernels
        kernels = [
            np.array([[1, 0, 0], [0, 0, 0], [0, 0, -1]]),  # Diagonal
            np.array([[0, 0, 1], [0, 0, 0], [-1, 0, 0]]),  # Other diagonal
            np.array([[0, 1, 0], [0, 0, 0], [0, -1, 0]]),  # Vertical
            np.array([[1, 0, -1], [0, 0, 0], [0, 0, 0]])   # Horizontal
        ]
        
        responses = []
        for kernel in kernels:
            response = cv2.filter2D(gray, -1, kernel)
            responses.append(np.std(response))
        
        # High variance in responses indicates directional blur
        motion_score = np.std(responses) / (np.mean(responses) + 1e-6)
        return min(1.0, motion_score)
    
    def _calculate_autism_score(self, results):
        """Calculate overall autism suitability score"""
        # Weights based on importance for autism education
        weights = {
            'person_count': 0.35,        # Most critical
            'background_simplicity': 0.20,
            'color_appropriateness': 0.15,
            'character_clarity': 0.12,
            'sensory_friendliness': 0.10,
            'focus_clarity': 0.08
        }
        
        total = 0.0
        for metric, weight in weights.items():
            if metric in results and 'score' in results[metric]:
                total += results[metric]['score'] * weight
        
        return total
    
    def _generate_recommendations(self, results):
        """Generate specific recommendations"""
        recommendations = []
        
        # Person count (most critical)
        person_count = results['person_count']['count']
        if person_count > 2:
            recommendations.append(f"🚨 CRITICAL: Reduce to 1-2 people (currently {person_count})")
        elif person_count == 0:
            recommendations.append("👤 Add a clear main character")
        
        # Background
        if results['background_simplicity']['score'] < 0.65:
            recommendations.append("🎨 Simplify background - remove distracting elements")
        
        # Colors
        if results['color_appropriateness']['dominant_colors'] > 6:
            recommendations.append("🌈 Reduce color palette to 4-6 colors")
        
        # Character clarity
        if results['character_clarity']['score'] < 0.6:
            recommendations.append("✏️ Improve character definition with clearer outlines")
        
        # Sensory
        if results['sensory_friendliness']['score'] < 0.6:
            recommendations.append("⚡ Reduce visual complexity to avoid sensory overload")
        
        return recommendations