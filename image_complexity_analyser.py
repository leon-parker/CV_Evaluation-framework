"""
Autism-Friendly Storyboard Image Analyzer
Analyzes images specifically for autism education requirements
Focuses on visual clarity, simplicity, and sensory-appropriate design
"""

import cv2
import numpy as np
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings("ignore")


class AutismFriendlyImageAnalyzer:
    """Specialized analyzer for autism storyboard image evaluation"""
    
    def __init__(self):
        print("🧩 Loading Autism-Friendly Image Analyzer...")
        
        try:
            # Load CLIP for semantic understanding
            self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
            
            if torch.cuda.is_available():
                self.clip_model = self.clip_model.to("cuda")
            
            # Face detection for person counting
            try:
                self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
                self.face_detection_available = True
            except:
                print("⚠️ Face detection not available")
                self.face_detection_available = False
            
            self.available = True
            print("✅ Autism-Friendly Image Analyzer ready")
            
        except Exception as e:
            print(f"❌ Failed to load analyzer: {e}")
            self.available = False
    
    def analyze_autism_suitability(self, image):
        """
        Analyze image suitability for autism storyboards
        Returns scores and recommendations for autism-friendly design
        """
        if not self.available:
            return {"error": "Analyzer not available"}
        
        # Convert PIL to cv2 if needed
        if isinstance(image, Image.Image):
            cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            pil_image = image
        else:
            cv_image = image
            pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        # Core autism-specific analyses
        results = {
            "person_count": self.count_people(cv_image),
            "background_simplicity": self.analyze_background_simplicity(cv_image),
            "color_appropriateness": self.analyze_color_appropriateness(cv_image),
            "character_clarity": self.analyze_character_clarity(cv_image),
            "sensory_friendliness": self.analyze_sensory_friendliness(cv_image),
            "focus_clarity": self.analyze_focus_clarity(cv_image)
        }
        
        # Calculate overall autism suitability
        results["autism_suitability"] = self.calculate_autism_suitability(results)
        results["autism_grade"] = self.get_autism_grade(results["autism_suitability"])
        results["recommendations"] = self.generate_recommendations(results)
        
        return results
    
    def count_people(self, cv_image):
        """Count number of people in image - critical for autism storyboards (max 2)"""
        people_count = 0
        detection_methods = []
        
        # Method 1: Face detection
        if self.face_detection_available:
            gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
            face_count = len(faces)
            people_count = max(people_count, face_count)
            detection_methods.append(f"Face detection: {face_count}")
        
        # Method 2: Human-like shape detection (simple contour analysis)
        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Look for person-like vertical contours
        person_like_shapes = 0
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 1000:  # Minimum size for person
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = h / w if w > 0 else 0
                
                # Person-like aspect ratio (taller than wide)
                if 1.5 <= aspect_ratio <= 4.0:
                    person_like_shapes += 1
        
        # Conservative estimate (don't over-count)
        shape_based_count = min(person_like_shapes, 3)  # Cap at 3 to avoid false positives
        people_count = max(people_count, shape_based_count)
        detection_methods.append(f"Shape detection: {shape_based_count}")
        
        # Determine compliance
        is_compliant = people_count <= 2
        compliance_level = "Excellent" if people_count <= 1 else "Good" if people_count == 2 else "Poor"
        
        return {
            "count": people_count,
            "is_compliant": is_compliant,
            "compliance_level": compliance_level,
            "detection_methods": detection_methods,
            "score": 1.0 if people_count <= 2 else max(0.0, 1.0 - (people_count - 2) * 0.3),
            "details": f"Autism storyboards should have maximum 2 people. Detected: {people_count}"
        }
    
    def analyze_background_simplicity(self, cv_image):
        """Analyze background simplicity - autism storyboards need clean, non-distracting backgrounds"""
        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        
        # Edge density in background regions
        edges = cv2.Canny(gray, 30, 100)  # Lower thresholds for subtle details
        edge_density = np.sum(edges > 0) / edges.size
        
        # Color variance (simpler = less color variation)
        color_variance = np.var(cv_image.reshape(-1, 3), axis=0).mean() / (255**2)
        
        # Texture complexity in background
        # Use image segmentation to identify likely background areas
        height, width = gray.shape
        border_region = np.zeros_like(gray)
        border_width = min(width//10, height//10, 50)  # Border region
        border_region[:border_width, :] = 1  # Top
        border_region[-border_width:, :] = 1  # Bottom  
        border_region[:, :border_width] = 1  # Left
        border_region[:, -border_width:] = 1  # Right
        
        # Background texture analysis
        background_variance = np.var(gray[border_region == 1]) / (255**2) if np.sum(border_region) > 0 else 0
        
        # Calculate simplicity score (higher = simpler background)
        edge_simplicity = 1.0 - min(1.0, edge_density * 10)  # Scale edge density
        color_simplicity = 1.0 - min(1.0, color_variance * 5)
        texture_simplicity = 1.0 - min(1.0, background_variance * 8)
        
        background_simplicity = (edge_simplicity * 0.4 + color_simplicity * 0.3 + texture_simplicity * 0.3)
        
        # Grade simplicity
        if background_simplicity >= 0.8:
            grade = "Excellent - Very clean background"
        elif background_simplicity >= 0.6:
            grade = "Good - Mostly clean background"
        elif background_simplicity >= 0.4:
            grade = "Moderate - Somewhat busy background"
        else:
            grade = "Poor - Very busy/distracting background"
        
        return {
            "score": float(background_simplicity),
            "edge_density": float(edge_density),
            "color_variance": float(color_variance),
            "texture_variance": float(background_variance),
            "grade": grade,
            "details": "Higher score = cleaner, less distracting background"
        }
    
    def analyze_color_appropriateness(self, cv_image):
        """Analyze color palette for autism-friendliness - avoid overwhelming colors"""
        # Convert to different color spaces
        rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        hsv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
        
        # Color count analysis
        pixels = rgb_image.reshape(-1, 3)
        unique_colors = len(np.unique(pixels.view(np.dtype((np.void, pixels.dtype.itemsize * pixels.shape[1])))))
        
        # Use k-means to find dominant colors
        try:
            n_clusters = min(8, unique_colors)
            if n_clusters > 1:
                kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                kmeans.fit(pixels)
                dominant_colors = len(kmeans.cluster_centers_)
            else:
                dominant_colors = 1
        except:
            dominant_colors = 5  # Conservative estimate
        
        # Color count score (fewer colors = better for autism)
        color_count_score = max(0.0, 1.0 - (dominant_colors - 3) * 0.15)  # Optimal: 3-4 colors
        
        # Saturation analysis (avoid over-saturated colors)
        saturation_values = hsv_image[:,:,1]
        avg_saturation = np.mean(saturation_values) / 255.0
        saturation_variance = np.var(saturation_values) / (255**2)
        
        # Moderate saturation is best (not too dull, not too bright)
        optimal_saturation = 0.6  # Target saturation
        saturation_score = 1.0 - abs(avg_saturation - optimal_saturation) * 2
        saturation_score = max(0.0, min(1.0, saturation_score))
        
        # Brightness analysis (consistent, comfortable brightness)
        brightness_values = hsv_image[:,:,2]
        avg_brightness = np.mean(brightness_values) / 255.0
        brightness_variance = np.var(brightness_values) / (255**2)
        
        # Good brightness: not too dark, not too bright, consistent
        brightness_score = 1.0 - abs(avg_brightness - 0.7) * 1.5  # Target: 70% brightness
        brightness_consistency = 1.0 - min(1.0, brightness_variance * 8)
        brightness_score = (brightness_score + brightness_consistency) / 2
        brightness_score = max(0.0, min(1.0, brightness_score))
        
        # Overall color appropriateness
        color_appropriateness = (color_count_score * 0.4 + saturation_score * 0.3 + brightness_score * 0.3)
        
        # FIX 1: ENSURE 0-1 RANGE FOR COLOR SCORE
        color_appropriateness = max(0.0, min(1.0, color_appropriateness))
        
        # Grade color appropriateness
        if color_appropriateness >= 0.8:
            grade = "Excellent - Autism-friendly color palette"
        elif color_appropriateness >= 0.6:
            grade = "Good - Suitable color palette"
        elif color_appropriateness >= 0.4:
            grade = "Moderate - Some color concerns"
        else:
            grade = "Poor - Overwhelming or inappropriate colors"
        
        return {
            "score": float(color_appropriateness),
            "dominant_colors": dominant_colors,
            "color_count_score": float(color_count_score),
            "avg_saturation": float(avg_saturation),
            "saturation_score": float(saturation_score),
            "avg_brightness": float(avg_brightness),
            "brightness_score": float(brightness_score),
            "grade": grade,
            "details": "Optimal: 3-4 colors, moderate saturation, consistent brightness"
        }
    
    def analyze_character_clarity(self, cv_image):
        """Analyze how clearly defined characters are - important for autism comprehension"""
        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        
        # FIX 2: LOWERED EDGE DETECTION THRESHOLDS
        edges = cv2.Canny(gray, 50, 150)  # CHANGED: Lower thresholds (was 100, 200)
        strong_edge_density = np.sum(edges > 0) / edges.size
        
        # Contrast analysis (clear distinction between elements)
        # Local contrast using Sobel operator
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        avg_contrast = np.mean(gradient_magnitude) / 255.0
        
        # Shape definition (how well-defined shapes are)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        shape_quality = 0
        if len(contours) > 0:
            # Analyze contour quality
            large_contours = [c for c in contours if cv2.contourArea(c) > 500]
            
            if large_contours:
                # Measure contour smoothness and closure
                smoothness_scores = []
                for contour in large_contours[:5]:  # Top 5 largest
                    epsilon = 0.02 * cv2.arcLength(contour, True)
                    approx = cv2.approxPolyDP(contour, epsilon, True)
                    smoothness = len(approx) / len(contour)  # Fewer points = smoother
                    smoothness_scores.append(smoothness)
                
                shape_quality = np.mean(smoothness_scores) if smoothness_scores else 0
        
        # FIX 3: INCREASED SCALING FOR EDGE CLARITY
        edge_clarity = min(1.0, strong_edge_density * 15)  # CHANGED: Increased scaling (was * 8)
        contrast_clarity = min(1.0, avg_contrast * 5)      # CHANGED: Increased scaling (was * 3)
        shape_clarity = min(1.0, shape_quality * 8)        # CHANGED: Increased scaling (was * 5)
        
        character_clarity = (edge_clarity * 0.4 + contrast_clarity * 0.4 + shape_clarity * 0.2)
        
        # FIX 4: ENSURE 0-1 RANGE FOR CHARACTER CLARITY
        character_clarity = max(0.0, min(1.0, character_clarity))
        
        # Grade clarity
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
        """Analyze sensory aspects - avoid overwhelming visual stimulation"""
        # Pattern repetition (can be overwhelming)
        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        
        # High frequency content (can be overstimulating)
        fft = np.fft.fft2(gray)
        fft_shift = np.fft.fftshift(fft)
        magnitude_spectrum = np.abs(fft_shift)
        
        h, w = magnitude_spectrum.shape
        center_h, center_w = h // 2, w // 2
        y, x = np.ogrid[:h, :w]
        center_distance = np.sqrt((x - center_w)**2 + (y - center_h)**2)
        
        # High frequency energy (overstimulating patterns)
        high_freq_mask = center_distance >= min(h, w) * 0.4
        total_energy = np.sum(magnitude_spectrum)
        high_freq_energy = np.sum(magnitude_spectrum[high_freq_mask])
        high_freq_ratio = high_freq_energy / total_energy if total_energy > 0 else 0
        
        # Flash/strobe detection (rapid intensity changes)
        # Simulate by checking local intensity variance
        kernel = np.ones((5,5), np.float32) / 25
        local_mean = cv2.filter2D(gray.astype(np.float32), -1, kernel)
        intensity_variance = np.var(gray.astype(np.float32) - local_mean) / (255**2)
        
        # Motion blur indicator (can help reduce overstimulation)
        motion_blur_kernel = np.zeros((9, 9))
        motion_blur_kernel[4, :] = 1/9  # Horizontal motion blur
        blurred = cv2.filter2D(gray, -1, motion_blur_kernel)
        blur_difference = np.mean(np.abs(gray.astype(float) - blurred.astype(float))) / 255.0
        
        # Calculate sensory friendliness
        pattern_score = 1.0 - min(1.0, high_freq_ratio * 3)  # Lower high-freq = better
        intensity_score = 1.0 - min(1.0, intensity_variance * 10)  # Lower variance = better
        motion_score = max(0.0, 1.0 - blur_difference * 2)  # Some blur can be good
        
        sensory_friendliness = (pattern_score * 0.5 + intensity_score * 0.3 + motion_score * 0.2)
        
        # FIX 5: ENSURE 0-1 RANGE FOR SENSORY SCORE
        sensory_friendliness = max(0.0, min(1.0, sensory_friendliness))
        
        # Grade sensory friendliness
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
        """Analyze focus clarity - autism storyboards should have one clear focal point"""
        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        
        # Center-weighted analysis (main subject usually in center)
        h, w = gray.shape
        y, x = np.ogrid[:h, :w]
        center_y, center_x = h // 2, w // 2
        
        # Create center weight mask (Gaussian)
        center_distance = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        max_distance = np.sqrt(center_x**2 + center_y**2)
        center_weights = np.exp(-center_distance**2 / (2 * (max_distance/3)**2))
        
        # Edge density weighted by center
        edges = cv2.Canny(gray, 100, 200)
        center_edge_density = np.sum(edges * center_weights) / np.sum(center_weights)
        peripheral_edge_density = np.sum(edges * (1 - center_weights)) / np.sum(1 - center_weights)
        
        # Focus ratio (center should have more detail than periphery)
        focus_ratio = center_edge_density / (peripheral_edge_density + 1e-6)
        focus_score = min(1.0, focus_ratio / 2.0)  # Normalize
        
        # Contrast focus (center should have higher contrast)
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        center_contrast = np.sum(gradient_magnitude * center_weights) / np.sum(center_weights)
        peripheral_contrast = np.sum(gradient_magnitude * (1 - center_weights)) / np.sum(1 - center_weights)
        contrast_ratio = center_contrast / (peripheral_contrast + 1e-6)
        contrast_focus = min(1.0, contrast_ratio / 2.0)
        
        # Overall focus clarity
        focus_clarity = (focus_score * 0.6 + contrast_focus * 0.4)
        
        # FIX 6: ENSURE 0-1 RANGE FOR FOCUS SCORE
        focus_clarity = max(0.0, min(1.0, focus_clarity))
        
        # Grade focus clarity
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
    
    def calculate_autism_suitability(self, results):
        """Calculate overall autism suitability score"""
        # Weights for autism-specific priorities
        weights = {
            "person_count": 0.25,           # CRITICAL: max 2 people
            "background_simplicity": 0.20,  # Clean backgrounds
            "character_clarity": 0.20,      # Clear character definition
            "color_appropriateness": 0.15,  # Sensory-appropriate colors
            "sensory_friendliness": 0.15,   # Avoid overstimulation
            "focus_clarity": 0.05          # Single focus point
        }
        
        total_score = 0
        total_weight = 0
        
        for metric, weight in weights.items():
            if metric in results and "score" in results[metric]:
                total_score += results[metric]["score"] * weight
                total_weight += weight
        
        autism_suitability = total_score / total_weight if total_weight > 0 else 0.5
        
        # FIX 7: ENSURE FINAL AUTISM SUITABILITY IS 0-1 RANGE
        return float(min(1.0, max(0.0, autism_suitability)))
    
    def get_autism_grade(self, suitability_score):
        """Convert autism suitability score to letter grade"""
        # FIX 8: LOWERED THRESHOLDS TO BE MORE REALISTIC
        if suitability_score >= 0.85:  # CHANGED: Was 0.9
            return "A+ (Excellent for autism storyboards)"
        elif suitability_score >= 0.75:  # CHANGED: Was 0.8
            return "A (Very suitable for autism storyboards)"
        elif suitability_score >= 0.65:  # CHANGED: Was 0.7
            return "B+ (Good for autism storyboards)"
        elif suitability_score >= 0.55:  # CHANGED: Was 0.6
            return "B (Acceptable for autism storyboards)"
        elif suitability_score >= 0.45:  # CHANGED: Was 0.5
            return "C+ (Needs some improvements)"
        elif suitability_score >= 0.35:  # CHANGED: Was 0.4
            return "C (Several issues to address)"
        elif suitability_score >= 0.25:  # CHANGED: Was 0.3
            return "D+ (Many issues for autism use)"
        elif suitability_score >= 0.15:  # CHANGED: Was 0.2
            return "D (Poor for autism storyboards)"
        else:
            return "F (Unsuitable for autism storyboards)"
    
    def generate_recommendations(self, results):
        """Generate specific recommendations for improving autism suitability"""
        recommendations = []
        
        # Person count recommendations
        person_count = results["person_count"]["count"]
        if person_count > 2:
            recommendations.append(f"🚨 CRITICAL: Reduce to max 2 people (currently {person_count})")
        elif person_count <= 1:
            recommendations.append("✅ Person count is excellent for autism storyboards")
        
        # Background recommendations
        bg_score = results["background_simplicity"]["score"]
        if bg_score < 0.6:
            recommendations.append("🎨 Simplify background - reduce clutter and details")
        
        # Color recommendations
        color_score = results["color_appropriateness"]["score"]
        color_count = results["color_appropriateness"]["dominant_colors"]
        if color_count > 6:
            recommendations.append(f"🌈 Reduce color count to 3-4 colors (currently {color_count})")
        if results["color_appropriateness"]["avg_saturation"] > 0.8:
            recommendations.append("🎨 Reduce color saturation - too bright/overwhelming")
        
        # Character clarity recommendations
        char_score = results["character_clarity"]["score"]
        if char_score < 0.6:
            recommendations.append("🖼️ Improve character clarity - strengthen edges and contrast")
        
        # Sensory recommendations
        sensory_score = results["sensory_friendliness"]["score"]
        if sensory_score < 0.6:
            recommendations.append("🧩 Reduce visual complexity - may be overstimulating")
        
        # Focus recommendations
        focus_score = results["focus_clarity"]["score"]
        if focus_score < 0.5:
            recommendations.append("🎯 Create clearer focal point - reduce background distractions")
        
        # Overall recommendations
        overall_score = results["autism_suitability"]
        if overall_score >= 0.75:  # CHANGED: Was 0.8
            recommendations.append("🎉 Excellent for autism education!")
        elif overall_score >= 0.55:  # CHANGED: Was 0.6
            recommendations.append("👍 Good for autism use with minor improvements")
        else:
            recommendations.append("⚠️ Needs significant improvements for autism suitability")
        
        return recommendations
    
    def generate_autism_report(self, results, save_path=None):
        """Generate detailed autism suitability report"""
        if "error" in results:
            return f"Error in autism analysis: {results['error']}"
        
        report = []
        report.append("AUTISM STORYBOARD SUITABILITY REPORT")
        report.append("=" * 45)
        report.append("")
        
        # Overall assessment
        report.append("OVERALL ASSESSMENT:")
        report.append(f"Autism Suitability: {results['autism_suitability']:.3f}")
        report.append(f"Grade: {results['autism_grade']}")
        report.append("")
        
        # Critical autism requirements
        report.append("AUTISM-SPECIFIC REQUIREMENTS:")
        report.append("-" * 30)
        
        # Person count (most critical)
        person_data = results["person_count"]
        status = "✅" if person_data["is_compliant"] else "❌"
        report.append(f"{status} Person Count: {person_data['count']} ({person_data['compliance_level']})")
        report.append(f"   → {person_data['details']}")
        report.append("")
        
        # Other metrics
        metrics = [
            ("Background Simplicity", "background_simplicity"),
            ("Character Clarity", "character_clarity"),
            ("Color Appropriateness", "color_appropriateness"),
            ("Sensory Friendliness", "sensory_friendliness"),
            ("Focus Clarity", "focus_clarity")
        ]
        
        for name, key in metrics:
            if key in results:
                score = results[key]["score"]
                grade = results[key]["grade"]
                report.append(f"{name}: {score:.3f} - {grade}")
                report.append("")
        
        # Recommendations
        report.append("RECOMMENDATIONS:")
        report.append("-" * 15)
        for rec in results["recommendations"]:
            report.append(f"• {rec}")
        
        report_text = "\n".join(report)
        
        if save_path:
            with open(save_path, 'w', encoding='utf-8') as f:
                f.write(report_text)
            print(f"📄 Autism suitability report saved: {save_path}")
        
        return report_text


def test_autism_analyzer():
    """Test the autism-friendly analyzer"""
    print("🧪 Testing Autism-Friendly Image Analyzer")
    print("=" * 45)
    
    analyzer = AutismFriendlyImageAnalyzer()
    
    if not analyzer.available:
        print("❌ Analyzer not available for testing")
        return
    
    # Create test images for autism scenarios
    
    # GOOD: Simple single character
    good_img = np.ones((512, 512, 3), dtype=np.uint8) * 240  # Clean light background
    cv2.circle(good_img, (256, 200), 80, (100, 150, 200), -1)  # Head
    cv2.rectangle(good_img, (200, 280), (312, 450), (120, 160, 180), -1)  # Body
    
    # BAD: Multiple people, busy background
    bad_img = np.random.randint(100, 200, (512, 512, 3), dtype=np.uint8)  # Noisy background
    # Add multiple faces
    cv2.circle(bad_img, (150, 150), 40, (255, 200, 180), -1)  # Person 1
    cv2.circle(bad_img, (350, 150), 40, (255, 190, 170), -1)  # Person 2  
    cv2.circle(bad_img, (250, 350), 40, (255, 210, 185), -1)  # Person 3
    
    test_images = [
        ("Good: Single character, clean", good_img),
        ("Bad: Multiple people, busy", bad_img)
    ]
    
    for name, img in test_images:
        print(f"\n🔍 Analyzing: {name}")
        pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        results = analyzer.analyze_autism_suitability(pil_img)
        
        print(f"Autism Suitability: {results['autism_suitability']:.3f}")
        print(f"Grade: {results['autism_grade']}")
        print(f"People Count: {results['person_count']['count']} ({'✅' if results['person_count']['is_compliant'] else '❌'})")
        print("Top Recommendations:")
        for i, rec in enumerate(results['recommendations'][:3]):
            print(f"  {i+1}. {rec}")
    
    print("\n✅ Autism analyzer test complete!")


if __name__ == "__main__":
    test_autism_analyzer()