"""
Autism-Integrated Cartoon Pipeline with IP-Adapter and Baked VAE Support
Complete integration of autism-friendly image analysis into the generation pipeline
Optimized for educational storyboards with character consistency
"""

import os
import torch
import time
import numpy as np
from diffusers import StableDiffusionXLPipeline
from PIL import Image
import cv2
import warnings
warnings.filterwarnings("ignore")

# Import modular components
from consistency_manager import ConsistencyManager
from quality_evaluator import QualityEvaluator  
from ip_adapter_manager import IPAdapterManager
from transformers import CLIPProcessor, CLIPModel
from sklearn.cluster import KMeans


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
        
        # Edge sharpness (clear boundaries)
        edges = cv2.Canny(gray, 100, 200)  # Higher thresholds for strong edges
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
        
        # Combine clarity metrics
        edge_clarity = min(1.0, strong_edge_density * 8)  # Scale edge density
        contrast_clarity = min(1.0, avg_contrast * 3)     # Scale contrast
        shape_clarity = min(1.0, shape_quality * 5)       # Scale shape quality
        
        character_clarity = (edge_clarity * 0.4 + contrast_clarity * 0.4 + shape_clarity * 0.2)
        
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
        return float(min(1.0, max(0.0, autism_suitability)))
    
    def get_autism_grade(self, suitability_score):
        """Convert autism suitability score to letter grade"""
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
        if overall_score >= 0.8:
            recommendations.append("🎉 Excellent for autism education!")
        elif overall_score >= 0.6:
            recommendations.append("👍 Good for autism use with minor improvements")
        else:
            recommendations.append("⚠️ Needs significant improvements for autism suitability")
        
        return recommendations


class AutismIntegratedCartoonPipeline:
    """Main pipeline with integrated autism-friendly evaluation"""
    
    def __init__(self, model_path, ip_adapter_path=None, config=None, enable_autism_scoring=True):
        print("🎨 Loading Autism-Integrated Cartoon Pipeline...")
        
        self.model_path = model_path
        self.ip_adapter_path = ip_adapter_path
        self.config = config or self._default_config()
        self.enable_autism_scoring = enable_autism_scoring
        
        # Initialize character reference tracking
        self.character_reference_image = None
        self.ip_adapter_loaded = False
        self.baked_vae_compatible = False
        
        # Check model exists
        if not os.path.exists(model_path):
            print(f"❌ Model not found: {model_path}")
            self.available = False
            return
        
        try:
            # Load main pipeline
            self._load_diffusion_pipeline()
            
            # CRITICAL: Test baked VAE compatibility first
            self._test_baked_vae_compatibility()
            
            # CRITICAL: ALWAYS load IP-Adapter - no conditional loading
            self._load_ip_adapter_mandatory()
            
            # CRITICAL: Fix tensor dtype issues for baked VAE
            self._fix_tensor_dtypes_baked_vae()
            
            # Test IP-Adapter with baked VAE
            self._test_ip_adapter_baked_vae_compatibility()
            
            # Initialize evaluation components
            self.consistency_manager = ConsistencyManager()
            self.quality_evaluator = QualityEvaluator()
            
            # Initialize Autism Analyzer
            if self.enable_autism_scoring:
                self.autism_analyzer = AutismFriendlyImageAnalyzer()
                print("🧩 Autism analyzer integrated into pipeline")
            else:
                self.autism_analyzer = None
                print("⚠️ Autism analyzer disabled")
            
            # Initialize IP-Adapter manager
            self.ip_adapter_manager = IPAdapterManager(
                base_pipeline=self.pipe,
                ip_adapter_path=ip_adapter_path
            )
            
            self.available = True
            print("✅ Autism-Integrated Cartoon Pipeline ready!")
            
        except Exception as e:
            print(f"❌ Pipeline loading failed: {e}")
            import traceback
            traceback.print_exc()
            self.available = False
    
    def _default_config(self):
        """Default generation configuration optimized for autism + baked VAE"""
        return {
            "generation": {
                "height": 1024,
                "width": 1024,
                "num_inference_steps": 30,
                "guidance_scale": 5.0,
                "num_images_per_prompt": 3
            },
            "selection": {
                "use_consistency": True,
                "use_ip_adapter": True,
                "use_autism_scoring": True,  # NEW: Enable autism scoring
                "quality_weight": 0.25,      # Reduced to make room for autism
                "consistency_weight": 0.25,
                "ip_adapter_weight": 0.25,
                "autism_weight": 0.25        # NEW: Autism score weight
            },
            "autism": {  # NEW: Autism-specific configuration
                "max_people": 2,
                "target_simplicity": 0.7,
                "target_clarity": 0.7,
                "enforce_requirements": True,
                "reject_threshold": 0.4  # Reject images below this autism score
            },
            "ip_adapter": {
                "character_weight": 0.3,  # LOWERED for baked VAE compatibility
                "update_reference_from_best": True,
                "fallback_to_clip": False
            },
            "baked_vae": {
                "use_conservative_optimizations": True,
                "disable_autocast_on_failure": True,
                "force_float32_fallback": True
            }
        }
    
    def _load_diffusion_pipeline(self):
        """Load pipeline optimized for baked VAE models"""
        print("🔧 Loading SDXL pipeline with baked VAE support...")
        self.pipe = StableDiffusionXLPipeline.from_single_file(
            self.model_path, 
            torch_dtype=torch.float16, 
            use_safetensors=True,
            variant="fp16"
        )
        
        self.pipe = self.pipe.to("cuda")
        
        # Conservative VAE optimizations for baked models
        try:
            # Try VAE tiling first (safer for baked VAE)
            if hasattr(self.pipe, 'enable_vae_tiling'):
                self.pipe.enable_vae_tiling()
                print("✅ VAE tiling enabled for baked VAE")
            else:
                # Fallback to slicing with caution
                self.pipe.enable_vae_slicing()
                print("✅ VAE slicing enabled (fallback)")
        except Exception as e:
            print(f"⚠️ VAE optimization skipped: {e}")
        
        print("✅ Baked VAE pipeline loaded")
    
    def _test_baked_vae_compatibility(self):
        """Test if baked VAE works correctly"""
        print("🧪 Testing baked VAE compatibility...")
        
        try:
            # Check VAE properties
            if hasattr(self.pipe, 'vae') and self.pipe.vae is not None:
                print(f"🔍 VAE type: {type(self.pipe.vae).__name__}")
                print(f"🔍 VAE device: {self.pipe.vae.device}")
                print(f"🔍 VAE dtype: {self.pipe.vae.dtype}")
                
                if hasattr(self.pipe.vae, 'config'):
                    scaling_factor = getattr(self.pipe.vae.config, 'scaling_factor', 'unknown')
                    print(f"🔍 VAE scaling factor: {scaling_factor}")
                    
                    if scaling_factor != 0.13025:
                        print(f"⚠️ Non-standard VAE scaling: {scaling_factor}")
            
            # Simple test generation
            test_prompt = "a simple red circle on white background"
            print("🧪 Testing basic generation...")
            
            result = self.pipe(
                prompt=test_prompt,
                num_images_per_prompt=1,
                num_inference_steps=10,
                height=512, width=512,
                guidance_scale=3.0
            )
            
            # Check if result is black
            test_image = result.images[0]
            img_array = np.array(test_image)
            
            # More sophisticated black image detection
            mean_value = np.mean(img_array)
            std_value = np.std(img_array)
            
            if mean_value < 15 and std_value < 10:  # Very dark with low variation
                print("❌ Baked VAE producing black/dark images")
                self.baked_vae_compatible = False
                
                # Try with different settings
                print("🔧 Trying VAE fallback settings...")
                return self._try_vae_fallback_test()
            else:
                print(f"✅ Baked VAE working (mean: {mean_value:.1f}, std: {std_value:.1f})")
                self.baked_vae_compatible = True
                return True
                
        except Exception as e:
            print(f"❌ Baked VAE test failed: {e}")
            self.baked_vae_compatible = False
            return False
    
    def _try_vae_fallback_test(self):
        """Try alternative settings for problematic baked VAE"""
        print("🔧 Trying VAE fallback configurations...")
        
        try:
            # Disable VAE optimizations
            if hasattr(self.pipe, 'disable_vae_slicing'):
                self.pipe.disable_vae_slicing()
            if hasattr(self.pipe, 'disable_vae_tiling'):
                self.pipe.disable_vae_tiling()
            
            # Test with float32 VAE
            original_vae_dtype = self.pipe.vae.dtype
            self.pipe.vae = self.pipe.vae.to(dtype=torch.float32)
            
            result = self.pipe(
                prompt="a simple red circle on white background",
                num_images_per_prompt=1,
                num_inference_steps=10,
                height=512, width=512,
                guidance_scale=3.0
            )
            
            test_image = result.images[0]
            img_array = np.array(test_image)
            mean_value = np.mean(img_array)
            
            # Restore VAE dtype
            self.pipe.vae = self.pipe.vae.to(dtype=original_vae_dtype)
            
            if mean_value > 15:
                print("✅ VAE fallback successful - float32 fixes the issue")
                self.config["baked_vae"]["force_float32_fallback"] = True
                self.baked_vae_compatible = True
                return True
            else:
                print("❌ VAE fallback still produces black images")
                return False
                
        except Exception as e:
            print(f"❌ VAE fallback test failed: {e}")
            return False
    
    def _load_ip_adapter_mandatory(self):
        """Load IP-Adapter with baked VAE considerations"""
        print("🎭 Loading IP-Adapter for baked VAE model...")
        
        try:
            print("📥 Loading IP-Adapter from HuggingFace...")
            self.pipe.load_ip_adapter(
                "h94/IP-Adapter", 
                subfolder="sdxl_models", 
                weight_name="ip-adapter_sdxl.bin",
                torch_dtype=torch.float16
            )
            
            # Set conservative initial scale for baked VAE
            initial_scale = self.config["ip_adapter"]["character_weight"]
            self.pipe.set_ip_adapter_scale(initial_scale)
            self.ip_adapter_loaded = True
            print(f"✅ IP-Adapter loaded with scale {initial_scale}")
            
        except Exception as e:
            print(f"❌ HuggingFace IP-Adapter failed: {e}")
            
            # Try local file if provided
            if self.ip_adapter_path and os.path.exists(self.ip_adapter_path):
                try:
                    print(f"📥 Trying local IP-Adapter: {self.ip_adapter_path}")
                    self.pipe.load_ip_adapter(self.ip_adapter_path, torch_dtype=torch.float16)
                    self.pipe.set_ip_adapter_scale(self.config["ip_adapter"]["character_weight"])
                    self.ip_adapter_loaded = True
                    print("✅ IP-Adapter loaded from local file")
                except Exception as e2:
                    print(f"❌ Local IP-Adapter failed: {e2}")
                    raise Exception(f"IP-Adapter loading failed: {e2}")
            else:
                raise Exception(f"IP-Adapter loading failed and no local backup: {e}")
    
    def _fix_tensor_dtypes_baked_vae(self):
        """Fix tensor dtypes specifically for baked VAE + IP-Adapter"""
        print("🔧 Fixing tensor dtypes for baked VAE + IP-Adapter...")
        
        try:
            # Handle baked VAE specifically
            if hasattr(self.pipe, 'vae') and self.pipe.vae is not None:
                print(f"🔍 Baked VAE device: {self.pipe.vae.device}")
                print(f"🔍 Baked VAE dtype: {self.pipe.vae.dtype}")
                
                # Ensure VAE is on CUDA with correct dtype
                self.pipe.vae = self.pipe.vae.to(device="cuda", dtype=torch.float16)
                print("✅ Baked VAE synced to CUDA float16")
            
            # IP-Adapter components
            if hasattr(self.pipe, 'image_encoder') and self.pipe.image_encoder is not None:
                self.pipe.image_encoder = self.pipe.image_encoder.to(device="cuda", dtype=torch.float16)
                print("✅ Image encoder synced with baked VAE")
            
            # UNet
            if hasattr(self.pipe, 'unet'):
                self.pipe.unet = self.pipe.unet.to(device="cuda", dtype=torch.float16)
                print("✅ UNet synced")
            
            # Text encoders
            if hasattr(self.pipe, 'text_encoder'):
                self.pipe.text_encoder = self.pipe.text_encoder.to(device="cuda", dtype=torch.float16)
            if hasattr(self.pipe, 'text_encoder_2'):
                self.pipe.text_encoder_2 = self.pipe.text_encoder_2.to(device="cuda", dtype=torch.float16)
            
            torch.cuda.synchronize()
            print("✅ Baked VAE + IP-Adapter tensor sync complete")
            
        except Exception as e:
            print(f"⚠️ Tensor sync warning: {e}")
    
    def _test_ip_adapter_baked_vae_compatibility(self):
        """Test IP-Adapter with baked VAE"""
        print("🧪 Testing IP-Adapter with baked VAE...")
        
        if not self.ip_adapter_loaded:
            print("⚠️ IP-Adapter not loaded, skipping compatibility test")
            return
        
        try:
            # Create simple test image
            dummy_image = Image.new('RGB', (512, 512), color=(128, 128, 128))
            
            # Test with minimal IP-Adapter influence
            self.pipe.set_ip_adapter_scale(0.1)
            
            result = self.pipe(
                prompt="a simple blue square",
                ip_adapter_image=dummy_image,
                num_images_per_prompt=1,
                num_inference_steps=10,
                height=512, width=512,
                guidance_scale=3.0
            )
            
            test_image = result.images[0]
            img_array = np.array(test_image)
            mean_value = np.mean(img_array)
            
            if mean_value < 15:
                print("❌ IP-Adapter + baked VAE producing black images")
                # Lower the default IP-Adapter scale even more
                self.config["ip_adapter"]["character_weight"] = 0.15
                print("🔧 Lowered IP-Adapter scale to 0.15")
            else:
                print(f"✅ IP-Adapter + baked VAE working (mean: {mean_value:.1f})")
            
        except Exception as e:
            print(f"❌ IP-Adapter + baked VAE test failed: {e}")
            # Emergency fallback - very low scale
            self.config["ip_adapter"]["character_weight"] = 0.05
            print("🚨 Emergency fallback: IP-Adapter scale set to 0.05")
    
    def _create_proper_dummy_image(self):
        """Create a simple dummy image for IP-Adapter"""
        # Simple gray image - no complex preprocessing
        return Image.new('RGB', (1024, 1024), color=(128, 128, 128))
    
    def set_character_reference_image(self, reference_image_path_or_image):
        """Set the character reference image for IP-Adapter consistency"""
        if isinstance(reference_image_path_or_image, str):
            reference_image = Image.open(reference_image_path_or_image).convert("RGB")
        else:
            reference_image = reference_image_path_or_image
        
        # Resize to consistent size
        if reference_image.size != (1024, 1024):
            reference_image = reference_image.resize((1024, 1024), Image.Resampling.LANCZOS)
        
        self.character_reference_image = reference_image
        print("🎭 Character reference set for IP-Adapter consistency")
        
        # Analyze autism suitability of reference if available
        if self.autism_analyzer and self.autism_analyzer.available:
            autism_result = self.autism_analyzer.analyze_autism_suitability(reference_image)
            print(f"🧩 Reference image autism score: {autism_result['autism_suitability']:.3f}")
            print(f"   Grade: {autism_result['autism_grade']}")
            if autism_result['autism_suitability'] < 0.6:
                print("⚠️ Warning: Reference image may not be ideal for autism storyboards")
        
        return reference_image
    
    def generate_single_image(self, prompt, negative_prompt="", use_ip_adapter=None, **kwargs):
        """Generate single best image with character consistency and autism scoring"""
        if not self.available:
            return None
        
        # Use IP-Adapter if loaded and reference exists
        if use_ip_adapter is None:
            use_ip_adapter = (
                self.ip_adapter_loaded and 
                self.character_reference_image is not None
            )
        
        result = self.generate_with_selection(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_images=self.config["generation"]["num_images_per_prompt"],
            use_ip_adapter=use_ip_adapter,
            **kwargs
        )
        
        if result:
            return {
                "image": result["best_image"],
                "score": result["best_score"],
                "autism_score": result.get("autism_score", 0.5),
                "autism_grade": result.get("autism_grade", "Not evaluated"),
                "generation_time": result["generation_time"],
                "used_ip_adapter": result.get("used_ip_adapter", False)
            }
        return None
    
    def generate_with_selection(self, prompt, negative_prompt="", num_images=3, use_ip_adapter=None, **kwargs):
        """Generate multiple images with autism-aware selection"""
        if not self.available:
            return None
        
        print(f"🎨 Generating {num_images} images for intelligent selection...")
        
        # Force IP-Adapter if loaded
        if use_ip_adapter is None:
            use_ip_adapter = self.ip_adapter_loaded
        
        # Prepare generation settings
        settings = self._prepare_generation_settings(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_images=num_images,
            **kwargs
        )
        
        start_time = time.time()
        
        # ===== BAKED VAE + IP-ADAPTER IMPLEMENTATION =====
        if use_ip_adapter and self.ip_adapter_loaded:
            if self.character_reference_image is not None:
                print("🎭 Using IP-Adapter with character reference")
                
                # Set IP-Adapter scale (lower for baked VAE)
                self.pipe.set_ip_adapter_scale(self.config["ip_adapter"]["character_weight"])
                
                # Try generation with baked VAE error handling
                result = self._generate_with_baked_vae_handling(
                    ip_adapter_image=self.character_reference_image,
                    settings=settings
                )
                used_ip_adapter = True
                
            else:
                # First image - use minimal IP-Adapter influence
                print("🎭 First image - minimal IP-Adapter influence")
                
                dummy_reference = self._create_proper_dummy_image()
                self.pipe.set_ip_adapter_scale(0.05)  # Very minimal for first image
                
                result = self._generate_with_baked_vae_handling(
                    ip_adapter_image=dummy_reference,
                    settings=settings
                )
                used_ip_adapter = True
        else:
            # Fallback without IP-Adapter
            print("🎨 Generating without IP-Adapter")
            result = self._generate_with_baked_vae_handling(settings=settings)
            used_ip_adapter = False
        
        if result is None or not result.images:
            raise Exception("Image generation failed - check baked VAE compatibility!")
        
        # Check for black images
        for i, img in enumerate(result.images):
            img_array = np.array(img)
            mean_val = np.mean(img_array)
            if mean_val < 15:
                print(f"⚠️ Image {i+1} appears very dark (mean: {mean_val:.1f})")
        
        gen_time = time.time() - start_time
        
        # Evaluate all generated images
        evaluation_results = self.quality_evaluator.evaluate_batch(result.images, prompt)
        
        # Select best image with autism awareness
        best_result = self._select_best_image_with_autism_scoring(
            images=result.images,
            evaluations=evaluation_results,
            prompt=prompt,
            use_ip_adapter=used_ip_adapter
        )
        
        # Store selected image for consistency tracking
        if self.consistency_manager.available and best_result:
            self.consistency_manager.store_selected_image(
                image=best_result["image"],
                prompt=prompt,
                tifa_score=best_result["tifa_score"]
            )
        
        # Update character reference from first generation
        if (self.character_reference_image is None and best_result and 
            self.config["ip_adapter"]["update_reference_from_best"]):
            print("🎭 Setting first generated image as character reference")
            self.set_character_reference_image(best_result["image"])
            # Increase IP-Adapter scale for subsequent images
            self.pipe.set_ip_adapter_scale(self.config["ip_adapter"]["character_weight"])
        
        return {
            "best_image": best_result["image"],
            "best_score": best_result["combined_score"],
            "best_index": best_result["index"],
            "autism_score": best_result.get("autism_score", 0.5),
            "autism_grade": best_result.get("autism_grade", "Not evaluated"),
            "autism_analysis": best_result.get("autism_analysis", {}),
            "all_images": result.images,
            "all_evaluations": evaluation_results,
            "generation_time": gen_time,
            "used_ip_adapter": used_ip_adapter,
            "consistency_used": len(self.consistency_manager.selected_images_history) > 0
        }
    
    def _generate_with_baked_vae_handling(self, ip_adapter_image=None, settings=None):
        """Generate with proper baked VAE error handling"""
        try:
            # First attempt - normal generation
            if ip_adapter_image is not None:
                result = self.pipe(ip_adapter_image=ip_adapter_image, **settings)
            else:
                result = self.pipe(**settings)
            
            # Quick black image check
            if result and result.images:
                first_img = np.array(result.images[0])
                if np.mean(first_img) > 15:  # Not black
                    return result
                else:
                    print("⚠️ Black image detected, trying fallback...")
            
        except Exception as e:
            print(f"⚠️ Generation failed: {e}, trying fallback...")
        
        # Fallback generation for baked VAE issues
        return self._baked_vae_fallback_generation(ip_adapter_image, settings)
    
    def _baked_vae_fallback_generation(self, ip_adapter_image, settings):
        """Fallback generation for baked VAE issues"""
        print("🔧 Using baked VAE fallback generation...")
        
        try:
            # Disable VAE optimizations
            if hasattr(self.pipe, 'disable_vae_slicing'):
                self.pipe.disable_vae_slicing()
            if hasattr(self.pipe, 'disable_vae_tiling'):
                self.pipe.disable_vae_tiling()
            
            # Try with float32 VAE if enabled in config
            if self.config["baked_vae"]["force_float32_fallback"]:
                original_vae_dtype = self.pipe.vae.dtype
                self.pipe.vae = self.pipe.vae.to(dtype=torch.float32)
                
                try:
                    if ip_adapter_image is not None:
                        # Lower IP-Adapter scale for fallback
                        original_scale = self.pipe.get_ip_adapter_scale() if hasattr(self.pipe, 'get_ip_adapter_scale') else None
                        self.pipe.set_ip_adapter_scale(0.1)
                        result = self.pipe(ip_adapter_image=ip_adapter_image, **settings)
                        if original_scale is not None:
                            self.pipe.set_ip_adapter_scale(original_scale)
                    else:
                        result = self.pipe(**settings)
                    
                    # Restore VAE dtype
                    self.pipe.vae = self.pipe.vae.to(dtype=original_vae_dtype)
                    
                    return result
                    
                except Exception as e:
                    # Restore VAE dtype even on failure
                    self.pipe.vae = self.pipe.vae.to(dtype=original_vae_dtype)
                    print(f"❌ Float32 fallback failed: {e}")
            
            # Final attempt without IP-Adapter
            if ip_adapter_image is not None:
                print("🚨 Final fallback: disabling IP-Adapter")
                return self.pipe(**settings)
            
            return None
            
        except Exception as e:
            print(f"❌ All fallback attempts failed: {e}")
            return None
        finally:
            # Re-enable optimizations
            try:
                if hasattr(self.pipe, 'enable_vae_tiling'):
                    self.pipe.enable_vae_tiling()
                elif hasattr(self.pipe, 'enable_vae_slicing'):
                    self.pipe.enable_vae_slicing()
            except:
                pass
    
    def _prepare_generation_settings(self, prompt, negative_prompt, num_images, **kwargs):
        """Prepare settings for image generation with autism considerations"""
        # Add autism-specific negative prompts
        autism_negative_terms = [
            "multiple people", "crowd", "group", "busy background", 
            "cluttered", "complex patterns", "overwhelming", "chaotic",
            "too many colors", "sensory overload", "distracting elements"
        ]
        
        # Combine with existing negative prompt
        full_negative = negative_prompt or "blurry, low quality, distorted, deformed"
        if self.enable_autism_scoring:
            full_negative = f"{full_negative}, {', '.join(autism_negative_terms)}"
        
        settings = {
            'prompt': prompt,
            'negative_prompt': full_negative,
            'num_images_per_prompt': num_images,
            **self.config["generation"],
            **kwargs
        }
        
        return settings
    
    def _select_best_image_with_autism_scoring(self, images, evaluations, prompt, use_ip_adapter):
        """Select best image using TIFA + CLIP + IP-Adapter consistency + Autism scoring"""
        is_first_image = len(self.consistency_manager.selected_images_history) == 0
        
        best_index = 0
        best_combined_score = -1
        best_autism_score = -1
        best_simplicity_score = -1  # ✅ NEW: Track individual simplicity score
        best_autism_analysis = None
        
        autism_results = []
        
        for idx, (image, evaluation) in enumerate(zip(images, evaluations)):
            tifa_score = evaluation["score"]
            
            # Check for black image and penalize
            img_array = np.array(image)
            mean_brightness = np.mean(img_array)
            brightness_penalty = 0
            
            if mean_brightness < 15:  # Very dark image
                brightness_penalty = 0.5  # Heavy penalty
                print(f"⚠️ Image {idx} is very dark (brightness: {mean_brightness:.1f})")
            elif mean_brightness < 30:  # Somewhat dark
                brightness_penalty = 0.2  # Moderate penalty
            
            # Calculate CLIP consistency if not first image
            clip_consistency = 0.5
            if not is_first_image and self.consistency_manager.available:
                image_embedding = self.consistency_manager.get_image_embedding(image)
                clip_consistency = self.consistency_manager.calculate_consistency_score(image_embedding)
            
            # Calculate IP-Adapter consistency if used
            ip_adapter_consistency = 0.5
            if use_ip_adapter and self.ip_adapter_manager and self.ip_adapter_manager.available:
                ip_adapter_consistency = self.ip_adapter_manager.get_consistency_score(image)
            
            # ✅ FIXED: Calculate simplicity score (individual metric)
            simplicity_score = 0.5
            autism_analysis = {}
            if self.enable_autism_scoring and self.autism_analyzer and self.autism_analyzer.available:
                autism_analysis = self.autism_analyzer.analyze_autism_suitability(image)
                simplicity_score = autism_analysis["autism_suitability"]  # ✅ This is just simplicity
                autism_results.append(autism_analysis)
                
                # ✅ FIXED: Print simplicity evaluation (not autism score)
                print(f"   Image {idx+1}: Simplicity score = {simplicity_score:.3f} ({autism_analysis['autism_grade']})")
                
                # Apply hard rejection for very poor simplicity scores
                if simplicity_score < self.config["autism"]["reject_threshold"]:
                    print(f"   ❌ Rejected: Simplicity score too low ({simplicity_score:.3f})")
                    brightness_penalty += 1.0  # Effectively reject this image
            else:
                autism_results.append({"autism_suitability": 0.5, "autism_grade": "Not evaluated"})
            
            # ✅ FIXED: Calculate the REAL autism score (weighted combination)
            if self.enable_autism_scoring and self.config["selection"]["use_autism_scoring"]:
                # The autism score IS the weighted combination of all 3 metrics
                autism_score = (
                    simplicity_score * 0.364 +    # SIMPLICITY - 36.4% (highest priority)
                    tifa_score * 0.333 +          # ACCURACY - 33.3% 
                    clip_consistency * 0.303      # CONSISTENCY - 30.3% (lowest priority)
                )
                
                # The combined score is the autism score minus penalties
                combined_score = autism_score - brightness_penalty
                
                print(f"   Image {idx+1}: Autism score = {autism_score:.3f} (weighted combination)")
                
            else:
                # Original scoring without autism
                autism_score = 0.5  # Default when autism scoring disabled
                
                if use_ip_adapter:
                    combined_score = (
                        tifa_score * self.config["selection"]["quality_weight"] +
                        clip_consistency * self.config["selection"]["consistency_weight"] +
                        ip_adapter_consistency * self.config["selection"]["ip_adapter_weight"] -
                        brightness_penalty
                    )
                else:
                    combined_score = (
                        tifa_score * self.config["selection"]["quality_weight"] +
                        clip_consistency * (self.config["selection"]["consistency_weight"] + 
                                          self.config["selection"]["ip_adapter_weight"]) -
                        brightness_penalty
                    )
            
            if combined_score > best_combined_score:
                best_combined_score = combined_score
                best_index = idx
                best_autism_score = autism_score  # ✅ This is now the weighted combination
                best_simplicity_score = simplicity_score  # ✅ Individual simplicity score
                best_autism_analysis = autism_analysis
        
        # Print autism recommendations for best image
        if best_autism_analysis and "recommendations" in best_autism_analysis:
            print("\n🧩 Autism recommendations for selected image:")
            for rec in best_autism_analysis["recommendations"][:3]:
                print(f"   {rec}")
        
        return {
            "image": images[best_index],
            "combined_score": best_combined_score,
            "tifa_score": evaluations[best_index]["score"],
            "autism_score": best_autism_score,          # ✅ FIXED: Weighted combination (36.4% + 33.3% + 30.3%)
            "simplicity_score": best_simplicity_score,  # ✅ NEW: Individual simplicity score
            "autism_grade": best_autism_analysis.get("autism_grade", "Not evaluated") if best_autism_analysis else "Not evaluated",
            "autism_analysis": best_autism_analysis,
            "index": best_index,
            "evaluation": evaluations[best_index]
        }
    
    def generate_autism_optimized_storyboard(self, prompts_list, character_reference_image=None, 
                                           enforce_autism_requirements=True, max_retries=3):
        """Generate storyboard specifically optimized for autism education"""
        if not self.available:
            return None
        
        print(f"🧩 Generating AUTISM-OPTIMIZED storyboard with {len(prompts_list)} prompts")
        
        # Ensure autism scoring is enabled
        original_autism_setting = self.enable_autism_scoring
        self.enable_autism_scoring = True
        
        # Set character reference if provided
        if character_reference_image is not None:
            self.set_character_reference_image(character_reference_image)
        
        sequence_results = []
        autism_scores = []
        retry_counts = []
        
        for prompt_idx, prompt in enumerate(prompts_list):
            print(f"\n📖 PROMPT {prompt_idx + 1}/{len(prompts_list)}: {prompt}")
            
            best_result = None
            retry_count = 0
            
            while retry_count < max_retries:
                result = self.generate_with_selection(
                    prompt=prompt,
                    num_images=self.config["generation"]["num_images_per_prompt"] + retry_count,  # Generate more on retries
                    use_ip_adapter=self.ip_adapter_loaded
                )
                
                if not result:
                    print(f"⚠️ Generation failed for prompt {prompt_idx + 1}")
                    retry_count += 1
                    continue
                
                autism_score = result.get("autism_score", 0.5)
                autism_grade = result.get("autism_grade", "Not evaluated")
                
                print(f"   Autism score: {autism_score:.3f} ({autism_grade})")
                
                # Check if meets autism requirements
                if enforce_autism_requirements and autism_score < 0.6:
                    print(f"   ⚠️ Below autism threshold (0.6), retry {retry_count + 1}/{max_retries}")
                    retry_count += 1
                    
                    # Adjust prompt for better autism compliance
                    if retry_count < max_retries:
                        prompt = self._enhance_prompt_for_autism(prompt)
                        print(f"   🔧 Enhanced prompt: {prompt[:80]}...")
                else:
                    best_result = result
                    break
            
            if best_result:
                sequence_results.append({
                    "prompt": prompt,
                    "prompt_index": prompt_idx,
                    "best_image": best_result["best_image"],
                    "final_score": best_result["best_score"],
                    "autism_score": best_result.get("autism_score", 0.5),
                    "autism_grade": best_result.get("autism_grade", "Not evaluated"),
                    "autism_analysis": best_result.get("autism_analysis", {}),
                    "used_ip_adapter": best_result.get("used_ip_adapter", False),
                    "retry_count": retry_count
                })
                autism_scores.append(best_result.get("autism_score", 0.5))
                retry_counts.append(retry_count)
            else:
                print(f"   ❌ Failed to generate suitable image after {max_retries} attempts")
        
        # Calculate autism statistics
        avg_autism_score = np.mean(autism_scores) if autism_scores else 0.0
        autism_compliance_rate = sum(1 for s in autism_scores if s >= 0.6) / len(autism_scores) if autism_scores else 0.0
        
        # Final consistency report
        final_consistency = {"available": False}
        if self.consistency_manager.available:
            final_consistency = self.consistency_manager.get_consistency_report()
        
        # Generate autism report
        autism_report = self._generate_autism_storyboard_report(
            sequence_results, autism_scores, retry_counts
        )
        
        print(f"\n🎉 AUTISM-OPTIMIZED STORYBOARD COMPLETE!")
        print(f"📖 Generated: {len(sequence_results)} images")
        print(f"🧩 Average autism score: {avg_autism_score:.3f}")
        print(f"✅ Autism compliance rate: {autism_compliance_rate:.1%}")
        print(f"🔄 Total retries: {sum(retry_counts)}")
        
        if final_consistency["available"]:
            print(f"📊 Overall CLIP Consistency: {final_consistency['consistency_grade']} "
                  f"(Score: {final_consistency['average_consistency']:.3f})")
        
        # Restore original autism setting
        self.enable_autism_scoring = original_autism_setting
        
        return {
            "sequence_results": sequence_results,
            "consistency_report": final_consistency,
            "autism_report": autism_report,
            "autism_statistics": {
                "average_score": avg_autism_score,
                "compliance_rate": autism_compliance_rate,
                "scores": autism_scores,
                "retry_counts": retry_counts,
                "total_retries": sum(retry_counts)
            },
            "ip_adapter_status": {
                "available": self.ip_adapter_loaded,
                "character_reference_set": self.character_reference_image is not None,
                "used_count": sum(1 for r in sequence_results if r.get("used_ip_adapter", False))
            },
            "total_prompts": len(prompts_list),
            "successful_generations": len(sequence_results)
        }
    
    def _enhance_prompt_for_autism(self, prompt):
        """Enhance prompt to improve autism suitability"""
        autism_enhancements = [
            "simple clean background",
            "clear character definition",
            "minimal details",
            "soft colors",
            "single focus",
            "calm expression",
            "uncluttered scene"
        ]
        
        # Add random autism-friendly enhancements
        import random
        enhancements_to_add = random.sample(autism_enhancements, 2)
        
        enhanced_prompt = prompt
        for enhancement in enhancements_to_add:
            if enhancement not in prompt.lower():
                enhanced_prompt += f", {enhancement}"
        
        return enhanced_prompt
    
    def _generate_autism_storyboard_report(self, sequence_results, autism_scores, retry_counts):
        """Generate detailed autism suitability report for storyboard"""
        report = {
            "summary": {
                "total_frames": len(sequence_results),
                "average_autism_score": np.mean(autism_scores) if autism_scores else 0.0,
                "min_autism_score": min(autism_scores) if autism_scores else 0.0,
                "max_autism_score": max(autism_scores) if autism_scores else 0.0,
                "compliant_frames": sum(1 for s in autism_scores if s >= 0.6),
                "total_retries": sum(retry_counts)
            },
            "frame_details": [],
            "common_issues": {},
            "recommendations": []
        }
        
        # Analyze each frame
        issue_counter = {}
        
        for result in sequence_results:
            autism_analysis = result.get("autism_analysis", {})
            
            frame_detail = {
                "prompt": result["prompt"],
                "autism_score": result.get("autism_score", 0.5),
                "autism_grade": result.get("autism_grade", "Not evaluated"),
                "retry_count": result.get("retry_count", 0),
                "key_metrics": {}
            }
            
            # Extract key metrics
            if autism_analysis:
                frame_detail["key_metrics"] = {
                    "person_count": autism_analysis.get("person_count", {}).get("count", "Unknown"),
                    "background_simplicity": autism_analysis.get("background_simplicity", {}).get("score", 0.0),
                    "color_appropriateness": autism_analysis.get("color_appropriateness", {}).get("score", 0.0),
                    "sensory_friendliness": autism_analysis.get("sensory_friendliness", {}).get("score", 0.0)
                }
                
                # Track common issues
                recommendations = autism_analysis.get("recommendations", [])
                for rec in recommendations:
                    if "CRITICAL" in rec:
                        issue_type = "critical_issues"
                    elif "Simplify" in rec or "Reduce" in rec:
                        issue_type = "simplification_needed"
                    else:
                        issue_type = "other_improvements"
                    
                    if issue_type not in issue_counter:
                        issue_counter[issue_type] = 0
                    issue_counter[issue_type] += 1
            
            report["frame_details"].append(frame_detail)
        
        # Summarize common issues
        report["common_issues"] = issue_counter
        
        # Generate overall recommendations
        if report["summary"]["average_autism_score"] < 0.6:
            report["recommendations"].append("⚠️ Overall autism suitability is LOW - significant improvements needed")
        elif report["summary"]["average_autism_score"] < 0.8:
            report["recommendations"].append("👍 Overall autism suitability is MODERATE - some improvements recommended")
        else:
            report["recommendations"].append("🎉 Overall autism suitability is EXCELLENT!")
        
        if issue_counter.get("critical_issues", 0) > 0:
            report["recommendations"].append("🚨 Address critical issues (e.g., too many people in scenes)")
        
        if issue_counter.get("simplification_needed", 0) > len(sequence_results) / 2:
            report["recommendations"].append("🎨 Consider simplifying backgrounds and reducing visual complexity")
        
        return report
    
    def generate_with_progressive_improvement(self, prompt, num_images=3, use_ip_adapter=None, 
                                             quality_threshold=0.5, max_iterations=3,
                                             autism_threshold=0.6):
        """Generate images with progressive improvement including autism optimization"""
        from prompt_improver import PromptImprover
        
        prompt_improver = PromptImprover(self.consistency_manager, self.quality_evaluator)
        
        current_prompt = prompt
        original_prompt = prompt
        current_negative = "blurry, low quality, distorted, deformed"
        
        best_image = None
        best_score = 0.0
        best_autism_score = 0.0
        score_progression = []
        autism_progression = []
        prompt_history = [prompt]
        progressive_improvement_applied = False
        
        for iteration in range(max_iterations):
            print(f"🔄 Iteration {iteration + 1}/{max_iterations}: {current_prompt[:60]}...")
            
            # Generate images using your existing selection method
            result = self.generate_with_selection(
                prompt=current_prompt,
                negative_prompt=current_negative,
                num_images=num_images,
                use_ip_adapter=use_ip_adapter
            )
            
            if not result:
                print(f"⚠️ Generation failed in iteration {iteration + 1}")
                break
            
            current_score = result["best_score"]
            current_autism_score = result.get("autism_score", 0.5)
            current_image = result["best_image"]
            
            score_progression.append(current_score)
            autism_progression.append(current_autism_score)
            
            # Update overall best if this iteration is better
            if current_score > best_score:
                best_score = current_score
                best_image = current_image
                best_autism_score = current_autism_score
            
            print(f"✨ Score this iteration: {current_score:.3f}")
            print(f"🧩 Autism score: {current_autism_score:.3f}")
            
            # Analyze the result using your PromptImprover
            is_first_selected = len(self.consistency_manager.selected_images_history) == 1
            analysis = prompt_improver.analyze_prompt_image_alignment(
                current_image, 
                current_prompt,
                is_first_image=is_first_selected
            )
            
            print(f"📊 CLIP similarity: {analysis['clip_similarity']:.3f}")
            print(f"📝 Caption: {analysis['caption'][:60]}...")
            
            # Check consistency issues
            consistency_issues = analysis.get("consistency_issues", {})
            if consistency_issues["severity"] != "low":
                print(f"⚠️ Consistency issues detected: {consistency_issues['severity']}")
            
            # Check if we should continue improving
            meets_quality = current_score >= quality_threshold
            meets_autism = current_autism_score >= autism_threshold or not self.enable_autism_scoring
            
            if meets_quality and meets_autism:
                print(f"✅ Quality threshold ({quality_threshold}) and autism threshold ({autism_threshold}) reached!")
                break
            
            # Only try to improve prompt if we haven't reached the last iteration
            if iteration < max_iterations - 1:
                # Generate improved prompts using your PromptImprover
                improved = prompt_improver.improve_prompts(
                    current_prompt, 
                    analysis, 
                    current_negative
                )
                
                # Add autism-specific improvements if score is low
                if self.enable_autism_scoring and current_autism_score < autism_threshold:
                    current_prompt = self._enhance_prompt_for_autism(improved["positive"])
                    print(f"🧩 Added autism enhancements to prompt")
                else:
                    current_prompt = improved["positive"]
                
                current_negative = improved["negative"]
                
                if current_prompt != prompt_history[-1]:
                    prompt_history.append(current_prompt)
                    progressive_improvement_applied = True
                    
                    print(f"🔧 Prompt refined: {current_prompt[:60]}...")
                    if improved["improvements"]:
                        print(f"✨ Applied improvements: {', '.join(improved['improvements'][:2])}...")
                else:
                    print("🤔 No prompt improvement suggested")
        
        # Show final improvement summary
        if len(score_progression) > 1:
            total_improvement = score_progression[-1] - score_progression[0]
            autism_improvement = autism_progression[-1] - autism_progression[0] if autism_progression else 0
            print(f"📈 Score progression: {score_progression[0]:.3f} → {score_progression[-1]:.3f} (Δ{total_improvement:+.3f})")
            if self.enable_autism_scoring:
                print(f"🧩 Autism progression: {autism_progression[0]:.3f} → {autism_progression[-1]:.3f} (Δ{autism_improvement:+.3f})")
        
        return {
            "best_image": best_image,
            "best_score": best_score,
            "autism_score": best_autism_score,
            "used_ip_adapter": use_ip_adapter and self.ip_adapter_loaded,
            "progressive_improvement": progressive_improvement_applied,
            "final_prompt": current_prompt,
            "original_prompt": original_prompt,
            "score_progression": score_progression,
            "autism_progression": autism_progression,
            "prompt_history": prompt_history,
            "iterations_completed": len(score_progression)
        }
    
    def get_pipeline_status(self):
        """Get comprehensive pipeline status including autism analyzer"""
        base_status = {
            "pipeline_available": self.available,
            "consistency_manager": self.consistency_manager.available if hasattr(self, 'consistency_manager') else False,
            "quality_evaluator": self.quality_evaluator.available if hasattr(self, 'quality_evaluator') else False,
            "autism_analyzer": self.autism_analyzer.available if self.autism_analyzer else False,
            "autism_scoring_enabled": self.enable_autism_scoring,
            "selected_images_count": len(self.consistency_manager.selected_images_history) if hasattr(self, 'consistency_manager') else 0,
            "model_path": self.model_path,
            "character_reference_set": self.character_reference_image is not None,
            "ip_adapter_loaded": self.ip_adapter_loaded,
            "baked_vae_compatible": self.baked_vae_compatible,
            "current_ip_adapter_scale": self.config["ip_adapter"]["character_weight"],
            "autism_config": self.config.get("autism", {})
        }
        
        return base_status
    
    def reset_all_memory(self):
        """Reset all consistency memory and references"""
        if hasattr(self, 'consistency_manager'):
            self.consistency_manager.reset_memory()
        self.character_reference_image = None
        if hasattr(self, 'ip_adapter_manager') and self.ip_adapter_manager:
            self.ip_adapter_manager.reset_references()
        print("🔄 Reset all memory - ready for new storyboard")
    
    def update_config(self, **config_updates):
        """Update pipeline configuration with autism considerations"""
        for key, value in config_updates.items():
            if key in self.config:
                if isinstance(self.config[key], dict) and isinstance(value, dict):
                    self.config[key].update(value)
                else:
                    self.config[key] = value
        
        # Update IP-Adapter scale if it changed
        if "ip_adapter" in config_updates and "character_weight" in config_updates["ip_adapter"]:
            if self.ip_adapter_loaded and hasattr(self.pipe, 'set_ip_adapter_scale'):
                new_scale = config_updates["ip_adapter"]["character_weight"]
                # Validate scale for baked VAE compatibility
                if not self.baked_vae_compatible and new_scale > 0.5:
                    print(f"⚠️ High IP-Adapter scale ({new_scale}) may cause issues with this baked VAE")
                self.pipe.set_ip_adapter_scale(new_scale)
        
        # Handle autism scoring toggle
        if "enable_autism_scoring" in config_updates:
            self.enable_autism_scoring = config_updates["enable_autism_scoring"]
            print(f"🧩 Autism scoring {'enabled' if self.enable_autism_scoring else 'disabled'}")
        
        print(f"🔧 Configuration updated: {config_updates}")
    
    def save_autism_report(self, sequence_result, output_dir):
        """Save comprehensive autism suitability report"""
        import os
        
        report_path = os.path.join(output_dir, "autism_suitability_report.txt")
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("AUTISM STORYBOARD SUITABILITY REPORT\n")
            f.write("=" * 50 + "\n\n")
            
            # Summary statistics
            if "autism_statistics" in sequence_result:
                stats = sequence_result["autism_statistics"]
                f.write("OVERALL AUTISM SUITABILITY:\n")
                f.write(f"Average Score: {stats['average_score']:.3f}\n")
                f.write(f"Compliance Rate: {stats['compliance_rate']:.1%}\n")
                f.write(f"Total Retries: {stats['total_retries']}\n\n")
            
            # Frame-by-frame analysis
            f.write("FRAME-BY-FRAME ANALYSIS:\n")
            f.write("-" * 30 + "\n")
            
            for i, result in enumerate(sequence_result.get("sequence_results", [])):
                f.write(f"\nFrame {i+1}:\n")
                f.write(f"Prompt: {result['prompt']}\n")
                f.write(f"Autism Score: {result.get('autism_score', 'N/A'):.3f}\n")
                f.write(f"Autism Grade: {result.get('autism_grade', 'Not evaluated')}\n")
                
                # Detailed metrics if available
                if "autism_analysis" in result and result["autism_analysis"]:
                    analysis = result["autism_analysis"]
                    f.write("Key Metrics:\n")
                    
                    if "person_count" in analysis:
                        f.write(f"  - Person Count: {analysis['person_count']['count']} ")
                        f.write(f"({'✅' if analysis['person_count']['is_compliant'] else '❌'})\n")
                    
                    if "background_simplicity" in analysis:
                        f.write(f"  - Background Simplicity: {analysis['background_simplicity']['score']:.3f} ")
                        f.write(f"({analysis['background_simplicity']['grade']})\n")
                    
                    if "color_appropriateness" in analysis:
                        f.write(f"  - Color Appropriateness: {analysis['color_appropriateness']['score']:.3f}\n")
                    
                    if "recommendations" in analysis:
                        f.write("Recommendations:\n")
                        for rec in analysis["recommendations"][:3]:
                            f.write(f"  • {rec}\n")
            
            # Common issues and recommendations
            if "autism_report" in sequence_result and sequence_result["autism_report"]:
                report = sequence_result["autism_report"]
                
                f.write("\n\nCOMMON ISSUES:\n")
                f.write("-" * 15 + "\n")
                for issue_type, count in report.get("common_issues", {}).items():
                    f.write(f"{issue_type}: {count} occurrences\n")
                
                f.write("\nOVERALL RECOMMENDATIONS:\n")
                f.write("-" * 23 + "\n")
                for rec in report.get("recommendations", []):
                    f.write(f"• {rec}\n")
        
        print(f"📄 Autism suitability report saved: {report_path}")
        return report_path


# Convenience function to create the pipeline
def create_autism_integrated_pipeline(model_path, ip_adapter_path=None, enable_autism_scoring=True):
    """Convenience function to create autism-integrated pipeline"""
    return AutismIntegratedCartoonPipeline(
        model_path=model_path,
        ip_adapter_path=ip_adapter_path,
        enable_autism_scoring=enable_autism_scoring
    )


# Test function
def test_autism_integration():
    """Test the autism-integrated pipeline"""
    print("🧪 Testing Autism-Integrated Pipeline")
    print("=" * 40)
    
    # Create test pipeline (you'll need to provide actual paths)
    pipeline = create_autism_integrated_pipeline(
        model_path="path/to/your/model.safetensors",
        ip_adapter_path="path/to/ip-adapter_sdxl.bin",
        enable_autism_scoring=True
    )
    
    if not pipeline.available:
        print("❌ Pipeline not available for testing")
        return
    
    # Test single image generation
    print("\n📸 Testing single image generation with autism scoring...")
    result = pipeline.generate_single_image(
        prompt="young boy reading a book, simple room, clean background, cartoon style",
        num_images=3
    )
    
    if result:
        print(f"✅ Image generated successfully")
        print(f"   Combined score: {result['score']:.3f}")
        print(f"   Autism score: {result['autism_score']:.3f}")
        print(f"   Autism grade: {result['autism_grade']}")
    
    # Test autism-optimized storyboard
    print("\n📚 Testing autism-optimized storyboard generation...")
    test_prompts = [
        "young boy waking up in bed, simple bedroom, cartoon style",
        "same boy brushing teeth, clean bathroom, cartoon style",
        "same boy eating breakfast, simple kitchen, cartoon style"
    ]
    
    storyboard_result = pipeline.generate_autism_optimized_storyboard(
        prompts_list=test_prompts,
        enforce_autism_requirements=True,
        max_retries=2
    )
    
    if storyboard_result:
        stats = storyboard_result["autism_statistics"]
        print(f"✅ Storyboard generated successfully")
        print(f"   Average autism score: {stats['average_score']:.3f}")
        print(f"   Compliance rate: {stats['compliance_rate']:.1%}")
        print(f"   Total retries: {stats['total_retries']}")
    
    print("\n✅ Autism integration test complete!")


if __name__ == "__main__":
    print("🎨 Autism-Integrated Cartoon Pipeline")
    print("📋 Features:")
    print("   - Autism suitability scoring for every image")
    print("   - Person count verification (max 2)")
    print("   - Background simplicity analysis")
    print("   - Sensory-friendly color evaluation")
    print("   - Character clarity assessment")
    print("   - Progressive prompt improvement with autism optimization")
    print("   - Detailed autism suitability reports")
    print("\n🚀 Ready for autism-friendly storyboard generation!")