"""
Computer Vision metrics for visual quality assessment
Adapted from the improved CV validation system
"""

import cv2
import numpy as np
from PIL import Image
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings('ignore')


class VisualQualityAnalyzer:
    """Analyzes technical visual quality and artifacts in images"""
    
    def __init__(self):
        self.artifact_detector = None
        self.quality_thresholds = {
            'blur_threshold': 100,
            'noise_threshold': 15,
            'brightness_range': (20, 240),
            'contrast_threshold': 20
        }
    
    def analyze_image_quality(self, image):
        """
        Comprehensive visual quality analysis
        Returns scores and detailed metrics
        """
        # Convert PIL to cv2 format
        if isinstance(image, Image.Image):
            img_array = np.array(image)
            if len(img_array.shape) == 3 and img_array.shape[2] == 3:
                cv_image = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            else:
                cv_image = img_array
        else:
            cv_image = image
        
        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        
        # Extract comprehensive metrics
        metrics = {}
        metrics.update(self._analyze_sharpness(gray))
        metrics.update(self._analyze_exposure(cv_image))
        metrics.update(self._analyze_noise(gray))
        metrics.update(self._analyze_contrast(gray))
        metrics.update(self._analyze_artifacts(cv_image, gray))
        
        # Calculate overall quality score
        quality_score = self._calculate_quality_score(metrics)
        
        return {
            'quality_score': quality_score,
            'metrics': metrics,
            'has_artifacts': quality_score < 0.7,
            'issues': self._identify_issues(metrics)
        }
    
    def _analyze_sharpness(self, gray):
        """Analyze image sharpness/blur"""
        # Laplacian variance (higher = sharper)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        laplacian_var = float(np.var(laplacian))
        
        # Gradient magnitude
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_mag = float(np.mean(np.sqrt(grad_x**2 + grad_y**2)))
        
        # Edge density
        edges = cv2.Canny(gray, 50, 150)
        edge_density = float(np.sum(edges > 0) / edges.size)
        
        # Multi-scale sharpness
        scales = [1, 2, 4]
        multi_scale_sharpness = []
        for scale in scales:
            if scale > 1:
                scaled = cv2.resize(gray, (gray.shape[1]//scale, gray.shape[0]//scale))
                scaled = cv2.resize(scaled, (gray.shape[1], gray.shape[0]))
            else:
                scaled = gray
            lap = cv2.Laplacian(scaled, cv2.CV_64F)
            multi_scale_sharpness.append(float(np.var(lap)))
        
        return {
            'laplacian_variance': laplacian_var,
            'gradient_magnitude': gradient_mag,
            'edge_density': edge_density,
            'multi_scale_sharpness_mean': float(np.mean(multi_scale_sharpness)),
            'is_blurry': laplacian_var < self.quality_thresholds['blur_threshold']
        }
    
    def _analyze_exposure(self, cv_image):
        """Analyze exposure quality"""
        # Brightness analysis
        mean_brightness = float(np.mean(cv_image))
        
        # Dynamic range
        p1, p5, p95, p99 = np.percentile(cv_image, [1, 5, 95, 99])
        effective_range = float(p99 - p1)
        core_range = float(p95 - p5)
        
        # Histogram analysis
        hist, _ = np.histogram(cv_image.flatten(), bins=64, range=[0, 256])
        hist_norm = hist / np.sum(hist)
        
        bright_pixels = float(np.sum(hist_norm[48:]))  # >75% brightness
        dark_pixels = float(np.sum(hist_norm[:16]))    # <25% brightness
        
        min_bright, max_bright = self.quality_thresholds['brightness_range']
        
        return {
            'mean_brightness': mean_brightness,
            'core_dynamic_range': core_range,
            'bright_pixel_ratio': bright_pixels,
            'dark_pixel_ratio': dark_pixels,
            'is_overexposed': mean_brightness > max_bright,
            'is_underexposed': mean_brightness < min_bright
        }
    
    def _analyze_noise(self, gray):
        """Analyze noise levels"""
        # MAD (Median Absolute Deviation) noise estimation
        median_blur = cv2.medianBlur(gray, 5)
        mad_noise = float(np.median(np.abs(gray.astype(np.float32) - median_blur.astype(np.float32))))
        
        # Gaussian noise estimation
        gaussian_blur = cv2.GaussianBlur(gray, (5,5), 1.0)
        noise_estimate = float(np.std(gray.astype(np.float32) - gaussian_blur.astype(np.float32)))
        
        # Local noise variation
        noise_patches = []
        patch_size = 20
        for i in range(0, gray.shape[0]-patch_size, patch_size):
            for j in range(0, gray.shape[1]-patch_size, patch_size):
                patch = gray[i:i+patch_size, j:j+patch_size]
                patch_blur = cv2.GaussianBlur(patch, (5,5), 1.0)
                patch_noise = np.std(patch.astype(np.float32) - patch_blur.astype(np.float32))
                noise_patches.append(patch_noise)
        
        noise_variation = float(np.std(noise_patches)) if noise_patches else 0.0
        
        return {
            'mad_noise': mad_noise,
            'gaussian_noise': noise_estimate,
            'noise_variation': noise_variation,
            'is_noisy': mad_noise > self.quality_thresholds['noise_threshold']
        }
    
    def _analyze_contrast(self, gray):
        """Analyze contrast quality"""
        # RMS contrast
        rms_contrast = float(np.std(gray))
        
        # Local contrast
        kernel = np.ones((9,9), np.float32) / 81
        local_means = cv2.filter2D(gray.astype(np.float32), -1, kernel)
        local_contrast_map = np.abs(gray.astype(np.float32) - local_means)
        local_contrast_mean = float(np.mean(local_contrast_map))
        
        return {
            'rms_contrast': rms_contrast,
            'local_contrast_mean': local_contrast_mean,
            'has_low_contrast': rms_contrast < self.quality_thresholds['contrast_threshold']
        }
    
    def _analyze_artifacts(self, cv_image, gray):
        """Detect specific visual artifacts"""
        artifacts = {
            'has_compression_artifacts': False,
            'has_banding': False,
            'has_color_artifacts': False
        }
        
        # JPEG compression artifacts (blockiness)
        dct_coeffs = cv2.dct(gray.astype(np.float32))
        high_freq_energy = np.sum(np.abs(dct_coeffs[8:, 8:]))
        total_energy = np.sum(np.abs(dct_coeffs))
        if total_energy > 0:
            compression_ratio = high_freq_energy / total_energy
            artifacts['has_compression_artifacts'] = compression_ratio < 0.1
        
        # Color banding detection
        color_unique = len(np.unique(cv_image.reshape(-1, 3), axis=0))
        expected_colors = cv_image.shape[0] * cv_image.shape[1] * 0.1  # 10% unique colors expected
        artifacts['has_banding'] = color_unique < expected_colors
        
        # Color channel artifacts
        b_mean = float(np.mean(cv_image[:,:,0]))
        g_mean = float(np.mean(cv_image[:,:,1]))
        r_mean = float(np.mean(cv_image[:,:,2]))
        
        # Check for severe channel imbalance
        channel_variance = np.var([b_mean, g_mean, r_mean])
        artifacts['has_color_artifacts'] = channel_variance > 1000
        
        return artifacts
    
    def _calculate_quality_score(self, metrics):
        """Calculate overall quality score from metrics"""
        score = 1.0
        
        # Sharpness penalty
        if metrics.get('is_blurry', False):
            score -= 0.3
        
        # Exposure penalty
        if metrics.get('is_overexposed', False) or metrics.get('is_underexposed', False):
            score -= 0.2
        
        # Noise penalty
        if metrics.get('is_noisy', False):
            score -= 0.15
        
        # Contrast penalty
        if metrics.get('has_low_contrast', False):
            score -= 0.1
        
        # Artifact penalties
        if metrics.get('has_compression_artifacts', False):
            score -= 0.1
        if metrics.get('has_banding', False):
            score -= 0.05
        if metrics.get('has_color_artifacts', False):
            score -= 0.1
        
        return max(0.0, min(1.0, score))
    
    def _identify_issues(self, metrics):
        """Identify specific quality issues"""
        issues = []
        
        if metrics.get('is_blurry', False):
            issues.append("Image is blurry")
        if metrics.get('is_overexposed', False):
            issues.append("Image is overexposed")
        if metrics.get('is_underexposed', False):
            issues.append("Image is underexposed")
        if metrics.get('is_noisy', False):
            issues.append("Image has excessive noise")
        if metrics.get('has_low_contrast', False):
            issues.append("Image has low contrast")
        if metrics.get('has_compression_artifacts', False):
            issues.append("JPEG compression artifacts detected")
        if metrics.get('has_banding', False):
            issues.append("Color banding detected")
        
        return issues