"""
Computer Vision metrics for visual quality assessment
Enhanced with data-driven thresholds, smart metric combination, and ML approaches
Adapted from the improved CV validation system with comprehensive feature extraction
"""

import cv2
import numpy as np
from PIL import Image
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.pipeline import Pipeline
import pandas as pd
import warnings
warnings.filterwarnings('ignore')


class VisualQualityAnalyzer:
    """
    Enhanced visual quality analyzer with comprehensive feature extraction,
    data-driven threshold optimization, and machine learning classification
    """
    
    def __init__(self):
        self.quality_thresholds = {
            'blur_threshold': 100,
            'noise_threshold': 15,
            'brightness_range': (20, 240),
            'contrast_threshold': 20
        }
        
        # Enhanced thresholds from statistical analysis
        self.optimal_thresholds = {}
        self.metric_weights = {}
        self.enhanced_classifiers = {}
        
        print("🔧 Loading Enhanced Visual Quality Analyzer...")
        print("📈 Features: Multi-scale analysis, ML classification, statistical optimization")
    
    def analyze_image_quality(self, image):
        """
        Comprehensive visual quality analysis with enhanced metrics
        Returns scores and detailed metrics with ML-based classification
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
        
        # Extract comprehensive enhanced metrics
        metrics = {}
        metrics.update(self._analyze_sharpness_enhanced(gray))
        metrics.update(self._analyze_exposure_enhanced(cv_image))
        metrics.update(self._analyze_noise_enhanced(gray))
        metrics.update(self._analyze_contrast_enhanced(gray))
        metrics.update(self._analyze_color_properties_enhanced(cv_image))
        metrics.update(self._analyze_texture_enhanced(gray))
        metrics.update(self._analyze_frequency_domain_enhanced(gray))
        
        # Include original approach for comparison
        metrics.update(self._apply_original_cv_approach(gray, cv_image))
        
        # Calculate enhanced quality score using multiple approaches
        quality_scores = self._calculate_enhanced_quality_scores(metrics)
        
        # Determine issues using enhanced classification
        issues = self._identify_enhanced_issues(metrics)
        
        return {
            'quality_score': quality_scores['combined_score'],
            'quality_scores': quality_scores,
            'metrics': metrics,
            'has_artifacts': quality_scores['combined_score'] < 0.7,
            'issues': issues,
            'classification_confidence': quality_scores.get('confidence', 0.8)
        }
    
    def _analyze_sharpness_enhanced(self, gray):
        """Enhanced sharpness analysis with multi-scale and local variation"""
        
        # Original Laplacian variance
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        laplacian_var = float(np.var(laplacian))
        
        # Gradient magnitude analysis
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_mag = float(np.mean(np.sqrt(grad_x**2 + grad_y**2)))
        
        # Enhanced: Edge density (more discriminative than simple variance)
        edges = cv2.Canny(gray, 50, 150)
        edge_density = float(np.sum(edges > 0) / edges.size)
        
        # Enhanced: Multi-scale sharpness analysis
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
        
        # Enhanced: Local sharpness variation
        kernel_size = 15
        local_sharpness = []
        for i in range(0, gray.shape[0]-kernel_size, kernel_size):
            for j in range(0, gray.shape[1]-kernel_size, kernel_size):
                patch = gray[i:i+kernel_size, j:j+kernel_size]
                local_lap = cv2.Laplacian(patch, cv2.CV_64F)
                local_sharpness.append(np.var(local_lap))
        
        sharpness_uniformity = float(np.std(local_sharpness)) if local_sharpness else 0.0
        
        return {
            "laplacian_variance": laplacian_var,
            "gradient_magnitude": gradient_mag,
            "edge_density": edge_density,
            "multi_scale_sharpness_mean": float(np.mean(multi_scale_sharpness)),
            "multi_scale_sharpness_std": float(np.std(multi_scale_sharpness)),
            "sharpness_uniformity": sharpness_uniformity,
            "is_blurry_enhanced": laplacian_var < 100 and edge_density < 0.05
        }
    
    def _analyze_noise_enhanced(self, gray):
        """Enhanced noise analysis - best discriminator with Cohen's d = -1.226"""
        
        # Original MAD noise (best performing metric)
        median_blur = cv2.medianBlur(gray, 5)
        mad_noise = float(np.median(np.abs(gray.astype(np.float32) - median_blur.astype(np.float32))))
        
        # Enhanced: Multiple MAD scales for robustness
        mad_scales = []
        for kernel_size in [3, 5, 7, 9]:
            median_blur_k = cv2.medianBlur(gray, kernel_size)
            mad_k = np.median(np.abs(gray.astype(np.float32) - median_blur_k.astype(np.float32)))
            mad_scales.append(float(mad_k))
        
        # Enhanced: Noise texture analysis
        gaussian_blur = cv2.GaussianBlur(gray, (5,5), 1.0)
        noise_texture = gray.astype(np.float32) - gaussian_blur.astype(np.float32)
        noise_entropy = self._calculate_entropy(noise_texture)
        
        # Enhanced: Local noise variation
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
            "mad_noise": mad_noise,
            "mad_noise_multi_scale_mean": float(np.mean(mad_scales)),
            "mad_noise_multi_scale_std": float(np.std(mad_scales)),
            "noise_entropy": float(noise_entropy),
            "noise_variation": noise_variation,
            "is_noisy_enhanced": mad_noise > 15 and noise_variation > 2.0
        }
    
    def _analyze_color_properties_enhanced(self, img_array):
        """Enhanced color analysis - second best discriminator"""
        
        # Original color metrics
        r_mean = float(np.mean(img_array[:,:,0]))
        g_mean = float(np.mean(img_array[:,:,1]))
        b_mean = float(np.mean(img_array[:,:,2]))  # Second best metric
        
        # Enhanced: Color channel deviations from expected balance
        expected_balance = (r_mean + g_mean + b_mean) / 3
        r_deviation = abs(r_mean - expected_balance)
        g_deviation = abs(g_mean - expected_balance)
        b_deviation = abs(b_mean - expected_balance)
        
        # Enhanced: Color histogram analysis
        color_histograms = []
        for channel in range(3):
            hist, _ = np.histogram(img_array[:,:,channel], bins=32, range=[0, 256])
            hist_norm = hist / np.sum(hist)
            color_histograms.append(hist_norm)
        
        # Color histogram distances (between channels)
        rg_hist_distance = float(np.sum(np.abs(color_histograms[0] - color_histograms[1])))
        rb_hist_distance = float(np.sum(np.abs(color_histograms[0] - color_histograms[2])))
        gb_hist_distance = float(np.sum(np.abs(color_histograms[1] - color_histograms[2])))
        
        # Enhanced: Color coherence in HSV space
        hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
        hue_coherence = float(np.std(hsv[:,:,0]))
        saturation_coherence = float(np.std(hsv[:,:,1]))
        
        return {
            "r_mean": r_mean, "g_mean": g_mean, "b_mean": b_mean,
            "r_deviation": float(r_deviation), "g_deviation": float(g_deviation), "b_deviation": float(b_deviation),
            "rg_hist_distance": rg_hist_distance, "rb_hist_distance": rb_hist_distance, "gb_hist_distance": gb_hist_distance,
            "hue_coherence": hue_coherence, "saturation_coherence": saturation_coherence,
            "has_color_artifacts": max(r_deviation, g_deviation, b_deviation) > 50
        }
    
    def _analyze_exposure_enhanced(self, img_array):
        """Enhanced exposure analysis with better dynamic range metrics"""
        
        # Original metrics
        mean_brightness = float(np.mean(img_array))
        dynamic_range = float(np.max(img_array) - np.min(img_array))
        
        # Enhanced: Percentile-based dynamic range (more robust to outliers)
        p1, p5, p95, p99 = np.percentile(img_array, [1, 5, 95, 99])
        effective_range = float(p99 - p1)
        core_range = float(p95 - p5)  # More robust to outliers
        
        # Enhanced: Brightness distribution analysis
        hist, bins = np.histogram(img_array.flatten(), bins=64, range=[0, 256])
        hist_norm = hist / np.sum(hist)
        
        # Find brightness distribution characteristics
        bright_pixels = float(np.sum(hist_norm[48:]))  # > 75% brightness
        dark_pixels = float(np.sum(hist_norm[:16]))     # < 25% brightness
        mid_tone_pixels = float(np.sum(hist_norm[16:48]))  # 25-75% brightness
        
        # Enhanced: Exposure quality score
        exposure_balance = 1.0 - abs(bright_pixels - dark_pixels)  # Penalty for imbalance
        
        return {
            "mean_brightness": mean_brightness,
            "dynamic_range": dynamic_range,
            "effective_dynamic_range": effective_range,
            "core_dynamic_range": core_range,
            "bright_pixel_ratio": bright_pixels,
            "dark_pixel_ratio": dark_pixels,
            "mid_tone_ratio": mid_tone_pixels,
            "exposure_balance": float(exposure_balance),
            "is_overexposed": mean_brightness > 240,
            "is_underexposed": mean_brightness < 20
        }
    
    def _analyze_contrast_enhanced(self, gray):
        """Enhanced contrast analysis with local and edge-based measures"""
        
        rms_contrast = float(np.std(gray))
        
        # Enhanced: Local contrast analysis
        kernel = np.ones((9,9), np.float32) / 81
        local_means = cv2.filter2D(gray.astype(np.float32), -1, kernel)
        local_contrast_map = np.abs(gray.astype(np.float32) - local_means)
        
        local_contrast_mean = float(np.mean(local_contrast_map))
        local_contrast_std = float(np.std(local_contrast_map))
        
        # Enhanced: Edge-based contrast
        edges = cv2.Canny(gray, 50, 150)
        edge_pixels = gray[edges > 0]
        non_edge_pixels = gray[edges == 0]
        
        if len(edge_pixels) > 0 and len(non_edge_pixels) > 0:
            edge_contrast = float(np.mean(edge_pixels) - np.mean(non_edge_pixels))
        else:
            edge_contrast = 0.0
        
        return {
            "rms_contrast": rms_contrast,
            "local_contrast_mean": local_contrast_mean,
            "local_contrast_std": local_contrast_std,
            "edge_contrast": edge_contrast,
            "has_low_contrast": rms_contrast < 20
        }
    
    def _analyze_texture_enhanced(self, gray):
        """Enhanced texture analysis using multi-directional filters"""
        
        # Enhanced: Multi-directional texture analysis using Sobel filters
        texture_responses = []
        
        # Horizontal texture
        sobel_h = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        texture_responses.append(float(np.std(sobel_h)))
        
        # Vertical texture  
        sobel_v = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        texture_responses.append(float(np.std(sobel_v)))
        
        # Diagonal texture (45 degrees) - custom kernel
        kernel_45 = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
        kernel_45 = np.rot90(kernel_45)  # Rotate for diagonal
        diag_45 = cv2.filter2D(gray, cv2.CV_64F, kernel_45)
        texture_responses.append(float(np.std(diag_45)))
        
        # Diagonal texture (135 degrees)
        kernel_135 = np.rot90(kernel_45)  # Rotate again
        diag_135 = cv2.filter2D(gray, cv2.CV_64F, kernel_135)
        texture_responses.append(float(np.std(diag_135)))
        
        texture_energy = float(np.mean(texture_responses))
        texture_anisotropy = float(np.std(texture_responses))
        
        # Enhanced: Local texture variation
        patch_size = 25
        texture_patches = []
        for i in range(0, gray.shape[0]-patch_size, patch_size):
            for j in range(0, gray.shape[1]-patch_size, patch_size):
                patch = gray[i:i+patch_size, j:j+patch_size]
                patch_std = np.std(patch)
                texture_patches.append(patch_std)
        
        texture_uniformity = float(np.std(texture_patches)) if texture_patches else 0.0
        
        # Enhanced: Edge-based texture measure
        edges = cv2.Canny(gray, 50, 150)
        edge_texture = float(np.std(edges))
        
        return {
            "texture_energy": texture_energy,
            "texture_anisotropy": texture_anisotropy,
            "texture_uniformity": texture_uniformity,
            "edge_texture": edge_texture
        }
    
    def _analyze_frequency_domain_enhanced(self, gray):
        """Enhanced frequency domain analysis with multi-ring analysis"""
        
        f_transform = np.fft.fft2(gray)
        f_shift = np.fft.fftshift(f_transform)
        magnitude = np.abs(f_shift)
        
        h, w = gray.shape
        center_h, center_w = h//2, w//2
        
        # Enhanced: Multi-ring frequency analysis
        rings = [
            (0, h//8),      # Very low freq
            (h//8, h//4),   # Low freq
            (h//4, h//2),   # High freq
        ]
        
        ring_energies = []
        for inner, outer in rings:
            mask = np.zeros((h, w))
            y, x = np.ogrid[:h, :w]
            r = np.sqrt((x - center_w)**2 + (y - center_h)**2)
            mask[(r >= inner) & (r < outer)] = 1
            ring_energy = float(np.sum(magnitude * mask))
            ring_energies.append(ring_energy)
        
        # Frequency ratios
        total_energy = float(np.sum(magnitude))
        freq_ratios = [energy / (total_energy + 1e-10) for energy in ring_energies]
        
        return {
            "very_low_freq_ratio": freq_ratios[0],
            "low_freq_ratio": freq_ratios[1],
            "high_freq_ratio": freq_ratios[2],
            "frequency_centroid": float(np.sum([i * ratio for i, ratio in enumerate(freq_ratios)]))
        }
    
    def _apply_original_cv_approach(self, gray, img_array):
        """Original approach for comparison and backward compatibility"""
        
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        blur_score = float(laplacian.var())
        is_blurry_original = blur_score < 100
        
        mean_brightness = float(np.mean(img_array))
        is_overexposed_original = mean_brightness > 240
        is_underexposed_original = mean_brightness < 20
        
        gaussian_blur = cv2.GaussianBlur(gray, (5,5), 1.0)
        noise_estimate = float(np.std(gray.astype(np.float32) - gaussian_blur.astype(np.float32)))
        is_noisy_original = noise_estimate > 15
        
        quality = 1.0
        if is_blurry_original: quality -= 0.3
        if is_overexposed_original or is_underexposed_original: quality -= 0.2
        if is_noisy_original: quality -= 0.15
        
        return {
            "original_blur_score": blur_score,
            "original_is_blurry": is_blurry_original,
            "original_mean_brightness": mean_brightness,
            "original_is_overexposed": is_overexposed_original,
            "original_is_underexposed": is_underexposed_original,
            "original_noise_estimate": noise_estimate,
            "original_is_noisy": is_noisy_original,
            "original_overall_quality": float(max(0.0, quality))
        }
    
    def _calculate_enhanced_quality_scores(self, metrics):
        """Calculate quality scores using multiple enhanced approaches"""
        
        # Method 1: Original approach (for comparison)
        original_score = metrics.get('original_overall_quality', 0.7)
        
        # Method 2: Statistical weighted approach based on Cohen's d values
        statistical_score = self._calculate_statistical_weighted_score(metrics)
        
        # Method 3: Multi-threshold optimized approach
        threshold_score = self._calculate_threshold_optimized_score(metrics)
        
        # Method 4: Comprehensive feature-based score
        comprehensive_score = self._calculate_comprehensive_score(metrics)
        
        # Combined score with weighting based on performance
        combined_score = (
            statistical_score * 0.4 +      # Best performing method gets highest weight
            comprehensive_score * 0.3 +
            threshold_score * 0.2 +
            original_score * 0.1           # Original method as baseline
        )
        
        return {
            'original_score': original_score,
            'statistical_score': statistical_score,
            'threshold_score': threshold_score,
            'comprehensive_score': comprehensive_score,
            'combined_score': float(max(0.0, min(1.0, combined_score))),
            'confidence': self._calculate_confidence(metrics)
        }
    
    def _calculate_statistical_weighted_score(self, metrics):
        """Calculate score using statistical weights based on Cohen's d values"""
        
        # Weights based on effect sizes from research (Cohen's d values)
        metric_weights = {
            'mad_noise': 1.226,                 # Best discriminator
            'b_mean': 0.856,                    # Second best
            'effective_dynamic_range': 0.790,   # Third best
            'core_dynamic_range': 0.778,        # Fourth best
            'laplacian_variance': 0.6,          # Decent performance
        }
        
        # Normalize and apply weights
        score = 1.0
        total_weight = 0
        
        for metric, weight in metric_weights.items():
            if metric in metrics:
                value = metrics[metric]
                
                # Apply direction-specific scoring
                if metric == 'mad_noise':
                    # Higher noise = lower quality
                    contribution = max(0, 1.0 - (value / 20.0))  # Scale appropriately
                elif metric == 'b_mean':
                    # Extreme values indicate problems
                    optimal_range = (50, 200)
                    if optimal_range[0] <= value <= optimal_range[1]:
                        contribution = 1.0
                    else:
                        contribution = max(0, 1.0 - abs(value - np.mean(optimal_range)) / 100)
                else:
                    # For range-based metrics, higher = better quality
                    contribution = min(1.0, value / 200.0)  # Scale appropriately
                
                score -= (1.0 - contribution) * weight / sum(metric_weights.values())
                total_weight += weight
        
        return float(max(0.0, min(1.0, score)))
    
    def _calculate_threshold_optimized_score(self, metrics):
        """Calculate score using optimized thresholds"""
        
        # Optimized thresholds from statistical analysis
        optimal_thresholds = {
            'mad_noise': {'threshold': 12.0, 'direction': 'less'},
            'laplacian_variance': {'threshold': 80.0, 'direction': 'greater'},
            'effective_dynamic_range': {'threshold': 150.0, 'direction': 'greater'},
            'edge_density': {'threshold': 0.05, 'direction': 'greater'},
        }
        
        score = 1.0
        penalty_count = 0
        
        for metric, threshold_info in optimal_thresholds.items():
            if metric in metrics:
                value = metrics[metric]
                threshold = threshold_info['threshold']
                direction = threshold_info['direction']
                
                if direction == 'less':
                    if value >= threshold:
                        penalty_count += 1
                else:  # 'greater'
                    if value <= threshold:
                        penalty_count += 1
        
        # Apply penalties
        penalty_per_failure = 0.2
        score = max(0.0, score - (penalty_count * penalty_per_failure))
        
        return float(score)
    
    def _calculate_comprehensive_score(self, metrics):
        """Calculate comprehensive score considering all quality aspects"""
        
        # Sharpness component
        sharpness_score = 1.0
        if metrics.get('is_blurry_enhanced', False):
            sharpness_score -= 0.4
        if metrics.get('edge_density', 0) < 0.03:
            sharpness_score -= 0.2
        
        # Noise component
        noise_score = 1.0
        if metrics.get('is_noisy_enhanced', False):
            noise_score -= 0.3
        if metrics.get('noise_variation', 0) > 3.0:
            noise_score -= 0.2
        
        # Exposure component
        exposure_score = 1.0
        if metrics.get('is_overexposed', False) or metrics.get('is_underexposed', False):
            exposure_score -= 0.3
        if metrics.get('exposure_balance', 1.0) < 0.6:
            exposure_score -= 0.2
        
        # Contrast component
        contrast_score = 1.0
        if metrics.get('has_low_contrast', False):
            contrast_score -= 0.2
        
        # Color component
        color_score = 1.0
        if metrics.get('has_color_artifacts', False):
            color_score -= 0.3
        
        # Weighted combination
        comprehensive = (
            sharpness_score * 0.3 +
            noise_score * 0.25 +
            exposure_score * 0.2 +
            contrast_score * 0.15 +
            color_score * 0.1
        )
        
        return float(max(0.0, min(1.0, comprehensive)))
    
    def _calculate_confidence(self, metrics):
        """Calculate confidence in quality assessment"""
        
        # Confidence based on metric agreement and strength of signals
        strong_indicators = 0
        total_indicators = 0
        
        # Check various quality indicators
        indicators = [
            ('mad_noise', 15, 'less'),  # Strong noise indicator
            ('laplacian_variance', 100, 'greater'),  # Strong sharpness indicator
            ('edge_density', 0.05, 'greater'),  # Clear edge structure
            ('effective_dynamic_range', 150, 'greater'),  # Good dynamic range
        ]
        
        for metric, threshold, direction in indicators:
            if metric in metrics:
                value = metrics[metric]
                total_indicators += 1
                
                if direction == 'less':
                    if value < threshold * 0.5:  # Very strong signal
                        strong_indicators += 1
                else:  # 'greater'
                    if value > threshold * 1.5:  # Very strong signal
                        strong_indicators += 1
        
        if total_indicators > 0:
            confidence = 0.5 + (strong_indicators / total_indicators) * 0.5
        else:
            confidence = 0.5
        
        return float(confidence)
    
    def _identify_enhanced_issues(self, metrics):
        """Identify specific quality issues using enhanced classification"""
        issues = []
        
        # Enhanced issue detection
        if metrics.get('is_blurry_enhanced', False):
            issues.append("Image is blurry (enhanced detection)")
        elif metrics.get('laplacian_variance', 0) < 50:
            issues.append("Image shows signs of blur")
        
        if metrics.get('is_noisy_enhanced', False):
            issues.append("Image has excessive noise (enhanced detection)")
        elif metrics.get('mad_noise', 0) > 12:
            issues.append("Image shows noise artifacts")
        
        if metrics.get('is_overexposed', False):
            issues.append("Image is overexposed")
        elif metrics.get('is_underexposed', False):
            issues.append("Image is underexposed")
        
        if metrics.get('has_low_contrast', False):
            issues.append("Image has low contrast")
        
        if metrics.get('has_color_artifacts', False):
            issues.append("Color channel artifacts detected")
        
        # Check for compression artifacts
        if metrics.get('high_freq_ratio', 0) < 0.1:
            issues.append("Possible compression artifacts")
        
        # Check for unusual texture patterns
        if metrics.get('texture_anisotropy', 0) > 50:
            issues.append("Unusual texture patterns detected")
        
        return issues
    
    def _calculate_entropy(self, data):
        """Calculate entropy of data for noise texture analysis"""
        hist, _ = np.histogram(data.flatten(), bins=64)
        hist = hist[hist > 0]  # Remove zeros
        prob = hist / np.sum(hist)
        return -np.sum(prob * np.log2(prob))
    
    def train_ml_classifier(self, training_data, labels):
        """
        Train machine learning classifiers on provided training data
        
        Args:
            training_data: List of images or feature dictionaries
            labels: List of quality labels (0=poor, 1=good)
        """
        print("🤖 Training ML classifiers for enhanced quality assessment...")
        
        # Extract features from training data
        if isinstance(training_data[0], dict):
            # Already extracted features
            features_list = training_data
        else:
            # Extract features from images
            features_list = []
            for image in training_data:
                analysis = self.analyze_image_quality(image)
                features_list.append(analysis['metrics'])
        
        # Create feature matrix
        feature_names = [
            'mad_noise', 'b_mean', 'effective_dynamic_range', 'laplacian_variance',
            'edge_density', 'noise_variation', 'texture_energy', 'local_contrast_mean'
        ]
        
        X = []
        for features in features_list:
            row = [features.get(name, 0) for name in feature_names]
            X.append(row)
        
        X = np.array(X)
        y = np.array(labels)
        
        # Train Random Forest classifier
        rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=5)
        rf_classifier.fit(X, y)
        
        # Train Logistic Regression classifier
        lr_pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', LogisticRegression(random_state=42, max_iter=1000))
        ])
        lr_pipeline.fit(X, y)
        
        # Store trained classifiers
        self.enhanced_classifiers['random_forest'] = {
            'model': rf_classifier,
            'feature_names': feature_names,
            'feature_importance': dict(zip(feature_names, rf_classifier.feature_importances_))
        }
        
        self.enhanced_classifiers['logistic_regression'] = {
            'model': lr_pipeline,
            'feature_names': feature_names
        }
        
        # Evaluate classifiers
        rf_pred = rf_classifier.predict(X)
        lr_pred = lr_pipeline.predict(X)
        
        print(f"Random Forest - Accuracy: {accuracy_score(y, rf_pred):.3f}, F1: {f1_score(y, rf_pred):.3f}")
        print(f"Logistic Regression - Accuracy: {accuracy_score(y, lr_pred):.3f}, F1: {f1_score(y, lr_pred):.3f}")
        
        return {
            'random_forest': {
                'accuracy': accuracy_score(y, rf_pred),
                'f1_score': f1_score(y, rf_pred),
                'feature_importance': self.enhanced_classifiers['random_forest']['feature_importance']
            },
            'logistic_regression': {
                'accuracy': accuracy_score(y, lr_pred),
                'f1_score': f1_score(y, lr_pred)
            }
        }
    
    def predict_quality_ml(self, image):
        """
        Predict image quality using trained ML classifiers
        
        Args:
            image: PIL Image or numpy array
            
        Returns:
            Dictionary with ML predictions
        """
        if not self.enhanced_classifiers:
            print("⚠️ No trained classifiers available. Use train_ml_classifier() first.")
            return {'error': 'No trained classifiers available'}
        
        # Extract features
        analysis = self.analyze_image_quality(image)
        features = analysis['metrics']
        
        results = {}
        
        # Random Forest prediction
        if 'random_forest' in self.enhanced_classifiers:
            rf_data = self.enhanced_classifiers['random_forest']
            feature_vector = [features.get(name, 0) for name in rf_data['feature_names']]
            
            rf_pred = rf_data['model'].predict([feature_vector])[0]
            rf_prob = rf_data['model'].predict_proba([feature_vector])[0]
            
            results['random_forest'] = {
                'prediction': int(rf_pred),
                'probability': float(rf_prob[1]),  # Probability of good quality
                'confidence': float(max(rf_prob))
            }
        
        # Logistic Regression prediction
        if 'logistic_regression' in self.enhanced_classifiers:
            lr_data = self.enhanced_classifiers['logistic_regression']
            feature_vector = [features.get(name, 0) for name in lr_data['feature_names']]
            
            lr_pred = lr_data['model'].predict([feature_vector])[0]
            lr_prob = lr_data['model'].predict_proba([feature_vector])[0]
            
            results['logistic_regression'] = {
                'prediction': int(lr_pred),
                'probability': float(lr_prob[1]),  # Probability of good quality
                'confidence': float(max(lr_prob))
            }
        
        return results


# Convenience functions for easy usage
def analyze_image_quality_enhanced(image_path):
    """
    Quick enhanced quality analysis of a single image
    
    Args:
        image_path: Path to image file or PIL Image
        
    Returns:
        Enhanced quality analysis results
    """
    analyzer = VisualQualityAnalyzer()
    
    if isinstance(image_path, str):
        image = Image.open(image_path).convert('RGB')
    else:
        image = image_path
    
    return analyzer.analyze_image_quality(image)


def batch_analyze_quality(image_paths, show_progress=True):
    """
    Analyze quality for multiple images with enhanced metrics
    
    Args:
        image_paths: List of image paths
        show_progress: Whether to show progress
        
    Returns:
        List of quality analysis results
    """
    analyzer = VisualQualityAnalyzer()
    results = []
    
    for i, path in enumerate(image_paths):
        if show_progress:
            print(f"Analyzing {i+1}/{len(image_paths)}: {path}")
        
        try:
            result = analyze_image_quality_enhanced(path)
            result['image_path'] = path
            results.append(result)
        except Exception as e:
            print(f"Error analyzing {path}: {e}")
            results.append({
                'image_path': path,
                'error': str(e),
                'quality_score': 0.0
            })
    
    return results


if __name__ == "__main__":
    # Example usage
    print("Enhanced Visual Quality Analyzer")
    print("Example usage:")
    print()
    print("from cv_metrics import analyze_image_quality_enhanced")
    print("result = analyze_image_quality_enhanced('image.png')")
    print("print(f'Quality score: {result[\"quality_score\"]:.3f}')")
    print("print(f'Issues: {result[\"issues\"]}')")