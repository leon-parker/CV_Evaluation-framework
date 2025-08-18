"""
Optimized Visual Quality Analyzer using Random Forest
Based on research showing 100% accuracy on SDXL autism-friendly content
Uses the 11 most discriminative features identified in the study
"""

import cv2
import numpy as np
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib
import warnings
import os
warnings.filterwarnings('ignore')


class VisualQualityAnalyzer:
    """
    Random Forest-based visual quality analyzer optimized for SDXL autism content
    Based on research achieving 100% accuracy with 11 selected features
    """
    
    def __init__(self):
        # The 11 most discriminative features from research
        self.feature_names = [
            'blue_channel_mean',      # Most important (0.180 Gini)
            'laplacian_variance',     # Second most important (0.160 Gini)
            'mad_noise',             # Noise detection
            'edge_density',          # Structural integrity
            'red_channel_mean',      # Color artifacts
            'green_channel_mean',    # Color balance
            'effective_dynamic_range', # Exposure quality
            'rms_contrast',          # Contrast measurement
            'texture_energy',        # Texture analysis
            'gradient_magnitude',    # Sharpness measure
            'frequency_centroid'     # Frequency domain analysis
        ]
        
        # Initialize classifier (matches research parameters)
        self.classifier = RandomForestClassifier(
            n_estimators=100,
            max_depth=5,
            random_state=42,
            class_weight='balanced'  # Handle any class imbalance
        )
        
        self.scaler = StandardScaler()
        self.is_trained = False
        
        print("🌲 Optimized Random Forest Visual Quality Analyzer")
        print("📊 Using 11 research-validated features for SDXL autism content")
        
        # Try to automatically load your trained model
        self._try_load_pretrained_model()
    
    def _try_load_pretrained_model(self):
        """Try to load your trained model from common locations"""
        model_paths = [
            'autism_quality_model.joblib',  # Your trained model!
            'models/autism_quality_model.joblib', 
            '../autism_quality_model.joblib',
            './autism_quality_model.joblib'
        ]
        
        for model_path in model_paths:
            if os.path.exists(model_path):
                if self.load_model(model_path):
                    print(f"✅ Loaded your trained model from {model_path}")
                    print("🎯 Using 100% accuracy Random Forest from your validation!")
                    return True
        
        print("🔄 No pre-trained model found, using fallback thresholds")
        print("💡 Train a model with train_on_autism_dataset() for best results")
        return False
    
    def extract_optimized_features(self, image):
        """
        Extract the 11 most discriminative features identified in research
        
        Args:
            image: PIL Image or numpy array
            
        Returns:
            Dictionary with the 11 key features
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
        
        features = {}
        
        # 1. Blue channel mean (most discriminative - 0.180 Gini importance)
        features['blue_channel_mean'] = float(np.mean(cv_image[:,:,0]))  # OpenCV uses BGR
        
        # 2. Laplacian variance (second most discriminative - 0.160 Gini importance)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        features['laplacian_variance'] = float(np.var(laplacian))
        
        # 3. MAD noise (critical for artifact detection)
        median_blur = cv2.medianBlur(gray, 5)
        features['mad_noise'] = float(np.median(np.abs(gray.astype(np.float32) - median_blur.astype(np.float32))))
        
        # 4. Edge density (structural integrity)
        edges = cv2.Canny(gray, 50, 150)
        features['edge_density'] = float(np.sum(edges > 0) / edges.size)
        
        # 5. Red channel mean (color artifacts)
        features['red_channel_mean'] = float(np.mean(cv_image[:,:,2]))  # BGR format
        
        # 6. Green channel mean (color balance)
        features['green_channel_mean'] = float(np.mean(cv_image[:,:,1]))
        
        # 7. Effective dynamic range (exposure quality)
        p1, p99 = np.percentile(cv_image, [1, 99])
        features['effective_dynamic_range'] = float(p99 - p1)
        
        # 8. RMS contrast
        features['rms_contrast'] = float(np.std(gray))
        
        # 9. Texture energy (multi-directional analysis)
        sobel_h = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobel_v = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        features['texture_energy'] = float(np.mean([np.std(sobel_h), np.std(sobel_v)]))
        
        # 10. Gradient magnitude (sharpness)
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        features['gradient_magnitude'] = float(np.mean(np.sqrt(grad_x**2 + grad_y**2)))
        
        # 11. Frequency centroid (frequency domain analysis)
        f_transform = np.fft.fft2(gray)
        f_shift = np.fft.fftshift(f_transform)
        magnitude = np.abs(f_shift)
        
        h, w = gray.shape
        center_h, center_w = h//2, w//2
        
        # Calculate frequency centroid
        y, x = np.ogrid[:h, :w]
        r = np.sqrt((x - center_w)**2 + (y - center_h)**2)
        
        # Weight by magnitude and calculate centroid
        total_magnitude = np.sum(magnitude)
        if total_magnitude > 0:
            features['frequency_centroid'] = float(np.sum(r * magnitude) / total_magnitude)
        else:
            features['frequency_centroid'] = 0.0
        
        return features
    
    def analyze_image_quality(self, image):
        """
        Analyze image quality using the optimized Random Forest approach
        
        Args:
            image: PIL Image or numpy array
            
        Returns:
            Quality analysis results with confidence scores
        """
        # Extract the 11 key features
        features = self.extract_optimized_features(image)
        
        if not self.is_trained:
            # Use fallback scoring if not trained
            return self._fallback_quality_analysis(features)
        
        # Prepare feature vector
        feature_vector = np.array([[features[name] for name in self.feature_names]])
        
        # Scale features
        feature_vector_scaled = self.scaler.transform(feature_vector)
        
        # Get prediction and probabilities
        prediction = self.classifier.predict(feature_vector_scaled)[0]
        probabilities = self.classifier.predict_proba(feature_vector_scaled)[0]
        
        # Calculate quality score (probability of good quality)
        quality_score = float(probabilities[1]) if len(probabilities) > 1 else float(prediction)
        
        # Get feature importance for this prediction
        feature_importance = dict(zip(self.feature_names, self.classifier.feature_importances_))
        
        # Identify specific issues based on feature values and importance
        issues = self._identify_issues_rf(features, feature_importance)
        
        return {
            'quality_score': quality_score,
            'prediction': int(prediction),  # 0=poor, 1=good
            'confidence': float(max(probabilities)),
            'features': features,
            'feature_importance': feature_importance,
            'has_artifacts': prediction == 0,
            'issues': issues,
            'method': 'your_trained_random_forest' if self.is_trained else 'fallback_thresholds'
        }
    
    def train_on_autism_dataset(self, images, labels, validation_split=0.2):
        """
        Train the Random Forest classifier on autism-specific SDXL data
        
        Args:
            images: List of PIL Images or numpy arrays
            labels: List of quality labels (0=poor/artifacts, 1=good)
            validation_split: Fraction of data for validation
            
        Returns:
            Training results and performance metrics
        """
        print(f"🎯 Training Random Forest on {len(images)} autism-specific images...")
        
        # Extract features for all images
        X = []
        for i, image in enumerate(images):
            if i % 20 == 0:
                print(f"   Extracting features: {i+1}/{len(images)}")
            
            features = self.extract_optimized_features(image)
            feature_vector = [features[name] for name in self.feature_names]
            X.append(feature_vector)
        
        X = np.array(X)
        y = np.array(labels)
        
        # Split data
        n_train = int(len(X) * (1 - validation_split))
        indices = np.random.permutation(len(X))
        
        X_train, X_val = X[indices[:n_train]], X[indices[n_train:]]
        y_train, y_val = y[indices[:n_train]], y[indices[n_train:]]
        
        # Scale features
        self.scaler.fit(X_train)
        X_train_scaled = self.scaler.transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        
        # Train Random Forest
        print("   🌲 Training Random Forest classifier...")
        self.classifier.fit(X_train_scaled, y_train)
        self.is_trained = True
        
        # Evaluate on validation set
        if len(X_val) > 0:
            val_pred = self.classifier.predict(X_val_scaled)
            val_prob = self.classifier.predict_proba(X_val_scaled)
            
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
            
            accuracy = accuracy_score(y_val, val_pred)
            precision = precision_score(y_val, val_pred, zero_division=0)
            recall = recall_score(y_val, val_pred, zero_division=0)
            f1 = f1_score(y_val, val_pred, zero_division=0)
            
            print(f"   📊 Validation Results:")
            print(f"      Accuracy:  {accuracy:.3f}")
            print(f"      Precision: {precision:.3f}")
            print(f"      Recall:    {recall:.3f}")
            print(f"      F1-Score:  {f1:.3f}")
        else:
            accuracy = precision = recall = f1 = 0.0
        
        # Feature importance analysis
        feature_importance = dict(zip(self.feature_names, self.classifier.feature_importances_))
        
        print(f"   🎯 Top 5 Most Important Features:")
        sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
        for feature, importance in sorted_features[:5]:
            print(f"      {feature}: {importance:.3f}")
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'feature_importance': feature_importance,
            'n_train': len(X_train),
            'n_val': len(X_val)
        }
    
    def _fallback_quality_analysis(self, features):
        """Fallback analysis when classifier isn't trained"""
        
        # Use research-based thresholds for key features
        quality_score = 1.0
        issues = []
        
        # Blue channel mean (most important feature)
        blue_mean = features['blue_channel_mean']
        if blue_mean < 50 or blue_mean > 200:  # Extreme values indicate artifacts
            quality_score -= 0.3
            issues.append("Color channel artifacts detected (blue channel)")
        
        # Laplacian variance (sharpness)
        if features['laplacian_variance'] < 80:  # Research threshold
            quality_score -= 0.25
            issues.append("Image appears blurry (low Laplacian variance)")
        
        # MAD noise
        if features['mad_noise'] > 12:  # Research threshold
            quality_score -= 0.2
            issues.append("Excessive noise detected")
        
        # Edge density (structural integrity)
        if features['edge_density'] < 0.05:
            quality_score -= 0.15
            issues.append("Poor structural definition (low edge density)")
        
        # Dynamic range
        if features['effective_dynamic_range'] < 150:
            quality_score -= 0.1
            issues.append("Limited dynamic range")
        
        quality_score = max(0.0, quality_score)
        
        return {
            'quality_score': quality_score,
            'prediction': 1 if quality_score >= 0.7 else 0,
            'confidence': 0.8,  # Lower confidence without ML
            'features': features,
            'has_artifacts': quality_score < 0.7,
            'issues': issues,
            'method': 'fallback_thresholds'
        }
    
    def _identify_issues_rf(self, features, feature_importance):
        """Identify specific issues using Random Forest insights"""
        issues = []
        
        # Use feature importance to weight issue detection
        top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:5]
        
        for feature_name, importance in top_features:
            value = features[feature_name]
            
            # Issue detection based on feature values and importance
            if feature_name == 'blue_channel_mean' and importance > 0.1:
                if value < 50 or value > 200:
                    issues.append(f"Color artifacts (blue channel: {value:.1f})")
            
            elif feature_name == 'laplacian_variance' and importance > 0.1:
                if value < 80:
                    issues.append(f"Blur detected (Laplacian: {value:.1f})")
            
            elif feature_name == 'mad_noise' and importance > 0.1:
                if value > 12:
                    issues.append(f"Noise artifacts (MAD: {value:.1f})")
            
            elif feature_name == 'edge_density' and importance > 0.1:
                if value < 0.05:
                    issues.append(f"Poor structure (edge density: {value:.3f})")
        
        return issues
    
    def save_model(self, filepath):
        """Save trained model and scaler"""
        if not self.is_trained:
            print("⚠️ Model not trained yet")
            return False
        
        model_data = {
            'classifier': self.classifier,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'is_trained': self.is_trained
        }
        
        joblib.dump(model_data, filepath)
        print(f"💾 Model saved to {filepath}")
        return True
    
    def load_model(self, filepath):
        """Load pre-trained model and scaler"""
        try:
            model_data = joblib.load(filepath)
            self.classifier = model_data['classifier']
            self.scaler = model_data['scaler']
            self.feature_names = model_data['feature_names']
            self.is_trained = model_data['is_trained']
            
            print(f"📂 Model loaded from {filepath}")
            return True
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            return False


# Convenience function for quick analysis
def analyze_quality_optimized(image_path):
    """
    Quick optimized quality analysis using Random Forest approach
    
    Args:
        image_path: Path to image or PIL Image
        
    Returns:
        Quality analysis results
    """
    analyzer = VisualQualityAnalyzer()
    
    if isinstance(image_path, str):
        image = Image.open(image_path).convert('RGB')
    else:
        image = image_path
    
    return analyzer.analyze_image_quality(image)


if __name__ == "__main__":
    print("Random Forest Visual Quality Analyzer")
    print("Based on research achieving 100% accuracy on SDXL autism content")
    print()
    print("Usage:")
    print("analyzer = VisualQualityAnalyzer()")
    print("result = analyzer.analyze_image_quality(image)")
    print("print(f'Quality: {result[\"quality_score\"]:.3f}')")