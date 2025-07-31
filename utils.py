"""
Shared utilities for the autism evaluation framework
Includes image processing, report generation, and visualization tools
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
from datetime import datetime
import cv2


class ImageUtils:
    """Image processing and manipulation utilities"""
    
    @staticmethod
    def load_image(image_path):
        """Load image from various formats"""
        if isinstance(image_path, str) or isinstance(image_path, Path):
            return Image.open(image_path).convert('RGB')
        elif isinstance(image_path, Image.Image):
            return image_path.convert('RGB')
        elif isinstance(image_path, np.ndarray):
            return Image.fromarray(image_path).convert('RGB')
        else:
            raise ValueError(f"Unsupported image type: {type(image_path)}")
    
    @staticmethod
    def resize_for_display(image, max_size=(800, 800)):
        """Resize image for display while maintaining aspect ratio"""
        image.thumbnail(max_size, Image.Resampling.LANCZOS)
        return image
    
    @staticmethod
    def create_image_grid(images, grid_size=None, padding=10):
        """Create a grid of images for comparison"""
        n_images = len(images)
        
        if grid_size is None:
            cols = int(np.ceil(np.sqrt(n_images)))
            rows = int(np.ceil(n_images / cols))
        else:
            rows, cols = grid_size
        
        # Ensure all images are same size
        max_width = max(img.width for img in images)
        max_height = max(img.height for img in images)
        
        # Create grid
        grid_width = cols * max_width + (cols + 1) * padding
        grid_height = rows * max_height + (rows + 1) * padding
        
        grid = Image.new('RGB', (grid_width, grid_height), color='white')
        
        for idx, img in enumerate(images):
            row = idx // cols
            col = idx % cols
            
            x = col * max_width + (col + 1) * padding
            y = row * max_height + (row + 1) * padding
            
            grid.paste(img, (x, y))
        
        return grid


class ReportGenerator:
    """Generate evaluation reports in various formats"""
    
    @staticmethod
    def generate_text_report(results, save_path=None):
        """Generate detailed text report"""
        report = []
        report.append("AUTISM STORYBOARD EVALUATION REPORT")
        report.append("=" * 60)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # Overall scores
        report.append("OVERALL ASSESSMENT")
        report.append("-" * 30)
        report.append(f"Combined Autism Score: {results['combined_score']:.3f}")
        report.append(f"Grade: {results['autism_grade']}")
        report.append(f"Image: {results.get('image_name', 'N/A')}")
        report.append(f"Prompt: {results.get('prompt', 'N/A')}")
        report.append("")
        
        # Individual metrics
        report.append("DETAILED METRICS")
        report.append("-" * 30)
        
        scores = results.get('scores', {})
        for metric, score in scores.items():
            report.append(f"{metric.replace('_', ' ').title()}: {score:.3f}")
        
        report.append("")
        
        # Autism-specific analysis
        if 'metrics' in results and 'complexity' in results['metrics']:
            complexity = results['metrics']['complexity']
            
            report.append("AUTISM-SPECIFIC ANALYSIS")
            report.append("-" * 30)
            
            # Person count
            person_data = complexity.get('person_count', {})
            report.append(f"Person Count: {person_data.get('count', 'N/A')} "
                         f"({'✓' if person_data.get('is_compliant') else '✗'})")
            
            # Background
            bg_data = complexity.get('background_simplicity', {})
            report.append(f"Background Simplicity: {bg_data.get('score', 0):.3f} "
                         f"({'Simple' if bg_data.get('is_simple') else 'Complex'})")
            
            # Colors
            color_data = complexity.get('color_appropriateness', {})
            report.append(f"Color Count: {color_data.get('dominant_colors', 'N/A')} colors")
            report.append(f"Color Appropriateness: {color_data.get('score', 0):.3f}")
            
            report.append("")
        
        # Recommendations
        report.append("RECOMMENDATIONS")
        report.append("-" * 30)
        for i, rec in enumerate(results.get('recommendations', []), 1):
            report.append(f"{i}. {rec}")
        
        # Save if requested
        if save_path:
            with open(save_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(report))
        
        return '\n'.join(report)
    
    @staticmethod
    def generate_json_report(results, save_path):
        """Save results as JSON for further analysis"""
        # Convert numpy types to Python types
        def convert_types(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.float32, np.float64)):
                return float(obj)
            elif isinstance(obj, (np.int32, np.int64)):
                return int(obj)
            elif isinstance(obj, dict):
                return {key: convert_types(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_types(item) for item in obj]
            return obj
        
        clean_results = convert_types(results)
        
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(clean_results, f, indent=2)
        
        return save_path


class VisualizationTools:
    """Create visual reports and comparisons"""
    
    @staticmethod
    def create_evaluation_dashboard(results, image, save_path=None):
        """Create a visual dashboard of evaluation results"""
        fig = plt.figure(figsize=(16, 10))
        gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
        
        # Display image
        ax_img = fig.add_subplot(gs[0:2, 0:2])
        if isinstance(image, str):
            image = Image.open(image)
        ax_img.imshow(image)
        ax_img.axis('off')
        ax_img.set_title(f"Evaluated Image\nGrade: {results['autism_grade']}", 
                        fontsize=14, fontweight='bold')
        
        # Overall score gauge
        ax_gauge = fig.add_subplot(gs[0, 2])
        VisualizationTools._create_gauge_chart(
            ax_gauge, 
            results['combined_score'], 
            "Autism Score"
        )
        
        # Metric breakdown
        ax_metrics = fig.add_subplot(gs[1, 2:4])
        scores = results.get('scores', {})
        if scores:
            metrics = list(scores.keys())
            values = list(scores.values())
            
            y_pos = np.arange(len(metrics))
            colors = ['green' if v >= 0.7 else 'orange' if v >= 0.5 else 'red' 
                     for v in values]
            
            bars = ax_metrics.barh(y_pos, values, color=colors, alpha=0.7)
            ax_metrics.set_yticks(y_pos)
            ax_metrics.set_yticklabels([m.replace('_', ' ').title() for m in metrics])
            ax_metrics.set_xlabel('Score')
            ax_metrics.set_xlim(0, 1)
            ax_metrics.set_title('Metric Breakdown', fontweight='bold')
            ax_metrics.grid(axis='x', alpha=0.3)
            
            # Add value labels
            for bar, value in zip(bars, values):
                ax_metrics.text(value + 0.01, bar.get_y() + bar.get_height()/2, 
                               f'{value:.2f}', va='center')
        
        # Autism-specific details
        ax_details = fig.add_subplot(gs[2, :])
        details_text = VisualizationTools._format_autism_details(results)
        ax_details.text(0.05, 0.95, details_text, transform=ax_details.transAxes,
                       verticalalignment='top', fontfamily='monospace',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        ax_details.axis('off')
        
        plt.suptitle('Autism Storyboard Evaluation Dashboard', 
                    fontsize=16, fontweight='bold')
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig
    
    @staticmethod
    def _create_gauge_chart(ax, value, title):
        """Create a gauge/speedometer chart"""
        # Create semicircle
        theta = np.linspace(0, np.pi, 100)
        radius = 1
        
        # Color zones
        colors = ['red', 'orange', 'yellow', 'lightgreen', 'green']
        zone_boundaries = [0, 0.4, 0.55, 0.7, 0.85, 1.0]
        
        for i in range(len(colors)):
            start_angle = zone_boundaries[i] * np.pi
            end_angle = zone_boundaries[i+1] * np.pi
            theta_zone = np.linspace(start_angle, end_angle, 20)
            
            x_outer = radius * np.cos(theta_zone)
            y_outer = radius * np.sin(theta_zone)
            x_inner = 0.7 * radius * np.cos(theta_zone)
            y_inner = 0.7 * radius * np.sin(theta_zone)
            
            vertices = list(zip(x_outer, y_outer)) + list(zip(x_inner[::-1], y_inner[::-1]))
            poly = plt.Polygon(vertices, color=colors[i], alpha=0.7)
            ax.add_patch(poly)
        
        # Add needle
        needle_angle = value * np.pi
        needle_x = 0.9 * radius * np.cos(needle_angle)
        needle_y = 0.9 * radius * np.sin(needle_angle)
        ax.plot([0, needle_x], [0, needle_y], 'k-', linewidth=3)
        ax.plot(0, 0, 'ko', markersize=10)
        
        # Formatting
        ax.set_xlim(-1.2, 1.2)
        ax.set_ylim(-0.1, 1.2)
        ax.set_aspect('equal')
        ax.axis('off')
        ax.text(0, -0.2, f'{value:.2f}', ha='center', fontsize=16, fontweight='bold')
        ax.set_title(title, fontsize=12, fontweight='bold', pad=20)
    
    @staticmethod
    def _format_autism_details(results):
        """Format autism-specific details for display"""
        details = []
        
        if 'metrics' in results and 'complexity' in results['metrics']:
            complexity = results['metrics']['complexity']
            
            # Person count
            person_count = complexity.get('person_count', {}).get('count', 'N/A')
            details.append(f"People in image: {person_count}")
            
            # Background
            bg_objects = complexity.get('background_simplicity', {}).get('object_count', 'N/A')
            details.append(f"Background objects: {bg_objects}")
            
            # Colors
            color_count = complexity.get('color_appropriateness', {}).get('dominant_colors', 'N/A')
            details.append(f"Dominant colors: {color_count}")
            
            # Key recommendations
            details.append("\nTop Recommendations:")
            for rec in results.get('recommendations', [])[:3]:
                details.append(f"• {rec}")
        
        return '\n'.join(details)
    
    @staticmethod
    def create_sequence_comparison(sequence_results, save_path=None):
        """Create visual comparison for sequence evaluation"""
        n_images = sequence_results['num_images']
        
        fig, axes = plt.subplots(2, n_images, figsize=(4*n_images, 8))
        if n_images == 1:
            axes = axes.reshape(2, 1)
        
        # Display images and scores
        for i, img_result in enumerate(sequence_results['image_results']):
            # Image
            ax_img = axes[0, i]
            # Note: Would need actual image data here
            ax_img.text(0.5, 0.5, f"Image {i+1}", ha='center', va='center')
            ax_img.set_title(f"Score: {img_result['combined_score']:.2f}")
            ax_img.axis('off')
            
            # Metrics bar
            ax_bar = axes[1, i]
            metrics = ['Visual', 'Prompt', 'Person', 'Background', 'Color']
            scores = [
                img_result['scores'].get('visual_quality', 0),
                img_result['scores'].get('prompt_faithfulness', 0),
                img_result['scores'].get('person_count', 0),
                img_result['scores'].get('background_simplicity', 0),
                img_result['scores'].get('color_appropriateness', 0)
            ]
            
            colors = ['green' if s >= 0.7 else 'orange' if s >= 0.5 else 'red' 
                     for s in scores]
            ax_bar.bar(metrics, scores, color=colors, alpha=0.7)
            ax_bar.set_ylim(0, 1)
            ax_bar.set_ylabel('Score')
            ax_bar.tick_params(axis='x', rotation=45)
        
        plt.suptitle(f"Storyboard Sequence Evaluation\n"
                    f"Overall Score: {sequence_results['overall_score']:.2f} "
                    f"({sequence_results['overall_grade']})",
                    fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig