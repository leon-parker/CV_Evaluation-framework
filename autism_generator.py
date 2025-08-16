#!/usr/bin/env python3
"""
Autism-Aware Storyboard Generator
Generates multiple versions, scores with autism evaluation framework, refines the best
Works alongside your existing autism_evaluator.py
"""

import os
import torch
import gc
from PIL import Image
from diffusers import StableDiffusionXLPipeline, StableDiffusionXLImg2ImgPipeline
from diffusers import EulerAncestralDiscreteScheduler
from compel import Compel, ReturnedEmbeddingsType
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import json
from datetime import datetime
import numpy as np

# Import your existing autism evaluation framework
from autism_evaluator import AutismStoryboardEvaluator
from evaluation_config import AUTISM_GUIDELINES


class AutismStoryboardGenerator:
    """
    Multi-generation system that creates autism-appropriate storyboards
    1. Generate 3 versions of each scene
    2. Score each with your autism evaluation framework
    3. Refine the best one using SDXL Refiner
    """
    
    def __init__(self, 
                 base_model_path: str,
                 refiner_model_path: Optional[str] = None,
                 device: str = "cuda" if torch.cuda.is_available() else "cpu",
                 verbose: bool = True):
        """
        Initialize the autism-aware generator
        
        Args:
            base_model_path: Path to base SDXL model (e.g., RealCartoon XL v7)
            refiner_model_path: Optional path to SDXL refiner model
            device: Computing device
            verbose: Whether to print detailed progress
        """
        self.base_model_path = base_model_path
        self.refiner_model_path = refiner_model_path
        self.device = device
        self.verbose = verbose
        
        # Generation components
        self.base_pipeline = None
        self.refiner_pipeline = None
        self.compel = None
        
        # Your autism evaluation framework
        self.autism_evaluator = None
        
        # Generation settings optimized for autism education
        self.generation_config = {
            'height': 1024,
            'width': 1024,
            'num_inference_steps': 25,
            'guidance_scale': 8.0,
            'refiner_strength': 0.3,  # Gentle refinement
            'refiner_steps': 15,
            'num_versions': 3  # Generate 3 versions to choose from
        }
        
        if self.verbose:
            print("🧩 Autism Storyboard Generator")
            print(f"   Base Model: {Path(base_model_path).name}")
            print(f"   Refiner: {'Available' if refiner_model_path else 'Not specified'}")
            print(f"   Device: {device}")
    
    def setup(self):
        """Initialize all pipelines and evaluators"""
        if self.verbose:
            print("\n🚀 Setting up generation and evaluation systems...")
        
        # Setup base generation pipeline
        if not self._setup_base_pipeline():
            return False
            
        # Setup refiner pipeline (optional)
        if self.refiner_model_path:
            self._setup_refiner_pipeline()
            
        # Setup autism evaluator (your existing framework)
        self._setup_autism_evaluator()
        
        if self.verbose:
            print("✅ All systems ready!")
        
        return True
    
    def _setup_base_pipeline(self):
        """Setup the base SDXL generation pipeline"""
        try:
            if self.verbose:
                print("   🎨 Loading base SDXL pipeline...")
            
            self.base_pipeline = StableDiffusionXLPipeline.from_single_file(
                self.base_model_path,
                torch_dtype=torch.float16,
                use_safetensors=True,
                safety_checker=None,
                requires_safety_checker=False
            ).to(self.device)
            
            self.base_pipeline.scheduler = EulerAncestralDiscreteScheduler.from_config(
                self.base_pipeline.scheduler.config
            )
            
            self.base_pipeline.enable_vae_tiling()
            self.base_pipeline.enable_model_cpu_offload()
            
            # Setup Compel for prompt weighting
            self.compel = Compel(
                tokenizer=[self.base_pipeline.tokenizer, self.base_pipeline.tokenizer_2],
                text_encoder=[self.base_pipeline.text_encoder, self.base_pipeline.text_encoder_2],
                returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED,
                requires_pooled=[False, True],
                truncate_long_prompts=False
            )
            
            return True
            
        except Exception as e:
            print(f"   ❌ Failed to setup base pipeline: {e}")
            return False
    
    def _setup_refiner_pipeline(self):
        """Setup SDXL refiner pipeline"""
        try:
            if self.verbose:
                print("   ✨ Loading SDXL refiner...")
                
            self.refiner_pipeline = StableDiffusionXLImg2ImgPipeline.from_single_file(
                self.refiner_model_path,
                torch_dtype=torch.float16,
                use_safetensors=True
            ).to(self.device)
            
            self.refiner_pipeline.enable_vae_tiling()
            
            return True
            
        except Exception as e:
            print(f"   ⚠️ Refiner setup failed: {e}")
            return False
    
    def _setup_autism_evaluator(self):
        """Setup your autism evaluation framework"""
        try:
            if self.verbose:
                print("   🧩 Loading autism evaluation framework...")
                
            self.autism_evaluator = AutismStoryboardEvaluator(
                device=self.device,
                verbose=False  # Keep it quiet during generation
            )
            
            return True
            
        except Exception as e:
            print(f"   ⚠️ Autism evaluator setup failed: {e}")
            return False
    
    def generate_autism_appropriate_image(self, 
                                        prompt: str,
                                        negative_prompt: str = None,
                                        output_dir: str = "generated_images",
                                        save_all_versions: bool = False) -> Dict:
        """
        Generate an autism-appropriate image using multi-generation and scoring
        
        Args:
            prompt: Text prompt for generation
            negative_prompt: Negative prompt
            output_dir: Directory to save results
            save_all_versions: Whether to save all 3 versions or just the best
            
        Returns:
            Dictionary with generation results and scores
        """
        if self.verbose:
            print(f"\n🎨 Generating autism-appropriate image")
            print(f"   Prompt: {prompt[:60]}...")
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Enhance prompts for autism education
        autism_prompt = self._enhance_prompt_for_autism(prompt)
        autism_negative = self._get_autism_negative_prompt(negative_prompt)
        
        if self.verbose:
            print(f"   Enhanced prompt: {autism_prompt[:60]}...")
        
        # Generate 3 versions
        versions = []
        for i in range(self.generation_config['num_versions']):
            if self.verbose:
                print(f"\n   🖼️ Generating version {i+1}/3...")
            
            # Use different seeds for variety
            seed = 42 + i * 1000
            
            try:
                image = self._generate_single_image(
                    autism_prompt, 
                    autism_negative, 
                    seed
                )
                
                if image:
                    # Save version
                    version_path = Path(output_dir) / f"version_{i+1}.png"
                    if save_all_versions:
                        image.save(version_path)
                    
                    versions.append({
                        'image': image,
                        'path': str(version_path) if save_all_versions else None,
                        'seed': seed,
                        'version': i + 1
                    })
                    
                    if self.verbose:
                        print(f"      ✅ Version {i+1} generated")
                else:
                    if self.verbose:
                        print(f"      ❌ Version {i+1} failed")
                        
            except Exception as e:
                print(f"      ❌ Version {i+1} error: {e}")
        
        if not versions:
            return {'error': 'Failed to generate any versions'}
        
        # Score all versions with autism framework
        if self.verbose:
            print(f"\n🧩 Scoring versions for autism appropriateness...")
        
        scored_versions = []
        for version in versions:
            try:
                # Create simple prompt for BLIP evaluation
                simple_prompt = self._simplify_prompt_for_blip(prompt)
                
                # Score with your autism framework
                score_result = self.autism_evaluator.evaluate_single_image(
                    image=version['image'],
                    prompt=simple_prompt,
                    save_report=False  # Don't save reports for versions
                )
                
                version['autism_score'] = score_result['combined_score']
                version['autism_grade'] = score_result['autism_grade']
                version['category_scores'] = score_result['category_scores']
                version['recommendations'] = score_result['recommendations']
                
                scored_versions.append(version)
                
                if self.verbose:
                    print(f"      Version {version['version']}: {version['autism_score']:.3f} - {version['autism_grade']}")
                    
            except Exception as e:
                print(f"      ⚠️ Scoring failed for version {version['version']}: {e}")
                version['autism_score'] = 0.0
                scored_versions.append(version)
        
        # Find best version
        best_version = max(scored_versions, key=lambda x: x['autism_score'])
        
        if self.verbose:
            print(f"\n🏆 Best version: {best_version['version']} (score: {best_version['autism_score']:.3f})")
        
        # Refine the best version if refiner available
        final_image = best_version['image']
        refined = False
        
        if self.refiner_pipeline and best_version['autism_score'] < 0.9:
            if self.verbose:
                print("   ✨ Refining best version...")
            
            try:
                refined_image = self._refine_image(
                    best_version['image'], 
                    autism_prompt
                )
                
                if refined_image:
                    # Quick score check for refined version
                    simple_prompt = self._simplify_prompt_for_blip(prompt)
                    refined_score = self.autism_evaluator.evaluate_single_image(
                        image=refined_image,
                        prompt=simple_prompt,
                        save_report=False
                    )
                    
                    # Use refined if it's better
                    if refined_score['combined_score'] > best_version['autism_score']:
                        final_image = refined_image
                        refined = True
                        best_version['autism_score'] = refined_score['combined_score']
                        best_version['autism_grade'] = refined_score['autism_grade']
                        
                        if self.verbose:
                            print(f"      ✅ Refinement improved score to {refined_score['combined_score']:.3f}")
                    else:
                        if self.verbose:
                            print(f"      ⚠️ Refinement didn't improve score, keeping original")
                            
            except Exception as e:
                print(f"      ⚠️ Refinement failed: {e}")
        
        # Save final result
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        final_path = Path(output_dir) / f"autism_appropriate_{timestamp}.png"
        final_image.save(final_path)
        
        # Generate final evaluation report
        simple_prompt = self._simplify_prompt_for_blip(prompt)
        final_evaluation = self.autism_evaluator.evaluate_single_image(
            image=final_image,
            prompt=simple_prompt,
            save_report=True,
            output_dir=output_dir
        )
        
        # Clean up GPU memory
        self._cleanup_memory()
        
        return {
            'final_image_path': str(final_path),
            'final_image': final_image,
            'final_score': best_version['autism_score'],
            'final_grade': best_version['autism_grade'],
            'final_evaluation': final_evaluation,
            'all_versions': scored_versions,
            'best_version_number': best_version['version'],
            'was_refined': refined,
            'generation_config': self.generation_config,
            'prompt_used': autism_prompt,
            'negative_prompt_used': autism_negative
        }
    
    def _generate_single_image(self, prompt: str, negative_prompt: str, seed: int) -> Optional[Image.Image]:
        """Generate a single image"""
        try:
            # Encode prompts with Compel
            positive_conditioning, positive_pooled = self.compel(prompt)
            negative_conditioning, negative_pooled = self.compel(negative_prompt)
            
            # Pad to same length
            [positive_conditioning, negative_conditioning] = self.compel.pad_conditioning_tensors_to_same_length(
                [positive_conditioning, negative_conditioning]
            )
            
            # Generate
            result = self.base_pipeline(
                prompt_embeds=positive_conditioning,
                pooled_prompt_embeds=positive_pooled,
                negative_prompt_embeds=negative_conditioning,
                negative_pooled_prompt_embeds=negative_pooled,
                height=self.generation_config['height'],
                width=self.generation_config['width'],
                num_inference_steps=self.generation_config['num_inference_steps'],
                guidance_scale=self.generation_config['guidance_scale'],
                generator=torch.Generator(self.device).manual_seed(seed)
            )
            
            return result.images[0] if result and result.images else None
            
        except Exception as e:
            print(f"Generation error: {e}")
            return None
    
    def _refine_image(self, image: Image.Image, prompt: str) -> Optional[Image.Image]:
        """Refine image using SDXL refiner"""
        try:
            result = self.refiner_pipeline(
                prompt=prompt,
                image=image,
                strength=self.generation_config['refiner_strength'],
                num_inference_steps=self.generation_config['refiner_steps']
            )
            
            return result.images[0] if result and result.images else None
            
        except Exception as e:
            print(f"Refinement error: {e}")
            return None
    
    def _enhance_prompt_for_autism(self, prompt: str) -> str:
        """Enhance prompt with autism-friendly elements"""
        autism_enhancements = [
            "cartoon style",
            "simple clean background", 
            "clear character definition",
            "bright friendly lighting",
            "minimal objects",
            "one character" if "character" in prompt.lower() or "person" in prompt.lower() or "boy" in prompt.lower() or "girl" in prompt.lower() else ""
        ]
        
        # Add enhancements that aren't already in prompt
        enhancements_to_add = []
        prompt_lower = prompt.lower()
        
        for enhancement in autism_enhancements:
            if enhancement and not any(word in prompt_lower for word in enhancement.split()):
                enhancements_to_add.append(enhancement)
        
        if enhancements_to_add:
            enhanced = f"{prompt}, {', '.join(enhancements_to_add)}"
        else:
            enhanced = prompt
            
        return enhanced
    
    def _get_autism_negative_prompt(self, negative_prompt: str = None) -> str:
        """Get negative prompt optimized for autism education"""
        base_negative = [
            "multiple people", "crowd", "many characters",
            "complex background", "cluttered", "busy scene",
            "too many objects", "detailed background",
            "realistic photo", "photorealistic",
            "dark", "scary", "frightening",
            "too many colors", "rainbow", "colorful patterns",
            "stripes", "checkerboard", "spiral patterns",
            "blurry", "low quality", "artifacts"
        ]
        
        if negative_prompt:
            return f"{negative_prompt}, {', '.join(base_negative)}"
        else:
            return ', '.join(base_negative)
    
    def _simplify_prompt_for_blip(self, prompt: str) -> str:
        """Simplify prompt for better BLIP evaluation"""
        # Extract key nouns and actions
        key_words = []
        words = prompt.lower().split()
        
        # Key autism education terms
        important_terms = [
            'boy', 'girl', 'child', 'person', 'character',
            'brushing', 'teeth', 'eating', 'breakfast', 'lunch', 'dinner',
            'washing', 'hands', 'bathroom', 'kitchen', 'bedroom',
            'reading', 'book', 'playing', 'toy', 'ball',
            'sitting', 'standing', 'walking', 'running'
        ]
        
        for word in words:
            clean_word = word.strip('.,!?;:')
            if clean_word in important_terms:
                key_words.append(clean_word)
        
        # Fallback to first few words if no key terms found
        if not key_words:
            key_words = words[:4]
        
        return ' '.join(key_words[:5])  # Max 5 words for BLIP
    
    def _cleanup_memory(self):
        """Clean up GPU memory"""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def cleanup(self):
        """Clean up all resources"""
        if self.base_pipeline:
            del self.base_pipeline
        if self.refiner_pipeline:
            del self.refiner_pipeline
        if self.compel:
            del self.compel
        
        self._cleanup_memory()


# Convenience function for easy usage
def generate_autism_storyboard_image(base_model_path: str, 
                                   prompt: str,
                                   refiner_model_path: str = None,
                                   output_dir: str = "autism_generated") -> Dict:
    """
    Quick function to generate autism-appropriate image
    
    Args:
        base_model_path: Path to base SDXL model
        prompt: Text prompt
        refiner_model_path: Optional refiner model path
        output_dir: Output directory
        
    Returns:
        Generation results
    """
    generator = AutismStoryboardGenerator(
        base_model_path=base_model_path,
        refiner_model_path=refiner_model_path,
        verbose=True
    )
    
    if not generator.setup():
        return {'error': 'Failed to setup generator'}
    
    try:
        result = generator.generate_autism_appropriate_image(
            prompt=prompt,
            output_dir=output_dir
        )
        return result
    finally:
        generator.cleanup()


if __name__ == "__main__":
    print("🧩 Autism Storyboard Generator")
    print("Generates multiple versions, scores with autism framework, refines the best")
    print("\nExample usage:")
    print("generator = AutismStoryboardGenerator('path/to/realcartoon_xl_v7.safetensors')")
    print("generator.setup()")
    print("result = generator.generate_autism_appropriate_image('boy brushing teeth')")
    print("print(f'Final score: {result[\"final_score\"]:.3f}')")