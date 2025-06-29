#!/usr/bin/env python3
"""
Standalone CLI tagger for Stable Diffusion images
Based on stable-diffusion-webui-wd14-tagger
"""

import argparse
import sys
from pathlib import Path
from PIL import Image
import json
import io
from typing import Dict, List, Tuple
from huggingface_hub import hf_hub_download
import numpy as np
from pandas import read_csv

# Image processing utilities
def fill_transparent(image: Image.Image, color='WHITE'):
    """Fill transparent areas with specified color"""
    image = image.convert('RGBA')
    new_image = Image.new('RGBA', image.size, color)
    new_image.paste(image, mask=image)
    image = new_image.convert('RGB')
    return image

def make_square(img, target_size):
    """Make an image square by padding"""
    old_size = img.shape[:2]
    desired_size = max(old_size)
    desired_size = max(desired_size, target_size)

    delta_w = desired_size - old_size[1]
    delta_h = desired_size - old_size[0]
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)

    color = [255, 255, 255]
    import cv2
    new_im = cv2.copyMakeBorder(
        img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color
    )
    return new_im

def smart_resize(img, size):
    """Resize an image intelligently"""
    import cv2
    if img.shape[0] > size:
        img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    elif img.shape[0] < size:
        img = cv2.resize(img, (size, size), interpolation=cv2.INTER_CUBIC)
    return img

class WaifuDiffusionTagger:
    """Waifu Diffusion tagger for anime images"""
    
    def __init__(self, model_name='wd14-vit-v2', threshold=0.35):
        self.model_name = model_name
        self.threshold = threshold
        self.model = None
        self.tags = None
        
        # Model configurations
        self.models = {
            'wd14-vit-v2': {
                'repo_id': 'SmilingWolf/wd-v1-4-vit-tagger-v2',
                'model_path': 'model.onnx',
                'tags_path': 'selected_tags.csv'
            },
            'wd14-convnext-v2': {
                'repo_id': 'SmilingWolf/wd-v1-4-convnext-tagger-v2',
                'model_path': 'model.onnx', 
                'tags_path': 'selected_tags.csv'
            },
            'wd14-swinv2-v1': {
                'repo_id': 'SmilingWolf/wd-v1-4-swinv2-tagger-v2',
                'model_path': 'model.onnx',
                'tags_path': 'selected_tags.csv'
            }
        }
        
    def load_model(self):
        """Load the ONNX model and tags"""
        if self.model_name not in self.models:
            raise ValueError(f"Model {self.model_name} not supported")
            
        config = self.models[self.model_name]
        
        # Download model and tags
        # print(f"Loading {self.model_name} model...")
        model_path = hf_hub_download(
            repo_id=config['repo_id'],
            filename=config['model_path']
        )
        tags_path = hf_hub_download(
            repo_id=config['repo_id'],
            filename=config['tags_path']
        )
        
        # Load ONNX model
        try:
            import onnxruntime as ort
        except ImportError:
            print("onnxruntime not found. Installing...")
            import subprocess
            subprocess.check_call([sys.executable, "-m", "pip", "install", "onnxruntime"])
            import onnxruntime as ort
            
        self.model = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
        
        # Load tags
        self.tags = read_csv(tags_path)
        # print(f"Loaded {len(self.tags)} tags")
        
    def preprocess_image(self, image: Image.Image) -> np.ndarray:
        """Preprocess image for inference"""
        # Get model input shape
        _, height, _, _ = self.model.get_inputs()[0].shape
        
        # Fill transparent areas
        image = fill_transparent(image)
        
        # Convert PIL to numpy array
        image = np.asarray(image)
        # PIL RGB to OpenCV BGR
        image = image[:, :, ::-1]
        
        # Make square and resize
        image = make_square(image, height)
        image = smart_resize(image, height)
        image = image.astype(np.float32)
        image = np.expand_dims(image, 0)
        
        return image
        
    def predict_tags(self, image: Image.Image) -> Tuple[Dict[str, float], Dict[str, float]]:
        """Predict tags for an image"""
        if self.model is None:
            self.load_model()
            
        # Preprocess image
        processed_image = self.preprocess_image(image)
        
        # Run inference
        input_name = self.model.get_inputs()[0].name
        output_name = self.model.get_outputs()[0].name
        predictions = self.model.run([output_name], {input_name: processed_image})[0]
        
        # Process predictions
        confidences = predictions[0]
        
        # Create tag dictionary
        tag_names = self.tags['name'].tolist()
        tag_confidences = dict(zip(tag_names, confidences))
        
        # Separate ratings and general tags
        ratings = {}
        tags = {}
        
        for i, (tag, conf) in enumerate(tag_confidences.items()):
            # First 4 are ratings: general, sensitive, questionable, explicit
            if i < 4:
                rating_names = ['general', 'sensitive', 'questionable', 'explicit']
                ratings[rating_names[i]] = float(conf)
            else:
                tags[tag] = float(conf)
                
        return ratings, tags
        
    def format_tags(self, tags: Dict[str, float], threshold: float = None) -> str:
        """Format tags above threshold as comma-separated string"""
        if threshold is None:
            threshold = self.threshold
            
        # Filter tags above threshold
        filtered_tags = {tag: conf for tag, conf in tags.items() 
                        if conf > threshold}
        
        # Sort by confidence
        sorted_tags = sorted(filtered_tags.items(), key=lambda x: x[1], reverse=True)
        
        # Format as tag names only
        tag_names = [tag.replace('_', ' ') for tag, _ in sorted_tags]
        
        return ', '.join(tag_names)

def main():
    parser = argparse.ArgumentParser(description='Tag anime images using WD14 tagger')
    parser.add_argument('image_path', help='Path to image file')
    parser.add_argument('--model', default='wd14-vit-v2', 
                       choices=['wd14-vit-v2', 'wd14-convnext-v2', 'wd14-swinv2-v1'],
                       help='Model to use for tagging')
    parser.add_argument('--threshold', type=float, default=0.35,
                       help='Confidence threshold for tags (default: 0.35)')
    parser.add_argument('--show-ratings', action='store_true',
                       help='Show content ratings')
    parser.add_argument('--show-confidence', action='store_true',
                       help='Show confidence scores with tags')
    
    args = parser.parse_args()
    
    # Check if image exists
    image_path = Path(args.image_path)
    if not image_path.exists():
        print(f"Error: Image file {image_path} not found", file=sys.stderr)
        sys.exit(1)
        
    try:
        # Load image
        image = Image.open(image_path)
        
        # Initialize tagger
        tagger = WaifuDiffusionTagger(
            model_name=args.model,
            threshold=args.threshold
        )
        
        # Predict tags
        ratings, tags = tagger.predict_tags(image)
        
        # Show ratings if requested
        if args.show_ratings:
            print("Ratings:")
            for rating, conf in ratings.items():
                print(f"  {rating}: {conf:.3f}")
            print()
            
        # Format and output tags
        if args.show_confidence:
            filtered_tags = {tag: conf for tag, conf in tags.items() 
                           if conf > args.threshold}
            sorted_tags = sorted(filtered_tags.items(), key=lambda x: x[1], reverse=True)
            tag_strings = [f"{tag.replace('_', ' ')}:{conf:.3f}" for tag, conf in sorted_tags]
            print(', '.join(tag_strings))
        else:
            formatted_tags = tagger.format_tags(tags)
            print(formatted_tags)
            
    except Exception as e:
        print(f"Error processing image: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == '__main__':
    main()
