#!/usr/bin/env python3
"""
Roof Texture Classification Inference Script
============================================

This script loads a trained roof texture classification model and performs
inference on new image patches.

Usage:
    python inference.py --model_path best_roof_texture_model.pth --image_path path/to/image.jpg
    python inference.py --model_path best_roof_texture_model.pth --batch_dir path/to/images/
"""

import argparse
import os
import glob
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import EfficientNet_B0_Weights
import matplotlib.pyplot as plt
from collections import Counter


class RoofTextureClassifier(nn.Module):
    """Custom CNN model for roof texture classification"""
    def __init__(self, num_classes=4, dropout_rate=0.5):
        super(RoofTextureClassifier, self).__init__()
        
        # Use EfficientNet-B0 as backbone
        self.backbone = models.efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
        
        # Modify classifier head
        num_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate, inplace=True),
            nn.Linear(num_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate, inplace=True),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, x):
        return self.backbone(x)


class RoofTextureInference:
    """Inference class for roof texture classification"""
    
    def __init__(self, model_path, device=None):
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.class_names = ['smooth', 'average', 'rough', 'no_contour']
        
        # Load model
        self.model = RoofTextureClassifier(num_classes=4)
        self.load_model(model_path)
        self.model.to(self.device)
        self.model.eval()
        
        # Setup transforms
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        print(f"✅ Model loaded successfully on {self.device}")
        print(f"📊 Classes: {self.class_names}")
    
    def load_model(self, model_path):
        """Load trained model weights"""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        print(f"📁 Model loaded from {model_path}")
        print(f"   Epoch: {checkpoint['epoch']}")
        print(f"   Loss: {checkpoint['loss']:.4f}")
        print(f"   Accuracy: {checkpoint['accuracy']:.2f}%")
    
    def preprocess_image(self, image_path):
        """Preprocess a single image"""
        try:
            # Load and convert image
            image = Image.open(image_path).convert('RGB')
            
            # Apply transforms
            image_tensor = self.transform(image).unsqueeze(0)
            
            return image_tensor, image
            
        except Exception as e:
            print(f"❌ Error preprocessing {image_path}: {str(e)}")
            return None, None
    
    def predict_single(self, image_path, show_image=False):
        """Predict class for a single image"""
        image_tensor, original_image = self.preprocess_image(image_path)
        
        if image_tensor is None:
            return None
        
        # Make prediction
        with torch.no_grad():
            image_tensor = image_tensor.to(self.device)
            output = self.model(image_tensor)
            probabilities = torch.softmax(output, dim=1)
            
            predicted_class = torch.argmax(output, dim=1).item()
            confidence = probabilities[0][predicted_class].item()
            
            # Get all class probabilities
            class_probs = probabilities[0].cpu().numpy()
        
        result = {
            'image_path': image_path,
            'predicted_class': self.class_names[predicted_class],
            'confidence': confidence,
            'class_probabilities': dict(zip(self.class_names, class_probs))
        }
        
        # Display results
        print(f"\n🔍 Prediction for: {os.path.basename(image_path)}")
        print(f"   Predicted: {result['predicted_class']}")
        print(f"   Confidence: {confidence:.4f}")
        print(f"   All probabilities:")
        for class_name, prob in result['class_probabilities'].items():
            print(f"     {class_name}: {prob:.4f}")
        
        # Show image if requested
        if show_image:
            self._display_prediction(original_image, result)
        
        return result
    
    def predict_batch(self, image_dir, show_images=False, max_images=10):
        """Predict classes for multiple images in a directory"""
        # Find all image files
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
        image_files = []
        
        for ext in image_extensions:
            image_files.extend(glob.glob(os.path.join(image_dir, ext)))
            image_files.extend(glob.glob(os.path.join(image_dir, ext.upper())))
        
        if not image_files:
            print(f"❌ No image files found in {image_dir}")
            return []
        
        print(f"📁 Found {len(image_files)} images in {image_dir}")
        
        # Limit number of images to process
        if len(image_files) > max_images:
            print(f"⚠️  Limiting to first {max_images} images")
            image_files = image_files[:max_images]
        
        results = []
        predictions = []
        
        for i, image_path in enumerate(image_files):
            print(f"\n📸 Processing image {i+1}/{len(image_files)}")
            result = self.predict_single(image_path, show_images)
            
            if result:
                results.append(result)
                predictions.append(result['predicted_class'])
        
        # Summary statistics
        if predictions:
            print(f"\n📊 Batch Prediction Summary:")
            print(f"   Total images processed: {len(results)}")
            
            # Count predictions
            pred_counts = Counter(predictions)
            for class_name, count in pred_counts.items():
                percentage = (count / len(predictions)) * 100
                print(f"   {class_name}: {count} ({percentage:.1f}%)")
        
        return results
    
    def _display_prediction(self, image, result):
        """Display image with prediction results"""
        plt.figure(figsize=(10, 6))
        
        # Show image
        plt.subplot(1, 2, 1)
        plt.imshow(image)
        plt.title(f"Input Image\n{os.path.basename(result['image_path'])}")
        plt.axis('off')
        
        # Show probabilities
        plt.subplot(1, 2, 2)
        classes = list(result['class_probabilities'].keys())
        probs = list(result['class_probabilities'].values())
        
        colors = ['green' if c == result['predicted_class'] else 'lightgray' for c in classes]
        bars = plt.bar(classes, probs, color=colors)
        
        plt.title(f"Class Probabilities\nPredicted: {result['predicted_class']} ({result['confidence']:.3f})")
        plt.ylabel('Probability')
        plt.ylim(0, 1)
        
        # Add value labels on bars
        for bar, prob in zip(bars, probs):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{prob:.3f}', ha='center', va='bottom')
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()


def main():
    """Main function for command line usage"""
    parser = argparse.ArgumentParser(description='Roof Texture Classification Inference')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to the trained model checkpoint')
    parser.add_argument('--image_path', type=str, default=None,
                       help='Path to a single image for prediction')
    parser.add_argument('--batch_dir', type=str, default=None,
                       help='Directory containing multiple images for batch prediction')
    parser.add_argument('--show_images', action='store_true',
                       help='Display images with predictions')
    parser.add_argument('--max_images', type=int, default=10,
                       help='Maximum number of images to process in batch mode')
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.image_path and not args.batch_dir:
        parser.error("Either --image_path or --batch_dir must be specified")
    
    if args.image_path and args.batch_dir:
        parser.error("Cannot specify both --image_path and --batch_dir")
    
    # Initialize inference
    try:
        inference = RoofTextureInference(args.model_path)
    except Exception as e:
        print(f"❌ Error initializing inference: {str(e)}")
        return
    
    # Perform inference
    if args.image_path:
        # Single image prediction
        if not os.path.exists(args.image_path):
            print(f"❌ Image file not found: {args.image_path}")
            return
        
        result = inference.predict_single(args.image_path, args.show_images)
        if result:
            print(f"\n✅ Prediction completed successfully!")
    
    elif args.batch_dir:
        # Batch prediction
        if not os.path.exists(args.batch_dir):
            print(f"❌ Directory not found: {args.batch_dir}")
            return
        
        results = inference.predict_batch(args.batch_dir, args.show_images, args.max_images)
        if results:
            print(f"\n✅ Batch prediction completed successfully!")
            print(f"📁 Processed {len(results)} images")


if __name__ == "__main__":
    main()