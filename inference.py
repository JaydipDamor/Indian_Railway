import torch
import torch.nn as nn
import timm
from PIL import Image
import torchvision.transforms as transforms
import cv2
import numpy as np
import os
from pathlib import Path

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class EfficientNetHandDetector(nn.Module):
    """Same model architecture as training"""
    def __init__(self, model_name='efficientnet_b0', num_classes=2, pretrained=True):
        super(EfficientNetHandDetector, self).__init__()
        
        self.backbone = timm.create_model(model_name, pretrained=pretrained)
        num_features = self.backbone.classifier.in_features
        
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        return self.backbone(x)

class HandDetectionInference:
    def __init__(self, model_path='best_2nd_angle_locopilot.pth'):
        """
        Initialize the inference pipeline
        
        Args:
            model_path: Path to the trained model checkpoint
        """
        self.device = device
        self.model = EfficientNetHandDetector(num_classes=2)
        
        # Load trained model
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            self.model.to(self.device)
            self.model.eval()
            print(f"Model loaded successfully from {model_path}")
            print(f"Model validation accuracy: {checkpoint.get('val_accuracy', 'N/A'):.2f}%")
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
        
        # Define preprocessing transform
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def preprocess_image(self, image_path):
        """
        Preprocess image for inference
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Preprocessed tensor
        """
        try:
            # Load image
            if isinstance(image_path, str):
                image = Image.open(image_path)
            else:
                image = image_path  # PIL Image object
            
            # Convert to RGB if needed
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Apply transforms
            image_tensor = self.transform(image).unsqueeze(0)  # Add batch dimension
            return image_tensor.to(self.device)
            
        except Exception as e:
            print(f"Error preprocessing image: {e}")
            return None
    
    def predict_single_image(self, image_path, return_confidence=True):
        """
        Predict hand presence in a single image
        
        Args:
            image_path: Path to image or PIL Image object
            return_confidence: Whether to return confidence scores
            
        Returns:
            Dictionary with prediction results
        """
        # Preprocess image
        image_tensor = self.preprocess_image(image_path)
        if image_tensor is None:
            return None
        
        # Make prediction
        with torch.no_grad():
            outputs = self.model(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            predicted_class = torch.argmax(outputs, dim=1).item()
            confidence = probabilities[0][predicted_class].item()
        
        # Interpret results
        class_names = ['No Hand', 'Hand Present']
        prediction = class_names[predicted_class]
        
        result = {
            'prediction': prediction,
            'predicted_class': predicted_class,
            'confidence': confidence
        }
        
        if return_confidence:
            result['probabilities'] = {
                'no_hand': probabilities[0][0].item(),
                'hand_present': probabilities[0][1].item()
            }
        
        return result
    
    def predict_batch(self, image_paths, batch_size=16):
        """
        Predict hand presence in multiple images
        
        Args:
            image_paths: List of image paths
            batch_size: Batch size for processing
            
        Returns:
            List of prediction results
        """
        results = []
        
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i+batch_size]
            batch_results = []
            
            for image_path in batch_paths:
                result = self.predict_single_image(image_path)
                if result:
                    result['image_path'] = image_path
                    batch_results.append(result)
            
            results.extend(batch_results)
        
        return results
    
    def predict_folder(self, folder_path, output_file=None):
        """
        Predict hand presence for all images in a folder
        
        Args:
            folder_path: Path to folder containing images
            output_file: Optional file to save results
            
        Returns:
            Dictionary with summary statistics
        """
        # Get all image files
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        image_paths = []
        
        for file_path in Path(folder_path).rglob('*'):
            if file_path.suffix.lower() in image_extensions:
                image_paths.append(str(file_path))
        
        print(f"Found {len(image_paths)} images in {folder_path}")
        
        # Make predictions
        results = self.predict_batch(image_paths)
        
        # Calculate statistics
        hand_count = sum(1 for r in results if r['predicted_class'] == 1)
        no_hand_count = len(results) - hand_count
        avg_confidence = np.mean([r['confidence'] for r in results])
        
        summary = {
            'total_images': len(results),
            'hand_detected': hand_count,
            'no_hand_detected': no_hand_count,
            'hand_percentage': (hand_count / len(results)) * 100 if results else 0,
            'average_confidence': avg_confidence,
            'results': results
        }
        
        # Save results if requested
        if output_file:
            import json
            with open(output_file, 'w') as f:
                json.dump(summary, f, indent=2)
            print(f"Results saved to {output_file}")
        
        return summary

def test_model_performance(model_path, test_data_dirs):
    """
    Test model performance on labeled test data
    
    Args:
        model_path: Path to trained model
        test_data_dirs: Dictionary with 'with_hand' and 'without_hand' folders
    """
    detector = HandDetectionInference(model_path)
    
    # Test on images with hands
    print("Testing on images WITH hands...")
    with_hand_results = detector.predict_folder(test_data_dirs['with_hand'])
    
    # Test on images without hands
    print("Testing on images WITHOUT hands...")
    without_hand_results = detector.predict_folder(test_data_dirs['without_hand'])
    
    # Calculate accuracy
    with_hand_correct = sum(1 for r in with_hand_results['results'] if r['predicted_class'] == 1)
    without_hand_correct = sum(1 for r in without_hand_results['results'] if r['predicted_class'] == 0)
    
    total_correct = with_hand_correct + without_hand_correct
    total_images = with_hand_results['total_images'] + without_hand_results['total_images']
    
    accuracy = (total_correct / total_images) * 100 if total_images > 0 else 0
    
    print(f"\n{'='*50}")
    print(f"MODEL PERFORMANCE SUMMARY")
    print(f"{'='*50}")
    print(f"Total test images: {total_images}")
    print(f"Correct predictions: {total_correct}")
    print(f"Overall accuracy: {accuracy:.2f}%")
    print(f"")
    print(f"With hand images: {with_hand_results['total_images']}")
    print(f"Correctly identified as having hand: {with_hand_correct}")
    print(f"Hand detection accuracy: {(with_hand_correct/with_hand_results['total_images'])*100:.2f}%")
    print(f"")
    print(f"Without hand images: {without_hand_results['total_images']}")
    print(f"Correctly identified as no hand: {without_hand_correct}")
    print(f"No-hand detection accuracy: {(without_hand_correct/without_hand_results['total_images'])*100:.2f}%")

def demo_inference():
    """
    Demo function showing how to use the inference pipeline
    """
    # Initialize detector
    detector = HandDetectionInference('best_2nd_angle_locopilot.pth')
    
    # Example 1: Single image prediction
   # print("Example 1: Single image prediction")
   # result = detector.predict_single_image(r'C:\Users\Hello\Downloads\Indian_Railway\movement\extracted_frames_for_2nd_pilotrighthand\cropped_frame_016588_t1658.80s.jpg')
   # if result:
       # print(f"Prediction: {result['prediction']}")
       # print(f"Confidence: {result['confidence']:.3f}")
       # print(f"Probabilities: {result['probabilities']}")
    
    # Example 2: Batch prediction
    print("\nExample 2: Batch prediction")
    test_images = [r"C:\Users\Hello\Downloads\Indian_Railway\movement\extracted_frames_for_2nd_pilotrighthand\cropped_frame_001708_t170.80s.jpg", r"C:\Users\Hello\Downloads\Indian_Railway\movement\extracted_frames_for_2nd_pilotrighthand\cropped_frame_001732_t173.20s.jpg", r"C:\Users\Hello\Downloads\Indian_Railway\movement\extracted_frames_for_2nd_pilotrighthand\cropped_frame_002786_t278.60s.jpg", r"C:\Users\Hello\Downloads\Indian_Railway\movement\extracted_frames_for_2nd_pilotrighthand\cropped_frame_003428_t342.80s.jpg",'extracted_frames_2nd_handle\cropped_frame_000008_t1.92s.jpg', 'extracted_frames_2nd_handle\cropped_frame_000010_t2.40s.jpg', 'extracted_frames_2nd_handle\cropped_frame_000012_t2.88s.jpg', 'extracted_frames_2nd_handle\cropped_frame_000014_t3.36s.jpg','extracted_frames_2nd_handle\cropped_frame_000016_t3.84s.jpg', 'extracted_frames_2nd_handle\cropped_frame_000018_t4.32s.jpg', 'extracted_frames_2nd_handle\cropped_frame_010794_t2590.56s.jpg' , 'extracted_frames_2nd_handle\cropped_frame_010792_t2590.08s.jpg', 'extracted_frames_2nd_handle\cropped_frame_010790_t2589.60s.jpg','extracted_frames_2nd_handle\cropped_frame_000330_t79.20s.jpg','extracted_frames_2nd_handle\cropped_frame_000328_t78.72s.jpg','extracted_frames_2nd_handle\cropped_frame_000326_t78.24s.jpg', 'extracted_frames_2nd_handle\cropped_frame_000326_t78.24s.jpg']  # Replace with actual paths
    batch_results = detector.predict_batch(test_images)
    
   # for result in batch_results:
       # print(f"{result['image_path']}: {result['prediction']} (conf: {result['confidence']:.3f})")
    
    # Example 3: Folder prediction
   # print("\nExample 3: Folder prediction")
   # folder_summary = detector.predict_folder('extracted_frames', 'results.json')
   # print(f"Processed {folder_summary['total_images']} images")
   # print(f"Hand detected in {folder_summary['hand_percentage']:.1f}% of images")

if __name__ == "__main__":
    # Test the model (uncomment and modify paths as needed)
    
    # Test model performance
     test_data_dirs = {
         'with_hand': 'test_2nd_angle_hand_image',
         'without_hand': 'test_2nd_angel_handle_images'
     }
     test_model_performance(r'best_2nd_angle_locopilot.pth', test_data_dirs)
    
    # Run demo
     demo_inference()