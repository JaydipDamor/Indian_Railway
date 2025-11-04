import gradio as gr
import cv2
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import timm  # For EfficientNet - same as training
import numpy as np
import os
import tempfile
from datetime import datetime, timedelta
import pandas as pd
from PIL import Image
import io
import base64

# Hand Detection Model Architecture
class EfficientNetHandDetector(nn.Module):
    """Hand detection model architecture"""
    def __init__(self, model_name='efficientnet_b0', num_classes=2, pretrained=False):
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

# Speedometer Detection Model Architecture
class EfficientNetSpeedometerDetector(nn.Module):
    """Speedometer detection model architecture with different classifier"""
    def __init__(self, model_name='efficientnet_b0', num_classes=2, pretrained=True):
        super(EfficientNetSpeedometerDetector, self).__init__()
        
        self.backbone = timm.create_model(model_name, pretrained=pretrained)
        num_features = self.backbone.classifier.in_features
        
        # Different classifier architecture for speedometer
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.2),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        return self.backbone(x)

class HandDetectionModel:
    def __init__(self, model_path, device='cpu'):
        """Initialize the hand detection model"""
        self.device = device
        self.model = self.load_model(model_path)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
    def load_model(self, model_path):
        """Load the trained hand detection model"""
        model = EfficientNetHandDetector(num_classes=2, pretrained=False)
        
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
            
            model.eval()
            model.to(self.device)
            print(f"Hand detection model loaded successfully from {model_path}")
            
        except Exception as e:
            print(f"Error loading hand model: {e}")
            raise
        
        return model
    
    def predict(self, image):
        """Predict if hand is present in the image"""
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(input_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][predicted_class].item()
        
        return predicted_class, confidence

class SpeedometerDetectionModel:
    def __init__(self, model_path, device='cpu'):
        """Initialize the speedometer detection model"""
        self.device = device
        self.model = self.load_model(model_path)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
    def load_model(self, model_path):
        """Load the trained speedometer detection model"""
        model = EfficientNetSpeedometerDetector(num_classes=2, pretrained=True)
        
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            else:
                model.load_state_dict(checkpoint, strict=False)
            
            model.eval()
            model.to(self.device)
            print(f"Speedometer model loaded successfully from {model_path}")
            
        except Exception as e:
            print(f"Error loading speedometer model: {e}")
            raise
        
        return model
    
    def predict(self, image):
        """Predict if train is moving or stopped"""
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(input_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][predicted_class].item()
        
        return predicted_class, confidence

class UnifiedVideoProcessor:
    def __init__(self, left_roi=None, right_roi=None, speedometer_roi=None):
        """
        Initialize unified video processor
        left_roi: (x, y, width, height) for left hand
        right_roi: (x, y, width, height) for right hand  
        speedometer_roi: (x, y, width, height) for speedometer
        """
        self.left_roi = left_roi
        self.right_roi = right_roi
        self.speedometer_roi = speedometer_roi
        
    def extract_frames_2fps(self, video_path):
        """Extract frames at 2 FPS and crop all ROIs"""
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_interval = int(fps / 2)
        
        frames_with_timestamps = []
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            if frame_count % frame_interval == 0:
                timestamp = frame_count / fps
                roi_frames = {}
                
                # Extract Left Hand ROI if specified
                if self.left_roi:
                    lx, ly, lw, lh = self.left_roi
                    left_roi_frame = frame[ly:ly+lh, lx:lx+lw]
                    roi_frames['left_hand'] = cv2.cvtColor(left_roi_frame, cv2.COLOR_BGR2RGB)
                
                # Extract Right Hand ROI if specified
                if self.right_roi:
                    rx, ry, rw, rh = self.right_roi
                    right_roi_frame = frame[ry:ry+rh, rx:rx+rw]
                    roi_frames['right_hand'] = cv2.cvtColor(right_roi_frame, cv2.COLOR_BGR2RGB)
                
                # Extract Speedometer ROI if specified
                if self.speedometer_roi:
                    sx, sy, sw, sh = self.speedometer_roi
                    speed_roi_frame = frame[sy:sy+sh, sx:sx+sw]
                    roi_frames['speedometer'] = cv2.cvtColor(speed_roi_frame, cv2.COLOR_BGR2RGB)
                
                frames_with_timestamps.append((roi_frames, timestamp, frame))
            
            frame_count += 1
        
        cap.release()
        return frames_with_timestamps
    
    def create_annotated_video(self, video_path, detections, output_path):
        """Create annotated video with all detection results"""
        cap = cv2.VideoCapture(video_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_count = 0
        detection_dict = {int(d['timestamp'] * fps): d for d in detections}
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Draw ROI rectangles
            if self.left_roi:
                lx, ly, lw, lh = self.left_roi
                cv2.rectangle(frame, (lx, ly), (lx + lw, ly + lh), (255, 255, 0), 2)  # Cyan
                cv2.putText(frame, "LEFT ROI", (lx, ly + lh + 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
            
            if self.right_roi:
                rx, ry, rw, rh = self.right_roi
                cv2.rectangle(frame, (rx, ry), (rx + rw, ry + rh), (0, 255, 255), 2)  # Yellow
                cv2.putText(frame, "RIGHT ROI", (rx, ry + rh + 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            
            if self.speedometer_roi:
                sx, sy, sw, sh = self.speedometer_roi
                cv2.rectangle(frame, (sx, sy), (sx + sw, sy + sh), (255, 0, 0), 2)  # Blue
                cv2.putText(frame, "SPEEDOMETER ROI", (sx, sy + sh + 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            
            # Add detection results
            if frame_count in detection_dict:
                detection = detection_dict[frame_count]
                
                # Left hand detection
                if 'left_hand_detected' in detection and self.left_roi:
                    lx, ly, lw, lh = self.left_roi
                    if detection['left_hand_detected']:
                        cv2.rectangle(frame, (lx, ly), (lx + lw, ly + lh), (0, 255, 0), 3)
                        text = f"LEFT HAND ({detection['left_confidence']:.2f})"
                        cv2.putText(frame, text, (lx, ly - 10), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    else:
                        cv2.rectangle(frame, (lx, ly), (lx + lw, ly + lh), (0, 0, 255), 3)
                
                # Right hand detection
                if 'right_hand_detected' in detection and self.right_roi:
                    rx, ry, rw, rh = self.right_roi
                    if detection['right_hand_detected']:
                        cv2.rectangle(frame, (rx, ry), (rx + rw, ry + rh), (0, 255, 0), 3)
                        text = f"RIGHT HAND ({detection['right_confidence']:.2f})"
                        cv2.putText(frame, text, (rx, ry - 10), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    else:
                        cv2.rectangle(frame, (rx, ry), (rx + rw, ry + rh), (0, 0, 255), 3)
                
                # Speedometer detection
                if 'train_moving' in detection and self.speedometer_roi:
                    sx, sy, sw, sh = self.speedometer_roi
                    if detection['train_moving']:
                        cv2.rectangle(frame, (sx, sy), (sx + sw, sy + sh), (0, 255, 0), 3)
                        text = f"TRAIN MOVING ({detection['speed_confidence']:.2f})"
                        cv2.putText(frame, text, (sx, sy - 10), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                        cv2.putText(frame, "STATUS: MOVING", (10, 70), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    else:
                        cv2.rectangle(frame, (sx, sy), (sx + sw, sy + sh), (0, 0, 255), 3)
                        text = f"TRAIN STOPPED ({detection['speed_confidence']:.2f})"
                        cv2.putText(frame, text, (sx, sy - 10), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                        cv2.putText(frame, "STATUS: STOPPED", (10, 70), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            # Add timestamp
            timestamp_text = f"Time: {frame_count/fps:.2f}s"
            cv2.putText(frame, timestamp_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            out.write(frame)
            frame_count += 1
        
        cap.release()
        out.release()

def process_unified_video(video_file, analysis_type, left_model_path=None, right_model_path=None, 
                         speedometer_model_path=None, confidence_threshold=0.7):
    """
    Main function to process video with unified detection
    """
    try:
        # ROI Configuration - Update these coordinates as needed
        LEFT_ROI_COORDINATES = (750, 200, 150, 100)      # Left hand area
        RIGHT_ROI_COORDINATES = (850, 320, 1000, 450)    # Right hand area  
        SPEEDOMETER_ROI_COORDINATES = (1040, 200, 100, 200)  # Speedometer area
        
        # Initialize components based on analysis type
        detections = []
        
        if analysis_type == "Hand Detection Only":
            if not left_model_path or not right_model_path:
                return "Error: Both left and right hand models are required for hand detection.", None, None, None, None
            
            left_hand_model = HandDetectionModel(left_model_path)
            right_hand_model = HandDetectionModel(right_model_path)
            video_processor = UnifiedVideoProcessor(
                left_roi=LEFT_ROI_COORDINATES, 
                right_roi=RIGHT_ROI_COORDINATES
            )
            
        elif analysis_type == "Speedometer Only":
            if not speedometer_model_path:
                return "Error: Speedometer model is required for speedometer analysis.", None, None, None, None
            
            speedometer_model = SpeedometerDetectionModel(speedometer_model_path)
            video_processor = UnifiedVideoProcessor(speedometer_roi=SPEEDOMETER_ROI_COORDINATES)
            
        else:  # Combined Analysis
            if not left_model_path or not right_model_path or not speedometer_model_path:
                return "Error: All models (left hand, right hand, speedometer) are required for combined analysis.", None, None, None, None
            
            left_hand_model = HandDetectionModel(left_model_path)
            right_hand_model = HandDetectionModel(right_model_path)
            speedometer_model = SpeedometerDetectionModel(speedometer_model_path)
            video_processor = UnifiedVideoProcessor(
                left_roi=LEFT_ROI_COORDINATES,
                right_roi=RIGHT_ROI_COORDINATES,
                speedometer_roi=SPEEDOMETER_ROI_COORDINATES
            )
        
        # Process video
        frames_data = video_processor.extract_frames_2fps(video_file)
        
        # Run detections
        for roi_frames, timestamp, original_frame in frames_data:
            detection_result = {
                'timestamp': timestamp,
                'time_formatted': str(timedelta(seconds=int(timestamp)))
            }
            
            # Hand detections
            if analysis_type in ["Hand Detection Only", "Combined Analysis"]:
                # Left hand
                if 'left_hand' in roi_frames:
                    left_prediction, left_confidence = left_hand_model.predict(roi_frames['left_hand'])
                    left_hand_detected = left_prediction == 1 and left_confidence >= confidence_threshold
                    detection_result.update({
                        'left_hand_detected': left_hand_detected,
                        'left_confidence': left_confidence
                    })
                
                # Right hand
                if 'right_hand' in roi_frames:
                    right_prediction, right_confidence = right_hand_model.predict(roi_frames['right_hand'])
                    right_hand_detected = right_prediction == 1 and right_confidence >= confidence_threshold
                    detection_result.update({
                        'right_hand_detected': right_hand_detected,
                        'right_confidence': right_confidence
                    })
            
            # Speedometer detection
            if analysis_type in ["Speedometer Only", "Combined Analysis"]:
                if 'speedometer' in roi_frames:
                    speed_prediction, speed_confidence = speedometer_model.predict(roi_frames['speedometer'])
                    train_moving = speed_prediction == 1 and speed_confidence >= confidence_threshold
                    detection_result.update({
                        'train_moving': train_moving,
                        'speed_confidence': speed_confidence,
                        'train_status': 'Moving' if train_moving else 'Stopped'
                    })
            
            detections.append(detection_result)
        
        # Generate report
        report = generate_unified_report(detections, analysis_type, confidence_threshold)
        
        # Create annotated video
        output_video_path = tempfile.mktemp(suffix='.mp4')
        video_processor.create_annotated_video(video_file, detections, output_video_path)
        
        # Create DataFrames
        left_df, right_df, speed_df = create_dataframes(detections, analysis_type)
        
        return report, output_video_path, left_df, right_df, speed_df
        
    except Exception as e:
        error_msg = f"Error processing video: {str(e)}"
        return error_msg, None, None, None, None

def generate_unified_report(detections, analysis_type, confidence_threshold):
    """Generate unified analysis report"""
    total_frames = len(detections)
    
    report = f"Railway Unified Detection Analysis Report\n"
    report += f"=" * 60 + "\n"
    report += f"Analysis Type: {analysis_type}\n"
    report += f"Video processed at 2 FPS\n"
    report += f"Total frames analyzed: {total_frames}\n"
    report += f"Confidence threshold: {confidence_threshold}\n\n"
    
    if analysis_type in ["Hand Detection Only", "Combined Analysis"]:
        # Left hand analysis
        left_detections = [d for d in detections if d.get('left_hand_detected', False)]
        report += f"LEFT HAND DETECTION RESULTS\n"
        report += f"-" * 40 + "\n"
        report += f"Left hand detections: {len(left_detections)}\n"
        
        if left_detections:
            report += f"Left Hand Detection Timeline:\n"
            for detection in left_detections:
                report += f"  Time: {detection['time_formatted']} ({detection['timestamp']:.2f}s) - "
                report += f"Confidence: {detection['left_confidence']:.3f}\n"
        else:
            report += f"No left hand detections found above confidence threshold.\n"
        
        report += f"\n"
        
        # Right hand analysis
        right_detections = [d for d in detections if d.get('right_hand_detected', False)]
        report += f"RIGHT HAND DETECTION RESULTS\n"
        report += f"-" * 40 + "\n"
        report += f"Right hand detections: {len(right_detections)}\n"
        
        if right_detections:
            report += f"Right Hand Detection Timeline:\n"
            for detection in right_detections:
                report += f"  Time: {detection['time_formatted']} ({detection['timestamp']:.2f}s) - "
                report += f"Confidence: {detection['right_confidence']:.3f}\n"
        else:
            report += f"No right hand detections found above confidence threshold.\n"
        
        report += f"\n"
    
    if analysis_type in ["Speedometer Only", "Combined Analysis"]:
        # Speedometer analysis
        moving_detections = [d for d in detections if d.get('train_moving', False)]
        stopped_detections = [d for d in detections if 'train_moving' in d and not d['train_moving']]
        
        moving_percentage = (len(moving_detections) / total_frames * 100) if total_frames > 0 else 0
        stopped_percentage = (len(stopped_detections) / total_frames * 100) if total_frames > 0 else 0
        
        report += f"TRAIN MOVEMENT ANALYSIS\n"
        report += f"-" * 40 + "\n"
        report += f"Frames with train moving: {len(moving_detections)} ({moving_percentage:.1f}%)\n"
        report += f"Frames with train stopped: {len(stopped_detections)} ({stopped_percentage:.1f}%)\n\n"
        
        if moving_detections:
            report += f"Train Moving Timeline:\n"
            for detection in moving_detections:
                report += f"  Time: {detection['time_formatted']} ({detection['timestamp']:.2f}s) - "
                report += f"Confidence: {detection['speed_confidence']:.3f}\n"
        
        report += f"\n"
    
    return report

def create_dataframes(detections, analysis_type):
    """Create DataFrames for different analysis types"""
    left_df = None
    right_df = None
    speed_df = None
    
    if analysis_type in ["Hand Detection Only", "Combined Analysis"]:
        # Left hand DataFrame
        left_df = pd.DataFrame([{
            'Timestamp (s)': d['timestamp'],
            'Time': d['time_formatted'],
            'Left Hand Detected': 'Yes' if d.get('left_hand_detected', False) else 'No',
            'Confidence': f"{d.get('left_confidence', 0):.3f}"
        } for d in detections if 'left_confidence' in d])
        
        # Right hand DataFrame
        right_df = pd.DataFrame([{
            'Timestamp (s)': d['timestamp'],
            'Time': d['time_formatted'],
            'Right Hand Detected': 'Yes' if d.get('right_hand_detected', False) else 'No',
            'Confidence': f"{d.get('right_confidence', 0):.3f}"
        } for d in detections if 'right_confidence' in d])
    
    if analysis_type in ["Speedometer Only", "Combined Analysis"]:
        # Speedometer DataFrame
        speed_df = pd.DataFrame([{
            'Timestamp (s)': d['timestamp'],
            'Time': d['time_formatted'],
            'Status': d.get('train_status', 'Unknown'),
            'Confidence': f"{d.get('speed_confidence', 0):.3f}"
        } for d in detections if 'speed_confidence' in d])
    
    return left_df, right_df, speed_df

def create_unified_interface():
    """Create unified Gradio interface"""
    with gr.Blocks(title="Unified Railway Detection System", theme=gr.themes.Soft()) as interface:
        gr.Markdown("""
        # 🚂 Unified Railway Detection System
        ### CCTV Engine Cockpit Analysis - Hand Detection & Speedometer Analysis
        
        Choose your analysis type and upload the corresponding models for comprehensive railway safety monitoring.
        """)
        
        with gr.Row():
            with gr.Column():
                analysis_type = gr.Radio(
                    choices=["Hand Detection Only", "Speedometer Only", "Combined Analysis"],
                    value="Combined Analysis",
                    label="Analysis Type"
                )
                
                video_input = gr.File(
                    label="Upload Video (MP4)", 
                    file_types=[".mp4"],
                    type="filepath"
                )
                
                gr.Markdown("### Model Files")
                with gr.Group():
                    left_model_input = gr.File(
                        label="Left Hand Model (.pth)", 
                        file_types=[".pth"],
                        type="filepath"
                    )
                    right_model_input = gr.File(
                        label="Right Hand Model (.pth)", 
                        file_types=[".pth"],
                        type="filepath"
                    )
                    speedometer_model_input = gr.File(
                        label="Speedometer Model (.pth)", 
                        file_types=[".pth"],
                        type="filepath"
                    )
                
                confidence_slider = gr.Slider(
                    minimum=0.1, 
                    maximum=1.0, 
                    value=0.7, 
                    step=0.05,
                    label="Confidence Threshold"
                )
                process_btn = gr.Button("🔍 Start Analysis", variant="primary", size="lg")
            
            with gr.Column():
                gr.Markdown("""
                ### Analysis Types:
                
                **🤲 Hand Detection Only**
                - Requires: Left Hand Model + Right Hand Model
                - Analyzes pilot hand movements in cockpit
                - Detects presence of hands in designated ROIs
                
                **⚡ Speedometer Only**
                - Requires: Speedometer Model
                - Analyzes train movement status
                - Detects Moving/Stopped based on speedometer
                
                **🔄 Combined Analysis**
                - Requires: All three models
                - Complete cockpit monitoring
                - Hand detection + Movement analysis
                
                ### ROI Colors:
                - **Cyan**: Left Hand ROI
                - **Yellow**: Right Hand ROI  
                - **Blue**: Speedometer ROI
                - **Green**: Detection positive
                - **Red**: Detection negative
                """)
        
        gr.Markdown("---")
        
        with gr.Row():
            with gr.Column():
                text_output = gr.Textbox(
                    label="📊 Analysis Report", 
                    lines=20, 
                    max_lines=25,
                    interactive=False
                )
            
            with gr.Column():
                video_output = gr.Video(
                    label="🎥 Annotated Video Output",
                    interactive=False
                )
        
        with gr.Row():
            with gr.Column():
                left_dataframe_output = gr.Dataframe(
                    label="📋 Left Hand Detection Results",
                    interactive=False,
                    wrap=True
                )
            
            with gr.Column():
                right_dataframe_output = gr.Dataframe(
                    label="📋 Right Hand Detection Results",
                    interactive=False,
                    wrap=True
                )
        
        with gr.Row():
            speed_dataframe_output = gr.Dataframe(
                label="📋 Train Movement Analysis Results",
                interactive=False,
                wrap=True
            )
        
        # Event handler
        process_btn.click(
            fn=process_unified_video,
            inputs=[
                video_input, 
                analysis_type, 
                left_model_input, 
                right_model_input, 
                speedometer_model_input, 
                confidence_slider
            ],
            outputs=[
                text_output, 
                video_output, 
                left_dataframe_output, 
                right_dataframe_output, 
                speed_dataframe_output
            ]
        )
        
        gr.Markdown("""
        ### Configuration & Notes:
        
        **ROI Coordinates (Update in code as needed):**
        ```python
        LEFT_ROI_COORDINATES = (750, 200, 150, 100)      # Left hand area
        RIGHT_ROI_COORDINATES = (850, 320, 150, 100)     # Right hand area  
        SPEEDOMETER_ROI_COORDINATES = (1040, 200, 100, 200)  # Speedometer area
        ```
        
        **Model Architectures:**
        - **Hand Models**: EfficientNet-B0 with simpler classifier
        - **Speedometer Model**: EfficientNet-B0 with BatchNorm classifier
        - Models are loaded with appropriate architectures automatically
        
        **Processing:**
        - Video processed at 2 FPS for efficiency
        - Independent analysis for each detection type
        - Combined results in unified interface
        - Export functionality available through data tables
        
        **Requirements:**
        - Upload only the models needed for your selected analysis type
        - All models should be trained PyTorch (.pth) files
        - Video should be MP4 format for best compatibility
        """)
    
    return interface

if __name__ == "__main__":
    # Create and launch the unified interface
    app = create_unified_interface()
    app.launch(
        server_name="localhost",
        server_port=7860,
        share=True,  # Set to True if you want a public link
        debug=True
    )