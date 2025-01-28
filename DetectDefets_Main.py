import sys
import os
import logging
import torch
import detectron2
from detectron2.utils.logger import setup_logger
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import MetadataCatalog
import cv2
import time
from typing import List, Dict, Optional
from dataclasses import dataclass
import numpy as np
import json

@dataclass
class DetectionConfig:
    """Configuration class for defect detection settings"""
    class_names: List[str] = None
    colors: List[List[int]] = None
    score_threshold: float = 0.7
    nms_threshold: float = 0.2
    model_path: str = None
    config_path: str = None
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

class DefectDetector:
    """Main class for sewer defect detection using Detectron2"""
    
    def __init__(self, config: DetectionConfig):
        # Configure logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        setup_logger()

        # Log system information
        self._log_system_info()
        
        # Initialize configuration
        self.cfg = self._initialize_config(config)
        
        # Set up dataset and metadata
        self._setup_dataset(config.class_names, config.colors)
        
        # Create predictor
        self.predictor = DefaultPredictor(self.cfg)

        torch.backends.cudnn.benchmark = True
        
        # Initialize detections list
        self.all_detections = []

    def _log_system_info(self) -> None:
        """Log system and CUDA information"""
        if torch.cuda.is_available():
            self.logger.info("CUDA is available!")
            self.logger.info(f"CUDA Version: {torch.version.cuda}")
        else:
            self.logger.warning("No CUDA available")

        torch_version = ".".join(torch.__version__.split(".")[:2])
        cuda_version = torch.version.cuda if torch.cuda.is_available() else "No CUDA"
        self.logger.info(f"PyTorch Version: {torch_version}")
        self.logger.info(f"CUDA Version: {cuda_version}")

    def _initialize_config(self, config: DetectionConfig) -> detectron2.config.CfgNode:
        """Initialize Detectron2 configuration"""
        cfg = get_cfg()
        cfg.DATASETS.TRAIN_REPEAT_FACTOR = [['my_dataset_train', 1.0]]
        cfg.MODEL.ROI_HEADS.CLASS_NAMES = config.class_names
        cfg.SOLVER.PATIENCE = 2000
        cfg.merge_from_file(config.config_path)
        cfg.MODEL.WEIGHTS = config.model_path
        cfg.MODEL.DEVICE = config.device
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = config.score_threshold
        cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = config.nms_threshold
        return cfg

    def _setup_dataset(self, class_names: List[str], colors: List[List[int]]) -> None:
        """Setup custom dataset and metadata"""
        MetadataCatalog.get("custom_dataset").set(thing_classes=class_names)
        self.metadata = MetadataCatalog.get("custom_dataset")
        self.metadata.thing_colors = colors

    def process_video(self, input_path: str, output_path: str, frame_interval: int = 24) -> None:
        """Process video file and detect defects"""
        try:
            start_time = time.time()
            cap = cv2.VideoCapture(input_path)
            
            if not cap.isOpened():
                self.logger.error(f"Error opening video file {input_path}")
                return

            # Initialize video writer
            writer = self._initialize_video_writer(cap, output_path)
            
            # Create output directory for frames
            frames_dir = os.path.splitext(output_path)[0] + "_frames"
            os.makedirs(frames_dir, exist_ok=True)
            
            # Process frames
            self._process_frames(cap, writer, frame_interval, frames_dir)
            
            # Save detections to JSON
            json_output_path = os.path.splitext(output_path)[0] + "_detections.json"
            with open(json_output_path, 'w') as f:
                json.dump(self.all_detections, f, indent=4)
            self.logger.info(f"Detection data saved to {json_output_path}")
            
            # Cleanup
            cap.release()
            writer.release()
            cv2.destroyAllWindows()

            self.logger.info(f"Video processing complete. Output saved to {output_path}")
            self.logger.info(f"Total processing time: {time.time() - start_time:.2f} seconds")
            
        except Exception as ex:
            self.logger.error(f"An error occurred during video analysis: {str(ex)}")

    def _initialize_video_writer(self, cap: cv2.VideoCapture, output_path: str) -> cv2.VideoWriter:
        """Initialize video writer with proper settings"""
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        return cv2.VideoWriter(
            output_path, 
            cv2.VideoWriter_fourcc(*"mp4v"), 
            fps, 
            (width, height)
        )

    def format_timestamp(self, seconds: float) -> str:
        """Convert seconds to HH:MM:SS format"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        seconds = int(seconds % 60)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    
    def _process_frames(self, cap: cv2.VideoCapture, writer: cv2.VideoWriter, frame_interval: int, frames_dir: str) -> None:
        """Process individual frames from the video"""
        frame_count = 0
        countPic = 0
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        output_frames = []  # Store frames with detections

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % frame_interval == 0:
                countPic += 1
                # Calculate timestamp in seconds
                timestamp = frame_count / fps
                
                output_frame, has_detections = self._process_single_frame(frame, timestamp)
                writer.write(output_frame)
                
                # Store frame info if defects were detected
                if has_detections:
                    output_frames.append({
                        'frame': output_frame,
                        'timestamp': timestamp,
                        'count': countPic
                    })
                
                # Display frame
                cv2.namedWindow("Output", cv2.WINDOW_NORMAL)
                cv2.resizeWindow("Output", 800, 600)
                cv2.imshow("Output", output_frame)
                
                if cv2.waitKey(int(1000 / fps)) & 0xFF == ord('q'):
                    break

            frame_count += 1

        # Save all frames with detections after processing
        for frame_info in output_frames:
            timestamp_str = self.format_timestamp(frame_info['timestamp']).replace(':', '_')
            frame_path = os.path.join(frames_dir, f"frame_{frame_info['count']}_{timestamp_str}.jpg")
            
            try:
                success = cv2.imwrite(frame_path, frame_info['frame'])
                if success:
                    self.logger.info(f"Saved frame with detections: {frame_path}")
                else:
                    self.logger.error(f"Failed to save frame: {frame_path}")
            except Exception as e:
                self.logger.error(f"Error saving frame {frame_path}: {str(e)}")

    def _process_single_frame(self, frame: np.ndarray, timestamp: float) -> tuple[np.ndarray, bool]:
        """Process a single frame and return the annotated result and detection data"""
        start_time = time.time()
        
        # Make predictions
        outputs = self.predictor(frame)
        instances = outputs["instances"].to("cpu")
        
        # Extract detection data
        detection_data = []
        boxes = instances.pred_boxes.tensor.numpy()
        scores = instances.scores.numpy()
        classes = instances.pred_classes.numpy()
        
        for box, score, class_id in zip(boxes, scores, classes):
            detection = {
                "bbox": box.tolist(),  # [x1, y1, x2, y2]
                "class": self.metadata.thing_classes[class_id],
                "confidence": float(score),
                "frame_time": time.time() - start_time,
                "timestamp_seconds": timestamp
            }
            detection_data.append(detection)
            
        # Log prediction time
        prediction_time = time.time() - start_time
        self.logger.info(f"Time per frame: {prediction_time:.2f} seconds")
        
        # Add detections to the list
        self.all_detections.extend(detection_data)
        
        # Visualize results
        v = Visualizer(frame[:, :, ::-1], metadata=self.metadata, scale=1.0, instance_mode= ColorMode.SEGMENTATION) # Dim input image to highlight predictions)
        v = v.draw_instance_predictions(instances)
        
        return v.get_image()[:, :, ::-1], len(detection_data) > 0

def main():
    # Define detection configuration with specific colors for each class
    config = DetectionConfig(
        class_names=[
            "Crack",           # Red for cracks/structural damage
            "Obstacle",        # Green for physical blockages 
            "Deposits",         # Blue for sediment/debris
            "Deformed",          # Yellow for pipe deformation
            "Broken",          # Magenta for severe breaks
            "Joint Displaced", # Cyan for joint issues
            "Surface Damage",  # Gray for surface deterioration
            "Root"            # Purple for root intrusion
        ],
        colors=[
            [255, 0, 0],      # Red - Crack
            [0, 255, 0],      # Green - Obstacle  
            [0, 0, 255],      # Blue - Deposit
            [255, 255, 0],    # Yellow - Deform
            [255, 0, 255],    # Magenta - Broken
            [0, 255, 255],    # Cyan - Joint Displaced
            [128, 128, 128],  # Gray - Surface Damage
            [128, 0, 128]     # Purple - Root
        ],
        model_path=os.path.join(r"Model", "model_final.pth"),
        config_path=os.path.join(r"Model", "mask_rcnn_X_101_32x8d_FPN_3x.yaml")
    )

    # Initialize detector
    detector = DefectDetector(config)

    # Process video
    input_path = r"C:\Users\sobha\Desktop\detectron2\Data\TestFilm\Closed circuit television (CCTV) sewer inspection.mp4"
    output_path = os.path.join("output", os.path.basename(input_path))
    
    detector.logger.info(f"Processing video: {input_path}")
    detector.process_video(input_path, output_path)

if __name__ == "__main__":
    main()
