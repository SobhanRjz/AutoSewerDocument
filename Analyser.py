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
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.data.transforms import ResizeShortestEdge

@dataclass
class TimingStats:
	"""Class to track timing statistics"""
	initialization_time: float = 0.0
	frame_collection_time: float = 0.0
	preprocessing_time: float = 0.0
	inference_time: float = 0.0
	visualization_time: float = 0.0
	total_time: float = 0.0

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
		init_start = time.time()
		
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
		
		# Initialize detections list and timing stats
		self.all_detections = []
		self.timing_stats = TimingStats()
		self.timing_stats.initialization_time = time.time() - init_start

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
		process_start = time.time()
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

		self.timing_stats.total_time = time.time() - process_start
		
		# Log timing statistics
		self.logger.info("\nTiming Statistics:")
		self.logger.info(f"Initialization Time: {self.timing_stats.initialization_time:.2f}s")
		self.logger.info(f"Frame Collection Time: {self.timing_stats.frame_collection_time:.2f}s")
		self.logger.info(f"Preprocessing Time: {self.timing_stats.preprocessing_time:.2f}s")
		self.logger.info(f"Inference Time: {self.timing_stats.inference_time:.2f}s")
		self.logger.info(f"Visualization Time: {self.timing_stats.visualization_time:.2f}s")
		self.logger.info(f"Total Time: {self.timing_stats.total_time:.2f}s")

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
		frame_collection_start = time.time()
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
				
				if has_detections:
					# for _ in range(6):
					# 	writer.write(output_frame)
					
				# Store frame info if defects were detected
				
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
			else:
				writer.write(frame)
			frame_count += 1

		self.timing_stats.frame_collection_time = time.time() - frame_collection_start
		
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
		preprocess_start = time.time()
		
		# Make predictions
		inference_start = time.time()
		self.timing_stats.preprocessing_time += inference_start - preprocess_start
		
		outputs = self.predictor(frame)
		instances = outputs["instances"].to("cpu")
		
		eachframe_time = time.time() - inference_start
		self.timing_stats.inference_time += time.time() - inference_start
		
		vis_start = time.time()
		
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
				"frame_time": time.time() - preprocess_start,
				"timestamp_seconds": timestamp
			}
			detection_data.append(detection)
			
		# Add detections to the list
		self.all_detections.extend(detection_data)
		
		# Visualize results
		v = Visualizer(frame[:, :, ::-1], metadata=self.metadata, scale=1.0, instance_mode= ColorMode.IMAGE_BW)
		v = v.draw_instance_predictions(instances)
		
		self.logger.info(f"Inference time for frame: {eachframe_time:.4f}s")
		self.timing_stats.visualization_time += time.time() - vis_start
		
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
	#input_path = r"C:\Users\sobha\Desktop\detectron2\Data\E.Hormozi\20240610\20240610_104450.AVI"
	input_path = r"C:\Users\sobha\Desktop\detectron2\Data\E.Hormozi\14030830\14030830\1104160202120663062-1104160202120663075\1.mpg"
	input_path = r"C:\Users\sobha\Desktop\detectron2\Data\E.Hormozi\08- 493.1 to 493\olympicSt25zdo4931Surveyupstream.mpg"

	output_path = os.path.join("output", os.path.basename(input_path))
	
	detector.logger.info(f"Processing video: {input_path}")
	detector.process_video(input_path, output_path)

if __name__ == "__main__":
	main()
