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
from typing import List, Dict
from dataclasses import dataclass
import numpy as np
import json
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.data.transforms import ResizeShortestEdge
import math
from concurrent.futures import ThreadPoolExecutor
from TextExtractor.ExtrctInfo import TextExtractor

@dataclass 
class TimingStats:
	frame_collection_time: float = 0.0
	preprocessing_time: float = 0.0
	inference_time: float = 0.0
	visualization_time: float = 0.0
	total_time: float = 0.0

@dataclass
class DetectionConfig:
	class_names: List[str]
	colors: List[List[int]]
	score_threshold: float = 0.7
	nms_threshold: float = 0.2
	model_path: str = None
	config_path: str = None
	device: str = "cuda" if torch.cuda.is_available() else "cpu"
	batch_size: int = 16
	Custom_batch: bool = True

class BatchDefectDetector:
	def __init__(self, config: DetectionConfig):
		# Configure logging with a more detailed format
		logging.basicConfig(
			level=logging.INFO,
			format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
			handlers=[
				logging.StreamHandler(sys.stdout),
				logging.FileHandler('defect_detection.log')
			]
		)
		self.logger = logging.getLogger(__name__)
		self.logger.setLevel(logging.INFO)

		self.cfg = self._initialize_config(config)
		self.model = build_model(self.cfg)
		DetectionCheckpointer(self.model).load(config.model_path)
		self.model.eval()
		
		if len(self.cfg.DATASETS.TEST):
			self.metadata = MetadataCatalog.get(self.cfg.DATASETS.TEST[0])

		self.aug = ResizeShortestEdge(
			[self.cfg.INPUT.MIN_SIZE_TEST, self.cfg.INPUT.MIN_SIZE_TEST], self.cfg.INPUT.MAX_SIZE_TEST
		)
		self.input_format = self.cfg.INPUT.FORMAT
		assert self.input_format in ["RGB", "BGR"], self.input_format
		self.predictor = DefaultPredictor(self.cfg)
		self.timing_stats = TimingStats()
		self.Custom_batch = config.Custom_batch
		self._setup_dataset(config.class_names, config.colors)
		self.all_detections = {}
		

	def _setup_dataset(self, class_names: List[str], colors: List[List[int]]) -> None:
		"""Setup custom dataset and metadata"""
		MetadataCatalog.get("custom_dataset").set(thing_classes=class_names)
		self.metadata = MetadataCatalog.get("custom_dataset")
		self.metadata.thing_colors = colors

	def _initialize_config(self, config: DetectionConfig):
		cfg = get_cfg()
		cfg.merge_from_file(config.config_path)
		cfg.MODEL.WEIGHTS = config.model_path
		cfg.MODEL.DEVICE = config.device
		cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = config.score_threshold
		cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = config.nms_threshold
		cfg.MODEL.ROI_HEADS.CLASS_NAMES = config.class_names
		return cfg

	def preprocess(self, images):
		preprocess_start = time.time()
		batch_inputs = []
		for img in images:
			if self.input_format == "RGB":
				img = img[:, :, ::-1]  # Convert BGR → RGB
			
			height, width = img.shape[:2]
			img_transformed = self.aug.get_transform(img).apply_image(img)
			img_tensor = torch.as_tensor(img_transformed.astype("float32").transpose(2, 0, 1))
			
			batch_inputs.append({
				"image": img_tensor.to(self.cfg.MODEL.DEVICE),
				"height": height,
				"width": width
			})
		self.timing_stats.preprocessing_time += time.time() - preprocess_start
		return batch_inputs
	
	def process_video(self, input_path: str, output_path: str, batch_size: int = 8):
		"""Process video file in batches for object detection
		
		Args:
			input_path (str): Path to input video file
			output_path (str): Path to save output video and detection results
			batch_size (int, optional): Number of frames to process in each batch. Defaults to 8.
		
		Returns:
			list: List of processed video frames
		"""
		from queue import Queue
		process_start = time.time()
		
		# Open video capture
		cap = cv2.VideoCapture(input_path)
		if not cap.isOpened():
			self.logger.error(f"Error opening video file {input_path}")
			return
		
		# Create output directory for frame images
		frames_dir = os.path.splitext(output_path)[0] + "_frames"
		os.makedirs(frames_dir, exist_ok=True)
		
		# Get video properties
		total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
		fps = int(cap.get(cv2.CAP_PROP_FPS))
		
		# Process 1 frame per second
		frame_interval = fps
		
		# Initialize storage for frames and timestamps
		frames_to_process = []
		timestamps_to_process = []
		frame_queue = Queue()
		
		# Start frame collection
		self.logger.info("Starting frame collection...")
		frame_collection_start = time.time()
		
		def read_frames():
			"""Read frames from video and add to queue"""
			frame_idx = 0
			while frame_idx < total_frames:
				ret, frame = cap.read()
				if not ret:
					break
				if frame_idx % frame_interval == 0:
					frame_queue.put((frame_idx, frame))
				frame_idx += 1
			frame_queue.put(None)  # Signal end
		
		def process_frames():
			"""Get frames from queue and store with timestamps"""
			while True:
				item = frame_queue.get()
				if item is None:
					break
				frame_idx, frame = item
				frames_to_process.append(frame)
				timestamps_to_process.append(frame_idx / fps)
		
		# Run frame reading and processing in parallel
		with ThreadPoolExecutor(max_workers=4) as executor:
			executor.submit(read_frames)
			executor.submit(process_frames)
		
		# Log frame collection stats
		self.timing_stats.frame_collection_time = time.time() - frame_collection_start
		self.logger.info(f"Frame collection complete. Total frames to process: {len(frames_to_process)}. "
						 f"Time taken: {self.timing_stats.frame_collection_time:.2f} seconds")
		
		# Initialize batch processing variables
		batch_start_time = time.time()
		total_batch_time = 0
		all_predictions = []  # Store all predictions
		all_frames = []      # Store all frames
		all_timestamps = []  # Store all timestamps
		
		# Process frames in batches
		total_batches = math.ceil(len(frames_to_process) / batch_size)
		self.logger.info(f"Starting batch processing of {len(frames_to_process)} frames in {total_batches} batches")
		
		for i in range(0, len(frames_to_process), batch_size):
			batch = frames_to_process[i:i + batch_size]
			timestamps = timestamps_to_process[i:i + batch_size]
			
			current_batch = i // batch_size + 1
			progress = int((current_batch / total_batches) * 50)  # 50 characters for progress bar
			progress_bar = '=' * progress + '-' * (50 - progress)
			self.logger.info(f"Processing batch {current_batch} of {total_batches} [{progress_bar}] {(current_batch/total_batches)*100:.1f}%")
			self.logger.info(f"Batch size: {len(batch)} frames")
			
			# Run inference on batch
			if self.Custom_batch:
				self.logger.info("Using custom batch processing")
				outputs = self._predict_batch(batch)
				all_predictions.extend(outputs)
				self.logger.info(f"Generated {len(outputs)} predictions using custom processing")
			else:
				self.logger.info("Using default batch processing") 
				predictions = self._predict_batch_default(batch)
				all_predictions.extend(predictions)
				self.logger.info(f"Generated {len(predictions)} predictions using default processing")
			
			# Store processed frames and timestamps
			all_frames.extend(batch)
			all_timestamps.extend(timestamps)
			self.logger.info(f"Total frames processed so far: {len(all_frames)}")
			
			# Log batch timing
			batch_time = time.time() - batch_start_time
			total_batch_time += batch_time
			self.logger.info(f"Batch time: {batch_time:.2f} seconds")
			self.logger.info(f"Total time spent processing batches: {total_batch_time:.2f} seconds")
			self.logger.info(f"Average time per batch: {total_batch_time/current_batch:.2f} seconds")
			batch_start_time = time.time()
		# Visualize and save results
		if self.Custom_batch:
			self._visualize_batch(all_frames, all_predictions, all_timestamps, frames_dir)
		else:
			self._visualize_batch_default(all_frames, all_predictions, all_timestamps)
		

		# Extract text information from frames with detections
		self.logger.info("Starting text extraction from frames with detections...")
		text_extractor = TextExtractor()
		for frameKey, frameDetections in self.all_detections.items():
			# Extract text from frame
			self.logger.info(f"Extracting text from frame {frameKey}...")
			text_info = text_extractor.extract_text_from_video_frame(frameDetections["frame_path"])
			
			# Add text info to frame detections
			split_text = text_info.split(" ")
			frameDetections["text_info"] = [item for item in split_text if 'm' in item]
			
			# Log extracted text
			self.logger.info(f"Frame {frameKey}: Extracted text: {text_info}")
			self.logger.info(f"Frame {frameKey}: Filtered text containing 'm': {frameDetections['text_info']}")
		self.logger.info("Text extraction complete for all frames with detections")



		# Save detections to JSON
		json_output_path = os.path.splitext(output_path)[0] + "_detections.json"
		with open(json_output_path, 'w') as f:
			json.dump(self.all_detections, f, indent=4)
		self.logger.info(f"Detection data saved to {json_output_path}")
		
		# Log final timing statistics
		self.timing_stats.total_time = time.time() - process_start
		self.logger.info("\nTiming Statistics:")
		self.logger.info(f"Frame Collection Time: {self.timing_stats.frame_collection_time:.2f}s")
		self.logger.info(f"Preprocessing Time: {self.timing_stats.preprocessing_time:.2f}s")
		self.logger.info(f"Inference Time: {self.timing_stats.inference_time:.2f}s")
		self.logger.info(f"Visualization Time: {self.timing_stats.visualization_time:.2f}s")
		self.logger.info(f"Total Time: {self.timing_stats.total_time:.2f}s")
		
		return frames_to_process

	def _predict_batch(self, batch):
		"""Run inference on a batch using custom processing"""
		batch_inputs = self.preprocess(batch)
		
		inference_start = time.time()
		with torch.no_grad():
			outputs = self.model(batch_inputs)
		self.timing_stats.inference_time += time.time() - inference_start
		
		return outputs
	
	def _predict_batch_default(self, batch):
		"""Run inference on a batch using default predictor"""
		inference_start = time.time()
		predictions = [self.predictor(img) for i, img in enumerate(batch)]
		self.timing_stats.inference_time += time.time() - inference_start
		return predictions
	
	def _visualize_batch(self, batch, outputs, timestamps, frames_dir):
		"""Visualize results from custom batch processing"""
		vis_start = time.time()
		preprocess_start = time.time()
		
		# Store frames with detections
		frames_with_detections = []
		
		frameCount = 0
		for frame_idx, (output, timestamp) in enumerate(zip(outputs, timestamps)):
			instances = output["instances"].to("cpu")
			
			# Only process frames that have detections
			if len(instances) > 0:
				frameCount += 1
				boxes = instances.pred_boxes.tensor.numpy()
				scores = instances.scores.numpy()
				classes = instances.pred_classes.numpy()
				
				
				self.all_detections[f"Image_{frameCount}"] = []
				


				for box, score, class_id in zip(boxes, scores, classes):
					detection = {
						"bbox": box.tolist(),  # [x1, y1, x2, y2]
						"class": self.metadata.thing_classes[class_id],
						"confidence": float(score),
						"frame_time": time.time() - preprocess_start,
						"timestamp_seconds": timestamp
					}
					self.all_detections[f"Image_{frameCount}"] = {"Detection":detection}
				

				v = Visualizer(batch[frame_idx][:, :, ::-1], metadata=self.metadata, scale=1.0)
				v = v.draw_instance_predictions(instances)
				frames_with_detections.append({
					'frame': v.get_image()[:, :, ::-1],
					'timestamp': timestamp,
					'frame_idx': frameCount
				})
		
		# Save all frames with detections after processing
		# Delete existing files in frames directory before saving new ones
		for file in os.listdir(frames_dir):
			if file.endswith('.jpg'):
				try:
					os.remove(os.path.join(frames_dir, file))
				except Exception as e:
					self.logger.error(f"Error deleting file {file}: {str(e)}")
		
		# Save new frames
		for frame_info in frames_with_detections:
			timestamp_str = self._format_timestamp(frame_info['timestamp']).replace(':', '_')
			frame_path = os.path.join(frames_dir, f"frame_{frame_info['frame_idx']}_{timestamp_str}.jpg")
			
			try:
				# Save the frame image
				cv2.imwrite(frame_path, frame_info['frame'])
				self.logger.info(f"Saved detection frame: {frame_path}")
				
				# Update detections with frame path
				image_key = f"Image_{frame_info['frame_idx']}"
				if image_key in self.all_detections:
					self.all_detections[image_key]["frame_path"] = frame_path
			except Exception as e:

				self.logger.error(f"Error saving frame {frame_path}: {str(e)}")
				
		self.timing_stats.visualization_time += time.time() - vis_start
	
	def _visualize_batch_default(self, batch, predictions, timestamps):
		"""Visualize results from default batch processing"""
		vis_start = time.time()
		for i, (frame, pred, timestamp) in enumerate(zip(batch, predictions, timestamps)):
			if pred is None:
				continue
				
			try:
				# Create visualizer
				v = Visualizer(frame[:, :, ::-1], 
							MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0]), 
							scale=1.2)
				
				# Draw predictions
				output = v.draw_instance_predictions(pred["instances"].to("cpu"))
				
				# Save frame
				timestamp_str = self._format_timestamp(timestamp).replace(':', '_')
				frame_path = os.path.join("frames", f"frame_{i}_{timestamp_str}.jpg")
				#cv2.imwrite(frame_path, v.get_image()[:, :, ::-1])
				self.logger.info(f"Saved detection frame: {frame_path}")
				
				# Write to video if needed
				# if writer is not None:
				# 	writer.write(v.get_image()[:, :, ::-1])
					
			except Exception as e:
				self.logger.error(f"Visualization failed for frame {i}: {str(e)}")
		self.timing_stats.visualization_time += time.time() - vis_start
	
	def _format_timestamp(self, seconds):
		hours = int(seconds // 3600)
		minutes = int((seconds % 3600) // 60)
		seconds = int(seconds % 60)
		return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

def main():
	config = DetectionConfig(
		class_names=["Crack", "Obstacle", "Deposits", "Deformed", "Broken", "Joint Displaced", "Surface Damage", "Root"],
		colors=[[255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0], [255, 0, 255], [0, 255, 255], [128, 128, 128], [128, 0, 128]],
		model_path="Model/model_final.pth",
		config_path="Model/mask_rcnn_X_101_32x8d_FPN_3x.yaml",
	)
	
	detector = BatchDefectDetector(config)

	# Process video
	#input_path = r"C:\Users\sobha\Desktop\detectron2\Data\TestFilm\Closed circuit television (CCTV) sewer inspection.mp4"
	input_path = r"C:\Users\sobha\Desktop\detectron2\Data\E.Hormozi\14030828\14030828\1104160202120636299-1104160202120633024\1.mpg"
	output_path = os.path.join("output", os.path.basename(input_path))
	
	detector.logger.info(f"Processing video: {input_path}")
	frames_to_process = detector.process_video(input_path, output_path, batch_size=config.batch_size)

if __name__ == "__main__":
	main()
