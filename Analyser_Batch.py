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
		logging.basicConfig(level=logging.INFO)
		self.logger = logging.getLogger(__name__)
		setup_logger()

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
		from queue import Queue
		process_start = time.time()
		cap = cv2.VideoCapture(input_path)
		if not cap.isOpened():
			self.logger.error(f"Error opening video file {input_path}")
			return
		
		writer = self._initialize_video_writer(cap, output_path)
		frames_dir = os.path.splitext(output_path)[0] + "_frames"
		os.makedirs(frames_dir, exist_ok=True)
		
		total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
		fps = int(cap.get(cv2.CAP_PROP_FPS))
		
		# Calculate frame_interval based on frames per second
		frame_interval = fps  # Process 1 frame per second
		
		frames_to_process = []
		timestamps_to_process = []
		frame_queue = Queue()
		
		self.logger.info("Starting frame collection...")
		frame_collection_start = time.time()
		def read_frames():
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
			while True:
				item = frame_queue.get()
				if item is None:
					break
				frame_idx, frame = item
				frames_to_process.append(frame)
				timestamps_to_process.append(frame_idx / fps)
		
		with ThreadPoolExecutor(max_workers=4) as executor:
			executor.submit(read_frames)
			executor.submit(process_frames)
		
		self.timing_stats.frame_collection_time = time.time() - frame_collection_start
		self.logger.info(f"Frame collection complete. Total frames to process: {len(frames_to_process)}. "
						 f"Time taken: {self.timing_stats.frame_collection_time:.2f} seconds")
		
		batch_start_time = time.time()
		total_batch_time = 0
		all_predictions = []  # Store all predictions
		all_frames = []      # Store all frames
		all_timestamps = []  # Store all timestamps
		
		for i in range(0, len(frames_to_process), batch_size):
			batch = frames_to_process[i:i + batch_size]
			timestamps = timestamps_to_process[i:i + batch_size]
			
			self.logger.info(f"Processing batch {i // batch_size + 1} of {math.ceil(len(frames_to_process) / batch_size)}")
			
			# Only do predictions during batch loop
			if self.Custom_batch:
				outputs = self._predict_batch(batch)
				all_predictions.extend(outputs)
			else:
				predictions = self._predict_batch_default(batch)
				all_predictions.extend(predictions)
			
			all_frames.extend(batch)
			all_timestamps.extend(timestamps)
			
			batch_time = time.time() - batch_start_time
			total_batch_time += batch_time
			self.logger.info(f"Batch time: {batch_time:.2f} seconds")
			self.logger.info(f"Total time spent processing batches: {total_batch_time:.2f} seconds")
			batch_start_time = time.time()
		
		# After all batches, visualize and save results
		if self.Custom_batch:
			self._visualize_batch(all_frames, all_predictions, all_timestamps, frames_dir)
		else:
			self._visualize_batch_default(all_frames, all_predictions, all_timestamps)
		
		self.timing_stats.total_time = time.time() - process_start
		self.logger.info("\nTiming Statistics:")
		self.logger.info(f"Frame Collection Time: {self.timing_stats.frame_collection_time:.2f}s")
		self.logger.info(f"Preprocessing Time: {self.timing_stats.preprocessing_time:.2f}s")
		self.logger.info(f"Inference Time: {self.timing_stats.inference_time:.2f}s")
		self.logger.info(f"Visualization Time: {self.timing_stats.visualization_time:.2f}s")
		self.logger.info(f"Total Time: {self.timing_stats.total_time:.2f}s")
	
	def _initialize_video_writer(self, cap, output_path):
		fourcc = cv2.VideoWriter_fourcc(*"mp4v")
		fps = int(cap.get(cv2.CAP_PROP_FPS))
		width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
		height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
		return cv2.VideoWriter(output_path, fourcc, fps, (width, height))
	
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
		
		# Store frames with detections
		frames_with_detections = []
		
		for frame_idx, (output, timestamp) in enumerate(zip(outputs, timestamps)):
			instances = output["instances"].to("cpu")
			
			# Only process frames that have detections
			if len(instances) > 0:
				v = Visualizer(batch[frame_idx][:, :, ::-1], metadata=self.metadata, scale=1.0)
				v = v.draw_instance_predictions(instances)
				frames_with_detections.append({
					'frame': v.get_image()[:, :, ::-1],
					'timestamp': timestamp,
					'frame_idx': frame_idx
				})
				#writer.write(v.get_image()[:, :, ::-1])
		
		# Save all frames with detections after processing
		for frame_info in frames_with_detections:
			timestamp_str = self._format_timestamp(frame_info['timestamp']).replace(':', '_')
			frame_path = os.path.join(frames_dir, f"frame_{frame_info['frame_idx']}_{timestamp_str}.jpg")
			try:
				cv2.imwrite(frame_path, frame_info['frame'])
				self.logger.info(f"Saved detection frame: {frame_path}")
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
	input_path = r"C:\Users\sobha\Desktop\detectron2\Data\TestFilm\Closed circuit television (CCTV) sewer inspection.mp4"
	output_path = os.path.join("output", os.path.basename(input_path))
	
	detector.logger.info(f"Processing video: {input_path}")
	detector.process_video(input_path, output_path, batch_size=config.batch_size)

if __name__ == "__main__":
	main()
