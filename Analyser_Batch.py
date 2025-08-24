import sys
import os
import logging
import torch
import cv2
import time
import json
import math
import numpy as np
from typing import List, Dict
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
from queue import Queue
from statistics import mean
from tqdm import tqdm
import traceback

from detectron2.utils.logger import setup_logger
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import MetadataCatalog
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.data.transforms import ResizeShortestEdge

from TextExtractor.ExtrctInfo import TextExtractor
from Reporter.ExcelReporter import ExcelReporter
from utils.logger import ProgressLogger


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
    custom_batch: bool = True

class BatchDefectDetector:
    def __init__(self, config: DetectionConfig):
        self._setup_logging()
        self.cfg = self._initialize_config(config)
        self._initialize_model(config)
        self._setup_dataset(config.class_names, config.colors)
        
        self.timing_stats = TimingStats()
        self.custom_batch = config.custom_batch
        self.all_detections = {}
        self._setup_progress_logger()

    def _setup_progress_logger(self):
        # Initialize progress logger
        self.progress_logger = ProgressLogger()
        # Initialize stages with their weights
        self.stages = {
            "initialization": 5,
            "frame extraction": 10,
            "Ai detection": 40,
            "text extraction": 30,
            "excel reporting": 15
        }
        self.progress_logger.start_process(self.stages)

        # Update progress for initialization stage
        self.progress_logger.update_stage_progress(
            "initialization",
            80.0,
            {"status": "Setting up detector configuration"}
        )

    def _setup_logging(self):
        """Configure logging with detailed format"""
        # Create logs directory if it doesn't exist
        log_dir = "logs"
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
            
        log_file = os.path.join(log_dir, "defect_detection.log")
        
        # Configure root logger
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(sys.stdout),
                logging.FileHandler(log_file, mode='a')
            ]
        )
        
        # Create logger for this class
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        
        # Add file handler with rotation
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(
            logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        )
        self.logger.addHandler(file_handler)
        
        # Log startup message
        self.logger.info("Initializing BatchDefectDetector")

    def _initialize_model(self, config: DetectionConfig):
        """Initialize model and related components"""
        self.model = build_model(self.cfg)
        DetectionCheckpointer(self.model).load(config.model_path)
        self.model.eval()

        if len(self.cfg.DATASETS.TEST):
            self.metadata = MetadataCatalog.get(self.cfg.DATASETS.TEST[0])

        self.aug = ResizeShortestEdge(
            [self.cfg.INPUT.MIN_SIZE_TEST, self.cfg.INPUT.MIN_SIZE_TEST],
            self.cfg.INPUT.MAX_SIZE_TEST
        )
        self.input_format = self.cfg.INPUT.FORMAT
        assert self.input_format in ["RGB", "BGR"], self.input_format
        self.predictor = DefaultPredictor(self.cfg)

    def _setup_dataset(self, class_names: List[str], colors: List[List[int]]) -> None:
        """Setup custom dataset metadata"""
        MetadataCatalog.get("custom_dataset").set(thing_classes=class_names)
        self.metadata = MetadataCatalog.get("custom_dataset")
        self.metadata.thing_colors = colors

    def _initialize_config(self, config: DetectionConfig):
        """Initialize Detectron2 configuration"""
        cfg = get_cfg()
        cfg.merge_from_file(config.config_path)
        cfg.MODEL.WEIGHTS = config.model_path
        cfg.MODEL.DEVICE = config.device
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = config.score_threshold
        cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = config.nms_threshold
        cfg.MODEL.ROI_HEADS.CLASS_NAMES = config.class_names
        return cfg

    def preprocess(self, images):
        """Preprocess batch of images for inference"""
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
        self.progress_logger.complete_stage(
            "initialization",
            {"status": "Detector configuration setup complete"}
        )

        """Process video file in batches for object detection"""
        process_start = time.time()
        
        frames_to_process, timestamps_to_process = self._collect_frames(input_path)
        
        # Process frames in batches
        all_predictions, all_frames, all_timestamps = self._process_batches(
            frames_to_process, 
            timestamps_to_process,
            batch_size
        )
        
        # Visualize and save results
        frames_dir = os.path.join(output_path, "frames")
        os.makedirs(frames_dir, exist_ok=True)
        
        if self.custom_batch:
            self._visualize_batch(all_frames, all_predictions, all_timestamps, frames_dir)
        else:
            self._visualize_batch_default(all_frames, all_predictions, all_timestamps)

        self._extract_and_process_text(all_frames)
        self.modify_detection_baseExperience()
        self._save_results(output_path)
        self._log_timing_stats(process_start)

        return frames_to_process
    
    def modify_detection_baseExperience(self):

        try:
            """Modify detection base experience"""
            # Group detections by text_info (distance)
            self.all_detectionsBaseCopy = self.all_detections.copy()
            distance_groups = {}
            for frame_key, frame_data in self.all_detections.items():
                distance = frame_data["text_info"]
                if distance not in distance_groups:
                    distance_groups[distance] = []
                distance_groups[distance].append((frame_key, frame_data))

            # Process detection groups based on detection count
            filtered_groups = {}
            
            for distance, group in distance_groups.items():
                # Check if all frames in this group have exactly 1 detection
                all_single_detection = all(len(frame_data["Detection"]) == 1 for _, frame_data in group)
                
                if all_single_detection:
                    # If all frames have exactly 1 detection, keep the most repetitive class
                    class_counts = {}
                    for _, frame_data in group:
                        defect_class = frame_data["Detection"][0]["class"]
                        class_counts[defect_class] = class_counts.get(defect_class, 0) + 1
                    
                    # Find the most common class
                    most_common_class = max(class_counts.items(), key=lambda x: x[1])[0] if class_counts else None
                    
                    # Keep only frames with the most common class
                    if most_common_class:
                        filtered_frames = []
                        for frame_key, frame_data in group:
                            if frame_data["Detection"][0]["class"] == most_common_class:
                                filtered_frames.append((frame_key, frame_data))
                        
                        filtered_groups[distance] = filtered_frames
                        self.logger.info(f"Distance {distance}: Kept {len(filtered_frames)} frames with most common class '{most_common_class}'")
                else:
                    # If some frames have multiple detections, keep only those with 2+ detections
                    multi_detection_frames = []
                    for frame_key, frame_data in group:
                        if len(frame_data["Detection"]) >= 2:
                            multi_detection_frames.append((frame_key, frame_data))
                    
                    if multi_detection_frames:
                        filtered_groups[distance] = multi_detection_frames
                        self.logger.info(f"Distance {distance}: Kept {len(multi_detection_frames)} frames with 2+ detections")
            
            
            # First, filter out duplicate detections within each frame
            for distance, group in filtered_groups.items():
                for frame_idx, (frame_key, frame_data) in enumerate(group):
                    # Track unique classes we've seen in this frame
                    seen_classes = set()
                    unique_detections = []
                    
                    # Process each detection and keep only unique classes
                    for detection in frame_data["Detection"]:
                        defect_class = detection["class"]
                        if defect_class not in seen_classes:
                            seen_classes.add(defect_class)
                            unique_detections.append(detection)
                    
                    # Update the frame data with unique detections
                    frame_data["Detection"] = unique_detections
                    group[frame_idx] = (frame_key, frame_data)
                    
                # Update the distance group with the modified frames
                distance_groups[distance] = group
                
            # Now filter out duplicate frames that have identical class combinations
            for distance, group in distance_groups.items():
                # Track unique class combinations we've seen
                unique_class_combinations = []
                unique_frames = []
                
                for frame_key, frame_data in group:
                    # Get sorted list of classes in this frame
                    frame_classes = sorted([d["class"] for d in frame_data["Detection"]])
                    class_combination = tuple(frame_classes)
                    
                    # If we haven't seen this combination before, keep it
                    if class_combination not in unique_class_combinations:
                        unique_class_combinations.append(class_combination)
                        unique_frames.append((frame_key, frame_data))
                
                # Update the distance group with unique frames
                distance_groups[distance] = unique_frames
                
            self.logger.info(f"Filtered detection groups to keep only unique class combinations")

            # Filter out frames that are too close to each other (less than 0.15m apart)
            self.logger.info("Filtering frames that are too close to each other (< 0.15m)")
            
            # Convert distance strings to floats and sort them
            distances = sorted([(d) for d in distance_groups.keys()])
            
            # Create a new filtered dictionary to store the results
            final_filtered_groups = {}
            
            # Process distances in order
            i = 0
            while i < len(distances):
                current_distance = distances[i]
                current_key = str(current_distance)
                
                # Add the first distance to our filtered results
                if i == 0:
                    final_filtered_groups[current_key] = distance_groups[current_key]
                    i += 1
                    continue
                
                # Get the previous distance we kept
                prev_distance = distances[i-1]
                prev_key = str(prev_distance)
                
                # Check if current distance is too close to previous one
                # Convert to float for numerical comparison
                if float(current_distance) - float(prev_distance) < 0.15:
                    # Compare the number of detections
                    current_frames = distance_groups[current_key]
                    prev_frames = distance_groups[current_key]  # Use from final_filtered_groups instead
                    
                    # Count total detections in current distance group
                    current_detection_count = sum(len(frame_data["Detection"]) for _, frame_data in current_frames)
                    
                    # Count total detections in previous distance group
                    prev_detection_count = sum(len(frame_data["Detection"]) for _, frame_data in prev_frames)
                    
                    if current_detection_count > prev_detection_count:
                        # Current has more detections, replace previous
                        if prev_key in final_filtered_groups:
                            del final_filtered_groups[prev_key]
                        final_filtered_groups[current_key] = current_frames
                    elif current_detection_count == prev_detection_count:
                        # Equal detection counts, compare confidence scores
                        current_max_confidence = max(
                            [detection.get("confidence", 0) for _, frame_data in current_frames 
                            for detection in frame_data["Detection"]], 
                            default=0
                        )
                        prev_max_confidence = max(
                            [detection.get("confidence", 0) for _, frame_data in prev_frames 
                            for detection in frame_data["Detection"]], 
                            default=0
                        )
                        
                        if current_max_confidence > prev_max_confidence:
                            # Current has higher confidence, replace previous
                            if prev_key in final_filtered_groups:
                                del final_filtered_groups[prev_key]
                            final_filtered_groups[current_key] = current_frames
                        # If previous has higher confidence, keep it (do nothing)
                else:
                    # Distance is not too close, keep current frame
                    final_filtered_groups[current_key] = distance_groups[current_key]
                
                i += 1
            
            # Replace distance_groups with our filtered version
            distance_groups = final_filtered_groups
            
            # Filter out similar detections across different distance groups using IoU
            self.logger.info("Filtering similar detections across distance groups based on bounding box similarity")
            
            def _calculate_bbox_similarity(self, bbox1, bbox2):
                """Calculate IoU (Intersection over Union) between two bounding boxes"""
                # Extract coordinates
                x1_1, y1_1, x2_1, y2_1 = bbox1
                x1_2, y1_2, x2_2, y2_2 = bbox2
                
                # Calculate intersection area
                x_left = max(x1_1, x1_2)
                y_top = max(y1_1, y1_2)
                x_right = min(x2_1, x2_2)
                y_bottom = min(y2_1, y2_2)
                
                # Check if there is an intersection
                if x_right < x_left or y_bottom < y_top:
                    return 0.0
                    
                intersection_area = (x_right - x_left) * (y_bottom - y_top)
                
                # Calculate union area
                bbox1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
                bbox2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
                union_area = bbox1_area + bbox2_area - intersection_area
                
                # Calculate IoU
                iou = intersection_area / union_area if union_area > 0 else 0.0
                return iou
            
            # Convert method to function for use within this scope
            calculate_bbox_similarity = lambda bbox1, bbox2: _calculate_bbox_similarity(self, bbox1, bbox2)
            
            # Group similar detections across distance groups
            similarity_groups = []
            processed_frames = set()
            
            # Sort distance groups by distance for consistent processing
            sorted_distances = sorted(distance_groups.keys(), key=lambda x: float(x))
            
            for i, distance1 in enumerate(sorted_distances):
                group1 = distance_groups[distance1]
                
                for frame1_key, frame1_data in group1:
                    if frame1_key in processed_frames:
                        continue
                    
                    # Create a new similarity group
                    current_group = [(frame1_key, frame1_data)]
                    processed_frames.add(frame1_key)
                    
                    # Compare with other distance groups
                    for j in range(i+1, len(sorted_distances)):
                        distance2 = sorted_distances[j]
                        group2 = distance_groups[distance2]
                        
                        for frame2_key, frame2_data in group2:
                            if frame2_key in processed_frames:
                                continue
                            
                            # Check if detections are similar using IoU
                            is_similar = False
                            
                            # Compare all detections between the two frames
                            for detection1 in frame1_data["Detection"]:
                                for detection2 in frame2_data["Detection"]:
                                    # Skip if classes are different
                                    if detection1["class"] != detection2["class"]:
                                        continue
                                        
                                    # Calculate IoU between bounding boxes
                                    iou = calculate_bbox_similarity(detection1["bbox"], detection2["bbox"])
                                    
                                    # If IoU is above threshold, consider them similar
                                    if iou > 0.6:  # Threshold can be adjusted
                                        is_similar = True
                                        break
                                
                                if is_similar:
                                    break
                            
                            # If similar, add to current group and mark as processed
                            if is_similar:
                                current_group.append((frame2_key, frame2_data))
                                processed_frames.add(frame2_key)
                    
                    # Add the group to similarity groups if it has at least one frame
                    if current_group:
                        similarity_groups.append(current_group)
            
            # For each similarity group, keep only the frame with highest confidence or most detections
            optimized_groups = {}
            
            for group in similarity_groups:
                if not group:
                    continue
                    
                # Find the best frame in the group
                best_frame_key = None
                best_frame_data = None
                max_detection_count = -1
                max_confidence = -1
                
                for frame_key, frame_data in group:
                    # Count detections
                    detection_count = len(frame_data["Detection"])
                    
                    # Calculate average confidence
                    avg_confidence = sum(d.get("confidence", 0) for d in frame_data["Detection"]) / detection_count if detection_count > 0 else 0
                    
                    # Prioritize by detection count, then by confidence
                    if detection_count > max_detection_count or (detection_count == max_detection_count and avg_confidence > max_confidence):
                        max_detection_count = detection_count
                        max_confidence = avg_confidence
                        best_frame_key = frame_key
                        best_frame_data = frame_data
                
                # Add the best frame to optimized groups
                if best_frame_key and best_frame_data:
                    distance = best_frame_data["text_info"]
                    if distance not in optimized_groups:
                        optimized_groups[distance] = []
                    optimized_groups[distance].append((best_frame_key, best_frame_data))
            
            # Replace distance_groups with optimized version
            distance_groups = optimized_groups
            
            self.logger.info(f"After similarity filtering: kept {len(distance_groups)} distance groups with {sum(len(group) for group in distance_groups.values())} frames")
            # Update all_detections with filtered results
            self.all_detections = {}
            for distance, frames in distance_groups.items():
                for frame_key, frame_data in frames:
                    self.all_detections[frame_key] = frame_data
            
            self.logger.info(f"After proximity filtering: kept {len(distance_groups)} distance groups")

        except Exception as e:
            self.logger.error(f"Error modifying detection base experience: {str(e)}")
        

        # Remove any detections not in filtered groups and their associated frames
        for key, frame_data in self.all_detectionsBaseCopy.items():
            if key not in self.all_detections:

                self.logger.info(f"Frame {key} was in original detections but removed in filtering")               # Remove associated frame image if it exists
                if "frame_path" in frame_data:
                    try:
                        os.remove(frame_data["frame_path"])
                        base_dir = os.path.dirname(frame_data["frame_path"])
                        Predictions_dir = os.path.join(base_dir, "predictions_frames")
                        PredImageName = os.path.join(Predictions_dir, os.path.basename(frame_data["frame_path"]))
                        os.remove(PredImageName)
                        self.logger.info(f"Removed frame image: {frame_data['frame_path']}")
                    except Exception as e:
                        self.logger.error(f"Error removing frame image {frame_data['frame_path']}: {str(e)}")


    def _collect_frames(self, input_path: str):
        """Collect frames from video file"""
        
        frames = []  # Initialize frames list
        cap = cv2.VideoCapture(input_path)
        ret, first_frame = cap.read()
        if ret:
            frames.append(first_frame)
            self.text_extractor._get_user_roi(first_frame)
        
        if not cap.isOpened():
            self.logger.error(f"Error opening video file {input_path}")
            return [], []
            
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        frame_interval = fps
        
        frames_to_process = []
        timestamps_to_process = []
        frame_queue = Queue()
        
        self.logger.info("Starting frame collection...")
        frame_collection_start = time.time()
        with ThreadPoolExecutor(max_workers=4) as executor:
            executor.submit(self._read_frames, cap, total_frames, frame_interval, frame_queue)
            executor.submit(self._process_frame_queue, frame_queue, frames_to_process, timestamps_to_process, fps)
            
        self.timing_stats.frame_collection_time = time.time() - frame_collection_start
        self.logger.info(f"Frame collection complete. Total frames: {len(frames_to_process)}. "
                        f"Time taken: {self.timing_stats.frame_collection_time:.2f}s")
        
        self.progress_logger.complete_stage(
            "frame extraction",
            {"status": "Frame collection complete"}
        )
                        

        return frames_to_process, timestamps_to_process

    def _read_frames(self, cap, total_frames, frame_interval, frame_queue):
        """Read frames from video capture"""
        frame_idx = 0
        pbar = tqdm(total=total_frames, desc="Reading frames",colour="cyan")
        while frame_idx < total_frames:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % frame_interval == 0:
                frame_queue.put((frame_idx, frame))
            frame_idx += 1
            pbar.update(1)
            
            # Update progress for frame extraction stage
            progress = (frame_idx / total_frames) * 100
            if frame_idx % 800 == 0:
                self.progress_logger.update_stage_progress(
                    "frame extraction",
                    progress,
                    {"frames_processed": frame_idx, "total_frames": total_frames}
                )
        frame_queue.put(None)
        pbar.close()

    def _process_frame_queue(self, frame_queue, frames_to_process, timestamps_to_process, fps):
        """Process frames from queue"""
        while True:
            item = frame_queue.get()
            if item is None:
                break
            frame_idx, frame = item
            frames_to_process.append(frame)
            timestamps_to_process.append(frame_idx / fps)

    def _process_batches(self, frames, timestamps, batch_size):
        """Process frames in batches"""
        total_batches = math.ceil(len(frames) / batch_size)
        self.logger.info(f"Starting batch processing of {len(frames)} frames in {total_batches} batches")
        
        all_predictions = []
        all_frames = []
        all_timestamps = []
        batch_start_time = time.time()
        total_batch_time = 0
        
        pbar = tqdm(range(0, len(frames), batch_size), desc="Processing batches",colour="cyan")
        for i in pbar:
            batch = frames[i:i + batch_size]
            batch_timestamps = timestamps[i:i + batch_size]

            current_batch = i // batch_size + 1
            pbar.set_description(f"Processing batch {current_batch}/{total_batches}")
            
            # Run inference
            if self.custom_batch:
                outputs = self._predict_batch(batch)
                all_predictions.extend(outputs)
            else:
                predictions = self._predict_batch_default(batch)
                all_predictions.extend(predictions)
            
            all_frames.extend(batch)
            all_timestamps.extend(batch_timestamps)
            
            # Log timing
            batch_time = time.time() - batch_start_time
            total_batch_time += batch_time
            pbar.set_postfix({"Batch time": f"{batch_time:.2f}s", "Total time": f"{total_batch_time:.2f}s"})
            batch_start_time = time.time()
            
            # Update progress for AI detection stage
            progress = (current_batch / total_batches) * 100
            if current_batch % 3 == 0:
                self.progress_logger.update_stage_progress(
                    "Ai detection",
                    progress,
                {
                    "batch": current_batch,
                    "total_batches": total_batches,
                    "batch_time": f"{batch_time:.2f}s",
                    "total_time": f"{total_batch_time:.2f}s"
                }
                )
        
        self.progress_logger.complete_stage(
            "Ai detection",
            {"status": "AI detection complete"}
        )
        return all_predictions, all_frames, all_timestamps

    def _predict_batch(self, batch):
        """Run inference on batch using custom processing"""
        batch_inputs = self.preprocess(batch)
        inference_start = time.time()
        with torch.no_grad():
            outputs = self.model(batch_inputs)
        self.timing_stats.inference_time += time.time() - inference_start
        return outputs

    def _predict_batch_default(self, batch):
        """Run inference on batch using default predictor"""
        inference_start = time.time()
        predictions = [self.predictor(img) for img in batch]
        self.timing_stats.inference_time += time.time() - inference_start
        return predictions

    def _visualize_batch(self, batch, outputs, timestamps, frames_dir):
        """Visualize results from custom batch processing"""
        vis_start = time.time()
        Normal_frames = []
        frames_with_detections = []
        frame_count = 0
        
        pbar = tqdm(enumerate(zip(outputs, timestamps)), total=len(outputs), desc="Visualizing detections",colour="cyan")
        for frame_idx, (output, timestamp) in pbar:
            instances = output["instances"].to("cpu")
            
            if len(instances) > 0:
                frame_count += 1
                self._process_detections(instances, frame_count, timestamp)
                
                frame_to_save = batch[frame_idx] 
                frame_to_save_with_predictions = self._draw_predictions(batch[frame_idx], instances)
                               
                Normal_frames.append({
                    'frame': frame_to_save,
                    'timestamp': timestamp,
                    'frame_idx': frame_count
                })
                frames_with_detections.append({
                    'frame': frame_to_save_with_predictions,
                    'timestamp': timestamp,
                    'frame_idx': frame_count
                })

        # Save frames with predictions
        predictions_frames_dir = os.path.join(frames_dir, "predictions_frames")
        os.makedirs(predictions_frames_dir, exist_ok=True)
        self._save_detection_frames(predictions_frames_dir, frames_with_detections, _IsNormalFrame=False)
        # Save normal frames
        self._save_detection_frames(frames_dir, Normal_frames, _IsNormalFrame=True)

        
        self.timing_stats.visualization_time += time.time() - vis_start

    def _process_detections(self, instances, frame_count, timestamp):
        """Process detection instances for a frame"""
        boxes = instances.pred_boxes.tensor.numpy()
        scores = instances.scores.numpy()
        classes = instances.pred_classes.numpy()
        
        detections = []
        for box, score, class_id in zip(boxes, scores, classes):
            detection = {
                "bbox": box.tolist(),
                "class": self.metadata.thing_classes[class_id],
                "confidence": float(score),
                "frame_time": time.time(),
                "timestamp_seconds": timestamp
            }
            detections.append(detection)
            
        self.all_detections[f"Image_{frame_count}"] = {"Detection": detections}

    def _should_draw_predictions(self):
        """Check if predictions should be drawn on frames"""
        return False

    def _draw_predictions(self, frame, instances):
        """Draw prediction visualizations on frame"""
        v = Visualizer(frame[:, :, ::-1], metadata=self.metadata, scale=1.0)
        v = v.draw_instance_predictions(instances)
        return v.get_image()[:, :, ::-1]

    def _save_detection_frames(self, frames_dir, frames_with_detections, _IsNormalFrame):
        """Save frames with detections to disk"""
        # Clear existing frames
        self._clear_existing_frames(frames_dir)
        
        # Save new frames
        pbar = tqdm(frames_with_detections, desc="Saving detection frames", colour="cyan")
        for frame_info in pbar:
            timestamp_str = self._format_timestamp(frame_info['timestamp']).replace(':', '_')
            frame_path = os.path.join(frames_dir, f"frame_{frame_info['frame_idx']}_{timestamp_str}.jpg")
            
            try:
                cv2.imwrite(frame_path, frame_info['frame'])
                self.logger.info(f"Saved detection frame: {frame_path}")
                
                # Update detection metadata with frame path if needed
                if _IsNormalFrame:
                    self._update_detection_metadata(frame_info['frame_idx'], frame_path)
            except Exception as e:
                self.logger.error(f"Error saving frame {frame_path}: {str(e)}")
    
    def _clear_existing_frames(self, frames_dir):
        """Clear existing jpg frames from the directory"""
        for file in os.listdir(frames_dir):
            if file.endswith('.jpg'):
                try:
                    os.remove(os.path.join(frames_dir, file))
                except Exception as e:
                    self.logger.error(f"Error deleting file {file}: {str(e)}")
    
    def _update_detection_metadata(self, frame_idx, frame_path):
        """Update detection metadata with frame path"""
        image_key = f"Image_{frame_idx}"
        if image_key in self.all_detections:
            self.all_detections[image_key]["frame_path"] = frame_path

    def _extract_and_process_text(self, frames):
        """Extract and process text from frames with detections"""
        self.logger.info("Starting text extraction from frames with detections...")
        text_extraction_start = time.time()
        

        
        
        self.distance_base = 0
        distance_means = []
        prev_timestamp = 0
        prev_distance = 0
        
        # Sort detections by timestamp
        self.all_detections = dict(sorted(
            self.all_detections.items(),
            key=lambda item: item[1]["Detection"][0]["timestamp_seconds"]
        ))
        
        pbar = tqdm(self.all_detections.items(), desc="Processing frame text", colour="cyan")
        total_frames = len(self.all_detections)
        
        for i, (frame_key, frame_data) in enumerate(pbar):
            result = self._process_frame_text(
                frame_key, frame_data,
                distance_means,
                prev_timestamp, prev_distance
            )
            
            # Update first_frame and prev values if needed

            prev_timestamp = result.get('timestamp', prev_timestamp)
            prev_distance = result.get('distance', prev_distance)
            
            # Update progress for text extraction stage
            progress = ((i + 1) / total_frames) * 100
            if i % 10 == 0:
                self.progress_logger.update_stage_progress(
                    "text extraction",
                    progress,
                    {"frames_processed": i + 1, "total_frames": total_frames}
                )
        self.progress_logger.complete_stage(
            "text extraction",
            {"status": "Text extraction complete"}
        )

        self.timing_stats.text_extraction_time = time.time() - text_extraction_start
        self.logger.info(f"Text extraction completed. Time taken: {self.timing_stats.text_extraction_time:.2f}s")


    def _process_frame_text(self, frame_key, frame_data,
                           distance_means,
                           prev_timestamp, prev_distance):
        """Process text for a single frame"""
        
        self.logger.info(f"Extracting text from frame {frame_key}...")
        result = {
            'timestamp': prev_timestamp,
            'distance': prev_distance
        }
        
        try:
            text_info = self.text_extractor.extract_text_from_video_frame(
                frame_path=frame_data["frame_path"],
                UseFullOCR=False
            )

            
            current_distance, current_timestamp = self._extract_distance_and_timestamp(
                text_info, frame_data, prev_distance
            )
            
                
            distance_means.append(current_distance)
            mean_distance = mean(distance_means)
            
            distance_diff = abs(current_distance - self.distance_base)
            time_diff = current_timestamp - prev_timestamp
            
            frame_data["text_info"] = self._update_distance(
                current_distance, distance_diff,
                time_diff, frame_data
            )
            
            result['timestamp'] = current_timestamp
            result['distance'] = float(frame_data["text_info"])
            self.logger.info(f"Extracted final distance info: {text_info}, first distance: {current_distance}")
            
        except (ValueError, IndexError) as e:
            self.logger.error(f"Error processing text in frame {frame_key}: {str(e)}")
            frame_data["text_info"] = str(self.distance_base)
            
        return result

    def _extract_distance_and_timestamp(self, text_info, frame_data, prev_distance):
        """Extract distance and timestamp from text"""
        
        split_text = text_info.split(" ")
        text = split_text[0]
        
        # Handle multiple decimals or missing decimal point
        if len(text.split('.')) > 2 or \
           (len(text) < len(str(prev_distance)) and '.' not in text):
            text_info = self.text_extractor.extract_text_from_video_frame(
                frame_path=frame_data["frame_path"],
                UseFullOCR=True
            )

            text = text_info.split(" ")[0]
            
        return float(text), frame_data["Detection"][0]["timestamp_seconds"]

    def _update_distance(self, current_distance, distance_diff,
                        time_diff, frame_data):
        """Update and validate distance measurement
        
        Args:
            current_distance (float): The current distance measurement
            distance_diff (float): Absolute difference from base distance
            time_diff (float): Time elapsed since previous measurement
            frame_data (dict): Frame metadata including path
            
        Returns:
            str: Updated distance value as string
        """
        # Avoid global variable usage by accessing class attribute instead
        
        # Define validation thresholds for distance measurements
        MAX_DISTANCE_DIFF = 2.0
        MIN_DISTANCE_DIFF = 0.25
        MAX_TIME_GAP = 20.0
        
        # Check if measurement is valid based on defined thresholds
        is_valid_measurement = (
            (current_distance >= self.distance_base or distance_diff < MIN_DISTANCE_DIFF) and
            distance_diff <= MAX_DISTANCE_DIFF and 
            time_diff < MAX_TIME_GAP
        )
        
        if is_valid_measurement:
            # Update base distance if current is greater
            self.distance_base = max(self.distance_base, current_distance)
            return str(self.distance_base)
        
        # If measurement is invalid, attempt to recover with full OCR
        text_info = self.text_extractor.extract_text_from_video_frame(
            frame_path=frame_data["frame_path"],
            UseFullOCR=True
        )
        
        try:
            text = text_info.split(" ")[0]
            new_distance = float(text)
            self.distance_base = max(self.distance_base, new_distance)
            return str(self.distance_base)
        except (ValueError, IndexError):
            # Return existing base distance if recovery fails
            self.logger.debug(f"Failed to extract valid distance from '{text_info}'")
            return str(self.distance_base)


    def _save_results(self, output_path):
        """Save detection results to JSON"""
        base_name = os.path.basename(output_path).split("_")[0]
        
        json_output_path = os.path.join(output_path,  "frames_detections.json")
        with open(json_output_path, 'w') as f:
            json.dump(self.all_detections, f, indent=4)
        self.logger.info(f"Detection data saved to {json_output_path}")

    def _log_timing_stats(self, process_start):
        """Log final timing statistics"""
        self.timing_stats.total_time = time.time() - process_start
        self.logger.info("\nTiming Statistics:")
        self.logger.info(f"Frame Collection Time: {self.timing_stats.frame_collection_time:.2f}s")
        self.logger.info(f"Preprocessing Time: {self.timing_stats.preprocessing_time:.2f}s")
        self.logger.info(f"Inference Time: {self.timing_stats.inference_time:.2f}s")
        self.logger.info(f"Visualization Time: {self.timing_stats.visualization_time:.2f}s")
        self.logger.info(f"Text extraction Time: {self.timing_stats.text_extraction_time:.2f}s")
        self.logger.info(f"Total Time: {self.timing_stats.total_time:.2f}s")


    def _format_timestamp(self, seconds):
        """Format seconds into HH:MM:SS string"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        seconds = int(seconds % 60)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

def main():
    try:
        if len(sys.argv) < 2:
            print("Usage: python YourPythonScript.py <video_path1> <video_path2> ...")
            return

        input_pathes = sys.argv[1:]
        print(input_pathes)
    
        # Get the directory where the current script is located
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))

        # Build the path to 'model/v2'
        model_path = os.path.join(BASE_DIR, 'Model', 'Model V.2.8.0', 'model_final.pth')
        config_path = os.path.join(BASE_DIR, 'Model', 'Model V.2.8.0', 'mask_rcnn_X_101_32x8d_FPN_3x.yaml')
        
        config = DetectionConfig(
            class_names=[
                "Crack", "Obstacle", "Deposits", "Deformed",
                "Broken", "Joint Displaced", "Surface Damage", "Root"
            ],
            colors=[
                [255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0],
                [255, 0, 255], [0, 255, 255], [128, 128, 128], [128, 0, 128]
            ],
            model_path=model_path,
            config_path=config_path,
            batch_size=32
        )
        
        detector = BatchDefectDetector(config)
        detector.text_extractor = TextExtractor()

        # input_pathes is now a list of video paths from command line arguments

        for input_path in input_pathes:
            if not os.path.isfile(input_path):
                print(f"Invalid video path provided: {input_path}")
                continue
            # Create output path by joining directory and filename using os.path for cross-platform compatibility
            output_path = os.path.join(os.path.dirname(input_path), os.path.splitext(os.path.basename(input_path))[0] + "_output")
            
            # Ensure output directory exists
            os.makedirs(output_path, exist_ok=True)

            
            detector.logger.info(f"Processing video: {input_path}")
            detector._setup_progress_logger()
            detector.process_video(input_path, output_path, batch_size=config.batch_size)
            reporter = ExcelReporter(
                input_path = output_path,
                excelOutPutName = "Condition-Details.xlsx",
                progress_logger = detector.progress_logger
            )
            reporter.generate_report()
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        time.sleep(15)
        return

if __name__ == "__main__":
    main()
