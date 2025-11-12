import sys
import os
import cv2
import torch
import json
import pickle
import numpy as np
from Analyser_Batch import BatchDefectDetector, DetectionConfig
from TextExtractor.ExtrctInfo import TextExtractor
from Reporter.ExcelReporter import ExcelReporter

class ComponentTester:
    """Test individual components of BatchDefectDetector."""
    
    def __init__(self, video_path: str, model_path: str, config_path: str, cache_dir: str = "test_cache"):
        self.video_path = video_path
        self.model_path = model_path
        self.config_path = config_path
        self.detector = None
        self.test_results = {}
        self.cache_dir = cache_dir
        self.use_cache = True
        os.makedirs(self.cache_dir, exist_ok=True)
        
    def setup_detector(self):
        """Initialize detector for testing."""
        config = DetectionConfig(
            class_names=[
                "Crack", "Obstacle", "Deposits", "Deformed",
                "Broken", "Joint Displaced", "Surface Damage", "Root"
            ],
            colors=[
                [255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0],
                [255, 0, 255], [0, 255, 255], [128, 128, 128], [128, 0, 128]
            ],
            model_path=self.model_path,
            config_path=self.config_path,
            batch_size=4
        )
        
        text_extractor = TextExtractor(lazy_load=True, use_cache=True)
        self.detector = BatchDefectDetector(config, text_extractor)
        print("✓ Detector initialized")
        
    def _save_frames_cache(self, frames, timestamps):
        """Save frames and timestamps to cache."""
        cache_path = os.path.join(self.cache_dir, "frames_cache.pkl")
        frames_dir = os.path.join(self.cache_dir, "frames")
        os.makedirs(frames_dir, exist_ok=True)
        
        # Save frames as individual images
        frame_files = []
        for idx, frame in enumerate(frames):
            frame_file = os.path.join(frames_dir, f"frame_{idx:05d}.jpg")
            cv2.imwrite(frame_file, frame)
            frame_files.append(frame_file)
        
        # Save metadata
        metadata = {
            'timestamps': timestamps,
            'frame_files': frame_files,
            'frame_count': len(frames),
            'frame_shape': frames[0].shape if frames else None
        }
        
        with open(cache_path, 'wb') as f:
            pickle.dump(metadata, f)
        
        print(f"✓ Cached {len(frames)} frames to {self.cache_dir}")
    
    def _load_frames_cache(self):
        """Load frames and timestamps from cache."""
        cache_path = os.path.join(self.cache_dir, "frames_cache.pkl")
        
        if not os.path.exists(cache_path):
            return None, None
        
        with open(cache_path, 'rb') as f:
            metadata = pickle.load(f)
        
        # Load frames from images
        frames = []
        for frame_file in metadata['frame_files']:
            if os.path.exists(frame_file):
                frame = cv2.imread(frame_file)
                frames.append(frame)
            else:
                print(f"✗ Cache frame missing: {frame_file}")
                return None, None
        
        print(f"✓ Loaded {len(frames)} frames from cache")
        return frames, metadata['timestamps']
    
    def test_frame_collection(self, use_cache=True):
        """Test frame collection framework."""
        print("\n=== Testing Frame Collection ===")
        
        # Try to load from cache first
        if use_cache:
            frames, timestamps = self._load_frames_cache()
            if frames is not None and timestamps is not None:
                print("✓ Using cached frames")
                self.test_results['frame_collection'] = {
                    'status': 'PASS (cached)',
                    'frame_count': len(frames),
                    'timestamp_count': len(timestamps),
                    'first_timestamp': timestamps[0] if timestamps else None,
                    'last_timestamp': timestamps[-1] if timestamps else None,
                    'frame_shape': frames[0].shape if frames else None
                }
                return frames, timestamps
        
        # Collect frames if cache not available or use_cache=False
        try:
            frames, timestamps = self.detector._collect_frames(self.video_path)
            
            # Save to cache
            self._save_frames_cache(frames, timestamps)
            
            self.test_results['frame_collection'] = {
                'status': 'PASS',
                'frame_count': len(frames),
                'timestamp_count': len(timestamps),
                'first_timestamp': timestamps[0] if timestamps else None,
                'last_timestamp': timestamps[-1] if timestamps else None,
                'frame_shape': frames[0].shape if frames else None
            }
            
            print(f"✓ Collected {len(frames)} frames")
            print(f"✓ Frame shape: {frames[0].shape}")
            print(f"✓ Timestamp range: {timestamps[0]:.2f}s - {timestamps[-1]:.2f}s")
            
            return frames, timestamps
            
        except Exception as e:
            self.test_results['frame_collection'] = {
                'status': 'FAIL',
                'error': str(e)
            }
            print(f"✗ Frame collection failed: {e}")
            raise
            
    def _save_predictions_cache(self, predictions):
        """Save predictions to cache."""
        cache_path = os.path.join(self.cache_dir, "predictions_cache.pkl")
        
        # Convert predictions to CPU and serialize
        serializable_predictions = []
        for pred in predictions:
            instances = pred["instances"].to("cpu")
            serializable_pred = {
                'pred_boxes': instances.pred_boxes.tensor.numpy(),
                'scores': instances.scores.numpy(),
                'pred_classes': instances.pred_classes.numpy(),
                'image_size': instances.image_size
            }
            
            if hasattr(instances, 'pred_masks') and instances.pred_masks is not None:
                serializable_pred['masks'] = instances.pred_masks.numpy()
            
            serializable_predictions.append(serializable_pred)
        
        with open(cache_path, 'wb') as f:
            pickle.dump(serializable_predictions, f)
        
        print(f"✓ Cached {len(predictions)} predictions to {self.cache_dir}")
    
    def _load_predictions_cache(self):
        """Load predictions from cache."""
        cache_path = os.path.join(self.cache_dir, "predictions_cache.pkl")
        
        if not os.path.exists(cache_path):
            return None
        
        with open(cache_path, 'rb') as f:
            serializable_predictions = pickle.load(f)
        
        # Reconstruct predictions
        from detectron2.structures import Instances, Boxes
        
        predictions = []
        for pred_data in serializable_predictions:
            instances = Instances(pred_data['image_size'])
            instances.pred_boxes = Boxes(torch.from_numpy(pred_data['pred_boxes']))
            instances.scores = torch.from_numpy(pred_data['scores'])
            instances.pred_classes = torch.from_numpy(pred_data['pred_classes'])
            
            if 'masks' in pred_data:
                instances.pred_masks = torch.from_numpy(pred_data['masks'])
            
            predictions.append({"instances": instances})
        
        print(f"✓ Loaded {len(predictions)} predictions from cache")
        return predictions
    
    def test_detection_ai(self, frames, batch_size=4, use_cache=True):
        """Test AI detection on collected frames."""
        print("\n=== Testing AI Detection ===")
        
        # Try to load from cache first
        if use_cache:
            all_predictions = self._load_predictions_cache()
            if all_predictions is not None:
                print("✓ Using cached predictions")
                
                # Calculate stats
                total_detections = sum(len(p["instances"]) for p in all_predictions)
                detection_stats = {}
                for output in all_predictions:
                    instances = output["instances"].to("cpu")
                    classes = instances.pred_classes.numpy()
                    for class_id in classes:
                        class_name = self.detector.metadata.thing_classes[class_id]
                        detection_stats[class_name] = detection_stats.get(class_name, 0) + 1
                
                self.test_results['detection_ai'] = {
                    'status': 'PASS (cached)',
                    'total_predictions': len(all_predictions),
                    'total_detections': total_detections,
                    'detections_by_class': detection_stats
                }
                
                print(f"✓ Processed {len(all_predictions)} frames")
                print(f"✓ Total detections: {total_detections}")
                print("✓ Detections by class:")
                for cls, count in detection_stats.items():
                    print(f"  - {cls}: {count}")
                
                return all_predictions
        
        # Run AI detection if cache not available
        try:
            all_predictions = []
            total_detections = 0
            
            for i in range(0, len(frames), batch_size):
                batch = frames[i:i + batch_size]
                outputs = self.detector._predict_batch(batch)
                all_predictions.extend(outputs)
                
                for output in outputs:
                    instances = output["instances"].to("cpu")
                    
                    total_detections += len(instances)
            
            # Save to cache
            self._save_predictions_cache(all_predictions)
            
            detection_stats = {}
            for output in all_predictions:
                instances = output["instances"].to("cpu")
                classes = instances.pred_classes.numpy()
                
                for class_id in classes:
                    class_name = self.detector.metadata.thing_classes[class_id]
                    detection_stats[class_name] = detection_stats.get(class_name, 0) + 1
            
            self.test_results['detection_ai'] = {
                'status': 'PASS',
                'total_predictions': len(all_predictions),
                'total_detections': total_detections,
                'detections_by_class': detection_stats
            }
            
            print(f"✓ Processed {len(all_predictions)} frames")
            print(f"✓ Total detections: {total_detections}")
            print("✓ Detections by class:")
            for cls, count in detection_stats.items():
                print(f"  - {cls}: {count}")
                
            return all_predictions
            
        except Exception as e:
            self.test_results['detection_ai'] = {
                'status': 'FAIL',
                'error': str(e)
            }
            print(f"✗ AI detection failed: {e}")
            raise
            
    def test_text_extraction(self, frames):
        """Test text extraction on frames with mock detections."""
        print("\n=== Testing Text Extraction ===")
        try:
            # Create mock detections for testing
            self.detector.all_detections = {}
            frame_count = 0
            
            # Process first 10 frames as sample
            sample_frames = frames[:min(10, len(frames))]
            output_dir = "test_output"
            frames_dir = os.path.join(output_dir, "frames")
            os.makedirs(frames_dir, exist_ok=True)
            
            for idx, frame in enumerate(sample_frames):
                frame_count += 1
                timestamp = idx * 1.0
                
                # Save frame for text extraction
                frame_path = os.path.join(frames_dir, f"frame_{frame_count}_00_00_{idx:02d}.jpg")
                cv2.imwrite(frame_path, frame)
                
                # Create mock detection
                self.detector.all_detections[f"Image_{frame_count}"] = {
                    "Detection": [{
                        "bbox": [100, 100, 200, 200],
                        "class": "Crack",
                        "confidence": 0.95,
                        "frame_time": 0.0,
                        "timestamp_seconds": timestamp
                    }],
                    "frame_path": frame_path
                }
            
            # Run text extraction
            self.detector._extract_and_process_text(sample_frames)
            
            extracted_distances = []
            for key, data in self.detector.all_detections.items():
                if "text_info" in data:
                    extracted_distances.append(float(data["text_info"]))
            
            self.test_results['text_extraction'] = {
                'status': 'PASS',
                'frames_processed': len(sample_frames),
                'distances_extracted': len(extracted_distances),
                'distance_range': f"{min(extracted_distances):.2f} - {max(extracted_distances):.2f}" if extracted_distances else "None"
            }
            
            print(f"✓ Processed {len(sample_frames)} frames")
            print(f"✓ Extracted {len(extracted_distances)} distances")
            if extracted_distances:
                print(f"✓ Distance range: {min(extracted_distances):.2f}m - {max(extracted_distances):.2f}m")
                
        except Exception as e:
            self.test_results['text_extraction'] = {
                'status': 'FAIL',
                'error': str(e)
            }
            print(f"✗ Text extraction failed: {e}")
            raise
            
    def test_root_classification(self, frames, predictions):
        """Test root detection classification."""
        print("\n=== Testing Root Classification ===")
        try:
            if self.detector.root_classifier is None:
                self.test_results['root_classification'] = {
                    'status': 'SKIP',
                    'reason': 'Root classifier not available'
                }
                print("⊘ Root classifier not initialized")
                return
                
            # Find Root detections
            root_count = 0
            classified_count = 0
            
            # Create temporary all_detections for testing
            temp_detections = {}
            root_class_idx = self.detector.metadata.thing_classes.index("Root")
            
            for frame_idx, (frame, output) in enumerate(zip(frames, predictions)):
                instances = output["instances"].to("cpu")
                
                if len(instances) == 0:
                    continue
                    
                classes = instances.pred_classes.numpy()
                boxes = instances.pred_boxes.tensor.numpy()
                
                has_root = False
                for class_id in classes:
                    if class_id == root_class_idx:
                        root_count += 1
                        has_root = True
                        
                if has_root:
                    temp_detections[f"Image_{frame_idx + 1}"] = {
                        "Detection": [{
                            "bbox": box.tolist(),
                            "class": self.detector.metadata.thing_classes[class_id],
                            "confidence": 0.9
                        } for box, class_id in zip(boxes, classes)]
                    }
            
            # Replace detector's all_detections temporarily
            original_detections = self.detector.all_detections
            self.detector.all_detections = temp_detections
            
            # Run classification
            self.detector._classify_root_detections(frames, predictions)
            
            # Count classified roots
            for data in self.detector.all_detections.values():
                for det in data.get("Detection", []):
                    if "root_subclass" in det:
                        classified_count += 1
            
            # Restore original detections
            self.detector.all_detections = original_detections
            
            self.test_results['root_classification'] = {
                'status': 'PASS',
                'root_detections': root_count,
                'classified': classified_count
            }
            
            print(f"✓ Found {root_count} Root detections")
            print(f"✓ Classified {classified_count} Root detections")
            
        except Exception as e:
            self.test_results['root_classification'] = {
                'status': 'FAIL',
                'error': str(e)
            }
            print(f"✗ Root classification failed: {e}")
            raise

    def test_excel_reporting(self, frames, predictions):
        """Test Excel report generation."""
        print("\n=== Testing Excel Reporting ===")
        try:
            # Check if root classification results are cached
            root_classification_cached = False
            test_results_path = "frames_detections.json"

            if os.path.exists(test_results_path):
                try:
                    with open(test_results_path, 'r') as f:
                        cached_results = json.load(f)
                    if 'root_classification' in cached_results and cached_results['root_classification'].get('status') == 'PASS':
                        root_classification_cached = True
                        print("✓ Using cached root classification results for Excel reporting")
                except (json.JSONDecodeError, KeyError):
                    pass

            # If root classification not cached, run it first
            if not root_classification_cached:
                print("⊘ Root classification not cached, running classification first...")
                self.test_root_classification(frames[:len(predictions)], predictions)

            # Create temporary all_detections for testing
            temp_detections = {}
            for frame_idx, (frame, output) in enumerate(zip(frames, predictions)):
                instances = output["instances"].to("cpu")

                if len(instances) == 0:
                    continue

                classes = instances.pred_classes.numpy()
                boxes = instances.pred_boxes.tensor.numpy()
                scores = instances.scores.numpy()

                detections = []
                for box, class_id, score in zip(boxes, classes, scores):
                    detection = {
                        "bbox": box.tolist(),
                        "class": self.detector.metadata.thing_classes[class_id],
                        "confidence": float(score),
                        "frame_time": 0.0,
                        "timestamp_seconds": frame_idx * 1.0
                    }
                    detections.append(detection)

                if detections:
                    temp_detections[f"Image_{frame_idx + 1}"] = {
                        "Detection": detections,
                        "frame_path": f"test_frame_{frame_idx}.jpg",
                        "text_info": str(frame_idx * 0.1)  # Mock distance
                    }

            # Temporarily replace detector's all_detections for root classification
            original_detections = self.detector.all_detections
            self.detector.all_detections = temp_detections

            # Apply root classification to the temp detections if classifier is available
            if self.detector.root_classifier is not None and not root_classification_cached:
                self.detector._classify_root_detections(frames, predictions)
                print("✓ Root classification applied for Excel report testing")

            # Get the updated detections with root subclasses
            temp_detections = self.detector.all_detections

            # Create test output directory
            test_output_dir = "test_excel_output"
            os.makedirs(test_output_dir, exist_ok=True)

            # Create Excel reporter
            reporter = ExcelReporter(
                input_path=test_output_dir,
                excelOutPutName="Test-Condition-Details.xlsx",
                progress_logger=self.detector.progress_logger
            )

            # Generate report with the classified detections
            reporter.generate_report()

            # Restore original detections
            self.detector.all_detections = original_detections

            # Check if Excel file was created
            excel_path = os.path.join(test_output_dir, "Test-Condition-Details.xlsx")
            if os.path.exists(excel_path):
                self.test_results['excel_reporting'] = {
                    'status': 'PASS',
                    'excel_file': excel_path,
                    'frames_processed': len(temp_detections)
                }
                print(f"✓ Excel report generated: {excel_path}")
                print(f"✓ Processed {len(temp_detections)} frames with detections")
            else:
                self.test_results['excel_reporting'] = {
                    'status': 'FAIL',
                    'reason': 'Excel file not created'
                }
                print("✗ Excel report generation failed - file not created")

        except Exception as e:
            self.test_results['excel_reporting'] = {
                'status': 'FAIL',
                'error': str(e)
            }
            print(f"✗ Excel reporting test failed: {e}")

    def save_results(self, output_path="frames_detections.json"):
        """Save test results to JSON."""
        test_output_dir = "test_excel_output"
        output_path = os.path.join(test_output_dir, "frames_detections.json")
        with open(output_path, 'w') as f:
            json.dump(self.test_results, f, indent=4)
        print(f"\n✓ Test results saved to {output_path}")

    def _save_results(self, output_path):
        """Save detection results to JSON (same as Analyser_Batch)"""
        base_name = os.path.basename(output_path).split("_")[0]

        json_output_path = os.path.join(output_path, "frames_detections.json")
        with open(json_output_path, 'w') as f:
            json.dump(self.detector.all_detections, f, indent=4)
        print(f"✓ Detection data saved to {json_output_path}")
        
    def run_all_tests(self):
        """Run all component tests sequentially."""
        print("\n" + "="*50)
        print("COMPONENT TESTING - Analyser Batch")
        print("="*50)
        
        try:
            self.setup_detector()
            
            # Test 1: Frame Collection
            frames, timestamps = self.test_frame_collection()
            
            # Test 2: AI Detection
            predictions = self.test_detection_ai(frames)  # Test on first 20 frames
            
            # Test 3: Text Extraction
            self.test_text_extraction(frames[:10])  # Test on first 10 frames
            
            # Test 4: Root Classification
            self.test_root_classification(frames[:len(predictions)], predictions)

            # Save results
            self.save_results()

            
            # Test 5: Excel Reporting
            self.test_excel_reporting(frames[:len(predictions)], predictions)

          
            
            # Print summary
            self._print_summary()
            
        except Exception as e:
            print(f"\n✗ Testing stopped due to error: {e}")
            self.save_results()
            
    def run_single_test(self, test_name: str):
        """Run a single component test.
        
        Args:
            test_name: Name of test to run ('frame_collection', 'detection_ai', 'text_extraction', 'root_classification', 'excel_reporting')
        """
        print("\n" + "="*50)
        print(f"COMPONENT TESTING - {test_name.replace('_', ' ').title()}")
        print("="*50)
        
        try:
            self.setup_detector()
            
            frames = None
            predictions = None
            
            # Collect frames if needed
            if test_name in ['frame_collection', 'detection_ai', 'text_extraction', 'root_classification', 'excel_reporting']:
                frames, timestamps = self.test_frame_collection(use_cache=self.use_cache)

            # Run AI detection if needed
            if test_name in ['detection_ai', 'root_classification', 'excel_reporting']:
                predictions = self.test_detection_ai(frames[:20], use_cache=self.use_cache)

            # Run the requested test
            if test_name == 'text_extraction':
                self.test_text_extraction(frames[:10])
            elif test_name == 'root_classification':
                self.test_root_classification(frames[:len(predictions)], predictions)
            elif test_name == 'excel_reporting':
                self.test_excel_reporting(frames[:len(predictions)], predictions)
            
            # Save results
            self.save_results()
            
            # Print summary
            self._print_summary()
            
        except Exception as e:
            print(f"\n✗ Testing stopped due to error: {e}")
            import traceback
            traceback.print_exc()
            self.save_results()
            
    def _print_summary(self):
        """Print test summary."""
        print("\n" + "="*50)
        print("TEST SUMMARY")
        print("="*50)
        
        passed = sum(1 for r in self.test_results.values() if r['status'] == 'PASS')
        failed = sum(1 for r in self.test_results.values() if r['status'] == 'FAIL')
        skipped = sum(1 for r in self.test_results.values() if r['status'] == 'SKIP')
        
        print(f"Total Tests: {len(self.test_results)}")
        print(f"✓ Passed: {passed}")
        print(f"✗ Failed: {failed}")
        print(f"⊘ Skipped: {skipped}")
        print("="*50)

    def clear_cache(self):
        """Clear all cached data."""
        import shutil
        if os.path.exists(self.cache_dir):
            shutil.rmtree(self.cache_dir)
            os.makedirs(self.cache_dir, exist_ok=True)
            print(f"✓ Cache cleared: {self.cache_dir}")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Test individual components of Analyser_Batch')
    parser.add_argument('--test', type=str, choices=['all', 'frame_collection', 'detection_ai', 'text_extraction', 'root_classification', 'excel_reporting'],
                       default='all', help='Specify which test to run')
    parser.add_argument('--video', type=str, 
                       default=r"C:\Users\sobha\Desktop\detectron2\Data\TestFilm\Closed circuit television (CCTV) sewer inspection.mp4",
                       help='Path to video file')
    parser.add_argument('--clear-cache', action='store_true', help='Clear cached frames and predictions before running tests')
    parser.add_argument('--no-cache', action='store_true', help='Skip using cache (force re-run all prerequisite tests)')
    parser.add_argument('--cache-dir', type=str, default='test_cache', help='Directory for caching frames and predictions')
    args = parser.parse_args()
    
    video_path = args.video
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, 'Model', 'Model V.2.8.0', 'model_final.pth')
    config_path = os.path.join(base_dir, 'Model', 'Model V.2.8.0', 'mask_rcnn_X_101_32x8d_FPN_3x.yaml')
    
    tester = ComponentTester(video_path, model_path, config_path, cache_dir=args.cache_dir)
    
    if args.clear_cache:
        tester.clear_cache()
        return
    
    # Set cache usage globally
    tester.use_cache = not args.no_cache
    
    if args.test == 'all':
        tester.run_all_tests()
    else:
        tester.run_single_test(args.test)

if __name__ == "__main__":
    main()

