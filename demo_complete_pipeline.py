"""Complete pipeline demo: Steps 1-4 with engineering metrics."""

import sys
import cv2
import numpy as np
import os
import pickle
import torch

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

from PipeCoverage import (
    CameraUndistorter,
    CameraIntrinsics,
    PipeRimDetector,
    PipeGeometry,
    MultiClassCoverageAnalyzer,
    EngineeringMetricsCalculator,
    LongitudinalAggregator
)


def create_multiclass_masks(h: int, w: int) -> dict:
    """Create synthetic masks for multiple defect classes."""
    cx, cy = w // 2, h // 2
    
    masks = {}
    
    # Root: top-right quadrant
    masks['Root'] = np.zeros((h, w), dtype=np.uint8)
    cv2.ellipse(masks['Root'], (cx, cy), (w//4, h//4), 0, -30, 80, 255, 30)
    
    # Deposits: bottom arc
    masks['Deposits'] = np.zeros((h, w), dtype=np.uint8)
    cv2.ellipse(masks['Deposits'], (cx, cy), (w//4, h//4), 0, 150, 210, 255, 25)
    
    # Crack: left side
    masks['Crack'] = np.zeros((h, w), dtype=np.uint8)
    cv2.ellipse(masks['Crack'], (cx, cy), (w//4, h//4), 0, 220, 280, 255, 15)
    
    # Other classes empty for this demo
    for cls in ['Obstacle', 'Deformed', 'Broken', 'Joint Displaced', 'Surface Damage']:
        masks[cls] = np.zeros((h, w), dtype=np.uint8)
    
    return masks


def load_cached_data():
    """Load frames and predictions from cache."""
    frames_cache = "test_cache/frames_cache.pkl"
    predictions_cache = "test_cache/predictions_cache.pkl"
    
    if not os.path.exists(frames_cache) or not os.path.exists(predictions_cache):
        return None, None, None
    
    # Load frames
    with open(frames_cache, 'rb') as f:
        frames_metadata = pickle.load(f)
    
    frames = []
    for frame_file in frames_metadata['frame_files']:
        if os.path.exists(frame_file):
            frame = cv2.imread(frame_file)
            frames.append(frame)
    
    # Load predictions
    with open(predictions_cache, 'rb') as f:
        predictions_data = pickle.load(f)
    
    timestamps = frames_metadata.get('timestamps', [i * 0.5 for i in range(len(frames))])
    
    return frames, predictions_data, timestamps


def convert_predictions_to_masks(pred_data, frame_shape, class_names):
    """Convert Detectron2 predictions to per-class masks."""
    h, w = frame_shape[:2]
    class_masks = {cls: np.zeros((h, w), dtype=np.uint8) for cls in class_names}
    
    # Reconstruct instances
    from detectron2.structures import Instances
    instances = Instances((h, w))
    instances.pred_boxes = torch.from_numpy(pred_data['boxes'])
    instances.pred_masks = torch.from_numpy(pred_data['masks'])
    instances.scores = torch.from_numpy(pred_data['scores'])
    instances.pred_classes = torch.from_numpy(pred_data['pred_classes'])
    
    # Extract per-class masks
    for i in range(len(instances)):
        class_id = int(instances.pred_classes[i].item())
        if class_id < len(class_names):
            class_name = class_names[class_id]
            mask = instances.pred_masks[i].numpy().astype(np.uint8) * 255
            class_masks[class_name] = np.maximum(class_masks[class_name], mask)
    
    return class_masks


def demo_complete_analysis(output_dir: str = "test_output/complete_pipeline", use_real_data: bool = True):
    """
    Demonstrate complete pipeline with all steps.
    
    Args:
        output_dir: Output directory
        use_real_data: If True, use cached real frames/predictions. If False, use synthetic.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n" + "="*70)
    print("COMPLETE PIPE COVERAGE ANALYSIS PIPELINE")
    print("Steps 1-4: Undistort → Rim → Angular → Engineering")
    print("="*70 + "\n")
    
    # Configuration
    CLASS_NAMES = [
        "Crack", "Obstacle", "Deposits", "Deformed",
        "Broken", "Joint Displaced", "Surface Damage", "Root"
    ]
    
    PIPE_GEOMETRY = PipeGeometry(inner_diameter_mm=600)  # 600mm diameter pipe
    
    print(f"Configuration:")
    print(f"  Pipe Inner Diameter: {PIPE_GEOMETRY.inner_diameter_mm} mm")
    print(f"  Pipe Circumference: {PIPE_GEOMETRY.circumference_mm:.0f} mm")
    print(f"  Defect Classes: {len(CLASS_NAMES)}")
    print()
    
    # Initialize components
    intrinsics = CameraIntrinsics(
        fx=1200, fy=1200, cx=640, cy=360,
        k1=-0.15, k2=0.03
    )
    undistorter = CameraUndistorter(intrinsics)
    rim_detector = PipeRimDetector(
        min_radius=150, max_radius=600,
        ransac_iterations=2000
    )
    coverage_analyzer = MultiClassCoverageAnalyzer(
        class_names=CLASS_NAMES,
        theta_bins=360,
        inner_radius_factor=0.8,
        outer_radius_factor=1.0
    )
    metrics_calculator = EngineeringMetricsCalculator(PIPE_GEOMETRY)
    longitudinal_aggregator = LongitudinalAggregator(segment_length_m=1.0)
    
    print("✓ Pipeline components initialized\n")
    
    # Load real data if available
    if use_real_data:
        frames, predictions_data, timestamps = load_cached_data()
        if frames is None:
            print("⊘ No cached data found, falling back to synthetic")
            use_real_data = False
        else:
            print(f"✓ Loaded {len(frames)} real frames from cache\n")
    
    if use_real_data:
        # Process real frames with real Detectron2 predictions
        print("Processing real video frames...")
        
        for idx, (frame, pred_data, timestamp) in enumerate(zip(frames, predictions_data, timestamps)):
            distance_m = idx * 1.0  # Simulate distance
            print(f"\n--- Frame {idx + 1}/{len(frames)} @ {distance_m:.1f}m ---")
            
            # Save original frame
            synthetic_frame_path = os.path.join(output_dir, f"real_frame_{idx:03d}_{distance_m:.1f}m.jpg")
            cv2.imwrite(synthetic_frame_path, frame)
            print(f"  ✓ Saved original frame: {synthetic_frame_path}")

        # STEP 1: Undistort
        undistorted = undistorter.undistort(frame)

        # Save undistorted frame
        undistorted_path = os.path.join(output_dir, f"step1_undistorted_{idx:03d}_{distance_m:.1f}m.jpg")
        cv2.imwrite(undistorted_path, undistorted)
        print(f"  ✓ Saved undistorted: {undistorted_path}")

        # STEP 2: Detect rim
        rim = rim_detector.detect(undistorted)
        if rim is None:
            print("  ✗ Rim detection failed")
            
        
        print(f"  ✓ Rim: center=({rim.center_x:.0f},{rim.center_y:.0f}), r={rim.radius:.0f}px")

        # Save rim detection visualization
        rim_vis = undistorted.copy()
        cx, cy, r = int(rim.center_x), int(rim.center_y), int(rim.radius)
        cv2.circle(rim_vis, (cx, cy), r, (0, 255, 255), 3)
        cv2.circle(rim_vis, (cx, cy), 7, (0, 255, 255), -1)
        cv2.putText(rim_vis, f"Rim: center=({cx},{cy}), r={r}px",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        rim_path = os.path.join(output_dir, f"step2_rim_detected_{idx:03d}_{distance_m:.1f}m.jpg")
        cv2.imwrite(rim_path, rim_vis)
        print(f"  ✓ Saved rim detection: {rim_path}")

        # Convert Detectron2 predictions to class masks
        class_masks = convert_predictions_to_masks(pred_data, frame.shape, CLASS_NAMES)
        
        # Undistort masks
        for cls in class_masks:
            class_masks[cls] = undistorter.undistort(class_masks[cls])
        
        # STEP 3: Angular coverage analysis
        coverage_result = coverage_analyzer.analyze(
            undistorted, class_masks, rim, frame_id=f"frame_{idx:03d}"
        )
        
        agg_pct = coverage_result.aggregate_any_defect.coverage_percentage
        print(f"  ✓ Angular coverage: {agg_pct:.1f}%")
        
        # Show top defect classes
        for cls, cls_result in coverage_result.per_class.items():
            if cls_result.angular_coverage.coverage_percentage > 1.0:
                print(f"    - {cls}: {cls_result.angular_coverage.coverage_percentage:.1f}%")
        
        # STEP 4: Engineering metrics
        eng_metrics = metrics_calculator.compute_metrics(
            coverage_result,
            distance_along_pipe_m=distance_m,
            timestamp_sec=idx * 0.5
        )
        
        blocked_mm = eng_metrics.aggregate_physical.blocked_arc_length_mm
        print(f"  ✓ Blocked arc: {blocked_mm:.0f} mm")
        
        # Add to longitudinal aggregator
        longitudinal_aggregator.add_frame(eng_metrics)
        
        # Save Step 4: Final coverage visualization
        vis = coverage_analyzer.visualize(undistorted, coverage_result, rim, class_masks, show_per_class=True)
        vis_path = os.path.join(output_dir, f"step4_coverage_final_{idx:03d}_{distance_m:.1f}m.jpg")
        cv2.imwrite(vis_path, vis)
        print(f"  ✓ Saved final coverage: {vis_path}")
    
    print("\n" + "="*70)
    print("LONGITUDINAL ANALYSIS")
    print("="*70)
    
    # STEP 4b: Longitudinal aggregation
    long_report = longitudinal_aggregator.compute_report(total_pipe_length_m=8.0)
    
    print(f"\nTotal Distance: {long_report.total_distance_m:.1f} m")
    print(f"Frames Analyzed: {long_report.total_frames}")
    print(f"Mean Coverage: {long_report.overall_mean_pct:.1f}%")
    print(f"P90 Coverage: {long_report.overall_p90_pct:.1f}%")
    print(f"Longitudinal Coverage: {long_report.longitudinal_coverage_pct:.1f}%")
    print(f"  ({long_report.frames_with_defect}/{long_report.total_frames} frames with defects)")
    
    print(f"\nPer-Meter Breakdown:")
    for seg in long_report.segments:
        if seg.frame_count > 0:
            print(f"  [{seg.start_distance_m:.0f}-{seg.end_distance_m:.0f}m]: "
                  f"Mean={seg.mean_coverage_pct:.1f}%, P90={seg.p90_coverage_pct:.1f}%")
    
    # Save detailed report
    long_report_text = longitudinal_aggregator.format_report(long_report)
    report_path = os.path.join(output_dir, "longitudinal_report.txt")
    with open(report_path, 'w') as f:
        f.write(long_report_text)
    print(f"\n✓ Detailed report: {report_path}")
    
    print("\n" + "="*70)
    print("PIPELINE COMPLETE")
    print(f"Output directory: {output_dir}")
    print("="*70 + "\n")


def create_synthetic_frame(width: int, height: int) -> np.ndarray:
    """Create synthetic sewer frame."""
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Gradient background
    for i in range(height):
        intensity = int(40 + (120 * i / height))
        frame[i, :] = [intensity * 0.6, intensity * 0.7, intensity * 0.8]
    
    # Noise
    noise = np.random.randint(-15, 15, (height, width, 3), dtype=np.int16)
    frame = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    
    # Pipe rim
    cx, cy = width // 2, height // 2
    radius = min(width, height) // 3
    cv2.circle(frame, (cx, cy), radius, (130, 140, 150), 3)
    
    return frame


if __name__ == "__main__":
    demo_complete_analysis()

