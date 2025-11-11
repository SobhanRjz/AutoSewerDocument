"""Demo using real cached Detectron2 predictions."""

import sys
import cv2
import numpy as np
import os
import pickle
import torch

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

from detectron2.structures import Instances
from PipeCoverage import (
    PipeRimDetector,
    PipeGeometry,
    MultiClassCoverageAnalyzer,
    EngineeringMetricsCalculator,
    LongitudinalAggregator
)


CLASS_NAMES = [
    "Crack", "Obstacle", "Deposits", "Deformed",
    "Broken", "Joint Displaced", "Surface Damage", "Root"
]


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


def convert_predictions_to_masks(pred_data, frame_shape):
    """Convert Detectron2 predictions to per-class masks."""
    h, w = frame_shape[:2]
    class_masks = {cls: np.zeros((h, w), dtype=np.uint8) for cls in CLASS_NAMES}
    
    # Reconstruct instances
    instances = Instances((h, w))
    instances.pred_boxes = torch.from_numpy(pred_data['pred_boxes'])
    instances.pred_masks = torch.from_numpy(pred_data['masks'])
    instances.scores = torch.from_numpy(pred_data['scores'])
    instances.pred_classes = torch.from_numpy(pred_data['pred_classes'])
    
    # Extract per-class masks
    for i in range(len(instances)):
        class_id = int(instances.pred_classes[i].item())
        if class_id < len(CLASS_NAMES):
            class_name = CLASS_NAMES[class_id]
            mask = instances.pred_masks[i].numpy().astype(np.uint8) * 255
            class_masks[class_name] = np.maximum(class_masks[class_name], mask)
    
    return class_masks


def main():
    output_dir = "test_output/real_data_pipeline"
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n" + "="*70)
    print("PIPE COVERAGE ANALYSIS - REAL DATA")
    print("Using cached Detectron2 predictions")
    print("="*70 + "\n")
    
    # Load data
    frames, predictions_data, timestamps = load_cached_data()
    
    if frames is None:
        print("✗ No cached data found in test_cache/")
        print("  Run test_analyser_components.py first to generate cache")
        return
    
    print(f"✓ Loaded {len(frames)} frames with predictions\n")
    
    # Initialize components
    pipe_geometry = PipeGeometry(inner_diameter_mm=600)
    rim_detector = PipeRimDetector(
        edge_low=50,
        edge_high=150,
        min_radius=400,
        max_radius=900,
        ransac_iterations=5000,
        ransac_threshold=5.0,
        temporal_alpha=0.7,
        use_hough=True,
        blur_kernel=5
    )
    coverage_analyzer = MultiClassCoverageAnalyzer(CLASS_NAMES, theta_bins=360)
    metrics_calculator = EngineeringMetricsCalculator(pipe_geometry)
    longitudinal_aggregator = LongitudinalAggregator(segment_length_m=1.0)
    
    print("✓ Pipeline initialized\n")
    
    # Process frames
    for idx, (frame, pred_data, timestamp) in enumerate(zip(frames, predictions_data, timestamps)):
        # Filter: only process frames where pred_classes contains 7 (Root)
        if idx != 82:
            continue
        if 7 not in pred_data['pred_classes']:
            continue

        distance_m = idx * 1.0
        print(f"--- Frame {idx + 1}/{len(frames)} @ {distance_m:.1f}m ---")
        
        # Save original
        orig_path = os.path.join(output_dir, f"step0_original_{idx:03d}.jpg")
        cv2.imwrite(orig_path, frame)
        
        # Detect rim (use fixed center for sewer cameras)
        rim = rim_detector.detect(frame)
        if rim is None:
            print("  ✗ Rim detection failed")
            continue

        # Save rim visualization
        rim_vis = frame.copy()
        cx, cy, r = int(rim.center_x), int(rim.center_y), int(rim.radius)
        cv2.circle(rim_vis, (cx, cy), r, (0, 255, 255), 3)
        cv2.circle(rim_vis, (cx, cy), 7, (0, 255, 255), -1)
        rim_path = os.path.join(output_dir, f"step1_rim_{idx:03d}.jpg")
        cv2.imwrite(rim_path, rim_vis)
        
        # Convert predictions to masks
        class_masks = convert_predictions_to_masks(pred_data, frame.shape)
        
        # Analyze coverage
        coverage = coverage_analyzer.analyze(frame, class_masks, rim, frame_id=f"frame_{idx:03d}")
        
        agg_pct = coverage.aggregate_any_defect.coverage_percentage
        area_pct = coverage.aggregate_any_defect.area_coverage_percentage
        print(f"  Angular Coverage: {agg_pct:.1f}% | Area Coverage: {area_pct:.1f}%")
        
        # Show defect classes
        for cls, result in coverage.per_class.items():
            pct = result.angular_coverage.coverage_percentage
            area = result.angular_coverage.area_coverage_percentage
            if pct > 1.0:
                print(f"    - {cls}: {pct:.1f}% (angular), {area:.1f}% (area)")
        
        # Engineering metrics
        metrics = metrics_calculator.compute_metrics(coverage, distance_along_pipe_m=distance_m)
        blocked_mm = metrics.aggregate_physical.blocked_arc_length_mm
        print(f"  Blocked arc: {blocked_mm:.0f} mm")
        
        longitudinal_aggregator.add_frame(metrics)
        
        # Save coverage visualization
        vis = coverage_analyzer.visualize(frame, coverage, rim, class_masks, show_per_class=True)
        vis_path = os.path.join(output_dir, f"step2_coverage_{idx:03d}.jpg")
        cv2.imwrite(vis_path, vis)
        print(f"  ✓ Saved to {output_dir}/")
        
    
    # Longitudinal report
    print("="*70)
    print("LONGITUDINAL ANALYSIS")
    print("="*70)
    
    report = longitudinal_aggregator.compute_report(total_pipe_length_m=float(len(frames)))
    
    print(f"\nTotal Distance: {report.total_distance_m:.1f} m")
    print(f"Frames: {report.total_frames}")
    print(f"Mean Coverage: {report.overall_mean_pct:.1f}%")
    print(f"P90 Coverage: {report.overall_p90_pct:.1f}%")
    print(f"Longitudinal Coverage: {report.longitudinal_coverage_pct:.1f}%\n")
    
    # Save report
    report_text = longitudinal_aggregator.format_report(report)
    report_path = os.path.join(output_dir, "longitudinal_report.txt")
    with open(report_path, 'w') as f:
        f.write(report_text)
    
    print(f"✓ Report saved: {report_path}")
    print(f"✓ All visualizations saved: {output_dir}/")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()

