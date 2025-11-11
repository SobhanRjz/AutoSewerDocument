"""Demo: Visualize pipe coverage analysis step-by-step."""

import sys
import cv2
import numpy as np
import os

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

from PipeCoverage import (
    CameraUndistorter,
    CameraIntrinsics,
    PipeRimDetector,
    CircumferenceCoverageAnalyzer,
    PipeCoverageProcessor
)


def create_synthetic_sewer_frame(width: int = 1280, height: int = 720) -> tuple[np.ndarray, np.ndarray]:
    """
    Create synthetic sewer pipe frame with root defect.
    
    Returns:
        (frame, defect_mask)
    """
    # Base frame with gradient (simulating lighting)
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    for i in range(height):
        intensity = int(50 + (150 * i / height))
        frame[i, :] = [intensity * 0.6, intensity * 0.7, intensity * 0.8]
    
    # Add noise
    noise = np.random.randint(-20, 20, (height, width, 3), dtype=np.int16)
    frame = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    
    # Draw pipe rim
    cx, cy = width // 2, height // 2
    radius = min(width, height) // 3
    cv2.circle(frame, (cx, cy), radius, (140, 150, 160), 4)
    
    # Add some texture on pipe wall
    for angle in range(0, 360, 15):
        rad = np.deg2rad(angle)
        x = int(cx + radius * np.cos(rad))
        y = int(cy + radius * np.sin(rad))
        cv2.circle(frame, (x, y), 3, (120, 130, 140), -1)
    
    # Create defect mask (root intrusion at top-right quadrant)
    mask = np.zeros((height, width), dtype=np.uint8)
    
    # Simulate root covering part of pipe circumference
    cv2.ellipse(mask, (cx, cy), (radius, radius), 0, -30, 80, 255, 35)
    
    # Add some irregular structure to the root
    for i in range(5):
        angle = np.random.uniform(-30, 80)
        rad = np.deg2rad(angle)
        x = int(cx + radius * np.cos(rad))
        y = int(cy + radius * np.sin(rad))
        blob_size = np.random.randint(15, 30)
        cv2.circle(mask, (x, y), blob_size, 255, -1)
    
    # Add root texture to frame
    root_color = (60, 80, 40)  # Brownish-green
    frame[mask > 0] = root_color
    
    return frame, mask


def demo_with_real_frame(frame_path: str, output_dir: str = "test_output/demo_steps"):
    """
    Process real frame and save each step.
    
    Args:
        frame_path: Path to input frame
        output_dir: Directory to save step outputs
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n{'='*70}")
    print(f"DEMO: Pipe Coverage Analysis Pipeline")
    print(f"Input: {frame_path}")
    print(f"{'='*70}\n")
    
    # Load frame
    frame = cv2.imread(frame_path)
    if frame is None:
        print(f"✗ Could not load frame: {frame_path}")
        return
    
    h, w = frame.shape[:2]
    print(f"Frame size: {w}x{h}\n")
    
    # Create synthetic mask for demo (since we don't have real Detectron2 mask)
    # In production, this would come from Detectron2 instance segmentation
    mask = np.zeros((h, w), dtype=np.uint8)
    cx, cy = w // 2, h // 2
    cv2.ellipse(mask, (cx, cy), (w // 4, h // 4), 0, 0, 90, 255, 35)
    
    # Step 0: Original frame
    step0_path = os.path.join(output_dir, "step0_original.jpg")
    cv2.imwrite(step0_path, frame)
    print(f"✓ Step 0: Original frame -> {step0_path}")
    
    # Step 0b: Defect mask
    step0b_path = os.path.join(output_dir, "step0b_defect_mask.jpg")
    mask_vis = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    cv2.imwrite(step0b_path, mask_vis)
    print(f"✓ Step 0b: Defect mask (from Detectron2) -> {step0b_path}")
    
    # STEP 1: Undistortion (optional, skip if no distortion)
    print(f"\n--- STEP 1: Camera Undistortion ---")
    
    # For demo, use moderate fisheye distortion
    intrinsics = CameraIntrinsics(
        fx=w * 0.8, fy=w * 0.8,
        cx=w / 2, cy=h / 2,
        k1=-0.15, k2=0.03  # Moderate barrel distortion
    )
    
    undistorter = CameraUndistorter(intrinsics, alpha=0.8)
    undistorted_frame = undistorter.undistort(frame)
    undistorted_mask = undistorter.undistort(mask)
    
    step1_path = os.path.join(output_dir, "step1_undistorted.jpg")
    cv2.imwrite(step1_path, undistorted_frame)
    print(f"✓ Undistorted frame -> {step1_path}")
    print(f"  Intrinsics: fx={intrinsics.fx:.0f}, k1={intrinsics.k1:.3f}")
    
    # STEP 2: Pipe Rim Detection
    print(f"\n--- STEP 2: Pipe Rim Detection ---")
    
    rim_detector = PipeRimDetector(
        min_radius=min(w, h) // 6,
        max_radius=min(w, h) // 2,
        ransac_iterations=2000,
        temporal_alpha=0.7
    )
    
    rim = rim_detector.detect(undistorted_frame)
    
    if rim is None:
        print("✗ Rim detection failed!")
        return
    
    # Visualize rim detection
    step2_frame = undistorted_frame.copy()
    cx, cy, r = int(rim.center_x), int(rim.center_y), int(rim.radius)
    cv2.circle(step2_frame, (cx, cy), r, (0, 255, 255), 3)
    cv2.circle(step2_frame, (cx, cy), 7, (0, 255, 255), -1)
    cv2.putText(step2_frame, f"Rim: center=({cx},{cy}), r={r}px",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    
    step2_path = os.path.join(output_dir, "step2_rim_detected.jpg")
    cv2.imwrite(step2_path, step2_frame)
    print(f"✓ Rim detected -> {step2_path}")
    print(f"  Center: ({cx}, {cy})")
    print(f"  Radius: {r}px")
    print(f"  Confidence: {rim.confidence:.2f}")
    
    # STEP 3: Coverage Analysis
    print(f"\n--- STEP 3: Circumference Coverage Analysis ---")
    
    analyzer = CircumferenceCoverageAnalyzer(angular_resolution=360)
    result = analyzer.analyze(undistorted_frame, undistorted_mask, rim)
    
    print(f"✓ Coverage computed:")
    print(f"  Fraction: {result.coverage_fraction:.3f}")
    print(f"  Percentage: {result.coverage_percentage:.1f}%")
    print(f"  Arc degrees: {result.covered_arc_degrees:.1f}°")
    print(f"  Circumference: {result.total_circumference_pixels:.0f}px")
    print(f"  Covered pixels: {result.covered_pixels}")
    
    # STEP 4: Final Visualization
    print(f"\n--- STEP 4: Final Visualization ---")
    
    vis_frame = analyzer.visualize_coverage(
        undistorted_frame, undistorted_mask, rim, result
    )
    
    step4_path = os.path.join(output_dir, "step4_final_visualization.jpg")
    cv2.imwrite(step4_path, vis_frame)
    print(f"✓ Final visualization -> {step4_path}")
    
    # Create side-by-side comparison
    comparison = np.hstack([
        cv2.resize(frame, (w//2, h//2)),
        cv2.resize(vis_frame, (w//2, h//2))
    ])
    comparison_path = os.path.join(output_dir, "comparison_before_after.jpg")
    cv2.imwrite(comparison_path, comparison)
    print(f"✓ Before/After comparison -> {comparison_path}")
    
    print(f"\n{'='*70}")
    print(f"Pipeline complete! All steps saved to: {output_dir}")
    print(f"{'='*70}\n")


def demo_with_synthetic_frame(output_dir: str = "test_output/demo_synthetic"):
    """
    Create and process synthetic sewer frame.
    
    Args:
        output_dir: Directory to save outputs
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n{'='*70}")
    print(f"DEMO: Synthetic Sewer Pipe with Root Defect")
    print(f"{'='*70}\n")
    
    # Create synthetic data
    frame, mask = create_synthetic_sewer_frame(1280, 720)
    
    # Save original
    cv2.imwrite(os.path.join(output_dir, "synthetic_frame.jpg"), frame)
    cv2.imwrite(os.path.join(output_dir, "synthetic_mask.jpg"), mask)
    print(f"✓ Created synthetic frame: 1280x720")
    
    # Process through pipeline
    processor = PipeCoverageProcessor()
    
    result, vis = processor.process_with_visualization(frame, mask)
    
    if result is not None:
        print(f"\n✓ Coverage: {result.coverage_percentage:.1f}%")
        print(f"  Arc: {result.covered_arc_degrees:.1f}°")
        print(f"  Rim radius: {result.rim_radius:.0f}px")
        
        vis_path = os.path.join(output_dir, "synthetic_result.jpg")
        cv2.imwrite(vis_path, vis)
        print(f"✓ Result saved: {vis_path}")
    else:
        print("✗ Processing failed")
    
    print(f"\n{'='*70}\n")


def demo_batch_processing(video_or_frames_dir: str, output_dir: str = "test_output/batch"):
    """
    Process multiple frames from video or directory.
    
    Args:
        video_or_frames_dir: Video file or directory with frames
        output_dir: Output directory
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n{'='*70}")
    print(f"DEMO: Batch Processing")
    print(f"{'='*70}\n")
    
    # Initialize processor (reuses temporal filtering across frames)
    processor = PipeCoverageProcessor()
    
    # Check if input is video or directory
    if os.path.isfile(video_or_frames_dir):
        cap = cv2.VideoCapture(video_or_frames_dir)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"Video: {video_or_frames_dir} ({frame_count} frames)")
        
        # Process every 30th frame
        frame_idx = 0
        results = []
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_idx % 30 == 0:
                # Create dummy mask (in production, use Detectron2)
                h, w = frame.shape[:2]
                mask = np.zeros((h, w), dtype=np.uint8)
                cv2.ellipse(mask, (w//2, h//2), (w//4, h//4), 0, 0, 90, 255, 30)
                
                result, vis = processor.process_with_visualization(frame, mask)
                
                if result is not None:
                    results.append(result.coverage_percentage)
                    
                    out_path = os.path.join(output_dir, f"frame_{frame_idx:05d}.jpg")
                    cv2.imwrite(out_path, vis)
                    print(f"  Frame {frame_idx}: {result.coverage_percentage:.1f}% coverage")
            
            frame_idx += 1
        
        cap.release()
        
        if results:
            avg_coverage = np.mean(results)
            print(f"\n✓ Processed {len(results)} frames")
            print(f"  Average coverage: {avg_coverage:.1f}%")
    
    print(f"\n{'='*70}\n")


if __name__ == "__main__":
    print("\n" + "="*70)
    print("PIPE COVERAGE ANALYSIS - DEMO SUITE")
    print("="*70)
    
    # Demo 1: Synthetic frame
    print("\n[Demo 1: Synthetic Frame]")
    demo_with_synthetic_frame()
    
    # Demo 2: Real frame if available
    real_frame_path = "temp_crops/full_frame_33_det_0.jpg"
    if os.path.exists(real_frame_path):
        print("\n[Demo 2: Real Sewer Frame]")
        demo_with_real_frame(real_frame_path)
    else:
        print(f"\n[Demo 2: Skipped - frame not found: {real_frame_path}]")
    
    print("\n" + "="*70)
    print("DEMOS COMPLETE!")
    print("="*70)
    print("\nTo run specific demos:")
    print("  1. Synthetic frame:  demo_with_synthetic_frame()")
    print("  2. Real frame:       demo_with_real_frame('path/to/frame.jpg')")
    print("  3. Batch processing: demo_batch_processing('path/to/video.mp4')")
    print("="*70 + "\n")

