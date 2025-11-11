# Pipe Circumference Coverage Analysis - Quick Start

## Overview

Distance-invariant pipe defect coverage measurement using **angular coverage** on pipe wall circumference, not raw pixels.

## Key Functions to Run

### 1. Basic Demo (Recommended Start)
```bash
python demo_pipe_coverage.py
```
**What it does:**
- Shows step-by-step process with saved frames at each stage
- Processes both synthetic and real sewer frames
- Output: `test_output/demo_steps/` with visualization at each step

### 2. Complete Pipeline Demo
```bash
python demo_complete_pipeline.py
```
**What it does:**
- Full 4-step pipeline: Undistort → Rim Detection → Angular Coverage → Engineering Metrics
- Multi-class defect analysis (Root, Crack, Deposits, etc.)
- Longitudinal aggregation per meter
- Output: `test_output/complete_pipeline/` with reports

### 3. Run Tests
```bash
python test_pipe_coverage.py
```
**What it does:**
- Comprehensive unit tests for all components
- Validates undistortion, rim detection, coverage analysis

## Pipeline Steps

### Step 1: Camera Undistortion
```python
from PipeCoverage import CameraUndistorter, CameraIntrinsics

intrinsics = CameraIntrinsics(
    fx=1200, fy=1200, cx=640, cy=360,
    k1=-0.15, k2=0.03  # Barrel distortion
)
undistorter = CameraUndistorter(intrinsics)
undistorted_frame = undistorter.undistort(frame)
```

### Step 2: Pipe Rim Detection (RANSAC + Temporal Smoothing)
```python
from PipeCoverage import PipeRimDetector

detector = PipeRimDetector(
    min_radius=150,
    max_radius=600,
    ransac_iterations=2000,
    temporal_alpha=0.7  # Temporal smoothing
)
rim = detector.detect(undistorted_frame)
# rim.center, rim.radius, rim.confidence
```

### Step 3: Angular Coverage Analysis
**Annulus-based measurement (0.8R to R) for distance-invariance**

```python
from PipeCoverage import MultiClassCoverageAnalyzer

analyzer = MultiClassCoverageAnalyzer(
    class_names=['Root', 'Crack', 'Deposits', ...],
    theta_bins=360,  # 1° resolution
    inner_radius_factor=0.8,  # Annulus inner = 0.8 × R
    outer_radius_factor=1.0   # Annulus outer = R
)

# class_masks: Dict[str, np.ndarray] from Detectron2
coverage = analyzer.analyze(frame, class_masks, rim)

# Results
print(coverage.aggregate_any_defect.coverage_percentage)  # e.g., 25.3%
print(coverage.aggregate_any_defect.covered_degrees)      # e.g., 91.1°

for class_name, result in coverage.per_class.items():
    print(f"{class_name}: {result.angular_coverage.coverage_percentage:.1f}%")
```

### Step 4: Engineering Metrics
**Convert angular coverage to physical measurements**

```python
from PipeCoverage import EngineeringMetricsCalculator, PipeGeometry

pipe = PipeGeometry(inner_diameter_mm=600)  # 600mm pipe
calculator = EngineeringMetricsCalculator(pipe)

metrics = calculator.compute_metrics(
    coverage,
    distance_along_pipe_m=5.2,  # From odometry/crawler
    timestamp_sec=10.5
)

# Physical measurements
print(f"Blocked Arc: {metrics.aggregate_physical.blocked_arc_length_mm:.0f} mm")
print(f"Circumference Coverage: {metrics.aggregate_physical.circumference_pct:.1f}%")
print(f"Pipe Circumference: {pipe.circumference_mm:.0f} mm")
```

### Step 4b: Longitudinal Aggregation
**Aggregate coverage along pipe distance**

```python
from PipeCoverage import LongitudinalAggregator

aggregator = LongitudinalAggregator(segment_length_m=1.0)

# Process video frames
for frame, masks, distance in video_frames:
    # ... steps 1-3 ...
    metrics = calculator.compute_metrics(coverage, distance_along_pipe_m=distance)
    aggregator.add_frame(metrics)

# Generate report
report = aggregator.compute_report(total_pipe_length_m=22.0)

print(f"Overall Mean Coverage: {report.overall_mean_pct:.1f}%")
print(f"P90 Coverage: {report.overall_p90_pct:.1f}%")
print(f"Longitudinal Coverage: {report.longitudinal_coverage_pct:.1f}%")
print(f"  (Defects in {report.frames_with_defect}/{report.total_frames} frames)")

# Per-meter breakdown
for seg in report.segments:
    print(f"[{seg.start_distance_m:.0f}-{seg.end_distance_m:.0f}m]: "
          f"Mean={seg.mean_coverage_pct:.1f}%, P90={seg.p90_coverage_pct:.1f}%")
```

## Integration with Existing Code

### Use with Detectron2 Predictions

```python
# In BatchDefectDetector after prediction
from PipeCoverage import (
    PipeRimDetector,
    MultiClassCoverageAnalyzer,
    EngineeringMetricsCalculator,
    PipeGeometry
)

# Initialize (once)
rim_detector = PipeRimDetector()
coverage_analyzer = MultiClassCoverageAnalyzer(class_names=self.cfg.MODEL.ROI_HEADS.CLASS_NAMES)
metrics_calculator = EngineeringMetricsCalculator(PipeGeometry(inner_diameter_mm=600))

# Process each frame
outputs = self.predictor(frame)
instances = outputs["instances"].to("cpu")

# Detect rim
rim = rim_detector.detect(frame)
if rim is None:
    continue

# Convert Detectron2 masks to class_masks dict
class_masks = {}
for class_name in self.class_names:
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    class_indices = instances.pred_classes == self.class_names.index(class_name)
    if np.any(class_indices):
        # Combine masks for this class
        for i, is_class in enumerate(class_indices):
            if is_class:
                instance_mask = instances.pred_masks[i].numpy()
                mask = np.maximum(mask, (instance_mask * 255).astype(np.uint8))
    class_masks[class_name] = mask

# Analyze coverage
coverage = coverage_analyzer.analyze(frame, class_masks, rim, frame_id=f"frame_{frame_idx}")
metrics = metrics_calculator.compute_metrics(coverage, distance_along_pipe_m=distance)

# Store results
detection_data['angular_coverage_pct'] = coverage.aggregate_any_defect.coverage_percentage
detection_data['blocked_arc_mm'] = metrics.aggregate_physical.blocked_arc_length_mm
detection_data['per_class_coverage'] = {
    cls: result.angular_coverage.coverage_percentage
    for cls, result in coverage.per_class.items()
}
```

## Output Files

After running demos:

1. **`test_output/demo_steps/`**
   - `step0_original.jpg` - Input frame
   - `step0b_defect_mask.jpg` - Detectron2 mask
   - `step1_undistorted.jpg` - After lens correction
   - `step2_rim_detected.jpg` - Detected pipe rim
   - `step4_final_visualization.jpg` - Coverage overlay
   - `comparison_before_after.jpg` - Side-by-side

2. **`test_output/complete_pipeline/`**
   - `frame_XXX_Y.Ym.jpg` - Coverage visualization per frame
   - `longitudinal_report.txt` - Detailed per-meter statistics

## Why This Approach?

### Distance-Invariant Measurement
- **Problem**: Raw pixel area changes dramatically with camera distance
- **Solution**: Angular coverage on circumference is consistent regardless of zoom/distance
- **Benefit**: "How many degrees of pipe are blocked?" is operator-friendly metric

### Annulus-Based Analysis
- **Annulus**: [0.8R, R] ring around pipe wall
- **Why**: Ignores center where near-camera defects explode in pixel size
- **Result**: Focuses on actual wall coverage, not depth artifacts

### Engineering Units
- Converts angular coverage to physical arc length (mm)
- Per-meter aggregation for longitudinal reports
- Mean and P90 statistics for severity assessment

## Key Classes

| Class | Purpose | Input | Output |
|-------|---------|-------|--------|
| `CameraUndistorter` | Fix lens distortion | Frame, intrinsics | Undistorted frame |
| `PipeRimDetector` | Detect pipe boundary | Frame | Rim (center, radius) |
| `MultiClassCoverageAnalyzer` | Measure angular coverage | Frame, masks, rim | Coverage % per class |
| `EngineeringMetricsCalculator` | Physical measurements | Coverage, pipe geometry | Arc length (mm), % |
| `LongitudinalAggregator` | Per-meter statistics | Metrics stream | Per-meter report |

## Contact & Support

All classes use dependency injection with explicit interfaces (`Protocol`).
Production-ready with comprehensive tests.

