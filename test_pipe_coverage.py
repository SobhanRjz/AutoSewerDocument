"""Comprehensive tests for pipe coverage measurement system."""

import sys
import cv2
import numpy as np
import os

# Fix Windows console encoding
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

from PipeCoverage import (
    CameraUndistorter,
    CameraIntrinsics,
    PipeRimDetector,
    PipeRim,
    CircumferenceCoverageAnalyzer,
    CoverageResult,
    PipeCoverageProcessor
)


class TestCameraUndistorter:
    """Test camera undistortion."""
    
    def test_identity_undistortion(self):
        """Test undistortion with zero distortion."""
        intrinsics = CameraIntrinsics(
            fx=800, fy=800, cx=640, cy=360,
            k1=0.0, k2=0.0, p1=0.0, p2=0.0
        )
        undistorter = CameraUndistorter(intrinsics)
        
        frame = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
        result = undistorter.undistort(frame)
        
        assert result.shape == frame.shape
        print("✓ Identity undistortion preserves shape")
    
    def test_fisheye_undistortion(self):
        """Test with typical fisheye distortion."""
        intrinsics = CameraIntrinsics(
            fx=600, fy=600, cx=640, cy=360,
            k1=-0.3, k2=0.1, p1=0.001, p2=0.001
        )
        undistorter = CameraUndistorter(intrinsics, alpha=0.5)
        
        frame = self._create_checkerboard(720, 1280)
        result = undistorter.undistort(frame)
        
        assert result.shape == frame.shape
        assert not np.array_equal(result, frame)
        print("✓ Fisheye undistortion modifies frame")
    
    def test_caching_different_sizes(self):
        """Test map recomputation on size change."""
        intrinsics = CameraIntrinsics(fx=500, fy=500, cx=320, cy=240)
        undistorter = CameraUndistorter(intrinsics)
        
        frame1 = np.zeros((480, 640, 3), dtype=np.uint8)
        frame2 = np.zeros((720, 1280, 3), dtype=np.uint8)
        
        result1 = undistorter.undistort(frame1)
        result2 = undistorter.undistort(frame2)
        
        assert result1.shape == frame1.shape
        assert result2.shape == frame2.shape
        print("✓ Undistortion handles multiple frame sizes")
    
    def _create_checkerboard(self, height: int, width: int) -> np.ndarray:
        """Create synthetic checkerboard pattern."""
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        square_size = 40
        
        for i in range(0, height, square_size):
            for j in range(0, width, square_size):
                if (i // square_size + j // square_size) % 2 == 0:
                    frame[i:i+square_size, j:j+square_size] = 255
        
        return frame


class TestPipeRimDetector:
    """Test pipe rim detection."""
    
    def test_perfect_circle_detection(self):
        """Test detection on synthetic perfect circle."""
        detector = PipeRimDetector(min_radius=100, max_radius=400)
        
        frame = self._create_circle_frame(480, 640, cx=320, cy=240, radius=200)
        rim = detector.detect(frame)
        
        assert rim is not None
        assert abs(rim.center_x - 320) < 10
        assert abs(rim.center_y - 240) < 10
        assert abs(rim.radius - 200) < 10
        print("✓ Perfect circle detected accurately")
    
    def test_noisy_circle_detection(self):
        """Test robustness to noise."""
        detector = PipeRimDetector(ransac_iterations=2000)
        
        frame = self._create_circle_frame(480, 640, cx=320, cy=240, radius=150)
        # Add noise
        noise = np.random.randint(0, 50, frame.shape, dtype=np.uint8)
        frame = cv2.add(frame, noise)
        
        rim = detector.detect(frame)
        
        assert rim is not None
        assert abs(rim.radius - 150) < 30
        print("✓ Noisy circle detected with RANSAC")
    
    def test_temporal_smoothing(self):
        """Test temporal filtering across frames."""
        detector = PipeRimDetector(temporal_alpha=0.8)
        
        # Frame sequence with slight jitter
        centers = [(320, 240), (322, 241), (318, 239), (321, 240)]
        radii = [200, 202, 198, 201]
        
        rims = []
        for (cx, cy), r in zip(centers, radii):
            frame = self._create_circle_frame(480, 640, cx, cy, radius=r)
            rim = detector.detect(frame)
            if rim:
                rims.append(rim)
        
        # Smoothed values should be less jittery
        assert len(rims) == 4
        
        # Check smoothing reduces variance
        raw_radii = np.array(radii)
        smoothed_radii = np.array([r.radius for r in rims])
        
        assert np.std(smoothed_radii) < np.std(raw_radii)
        print("✓ Temporal smoothing reduces jitter")
    
    def test_no_circle_fallback(self):
        """Test fallback on detection failure."""
        detector = PipeRimDetector()
        
        # Blank frame - no edges
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        rim = detector.detect(frame)
        
        assert rim is None
        print("✓ Returns None on detection failure")
    
    def test_reset_temporal_state(self):
        """Test state reset between videos."""
        detector = PipeRimDetector(temporal_alpha=0.9)
        
        frame1 = self._create_circle_frame(480, 640, cx=320, cy=240, radius=200)
        rim1 = detector.detect(frame1)
        
        detector.reset()
        
        frame2 = self._create_circle_frame(480, 640, cx=400, cy=300, radius=150)
        rim2 = detector.detect(frame2)
        
        # After reset, new detection shouldn't be smoothed with old
        assert abs(rim2.center_x - 400) < 20
        print("✓ Reset clears temporal state")
    
    def _create_circle_frame(
        self, height: int, width: int, cx: int, cy: int, radius: int
    ) -> np.ndarray:
        """Create frame with circle edge."""
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        cv2.circle(frame, (cx, cy), radius, (255, 255, 255), 2)
        return frame


class TestCircumferenceCoverageAnalyzer:
    """Test coverage measurement."""
    
    def test_full_coverage(self):
        """Test 100% coverage."""
        analyzer = CircumferenceCoverageAnalyzer(angular_resolution=360)
        
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        rim = PipeRim(center_x=320, center_y=240, radius=150)
        
        # Full ring mask
        mask = self._create_ring_mask(480, 640, 320, 240, 150, thickness=50)
        
        result = analyzer.analyze(frame, mask, rim)
        
        assert result.coverage_fraction > 0.9
        assert result.covered_arc_degrees > 300
        print(f"✓ Full coverage: {result.coverage_percentage:.1f}%")
    
    def test_partial_coverage(self):
        """Test partial arc coverage."""
        analyzer = CircumferenceCoverageAnalyzer()
        
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        rim = PipeRim(center_x=320, center_y=240, radius=150)
        
        # Quarter arc mask (top-right quadrant)
        mask = self._create_arc_mask(480, 640, 320, 240, 150, 0, 90)
        
        result = analyzer.analyze(frame, mask, rim)
        
        assert 0.2 < result.coverage_fraction < 0.35
        assert 70 < result.covered_arc_degrees < 130
        print(f"✓ Partial coverage: {result.coverage_percentage:.1f}%")
    
    def test_zero_coverage(self):
        """Test empty mask."""
        analyzer = CircumferenceCoverageAnalyzer()
        
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        rim = PipeRim(center_x=320, center_y=240, radius=150)
        mask = np.zeros((480, 640), dtype=np.uint8)
        
        result = analyzer.analyze(frame, mask, rim)
        
        assert result.coverage_fraction == 0.0
        assert result.covered_pixels == 0
        print("✓ Zero coverage detected")
    
    def test_off_rim_defect_ignored(self):
        """Test that defects far from rim are ignored."""
        analyzer = CircumferenceCoverageAnalyzer()
        
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        rim = PipeRim(center_x=320, center_y=240, radius=150)
        
        # Mask in center (far from rim)
        mask = np.zeros((480, 640), dtype=np.uint8)
        cv2.circle(mask, (320, 240), 30, 255, -1)
        
        result = analyzer.analyze(frame, mask, rim)
        
        # Should have very low coverage since defect is far from rim
        assert result.coverage_fraction < 0.1
        print("✓ Off-rim defects filtered out")
    
    def test_visualization(self):
        """Test visualization generation."""
        analyzer = CircumferenceCoverageAnalyzer()
        
        frame = np.random.randint(50, 150, (480, 640, 3), dtype=np.uint8)
        rim = PipeRim(center_x=320, center_y=240, radius=150)
        mask = self._create_arc_mask(480, 640, 320, 240, 150, 45, 135)
        
        result = analyzer.analyze(frame, mask, rim)
        vis = analyzer.visualize_coverage(frame, mask, rim, result)
        
        assert vis.shape == frame.shape
        assert not np.array_equal(vis, frame)
        print("✓ Visualization generated")
    
    def _create_ring_mask(
        self, height: int, width: int, cx: int, cy: int, radius: int, thickness: int
    ) -> np.ndarray:
        """Create ring-shaped mask."""
        mask = np.zeros((height, width), dtype=np.uint8)
        cv2.circle(mask, (cx, cy), radius + thickness // 2, 255, thickness)
        return mask
    
    def _create_arc_mask(
        self, height: int, width: int, cx: int, cy: int, radius: int,
        start_angle: float, end_angle: float
    ) -> np.ndarray:
        """Create arc-shaped mask."""
        mask = np.zeros((height, width), dtype=np.uint8)
        
        # Draw arc using ellipse function
        cv2.ellipse(
            mask, (cx, cy), (radius, radius),
            0, start_angle, end_angle, 255, 30
        )
        
        return mask


class TestPipeCoverageProcessor:
    """Test complete pipeline."""
    
    def test_full_pipeline_no_undistortion(self):
        """Test pipeline without undistortion."""
        processor = PipeCoverageProcessor()
        
        frame = self._create_test_frame(480, 640)
        mask = self._create_test_mask(480, 640)
        
        result = processor.process(frame, mask)
        
        assert result is not None
        assert 0 <= result.coverage_fraction <= 1.0
        print(f"✓ Pipeline (no undistort): {result.coverage_percentage:.1f}%")
    
    def test_full_pipeline_with_undistortion(self):
        """Test complete pipeline with undistortion."""
        intrinsics = CameraIntrinsics(
            fx=600, fy=600, cx=320, cy=240,
            k1=-0.2, k2=0.05
        )
        undistorter = CameraUndistorter(intrinsics)
        processor = PipeCoverageProcessor(undistorter=undistorter)
        
        frame = self._create_test_frame(480, 640)
        mask = self._create_test_mask(480, 640)
        
        result = processor.process(frame, mask)
        
        assert result is not None
        print(f"✓ Pipeline (with undistort): {result.coverage_percentage:.1f}%")
    
    def test_pipeline_with_visualization(self):
        """Test visualization output."""
        processor = PipeCoverageProcessor()
        
        frame = self._create_test_frame(480, 640)
        mask = self._create_test_mask(480, 640)
        
        result, vis_frame = processor.process_with_visualization(frame, mask)
        
        assert result is not None
        assert vis_frame.shape == frame.shape
        print("✓ Pipeline with visualization")
    
    def test_pipeline_rim_detection_failure(self):
        """Test graceful failure on bad input."""
        processor = PipeCoverageProcessor()
        
        # Blank frame - rim detection should fail
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        mask = np.zeros((480, 640), dtype=np.uint8)
        
        result = processor.process(frame, mask)
        
        assert result is None
        print("✓ Pipeline handles detection failure")
    
    def test_pipeline_reset(self):
        """Test state reset."""
        processor = PipeCoverageProcessor()
        
        frame1 = self._create_test_frame(480, 640)
        mask1 = self._create_test_mask(480, 640)
        
        result1 = processor.process(frame1, mask1)
        
        processor.reset()
        
        result2 = processor.process(frame1, mask1)
        
        # Both should succeed
        assert result1 is not None
        assert result2 is not None
        print("✓ Pipeline reset works")
    
    def _create_test_frame(self, height: int, width: int) -> np.ndarray:
        """Create synthetic sewer frame."""
        frame = np.random.randint(30, 100, (height, width, 3), dtype=np.uint8)
        
        # Draw pipe rim
        cx, cy = width // 2, height // 2
        radius = min(width, height) // 3
        cv2.circle(frame, (cx, cy), radius, (150, 150, 150), 3)
        
        return frame
    
    def _create_test_mask(self, height: int, width: int) -> np.ndarray:
        """Create synthetic defect mask."""
        mask = np.zeros((height, width), dtype=np.uint8)
        
        # Arc of defects on rim
        cx, cy = width // 2, height // 2
        radius = min(width, height) // 3
        cv2.ellipse(mask, (cx, cy), (radius, radius), 0, 30, 120, 255, 25)
        
        return mask


class TestIntegrationWithRealFrames:
    """Test with actual sewer inspection frames if available."""
    
    def test_real_frame_processing(self):
        """Test on real frame from temp_crops."""
        test_frame_path = "temp_crops/full_frame_33_det_0.jpg"
        
        if not os.path.exists(test_frame_path):
            print("⊘ Real frame test skipped (frame not found)")
            return
        
        frame = cv2.imread(test_frame_path)
        if frame is None:
            print("⊘ Real frame test skipped (could not load)")
            return
        
        # Create synthetic mask for testing
        h, w = frame.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        cx, cy = w // 2, h // 2
        cv2.ellipse(mask, (cx, cy), (w // 4, h // 4), 0, 0, 90, 255, 30)
        
        processor = PipeCoverageProcessor()
        result, vis = processor.process_with_visualization(frame, mask)
        
        if result is not None:
            print(f"✓ Real frame: {result.coverage_percentage:.1f}% coverage")
            
            # Save visualization
            output_dir = "test_output"
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, "coverage_test.jpg")
            cv2.imwrite(output_path, vis)
            print(f"  Saved visualization to {output_path}")
        else:
            print("⊘ Real frame test: rim detection failed")


def run_all_tests():
    """Execute all test suites."""
    print("\n" + "="*60)
    print("PIPE COVERAGE MEASUREMENT SYSTEM - TEST SUITE")
    print("="*60)
    
    test_suites = [
        ("Camera Undistorter", TestCameraUndistorter),
        ("Pipe Rim Detector", TestPipeRimDetector),
        ("Coverage Analyzer", TestCircumferenceCoverageAnalyzer),
        ("Coverage Processor", TestPipeCoverageProcessor),
        ("Real Frame Integration", TestIntegrationWithRealFrames)
    ]
    
    for suite_name, test_class in test_suites:
        print(f"\n[{suite_name}]")
        tester = test_class()
        
        for method_name in dir(tester):
            if method_name.startswith("test_"):
                try:
                    method = getattr(tester, method_name)
                    method()
                except Exception as e:
                    print(f"✗ {method_name} FAILED: {e}")
    
    print("\n" + "="*60)
    print("TEST SUITE COMPLETE")
    print("="*60 + "\n")


if __name__ == "__main__":
    run_all_tests()

