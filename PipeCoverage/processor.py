"""Unified processor for pipe coverage analysis pipeline."""

import numpy as np
from typing import Optional
from .camera_undistort import CameraUndistorter, IUndistorter
from .rim_detector import PipeRimDetector, IRimDetector, PipeRim
from .coverage_analyzer import CircumferenceCoverageAnalyzer, ICoverageAnalyzer, CoverageResult


class PipeCoverageProcessor:
    """
    Complete pipeline: undistort → detect rim → measure coverage.
    
    Implements steps 1-2 of 7-step circumference coverage workflow:
    1. Undistort frame with camera intrinsics
    2. Detect pipe rim with RANSAC + temporal filtering
    (Steps 3-7 handled by subclasses or extensions)
    """
    
    def __init__(
        self,
        undistorter: Optional[IUndistorter] = None,
        rim_detector: Optional[IRimDetector] = None,
        coverage_analyzer: Optional[ICoverageAnalyzer] = None
    ):
        """
        Args:
            undistorter: Camera undistortion processor. None=skip undistortion
            rim_detector: Pipe rim detector. None=use default
            coverage_analyzer: Coverage measurement. None=use default
        """
        self._undistorter = undistorter
        self._rim_detector = rim_detector or PipeRimDetector()
        self._coverage_analyzer = coverage_analyzer or CircumferenceCoverageAnalyzer()
    
    def process(
        self,
        frame: np.ndarray,
        mask: np.ndarray
    ) -> Optional[CoverageResult]:
        """
        Process frame and defect mask to compute coverage.
        
        Args:
            frame: BGR input frame
            mask: Binary defect mask (0=background, 255=defect)
            
        Returns:
            Coverage result or None if rim detection fails
        """
        # Step 1: Undistort
        if self._undistorter is not None:
            frame = self._undistorter.undistort(frame)
            mask = self._undistorter.undistort(mask)
        
        # Step 2: Detect rim
        rim = self._rim_detector.detect(frame)
        
        if rim is None:
            return None
        
        # Compute coverage
        result = self._coverage_analyzer.analyze(frame, mask, rim)
        
        return result
    
    def process_with_visualization(
        self,
        frame: np.ndarray,
        mask: np.ndarray
    ) -> tuple[Optional[CoverageResult], np.ndarray]:
        """
        Process and generate visualization.
        
        Args:
            frame: BGR input frame
            mask: Binary defect mask
            
        Returns:
            (coverage_result, annotated_frame)
        """
        # Process
        undistorted_frame = frame
        undistorted_mask = mask
        
        if self._undistorter is not None:
            undistorted_frame = self._undistorter.undistort(frame)
            undistorted_mask = self._undistorter.undistort(mask)
        
        rim = self._rim_detector.detect(undistorted_frame)
        
        if rim is None:
            return None, undistorted_frame
        
        result = self._coverage_analyzer.analyze(undistorted_frame, undistorted_mask, rim)
        
        # Visualize
        vis_frame = self._coverage_analyzer.visualize_coverage(
            undistorted_frame,
            undistorted_mask,
            rim,
            result
        )
        
        return result, vis_frame
    
    def reset(self) -> None:
        """Reset temporal state (e.g., between videos)."""
        self._rim_detector.reset()

