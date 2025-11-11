"""Circumference coverage analyzer for pipe wall defects."""

import cv2
import numpy as np
from dataclasses import dataclass
from typing import Optional, Protocol
from .rim_detector import PipeRim


@dataclass
class CoverageResult:
    """Defect coverage measurement on pipe wall."""
    coverage_fraction: float
    covered_arc_degrees: float
    total_circumference_pixels: float
    covered_pixels: int
    rim_center: tuple[float, float]
    rim_radius: float
    
    @property
    def coverage_percentage(self) -> float:
        """Coverage as percentage."""
        return self.coverage_fraction * 100.0


class ICoverageAnalyzer(Protocol):
    """Interface for coverage analysis."""
    def analyze(self, frame: np.ndarray, mask: np.ndarray, rim: PipeRim) -> CoverageResult: ...


class CircumferenceCoverageAnalyzer:
    """Measure defect coverage as fraction of pipe wall circumference."""
    
    def __init__(self, angular_resolution: int = 360):
        """
        Args:
            angular_resolution: Number of angular bins for coverage [360=1deg bins]
        """
        self._angular_resolution = angular_resolution
    
    def analyze(self, frame: np.ndarray, mask: np.ndarray, rim: PipeRim) -> CoverageResult:
        """
        Compute circumference coverage.
        
        Args:
            frame: BGR input frame (for reference)
            mask: Binary mask of defect (0=background, 255=defect)
            rim: Detected pipe rim geometry
            
        Returns:
            Coverage measurement result
        """
        h, w = mask.shape[:2]
        cx, cy = rim.center
        r = rim.radius
        
        # Create angular coverage map
        angular_coverage = self._compute_angular_coverage(mask, cx, cy, r, h, w)
        
        # Count covered angular bins
        covered_bins = np.sum(angular_coverage > 0)
        coverage_fraction = covered_bins / self._angular_resolution
        
        # Convert to arc degrees
        covered_arc_degrees = coverage_fraction * 360.0
        
        # Total circumference
        total_circumference = 2 * np.pi * r
        
        # Count covered pixels
        covered_pixels = int(np.sum(mask > 0))
        
        return CoverageResult(
            coverage_fraction=coverage_fraction,
            covered_arc_degrees=covered_arc_degrees,
            total_circumference_pixels=total_circumference,
            covered_pixels=covered_pixels,
            rim_center=(cx, cy),
            rim_radius=r
        )
    
    def _compute_angular_coverage(
        self,
        mask: np.ndarray,
        cx: float,
        cy: float,
        radius: float,
        height: int,
        width: int
    ) -> np.ndarray:
        """
        Compute angular coverage histogram.
        
        Returns:
            1D array of shape (angular_resolution,) with coverage counts per bin
        """
        # Get all defect pixel coordinates
        y_coords, x_coords = np.where(mask > 0)
        
        if len(x_coords) == 0:
            return np.zeros(self._angular_resolution, dtype=np.int32)
        
        # Compute angles relative to rim center
        dx = x_coords - cx
        dy = y_coords - cy
        
        # Filter pixels near rim (within reasonable distance)
        distances = np.sqrt(dx**2 + dy**2)
        rim_tolerance = radius * 0.3  # Consider pixels within 30% of radius
        near_rim_mask = np.abs(distances - radius) < rim_tolerance
        
        if np.sum(near_rim_mask) == 0:
            return np.zeros(self._angular_resolution, dtype=np.int32)
        
        dx_filtered = dx[near_rim_mask]
        dy_filtered = dy[near_rim_mask]
        
        # Compute angles [0, 2*pi)
        angles = np.arctan2(dy_filtered, dx_filtered)
        angles = (angles + 2 * np.pi) % (2 * np.pi)
        
        # Bin into angular segments
        bin_size = 2 * np.pi / self._angular_resolution
        bin_indices = (angles / bin_size).astype(np.int32)
        bin_indices = np.clip(bin_indices, 0, self._angular_resolution - 1)
        
        # Count coverage per bin
        angular_coverage = np.bincount(bin_indices, minlength=self._angular_resolution)
        
        return angular_coverage
    
    def visualize_coverage(
        self,
        frame: np.ndarray,
        mask: np.ndarray,
        rim: PipeRim,
        result: CoverageResult
    ) -> np.ndarray:
        """
        Draw coverage visualization on frame.
        
        Args:
            frame: BGR input frame
            mask: Binary defect mask
            rim: Pipe rim
            result: Coverage result
            
        Returns:
            Annotated BGR frame
        """
        vis = frame.copy()
        
        # Draw rim circle
        cx, cy, r = int(rim.center_x), int(rim.center_y), int(rim.radius)
        cv2.circle(vis, (cx, cy), r, (0, 255, 255), 2)
        cv2.circle(vis, (cx, cy), 5, (0, 255, 255), -1)
        
        # Overlay mask
        mask_colored = np.zeros_like(vis)
        mask_colored[mask > 0] = (0, 0, 255)
        vis = cv2.addWeighted(vis, 0.7, mask_colored, 0.3, 0)
        
        # Draw coverage info
        info_text = [
            f"Coverage: {result.coverage_percentage:.1f}%",
            f"Arc: {result.covered_arc_degrees:.1f} deg",
            f"Radius: {result.rim_radius:.0f}px"
        ]
        
        y_offset = 30
        for text in info_text:
            cv2.putText(
                vis, text, (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
            )
            y_offset += 30
        
        return vis

