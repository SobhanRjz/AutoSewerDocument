"""Polar unwrapping and annulus-based angular coverage measurement."""

import cv2
import numpy as np
from dataclasses import dataclass
from typing import Protocol
from .rim_detector import PipeRim


@dataclass
class AngularCoverage:
    """Angular coverage measurement in pipe wall annulus."""
    covered_bins: int
    total_bins: int
    coverage_fraction: float
    covered_degrees: float
    bin_mask: np.ndarray  # Boolean array indicating covered bins
    area_coverage_fraction: float = 0.0  # Fraction of pipe area covered by defect
    
    @property
    def coverage_percentage(self) -> float:
        return self.coverage_fraction * 100.0
    
    @property
    def area_coverage_percentage(self) -> float:
        return self.area_coverage_fraction * 100.0


class IPolarAnalyzer(Protocol):
    """Interface for polar unwrap analysis."""
    def analyze_angular_coverage(
        self, mask: np.ndarray, rim: PipeRim
    ) -> AngularCoverage: ...


class PolarUnwrapAnalyzer:
    """
    Measure angular coverage in wall annulus using polar unwrapping.
    
    Step 3: Defines annulus [0.8R, R], unwraps to polar strip, measures θ coverage.
    """
    
    def __init__(
        self,
        theta_bins: int = 360,
        inner_radius_factor: float = 0.6,
        outer_radius_factor: float = 1.1
    ):
        """
        Args:
            theta_bins: Angular resolution (360 = 1° per bin)
            inner_radius_factor: Inner annulus radius = factor × R
            outer_radius_factor: Outer annulus radius = factor × R
        """
        self._theta_bins = theta_bins
        self._r_inner_factor = inner_radius_factor
        self._r_outer_factor = outer_radius_factor
    
    def analyze_angular_coverage(
        self,
        mask: np.ndarray,
        rim: PipeRim
    ) -> AngularCoverage:
        """
        Compute angular coverage in wall annulus.
        
        Args:
            mask: Binary defect mask (0=background, 255=defect)
            rim: Detected pipe rim
            
        Returns:
            Angular coverage measurement
        """
        cx, cy = rim.center
        R = rim.radius
        
        # Define annulus
        r_inner = R * self._r_inner_factor
        r_outer = R * self._r_outer_factor
        
        # Get defect pixels
        y_coords, x_coords = np.where(mask > 0)
        
        if len(x_coords) == 0:
            return AngularCoverage(
                covered_bins=0,
                total_bins=self._theta_bins,
                coverage_fraction=0.0,
                covered_degrees=0.0,
                bin_mask=np.zeros(self._theta_bins, dtype=bool),
                area_coverage_fraction=0.0
            )
        
        # Compute polar coordinates
        dx = x_coords - cx
        dy = y_coords - cy
        distances = np.sqrt(dx**2 + dy**2)
        
        # Filter to annulus
        in_annulus = (distances >= r_inner) & (distances <= r_outer)
        
                # Compute area coverage: defect pixels / circle area detected by rim
        defect_pixel_count = len(x_coords)
        circle_area = np.pi * R ** 2
        area_coverage_fraction = defect_pixel_count / circle_area if circle_area > 0 else 0.0
        
        if np.sum(in_annulus) == 0:
            return AngularCoverage(
                covered_bins=0,
                total_bins=self._theta_bins,
                coverage_fraction=0.0,
                covered_degrees=0.0,
                bin_mask=np.zeros(self._theta_bins, dtype=bool),
                area_coverage_fraction=area_coverage_fraction
            )
        
        # Compute angles for pixels in annulus
        dx_annulus = dx[in_annulus]
        dy_annulus = dy[in_annulus]
        angles = np.arctan2(dy_annulus, dx_annulus)
        angles = (angles + 2 * np.pi) % (2 * np.pi)
        
        # Bin into angular segments
        bin_indices = (angles / (2 * np.pi) * self._theta_bins).astype(np.int32)
        bin_indices = np.clip(bin_indices, 0, self._theta_bins - 1)
        
        # Mark covered bins
        bin_mask = np.zeros(self._theta_bins, dtype=bool)
        bin_mask[bin_indices] = True
        
        covered_bins = np.sum(bin_mask)
        coverage_fraction = covered_bins / self._theta_bins
        covered_degrees = coverage_fraction * 360.0
        

        return AngularCoverage(
            covered_bins=covered_bins,
            total_bins=self._theta_bins,
            coverage_fraction=coverage_fraction,
            covered_degrees=covered_degrees,
            bin_mask=bin_mask,
            area_coverage_fraction=area_coverage_fraction
        )
    
    def create_polar_unwrap(
        self,
        frame: np.ndarray,
        rim: PipeRim,
        radial_samples: int = 100
    ) -> np.ndarray:
        """
        Unwrap annulus to rectangular strip for visualization.
        
        Args:
            frame: Input BGR frame
            rim: Pipe rim
            radial_samples: Number of radial samples
            
        Returns:
            Unwrapped strip [radial_samples × theta_bins, 3]
        """
        cx, cy = rim.center
        R = rim.radius
        
        r_inner = R * self._r_inner_factor
        r_outer = R * self._r_outer_factor
        
        unwrapped = np.zeros((radial_samples, self._theta_bins, 3), dtype=np.uint8)
        
        for theta_idx in range(self._theta_bins):
            theta = (theta_idx / self._theta_bins) * 2 * np.pi
            
            for r_idx in range(radial_samples):
                # Linear interpolation in radius
                r = r_inner + (r_outer - r_inner) * (r_idx / radial_samples)
                
                x = int(cx + r * np.cos(theta))
                y = int(cy + r * np.sin(theta))
                
                h, w = frame.shape[:2]
                if 0 <= y < h and 0 <= x < w:
                    unwrapped[r_idx, theta_idx] = frame[y, x]
        
        return unwrapped
    
    def visualize_annulus(
        self,
        frame: np.ndarray,
        rim: PipeRim,
        coverage: AngularCoverage
    ) -> np.ndarray:
        """
        Draw annulus and covered angular sectors.
        
        Args:
            frame: Input BGR frame
            rim: Pipe rim
            coverage: Angular coverage result
            
        Returns:
            Annotated frame
        """
        vis = frame.copy()
        cx, cy = int(rim.center_x), int(rim.center_y)
        R = int(rim.radius)
        
        r_inner = int(R * self._r_inner_factor)
        r_outer = int(R * self._r_outer_factor)
        
        # Draw annulus boundaries
        cv2.circle(vis, (cx, cy), r_inner, (100, 100, 100), 2)
        cv2.circle(vis, (cx, cy), r_outer, (200, 200, 200), 2)
        
        # Draw covered angular sectors
        for bin_idx, is_covered in enumerate(coverage.bin_mask):
            if is_covered:
                theta_start = (bin_idx / self._theta_bins) * 360
                theta_end = ((bin_idx + 1) / self._theta_bins) * 360
                
                # Draw arc in the annulus
                cv2.ellipse(
                    vis, (cx, cy),
                    ((r_inner + r_outer) // 2, (r_inner + r_outer) // 2),
                    0, theta_start, theta_end,
                    (0, 255, 0), max(1, (r_outer - r_inner) // 2)
                )
        
        return vis

