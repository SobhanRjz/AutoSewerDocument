"""Engineering metrics: physical measurements from angular coverage."""

import numpy as np
from dataclasses import dataclass
from typing import Optional
from .multiclass_coverage import MultiClassCoverageResult


@dataclass
class PipeGeometry:
    """Physical pipe dimensions."""
    inner_diameter_mm: float
    
    @property
    def circumference_mm(self) -> float:
        """Inner circumference in mm."""
        return np.pi * self.inner_diameter_mm
    
    @property
    def radius_mm(self) -> float:
        """Inner radius in mm."""
        return self.inner_diameter_mm / 2.0


@dataclass
class PhysicalCoverageMetrics:
    """Physical coverage measurements."""
    blocked_arc_length_mm: float
    circumference_pct: float
    circumference_mm: float
    theta_bins: int
    arc_length_per_bin_mm: float
    distance_along_pipe_m: Optional[float] = None


@dataclass
class EngineeringMetrics:
    """Complete engineering metrics for a frame."""
    aggregate_physical: PhysicalCoverageMetrics
    per_class_physical: dict[str, PhysicalCoverageMetrics]
    pipe_geometry: PipeGeometry
    frame_id: Optional[str] = None
    timestamp_sec: Optional[float] = None


class EngineeringMetricsCalculator:
    """
    Convert angular coverage to engineering measurements.
    
    Step 4: Turn angular coverage into physical measurements (mm, %).
    """
    
    def __init__(self, pipe_geometry: PipeGeometry):
        """
        Args:
            pipe_geometry: Physical pipe dimensions
        """
        self._pipe = pipe_geometry
    
    def compute_metrics(
        self,
        coverage_result: MultiClassCoverageResult,
        distance_along_pipe_m: Optional[float] = None,
        timestamp_sec: Optional[float] = None
    ) -> EngineeringMetrics:
        """
        Compute physical metrics from angular coverage.
        
        Args:
            coverage_result: Multi-class coverage result
            distance_along_pipe_m: Distance along pipe (from odometry)
            timestamp_sec: Frame timestamp
            
        Returns:
            Engineering metrics
        """
        # Aggregate metrics
        agg_cov = coverage_result.aggregate_any_defect
        aggregate_physical = self._compute_physical_metrics(
            agg_cov.coverage_fraction,
            agg_cov.total_bins,
            distance_along_pipe_m
        )
        
        # Per-class metrics
        per_class_physical = {}
        for class_name, class_result in coverage_result.per_class.items():
            cov = class_result.angular_coverage
            per_class_physical[class_name] = self._compute_physical_metrics(
                cov.coverage_fraction,
                cov.total_bins,
                distance_along_pipe_m
            )
        
        return EngineeringMetrics(
            aggregate_physical=aggregate_physical,
            per_class_physical=per_class_physical,
            pipe_geometry=self._pipe,
            frame_id=coverage_result.frame_id,
            timestamp_sec=timestamp_sec
        )
    
    def _compute_physical_metrics(
        self,
        coverage_fraction: float,
        theta_bins: int,
        distance_along_pipe_m: Optional[float]
    ) -> PhysicalCoverageMetrics:
        """Compute physical measurements from coverage fraction."""
        circumference_mm = self._pipe.circumference_mm
        arc_length_per_bin_mm = circumference_mm / theta_bins
        blocked_arc_length_mm = coverage_fraction * circumference_mm
        
        return PhysicalCoverageMetrics(
            blocked_arc_length_mm=blocked_arc_length_mm,
            circumference_pct=coverage_fraction * 100.0,
            circumference_mm=circumference_mm,
            theta_bins=theta_bins,
            arc_length_per_bin_mm=arc_length_per_bin_mm,
            distance_along_pipe_m=distance_along_pipe_m
        )
    
    def format_report(self, metrics: EngineeringMetrics) -> str:
        """
        Format engineering metrics as readable report.
        
        Args:
            metrics: Engineering metrics
            
        Returns:
            Formatted text report
        """
        lines = []
        lines.append("="*60)
        lines.append("ENGINEERING METRICS REPORT")
        lines.append("="*60)
        
        if metrics.frame_id:
            lines.append(f"Frame: {metrics.frame_id}")
        if metrics.timestamp_sec is not None:
            lines.append(f"Timestamp: {metrics.timestamp_sec:.2f}s")
        
        lines.append(f"\nPipe Geometry:")
        lines.append(f"  Inner Diameter: {metrics.pipe_geometry.inner_diameter_mm:.0f} mm")
        lines.append(f"  Circumference: {metrics.pipe_geometry.circumference_mm:.0f} mm")
        
        if metrics.aggregate_physical.distance_along_pipe_m is not None:
            lines.append(f"  Distance Along Pipe: {metrics.aggregate_physical.distance_along_pipe_m:.2f} m")
        
        lines.append(f"\nAggregate (Any Defect):")
        agg = metrics.aggregate_physical
        lines.append(f"  Circumference Coverage: {agg.circumference_pct:.1f}%")
        lines.append(f"  Blocked Arc Length: {agg.blocked_arc_length_mm:.0f} mm")
        lines.append(f"  Arc per Bin: {agg.arc_length_per_bin_mm:.2f} mm")
        
        lines.append(f"\nPer-Class Coverage:")
        for class_name, phys in metrics.per_class_physical.items():
            if phys.circumference_pct > 0.1:  # Only show non-zero
                lines.append(f"  {class_name}:")
                lines.append(f"    Coverage: {phys.circumference_pct:.1f}%")
                lines.append(f"    Blocked Arc: {phys.blocked_arc_length_mm:.0f} mm")
        
        lines.append("="*60)
        
        return "\n".join(lines)

