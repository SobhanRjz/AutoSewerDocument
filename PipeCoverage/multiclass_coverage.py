"""Multi-class coverage analysis with per-class and aggregate metrics."""

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional
from .rim_detector import PipeRim
from .polar_unwrap import PolarUnwrapAnalyzer, AngularCoverage


@dataclass
class ClassCoverageResult:
    """Coverage for a single defect class."""
    class_name: str
    angular_coverage: AngularCoverage
    pixel_count: int


@dataclass
class MultiClassCoverageResult:
    """Coverage results for multiple defect classes."""
    per_class: Dict[str, ClassCoverageResult]
    aggregate_any_defect: AngularCoverage
    frame_id: Optional[str] = None
    
    def get_summary(self) -> Dict[str, float]:
        """Get summary statistics as percentages."""
        summary = {
            'any_defect_pct': self.aggregate_any_defect.coverage_percentage
        }
        
        for class_name, result in self.per_class.items():
            summary[f'{class_name}_pct'] = result.angular_coverage.coverage_percentage
        
        return summary


class MultiClassCoverageAnalyzer:
    """
    Analyze angular coverage per defect class and aggregate.
    
    Step 3 (multi-class): Measure circumference coverage for roots, grease, 
    deposits, etc., plus an "any defect" aggregate.
    """
    
    def __init__(
        self,
        class_names: List[str],
        theta_bins: int = 360,
        inner_radius_factor: float = 0.8,
        outer_radius_factor: float = 1.0
    ):
        """
        Args:
            class_names: List of defect class names
            theta_bins: Angular resolution
            inner_radius_factor: Inner annulus radius factor
            outer_radius_factor: Outer annulus radius factor
        """
        self._class_names = class_names
        self._polar_analyzer = PolarUnwrapAnalyzer(
            theta_bins=theta_bins,
            inner_radius_factor=inner_radius_factor,
            outer_radius_factor=outer_radius_factor
        )
    
    def analyze(
        self,
        frame: np.ndarray,
        class_masks: Dict[str, np.ndarray],
        rim: PipeRim,
        frame_id: Optional[str] = None
    ) -> MultiClassCoverageResult:
        """
        Analyze coverage for each class and aggregate.
        
        Args:
            frame: Input BGR frame
            class_masks: Dict mapping class_name → binary mask
            rim: Detected pipe rim
            frame_id: Optional frame identifier
            
        Returns:
            Multi-class coverage result
        """
        per_class = {}
        
        # Analyze each class
        for class_name in self._class_names:
            mask = class_masks.get(class_name, np.zeros_like(frame[:, :, 0]))
            
            coverage = self._polar_analyzer.analyze_angular_coverage(mask, rim)
            pixel_count = int(np.sum(mask > 0))
            
            per_class[class_name] = ClassCoverageResult(
                class_name=class_name,
                angular_coverage=coverage,
                pixel_count=pixel_count
            )
        
        # Compute aggregate "any defect"
        aggregate_mask = np.zeros_like(frame[:, :, 0], dtype=np.uint8)
        for mask in class_masks.values():
            aggregate_mask = np.maximum(aggregate_mask, mask)
        
        aggregate_coverage = self._polar_analyzer.analyze_angular_coverage(
            aggregate_mask, rim
        )
        
        return MultiClassCoverageResult(
            per_class=per_class,
            aggregate_any_defect=aggregate_coverage,
            frame_id=frame_id
        )
    
    def visualize(
        self,
        frame: np.ndarray,
        result: MultiClassCoverageResult,
        rim: PipeRim,
        class_masks: Dict[str, np.ndarray],
        show_per_class: bool = False
    ) -> np.ndarray:
        """
        Visualize coverage with annotations.
        
        Args:
            frame: Input frame
            result: Coverage result
            rim: Pipe rim
            class_masks: Dict mapping class_name → binary mask
            show_per_class: If True, show per-class breakdown
            
        Returns:
            Annotated frame
        """
        vis = self._polar_analyzer.visualize_annulus(
            frame, rim, result.aggregate_any_defect
        )
        
        # Draw defect masks with transparency
        mask_overlay = vis.copy()
        colors = [
            (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
            (255, 0, 255), (0, 255, 255), (128, 0, 128), (255, 128, 0)
        ]
        
        for idx, (class_name, class_result) in enumerate(result.per_class.items()):
            if class_result.pixel_count > 0:
                mask = class_masks.get(class_name, np.zeros_like(frame[:, :, 0]))
                color = colors[idx % len(colors)]
                mask_overlay[mask > 0] = color
        
        vis = cv2.addWeighted(vis, 0.7, mask_overlay, 0.3, 0)
        
        # Draw rim center
        cx, cy = int(rim.center_x), int(rim.center_y)
        cv2.circle(vis, (cx, cy), 5, (0, 255, 255), -1)
        
        # Add text annotations
        y_offset = 30
        
        # Aggregate
        ang = result.aggregate_any_defect.coverage_percentage
        area = result.aggregate_any_defect.area_coverage_percentage
        text = f"Any Defect: Ang={ang:.1f}% Area={area:.1f}%"
        cv2.putText(vis, text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 255, 0), 2)
        y_offset += 30
        
        # Per-class (if requested)
        if show_per_class:
            for idx, (class_name, class_result) in enumerate(result.per_class.items()):
                if class_result.pixel_count > 0:
                    ang = class_result.angular_coverage.coverage_percentage
                    area = class_result.angular_coverage.area_coverage_percentage
                    color = colors[idx % len(colors)]
                    text = f"{class_name}: Ang={ang:.1f}% Area={area:.1f}%"
                    cv2.putText(vis, text, (10, y_offset),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                    y_offset += 25
        
        return vis


import cv2  # Import needed for visualize method

