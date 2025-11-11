"""Longitudinal aggregation: analyze coverage along pipe distance."""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from collections import defaultdict
from .engineering_metrics import EngineeringMetrics, PhysicalCoverageMetrics


@dataclass
class MeterSegmentStats:
    """Statistics for a 1-meter pipe segment."""
    start_distance_m: float
    end_distance_m: float
    frame_count: int
    mean_coverage_pct: float
    p90_coverage_pct: float
    max_coverage_pct: float
    frames_with_defect: int
    defect_presence_pct: float
    per_class_mean: Dict[str, float] = field(default_factory=dict)


@dataclass
class LongitudinalReport:
    """Complete longitudinal analysis report."""
    total_distance_m: float
    segments: List[MeterSegmentStats]
    overall_mean_pct: float
    overall_p90_pct: float
    total_frames: int
    frames_with_defect: int
    longitudinal_coverage_pct: float  # % of pipe length with defects
    
    def get_segment_at_distance(self, distance_m: float) -> Optional[MeterSegmentStats]:
        """Get segment containing given distance."""
        for seg in self.segments:
            if seg.start_distance_m <= distance_m < seg.end_distance_m:
                return seg
        return None


class LongitudinalAggregator:
    """
    Aggregate coverage metrics along pipe distance.
    
    Step 4 (longitudinal): Aggregate per meter with mean/P90 coverage, 
    compute longitudinal coverage (% of pipe with defects).
    """
    
    def __init__(self, segment_length_m: float = 1.0):
        """
        Args:
            segment_length_m: Segment length for aggregation (default 1m)
        """
        self._segment_length = segment_length_m
        self._metrics_buffer: List[EngineeringMetrics] = []
    
    def add_frame(self, metrics: EngineeringMetrics) -> None:
        """
        Add frame metrics to buffer.
        
        Args:
            metrics: Engineering metrics for a frame
        """
        self._metrics_buffer.append(metrics)
    
    def compute_report(self, total_pipe_length_m: Optional[float] = None) -> LongitudinalReport:
        """
        Compute longitudinal aggregation report.
        
        Args:
            total_pipe_length_m: Total pipe length. If None, inferred from data
            
        Returns:
            Longitudinal analysis report
        """
        if not self._metrics_buffer:
            return self._empty_report()
        
        # Extract distances
        distances = []
        for m in self._metrics_buffer:
            if m.aggregate_physical.distance_along_pipe_m is not None:
                distances.append(m.aggregate_physical.distance_along_pipe_m)
        
        if not distances:
            # No distance data - treat as single segment
            return self._compute_single_segment_report()
        
        # Determine pipe length
        if total_pipe_length_m is None:
            total_pipe_length_m = max(distances) + self._segment_length
        
        # Group frames by segment
        segment_frames = self._group_by_segments(total_pipe_length_m)
        
        # Compute stats per segment
        segments = []
        for start_dist, frames in sorted(segment_frames.items()):
            stats = self._compute_segment_stats(
                start_dist,
                start_dist + self._segment_length,
                frames
            )
            segments.append(stats)
        
        # Overall statistics
        all_coverages = [
            m.aggregate_physical.circumference_pct
            for m in self._metrics_buffer
        ]
        
        frames_with_defect = sum(
            1 for m in self._metrics_buffer
            if m.aggregate_physical.circumference_pct > 0.1
        )
        
        segments_with_defect = sum(
            1 for seg in segments if seg.defect_presence_pct > 0
        )
        
        longitudinal_coverage_pct = (
            segments_with_defect / len(segments) * 100.0
            if segments else 0.0
        )
        
        return LongitudinalReport(
            total_distance_m=total_pipe_length_m,
            segments=segments,
            overall_mean_pct=float(np.mean(all_coverages)),
            overall_p90_pct=float(np.percentile(all_coverages, 90)),
            total_frames=len(self._metrics_buffer),
            frames_with_defect=frames_with_defect,
            longitudinal_coverage_pct=longitudinal_coverage_pct
        )
    
    def reset(self) -> None:
        """Clear buffer."""
        self._metrics_buffer.clear()
    
    def _group_by_segments(self, total_length: float) -> Dict[float, List[EngineeringMetrics]]:
        """Group frames by meter segments."""
        segments = defaultdict(list)
        
        for metrics in self._metrics_buffer:
            dist = metrics.aggregate_physical.distance_along_pipe_m
            if dist is None:
                continue
            
            # Find segment start
            segment_idx = int(dist / self._segment_length)
            segment_start = segment_idx * self._segment_length
            
            segments[segment_start].append(metrics)
        
        return segments
    
    def _compute_segment_stats(
        self,
        start_dist: float,
        end_dist: float,
        frames: List[EngineeringMetrics]
    ) -> MeterSegmentStats:
        """Compute statistics for a segment."""
        if not frames:
            return MeterSegmentStats(
                start_distance_m=start_dist,
                end_distance_m=end_dist,
                frame_count=0,
                mean_coverage_pct=0.0,
                p90_coverage_pct=0.0,
                max_coverage_pct=0.0,
                frames_with_defect=0,
                defect_presence_pct=0.0
            )
        
        # Aggregate coverage
        coverages = [m.aggregate_physical.circumference_pct for m in frames]
        frames_with_defect = sum(1 for c in coverages if c > 0.1)
        
        # Per-class mean
        per_class_mean = {}
        class_names = frames[0].per_class_physical.keys()
        for class_name in class_names:
            class_coverages = [
                m.per_class_physical[class_name].circumference_pct
                for m in frames
            ]
            per_class_mean[class_name] = float(np.mean(class_coverages))
        
        return MeterSegmentStats(
            start_distance_m=start_dist,
            end_distance_m=end_dist,
            frame_count=len(frames),
            mean_coverage_pct=float(np.mean(coverages)),
            p90_coverage_pct=float(np.percentile(coverages, 90)),
            max_coverage_pct=float(np.max(coverages)),
            frames_with_defect=frames_with_defect,
            defect_presence_pct=(frames_with_defect / len(frames)) * 100.0,
            per_class_mean=per_class_mean
        )
    
    def _compute_single_segment_report(self) -> LongitudinalReport:
        """Compute report when no distance data available."""
        all_coverages = [
            m.aggregate_physical.circumference_pct
            for m in self._metrics_buffer
        ]
        
        frames_with_defect = sum(
            1 for c in all_coverages if c > 0.1
        )
        
        segment = MeterSegmentStats(
            start_distance_m=0.0,
            end_distance_m=self._segment_length,
            frame_count=len(self._metrics_buffer),
            mean_coverage_pct=float(np.mean(all_coverages)),
            p90_coverage_pct=float(np.percentile(all_coverages, 90)),
            max_coverage_pct=float(np.max(all_coverages)),
            frames_with_defect=frames_with_defect,
            defect_presence_pct=(frames_with_defect / len(self._metrics_buffer)) * 100.0
        )
        
        return LongitudinalReport(
            total_distance_m=self._segment_length,
            segments=[segment],
            overall_mean_pct=segment.mean_coverage_pct,
            overall_p90_pct=segment.p90_coverage_pct,
            total_frames=len(self._metrics_buffer),
            frames_with_defect=frames_with_defect,
            longitudinal_coverage_pct=100.0 if frames_with_defect > 0 else 0.0
        )
    
    def _empty_report(self) -> LongitudinalReport:
        """Create empty report."""
        return LongitudinalReport(
            total_distance_m=0.0,
            segments=[],
            overall_mean_pct=0.0,
            overall_p90_pct=0.0,
            total_frames=0,
            frames_with_defect=0,
            longitudinal_coverage_pct=0.0
        )
    
    def format_report(self, report: LongitudinalReport) -> str:
        """Format longitudinal report as text."""
        lines = []
        lines.append("="*70)
        lines.append("LONGITUDINAL COVERAGE REPORT")
        lines.append("="*70)
        
        lines.append(f"\nOverall Statistics:")
        lines.append(f"  Total Distance: {report.total_distance_m:.1f} m")
        lines.append(f"  Total Frames: {report.total_frames}")
        lines.append(f"  Frames with Defects: {report.frames_with_defect}")
        lines.append(f"  Mean Coverage: {report.overall_mean_pct:.1f}%")
        lines.append(f"  P90 Coverage: {report.overall_p90_pct:.1f}%")
        lines.append(f"  Longitudinal Coverage: {report.longitudinal_coverage_pct:.1f}%")
        lines.append(f"    ({len([s for s in report.segments if s.defect_presence_pct > 0])}/{len(report.segments)} segments with defects)")
        
        lines.append(f"\nPer-Meter Segments:")
        for seg in report.segments:
            if seg.frame_count > 0:
                lines.append(f"  [{seg.start_distance_m:.1f}m - {seg.end_distance_m:.1f}m]:")
                lines.append(f"    Frames: {seg.frame_count}, Mean: {seg.mean_coverage_pct:.1f}%, P90: {seg.p90_coverage_pct:.1f}%")
                if seg.per_class_mean:
                    top_classes = sorted(
                        seg.per_class_mean.items(),
                        key=lambda x: x[1],
                        reverse=True
                    )[:3]
                    for cls, pct in top_classes:
                        if pct > 0.1:
                            lines.append(f"      {cls}: {pct:.1f}%")
        
        lines.append("="*70)
        
        return "\n".join(lines)

