"""Pipe circumference coverage measurement system."""

from .camera_undistort import CameraUndistorter, CameraIntrinsics
from .rim_detector import PipeRimDetector, PipeRim
from .coverage_analyzer import CircumferenceCoverageAnalyzer, CoverageResult
from .processor import PipeCoverageProcessor
from .polar_unwrap import PolarUnwrapAnalyzer, AngularCoverage
from .multiclass_coverage import (
    MultiClassCoverageAnalyzer,
    MultiClassCoverageResult,
    ClassCoverageResult
)
from .engineering_metrics import (
    EngineeringMetricsCalculator,
    EngineeringMetrics,
    PhysicalCoverageMetrics,
    PipeGeometry
)
from .longitudinal_aggregator import (
    LongitudinalAggregator,
    LongitudinalReport,
    MeterSegmentStats
)

__all__ = [
    'CameraUndistorter',
    'CameraIntrinsics',
    'PipeRimDetector',
    'PipeRim',
    'CircumferenceCoverageAnalyzer',
    'CoverageResult',
    'PipeCoverageProcessor',
    'PolarUnwrapAnalyzer',
    'AngularCoverage',
    'MultiClassCoverageAnalyzer',
    'MultiClassCoverageResult',
    'ClassCoverageResult',
    'EngineeringMetricsCalculator',
    'EngineeringMetrics',
    'PhysicalCoverageMetrics',
    'PipeGeometry',
    'LongitudinalAggregator',
    'LongitudinalReport',
    'MeterSegmentStats'
]

