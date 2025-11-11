"""Few-shot classification module for root type detection."""

from .interface import IFewShotClassifier
from .root_classifier import RootClassifier

__all__ = ["IFewShotClassifier", "RootClassifier"]

