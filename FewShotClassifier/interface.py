"""Interface for few-shot classification."""

from abc import ABC, abstractmethod
from typing import Dict, Any
import numpy as np


class IFewShotClassifier(ABC):
    """Interface for few-shot classification models."""
    
    @abstractmethod
    def predict(self, frame: np.ndarray) -> Dict[str, Any]:
        """Classify image crop into subcategory.
        
        Args:
            frame: Input image as numpy array (BGR format)
            
        Returns:
            Dictionary with 'class' and 'confidence' keys
        """
        pass
    
    @abstractmethod
    def is_loaded(self) -> bool:
        """Check if model is loaded and ready."""
        pass

