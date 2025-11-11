"""Root type classifier using prototypical networks."""

import os
import torch
import numpy as np
from pathlib import Path
from typing import Dict, Any
from PIL import Image
from torchvision import transforms
import logging

from .interface import IFewShotClassifier


class RootClassifier(IFewShotClassifier):
    """Classifies root detections into mass, tap, or fine."""
    
    CLASS_NAMES = ["mass", "tap", "fine"]
    
    def __init__(
        self,
        model_path: str,
        support_set_dir: str,
        device: str = None
    ):
        """Initialize classifier.
        
        Args:
            model_path: Path to trained model weights
            support_set_dir: Directory containing support set images
            device: Device for inference ('cuda' or 'cpu')
        """
        self.logger = logging.getLogger(__name__)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_path = model_path
        self.support_set_dir = support_set_dir
        
        self.model = None
        self.support_tensors = None
        self.support_labels = None
        self.transform = None
        
        self._load_model()
        self._load_support_set()
    
    def _load_model(self) -> None:
        """Load prototypical network model."""
        from .proto_model import PrototypicalNetwork, create_backbone
        
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model not found: {self.model_path}")
        
        backbone = create_backbone("resnet18", pretrained=False)
        self.model = PrototypicalNetwork(backbone, freeze_backbone=False)
        
        state_dict = torch.load(self.model_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()
        
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])
        
        self.logger.info(f"Loaded model from {self.model_path}")
    
    def _load_support_set(self) -> None:
        """Load support set images."""
        support_dir = Path(self.support_set_dir)
        if not support_dir.exists():
            raise FileNotFoundError(f"Support set not found: {support_dir}")
        
        support_images = []
        support_labels = []
        
        for class_idx, class_name in enumerate(self.CLASS_NAMES):
            for shot_idx in [1, 2]:
                img_path = support_dir / f"root_{class_name}_{shot_idx}.jpg"
                if not img_path.exists():
                    raise FileNotFoundError(f"Support image not found: {img_path}")
                
                img = Image.open(img_path).convert('RGB')
                img_tensor = self.transform(img)
                support_images.append(img_tensor)
                support_labels.append(class_idx)
        
        self.support_tensors = torch.stack(support_images).to(self.device)
        self.support_labels = torch.tensor(support_labels).to(self.device)
        
        self.logger.info(f"Loaded support set with {len(support_images)} images")
    
    def predict(self, frame: np.ndarray) -> Dict[str, Any]:
        """Classify root detection into subcategory.
        
        Args:
            frame: BGR image crop from detection
            
        Returns:
            Dict with 'class' and 'confidence' keys
        """
        if not self.is_loaded():
            raise RuntimeError("Model not loaded")
        
        rgb_frame = frame[:, :, ::-1]
        pil_img = Image.fromarray(rgb_frame)
        query_tensor = self.transform(pil_img).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            scores = self.model(self.support_tensors, self.support_labels, query_tensor)
            probs = torch.softmax(scores, dim=1)
            confidence, pred = torch.max(probs, 1)
        
        return {
            "class": self.CLASS_NAMES[pred.item()],
            "confidence": float(confidence.item())
        }
    
    def is_loaded(self) -> bool:
        """Check if model and support set are loaded."""
        return (
            self.model is not None and 
            self.support_tensors is not None and 
            self.support_labels is not None
        )

