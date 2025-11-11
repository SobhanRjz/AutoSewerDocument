# Few-Shot Classification Module

## Overview
This module provides few-shot classification for root defect subtypes using Prototypical Networks.

## Architecture
- **Interface**: `IFewShotClassifier` - Abstract interface for few-shot classifiers
- **Implementation**: `RootClassifier` - Classifies roots into mass, tap, or fine
- **Model**: Prototypical Network with ResNet-18 backbone

## Root Classification
The `RootClassifier` distinguishes between three root types:
- **mass**: Root mass formations
- **tap**: Tap root intrusions  
- **fine**: Fine root intrusions

## Model Setup
1. Model weights: `Model/MiniModel/Root/model.pth`
2. Support set: `Model/MiniModel/Root/Support_Set/`
   - Requires 2 support images per class (6 total)
   - Format: `root_{class}_{1,2}.jpg`

## Integration
The classifier is automatically initialized in `BatchDefectDetector` and applied after primary detection:
1. Primary model detects "Root" class
2. Few-shot classifier crops root region
3. Classifier predicts subtype (mass/tap/fine)
4. Results stored in detection JSON with `root_subclass` and `root_subclass_confidence` fields

## Usage
```python
from FewShotClassifier import RootClassifier

classifier = RootClassifier(
    model_path="path/to/model.pth",
    support_set_dir="path/to/support_set",
    device="cuda"
)

result = classifier.predict(crop_image)
# Returns: {"class": "mass", "confidence": 0.95}
```

## Extension
To add more few-shot classifiers:
1. Implement `IFewShotClassifier` interface
2. Follow the pattern in `RootClassifier`
3. Integrate in `BatchDefectDetector._classify_root_detections()`

