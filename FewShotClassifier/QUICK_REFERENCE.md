# Few-Shot Classifier Quick Reference

## Files Created

```
FewShotClassifier/
├── __init__.py           # Module exports
├── interface.py          # IFewShotClassifier interface
├── proto_model.py        # Prototypical Network
├── root_classifier.py    # RootClassifier implementation
├── README.md            # Usage documentation
├── ARCHITECTURE.md      # System design
└── QUICK_REFERENCE.md   # This file
```

## Model Setup Required

Ensure these files exist:
```
Model/MiniModel/Root/
├── model.pth                 # Trained weights
└── Support_Set/
    ├── root_mass_1.jpg       # 6 support images
    ├── root_mass_2.jpg       # (2 per class)
    ├── root_tap_1.jpg
    ├── root_tap_2.jpg
    ├── root_fine_1.jpg
    └── root_fine_2.jpg
```

## Integration Summary

### Imports Added
```python
from FewShotClassifier import RootClassifier
```

### Initialization (BatchDefectDetector)
```python
def _initialize_few_shot_classifier(self):
    self.root_classifier = RootClassifier(
        model_path="Model/MiniModel/Root/model.pth",
        support_set_dir="Model/MiniModel/Root/Support_Set",
        device=self.cfg.MODEL.DEVICE
    )
```

### Classification Hook
```python
# In process_video(), after _process_batches()
self._classify_root_detections(all_frames, all_predictions)
```

### Output Format
JSON detection with Root subclass:
```json
{
  "bbox": [x1, y1, x2, y2],
  "class": "Root",
  "confidence": 0.92,
  "root_subclass": "mass",
  "root_subclass_confidence": 0.87
}
```

## API Usage

```python
from FewShotClassifier import RootClassifier

# Initialize
classifier = RootClassifier(
    model_path="path/to/model.pth",
    support_set_dir="path/to/support_set",
    device="cuda"  # or "cpu"
)

# Check if loaded
if classifier.is_loaded():
    # Predict on crop (BGR numpy array)
    result = classifier.predict(crop_image)
    print(result)  # {"class": "mass", "confidence": 0.89}
```

## Class Names
- `"mass"` - Root mass
- `"tap"` - Tap root  
- `"fine"` - Fine root

## Dependencies
- torch
- torchvision
- PIL
- numpy

## Error Handling
- Classifier fails gracefully if model not found
- Logs warnings, continues without few-shot classification
- Per-detection try/catch prevents pipeline failures

## Performance
- GPU: ~5-10ms per crop
- CPU: ~20-50ms per crop
- Support set cached in memory
- Minimal overhead on primary detection

## Extending to Other Classes

1. Create new classifier:
```python
from FewShotClassifier import IFewShotClassifier

class MyClassifier(IFewShotClassifier):
    def predict(self, frame):
        # Implementation
        return {"class": "subtype", "confidence": 0.9}
    
    def is_loaded(self):
        return self.model is not None
```

2. Initialize in BatchDefectDetector:
```python
self.my_classifier = MyClassifier(...)
```

3. Add classification method:
```python
def _classify_my_detections(self, frames, predictions):
    # Follow pattern from _classify_root_detections
```

4. Update _process_detections to store subclass

## Troubleshooting

**Model not found**: Check path to `model.pth`
**Support set error**: Verify 6 images exist with correct naming
**CUDA out of memory**: Use `device="cpu"`
**Import error**: Ensure module in Python path
**No predictions**: Check if Root class detected in primary model

