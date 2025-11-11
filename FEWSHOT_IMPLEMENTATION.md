# Few-Shot Classification Implementation

## Overview
Integrated few-shot learning for Root defect subtype classification (mass, tap, fine) using Prototypical Networks.

## Components Created

### 1. FewShotClassifier Module (`FewShotClassifier/`)
- `interface.py` - Abstract interface `IFewShotClassifier`
- `proto_model.py` - Prototypical Network implementation
- `root_classifier.py` - Root subtype classifier
- `README.md` - Module documentation

### 2. Model Structure
```
Model/MiniModel/Root/
├── model.pth                    # Trained prototypical network
└── Support_Set/
    ├── root_mass_1.jpg          # Support examples (2 per class)
    ├── root_mass_2.jpg
    ├── root_tap_1.jpg
    ├── root_tap_2.jpg
    ├── root_fine_1.jpg
    └── root_fine_2.jpg
```

## Integration Points

### BatchDefectDetector Changes

1. **Import** (line 31)
```python
from FewShotClassifier import RootClassifier
```

2. **Initialization** (line 66)
```python
self._initialize_few_shot_classifier()
```

3. **Method: _initialize_few_shot_classifier** (lines 88-103)
Loads model and support set, handles failures gracefully.

4. **Process Flow** (line 233)
```python
# After batch inference
self._classify_root_detections(all_frames, all_predictions)
```

5. **Method: _classify_root_detections** (lines 829-890)
- Filters Root detections from primary model
- Extracts bounding box crops
- Applies few-shot classifier
- Attaches subclass to instances

6. **Detection Storage** (lines 952-971)
Enhanced `_process_detections` to store:
```json
{
  "class": "Root",
  "confidence": 0.95,
  "root_subclass": "mass",
  "root_subclass_confidence": 0.89
}
```

## Architecture

### Prototypical Network
- Backbone: ResNet-18 (pretrained, layer4 unfrozen)
- Few-shot: 3-way, 2-shot per class
- Metric: Euclidean distance to class prototypes
- Output: Softmax scores over 3 classes

### Interface Pattern
```python
class IFewShotClassifier(ABC):
    @abstractmethod
    def predict(self, frame: np.ndarray) -> Dict[str, Any]:
        pass
    
    @abstractmethod
    def is_loaded(self) -> bool:
        pass
```

## Usage Flow

1. Primary model detects "Root" class
2. For each Root detection:
   - Extract bounding box crop from frame
   - Pass crop to `RootClassifier.predict()`
   - Receive subclass (mass/tap/fine) + confidence
3. Store results in detection metadata
4. Export to JSON and Excel reports

## Extension Guide

To add new few-shot classifiers (e.g., Crack subtypes):

1. Implement `IFewShotClassifier`:
```python
class CrackClassifier(IFewShotClassifier):
    def predict(self, frame: np.ndarray) -> Dict[str, Any]:
        # Return {"class": "longitudinal", "confidence": 0.92}
```

2. Initialize in `BatchDefectDetector.__init__()`:
```python
self.crack_classifier = CrackClassifier(...)
```

3. Add classification logic in `_classify_root_detections()` pattern:
```python
def _classify_crack_detections(self, frames, predictions):
    crack_class_idx = self.metadata.thing_classes.index("Crack")
    # Similar pattern as root classification
```

4. Update `_process_detections()` to store subclass fields

## Performance Notes

- Classifier loaded once at initialization (cached in GPU memory)
- Support set preprocessed and cached
- Inference: ~5-10ms per crop (GPU), ~20-50ms (CPU)
- Minimal overhead: only processes Root detections

## Dependencies

- torch, torchvision (ResNet-18 backbone)
- PIL (image preprocessing)
- numpy (array operations)

## Error Handling

- Graceful degradation if model not found
- Logs warnings if classifier unavailable
- Continues primary detection if few-shot fails
- Per-detection try/catch for robustness

