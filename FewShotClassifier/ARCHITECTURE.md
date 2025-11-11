# Few-Shot Classifier Architecture

## System Flow

```
Video Input
    ↓
Frame Extraction (FFmpeg)
    ↓
Primary Detection (Detectron2)
    ↓
Root Detected? ──No──→ Continue
    ↓ Yes
Few-Shot Classification
    ├─ Extract Crop
    ├─ Preprocess (224x224)
    ├─ Feature Extraction (ResNet-18)
    ├─ Compute Prototypes
    ├─ Distance Metric
    └─ Softmax → [mass, tap, fine]
    ↓
Attach Subclass to Detection
    ↓
Visualization & Export
```

## Class Diagram

```
┌─────────────────────────┐
│  IFewShotClassifier     │ (ABC)
├─────────────────────────┤
│ + predict(frame)        │
│ + is_loaded() → bool    │
└───────────▲─────────────┘
            │
            │ implements
            │
┌───────────┴─────────────┐
│   RootClassifier        │
├─────────────────────────┤
│ - model                 │
│ - support_tensors       │
│ - support_labels        │
│ - transform             │
├─────────────────────────┤
│ + __init__(model_path)  │
│ + predict(frame)        │
│ + is_loaded()           │
│ - _load_model()         │
│ - _load_support_set()   │
└─────────────────────────┘

┌─────────────────────────┐
│  PrototypicalNetwork    │
├─────────────────────────┤
│ - backbone: ResNet-18   │
├─────────────────────────┤
│ + forward(support,      │
│           labels,        │
│           query)         │
│ - _freeze_backbone()    │
└─────────────────────────┘
```

## Integration with BatchDefectDetector

```
BatchDefectDetector
│
├─ __init__()
│   └─ _initialize_few_shot_classifier()
│       └─ self.root_classifier = RootClassifier(...)
│
├─ process_video()
│   ├─ _collect_frames()
│   ├─ _process_batches()
│   ├─ _classify_root_detections()  ← NEW
│   │   └─ For each Root detection:
│   │       ├─ Extract crop from frame
│   │       └─ root_classifier.predict(crop)
│   ├─ _visualize_batch()
│   └─ _save_results()
│
└─ _process_detections()
    └─ Attach root_subclass to detection dict
```

## Data Flow

```
Frame [H,W,3]
    ↓
Root BBox [x1,y1,x2,y2]
    ↓
Crop [H',W',3]
    ↓
Resize → [224,224,3]
    ↓
ToTensor → [3,224,224]
    ↓
ResNet-18 → [512] features
    ↓
Compute prototypes:
  mass_proto = mean(support_mass_features)
  tap_proto  = mean(support_tap_features)
  fine_proto = mean(support_fine_features)
    ↓
Distances:
  d_mass = ||query_feat - mass_proto||
  d_tap  = ||query_feat - tap_proto||
  d_fine = ||query_feat - fine_proto||
    ↓
Scores = -distances
    ↓
Softmax → [P(mass), P(tap), P(fine)]
    ↓
argmax → predicted class
```

## Support Set Structure

```
Support_Set/
├── root_mass_1.jpg  ─┐
├── root_mass_2.jpg  ─┤ Class 0: mass
├── root_tap_1.jpg   ─┤ Class 1: tap
├── root_tap_2.jpg   ─┤
├── root_fine_1.jpg  ─┤ Class 2: fine
└── root_fine_2.jpg  ─┘

Loaded as:
support_tensors: [6, 3, 224, 224]
support_labels:  [0, 0, 1, 1, 2, 2]
```

## Extensibility

To add more classifiers:

1. Create new classifier class implementing `IFewShotClassifier`
2. Initialize in `BatchDefectDetector.__init__()`
3. Add classification method following `_classify_root_detections` pattern
4. Update `_process_detections()` to store subclass fields

Example for Crack subtype:
```python
self.crack_classifier = CrackClassifier(
    model_path="Model/MiniModel/Crack/model.pth",
    support_set_dir="Model/MiniModel/Crack/Support_Set"
)
```

## Performance Characteristics

| Metric | Value |
|--------|-------|
| Model size | ~45MB (ResNet-18) |
| Support set | 6 images (cached) |
| Inference (GPU) | 5-10ms per crop |
| Inference (CPU) | 20-50ms per crop |
| Memory overhead | ~200MB (model + support) |
| Accuracy | Depends on support set quality |

## Design Principles

1. **Interface-based**: Easy to extend with new classifiers
2. **Lazy loading**: Fails gracefully if model unavailable
3. **Minimal coupling**: Independent from primary detector
4. **Efficient**: Support set cached, single pass inference
5. **Production-ready**: Error handling, logging, type hints

