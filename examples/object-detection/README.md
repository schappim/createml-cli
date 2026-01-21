# Object Detection Example

This example shows the data format for training object detection models to detect and locate objects in images.

## Directory Structure

```
object-detection/
├── annotations.json
└── images/
    ├── street_001.jpg
    ├── street_002.jpg
    ├── park_001.jpg
    └── ...
```

## Data Format

The `annotations.json` file contains bounding box annotations for each image:

```json
[
  {
    "image": "images/street_001.jpg",
    "annotations": [
      {
        "label": "car",
        "coordinates": {"x": 150, "y": 200, "width": 300, "height": 180}
      },
      {
        "label": "person",
        "coordinates": {"x": 500, "y": 180, "width": 80, "height": 200}
      }
    ]
  }
]
```

### Coordinate Format

- `x` - X position of the bounding box center
- `y` - Y position of the bounding box center
- `width` - Width of the bounding box
- `height` - Height of the bounding box

## Training

**Note:** Add your own images to the `images/` directory and update `annotations.json` accordingly.

```bash
createml object-detect . -o ObjectDetector.mlmodel
```

### With Options

```bash
createml object-detect . -o ObjectDetector.mlmodel \
  --iterations 1000 \
  --batch-size 16 \
  --validation ../validation_data/ \
  --author "Your Name"
```

## Output

```
Loading training data from ...
Configuring training parameters...
Training object detector (max 500 iterations)...
This may take a while depending on your dataset size...
Saving model to ObjectDetector.mlmodel...

Training Complete!
==================================================

Model saved to: ObjectDetector.mlmodel

Classes: bicycle, car, dog, person

Metrics (mAP @ IoU 0.5):
  Training mAP:        85.50%
  Validation mAP:      82.30%
  Training duration:   342.15s
```

## Tips

1. **Image Quality** - Use high-resolution images (at least 416x416 pixels)
2. **Annotations** - Ensure bounding boxes tightly fit objects
3. **Balance** - Include similar numbers of examples for each class
4. **Variety** - Include objects at different scales, angles, and lighting
5. **Dataset Size** - Aim for at least 50-100 images per class
