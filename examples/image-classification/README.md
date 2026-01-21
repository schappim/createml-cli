# Image Classification Example

Train image classifiers by organizing images into labeled subdirectories.

## Directory Structure

```
training_data/
├── cats/
│   ├── cat_001.jpg
│   ├── cat_002.jpg
│   └── ...
├── dogs/
│   ├── dog_001.jpg
│   ├── dog_002.jpg
│   └── ...
└── birds/
    ├── bird_001.jpg
    └── ...
```

Each subdirectory name becomes a class label.

## Training

```bash
createml image training_data/ -o PetClassifier.mlmodel
```

### With Options

```bash
createml image training_data/ -o PetClassifier.mlmodel \
  --iterations 50 \
  --validation validation_data/ \
  --author "Your Name" \
  --description "Pet classifier model" \
  --no-augmentation
```

## Output

```
Loading training data from training_data/...
Configuring training parameters...
Training image classifier (max 25 iterations)...
Saving model to PetClassifier.mlmodel...

Training Complete!
==================================================

Model saved to: PetClassifier.mlmodel

Classes: birds, cats, dogs

Metrics:
  Training accuracy:   98.50%
  Validation accuracy: 95.20%
  Training duration:   45.32s
```

## Supported Formats

- JPEG (.jpg, .jpeg)
- PNG (.png)
- HEIC (.heic)
- TIFF (.tiff, .tif)

## Tips

1. **Consistency** - Use similar image sizes and quality across classes
2. **Balance** - Include similar numbers of images per class
3. **Variety** - Include images with different backgrounds, lighting, angles
4. **Minimum** - Aim for at least 10-20 images per class (more is better)
5. **Augmentation** - Enabled by default, helps with small datasets
