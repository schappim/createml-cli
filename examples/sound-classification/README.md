# Sound Classification Example

Train sound classifiers by organizing audio files into labeled subdirectories.

## Directory Structure

```
sounds/
├── dog_bark/
│   ├── bark_001.wav
│   ├── bark_002.wav
│   └── ...
├── cat_meow/
│   ├── meow_001.wav
│   ├── meow_002.wav
│   └── ...
└── bird_chirp/
    ├── chirp_001.wav
    └── ...
```

Each subdirectory name becomes a class label.

## Training

```bash
createml sound sounds/ -o SoundClassifier.mlmodel
```

### With Options

```bash
createml sound sounds/ -o SoundClassifier.mlmodel \
  --validation validation_sounds/ \
  --author "Your Name" \
  --description "Animal sound classifier"
```

## Output

```
Loading training data from sounds/...
Configuring training parameters...
Training sound classifier...
Saving model to SoundClassifier.mlmodel...

Training Complete!
==================================================

Model saved to: SoundClassifier.mlmodel

Classes: bird_chirp, cat_meow, dog_bark

Metrics:
  Training accuracy:   96.50%
  Validation accuracy: 92.30%
  Training duration:   28.45s
```

## Supported Formats

- WAV (.wav)
- AIFF (.aiff, .aif)
- CAF (.caf)
- MP3 (.mp3)
- M4A (.m4a)

## Tips

1. **Duration** - Audio clips should be at least 1-3 seconds long
2. **Quality** - Use consistent sample rates (44.1kHz recommended)
3. **Clarity** - Ensure the target sound is clearly audible
4. **Variety** - Include recordings from different sources/environments
5. **Balance** - Include similar numbers of clips per class
6. **Minimum** - Aim for at least 10-20 clips per class
