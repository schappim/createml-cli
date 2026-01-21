# createml-cli

A native command-line interface for training Core ML models on macOS using Apple's Create ML framework. Train image classifiers, text classifiers, sound classifiers, and tabular models without Xcode or Python.

## Features

- **Image Classification** - Train models from labeled image directories
- **Text Classification** - Train sentiment analysis, spam detection, etc.
- **Sound Classification** - Train audio classification models
- **Tabular Classification/Regression** - Train models on CSV/JSON data

## Installation

### Homebrew (Recommended)

```bash
brew tap schappim/createml-cli
brew install createml-cli
```

### Download Binary

Download the latest release from [GitHub Releases](https://github.com/schappim/createml-cli/releases):

```bash
curl -L https://github.com/schappim/createml-cli/releases/download/v1.0.0/createml-1.0.0-macos.tar.gz -o createml.tar.gz
tar -xzf createml.tar.gz
sudo mv createml /usr/local/bin/
```

### Build from Source

Requires macOS 14+ and Swift 5.9+

```bash
git clone https://github.com/schappim/createml-cli.git
cd createml-cli
swift build -c release
sudo cp .build/release/createml /usr/local/bin/
```

## Usage

### Train an Image Classifier

Organize your training images into subdirectories by class:

```
training_data/
├── cats/
│   ├── cat1.jpg
│   ├── cat2.jpg
│   └── ...
├── dogs/
│   ├── dog1.jpg
│   ├── dog2.jpg
│   └── ...
└── birds/
    ├── bird1.jpg
    └── ...
```

Train the model:

```bash
createml image training_data/ -o PetClassifier.mlmodel
```

Output:
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

Options:
```bash
createml image training_data/ -o Model.mlmodel \
  --iterations 50 \
  --validation validation_data/ \
  --author "Your Name" \
  --description "Pet classifier model" \
  --no-augmentation \
  --json
```

### Train a Text Classifier

Create a CSV file with `text` and `label` columns:

```csv
text,label
"I love this product!",positive
"Terrible experience",negative
"It's okay",neutral
```

Train the model:

```bash
createml text sentiment.csv -o SentimentClassifier.mlmodel
```

Output:
```
Loading training data from sentiment.csv...
Found 1000 training examples...
Training text classifier using Maximum Entropy...
Saving model to SentimentClassifier.mlmodel...

Training Complete!
==================================================

Model saved to: SentimentClassifier.mlmodel

Classes: negative, neutral, positive

Metrics:
  Training accuracy:   95.40%
  Training duration:   2.15s
```

Options:
```bash
createml text data.csv -o Model.mlmodel \
  --text-column "review" \
  --label-column "sentiment" \
  --algorithm transfer \
  --json
```

### Train a Sound Classifier

Organize audio files into subdirectories by class:

```
sounds/
├── dog_bark/
│   ├── bark1.wav
│   └── bark2.wav
├── cat_meow/
│   ├── meow1.wav
│   └── meow2.wav
└── bird_chirp/
    └── chirp1.wav
```

Train:

```bash
createml sound sounds/ -o SoundClassifier.mlmodel
```

### Train a Tabular Classifier

Create a CSV with feature columns and a target column:

```csv
sepal_length,sepal_width,petal_length,petal_width,species
5.1,3.5,1.4,0.2,setosa
7.0,3.2,4.7,1.4,versicolor
6.3,3.3,6.0,2.5,virginica
```

Train a classifier:

```bash
createml tabular iris.csv -o IrisClassifier.mlmodel -t species
```

Output:
```
Loading training data from iris.csv...
Found 150 training examples...
Training tabular classifier on 4 features...
Saving model to IrisClassifier.mlmodel...

Training Complete!
==================================================

Model saved to: IrisClassifier.mlmodel

Metrics:
  Training accuracy:   100.00%
  Validation accuracy: 97.33%
  Training duration:   0.45s
```

Train a regressor for numeric targets:

```bash
createml tabular housing.csv -o PricePredictor.mlmodel -t price --type regressor
```

Options:
```bash
createml tabular data.csv -o Model.mlmodel \
  -t target_column \
  --type classifier \
  --algorithm boostedtree \
  --max-depth 10 \
  --max-iterations 100 \
  --json
```

## Command Reference

| Command | Description |
|---------|-------------|
| `createml image <dir> -o <output>` | Train image classifier |
| `createml text <csv> -o <output>` | Train text classifier |
| `createml sound <dir> -o <output>` | Train sound classifier |
| `createml tabular <csv> -o <output> -t <target>` | Train tabular model |

### Algorithms

**Text Classification:**
- `maxent` (default) - Maximum Entropy classifier, fast and lightweight
- `transfer` - Transfer learning with word embeddings, more accurate

**Tabular:**
- `auto` (default) - Automatically select best algorithm
- `randomforest` - Random Forest ensemble
- `boostedtree` - Gradient Boosted Trees
- `decisiontree` - Single Decision Tree
- `linear` - Linear Regression (regressor only)
- `logistic` - Logistic Regression (classifier only)

## JSON Output

Add `--json` to any command for machine-readable output:

```bash
createml text data.csv -o Model.mlmodel --json
```

```json
{
  "modelPath": "/path/to/Model.mlmodel",
  "trainingAccuracy": 95.4,
  "trainingDurationSeconds": 2.15,
  "classLabels": ["negative", "neutral", "positive"]
}
```

## Using Trained Models

Use your trained models with the [coreml-cli](https://github.com/schappim/coreml-cli) tool:

```bash
# Inspect the model
coreml inspect SentimentClassifier.mlmodel

# Run inference
coreml predict SentimentClassifier.mlmodel --input "I love this!"

# Benchmark performance
coreml benchmark SentimentClassifier.mlmodel --input sample.txt
```

## Requirements

- macOS 14.0 or later
- Apple Silicon or Intel Mac
- Training data in supported formats

## License

MIT License - see [LICENSE](LICENSE) for details.

## Contributing

Contributions welcome! Please open an issue or submit a pull request.

## See Also

- [coreml-cli](https://github.com/schappim/coreml-cli) - CLI for Core ML inference and model inspection
- [Apple Create ML Documentation](https://developer.apple.com/documentation/createml)
