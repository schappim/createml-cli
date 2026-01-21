# Text Classification Example

This example trains a sentiment analysis model to classify text as positive, negative, or neutral.

## Data Format

The training data is a CSV file with `text` and `label` columns:

```csv
text,label
"I love this product!",positive
"Terrible experience",negative
"It's okay",neutral
```

## Training

```bash
createml text sentiment.csv -o SentimentClassifier.mlmodel
```

### With Options

```bash
createml text sentiment.csv -o SentimentClassifier.mlmodel \
  --algorithm transfer \
  --author "Your Name" \
  --description "Sentiment analysis model"
```

## Algorithms

- `maxent` (default) - Maximum Entropy classifier, fast and lightweight
- `transfer` - Transfer learning with word embeddings, more accurate but slower

## Output

```
Loading training data from sentiment.csv...
Found 30 training examples...
Training text classifier using Maximum Entropy...
Saving model to SentimentClassifier.mlmodel...

Training Complete!
==================================================

Model saved to: SentimentClassifier.mlmodel

Classes: negative, neutral, positive

Metrics:
  Training accuracy:   100.00%
  Training duration:   0.12s
```
