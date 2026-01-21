# createml-cli Examples

This directory contains example data and documentation for each model type supported by createml-cli.

## Available Examples

| Directory | Model Type | Data Format |
|-----------|-----------|-------------|
| [text-classification](text-classification/) | Text Classification | CSV with text and labels |
| [word-tagging](word-tagging/) | Word Tagging (NER, POS) | JSON with tokens and labels |
| [tabular-classification](tabular-classification/) | Tabular Classification | CSV with features and target |
| [tabular-regression](tabular-regression/) | Tabular Regression | CSV with features and numeric target |
| [recommendation](recommendation/) | Recommendation | CSV with user-item interactions |
| [object-detection](object-detection/) | Object Detection | JSON annotations + images |
| [image-classification](image-classification/) | Image Classification | Labeled subdirectories |
| [sound-classification](sound-classification/) | Sound Classification | Labeled subdirectories |

## Quick Start

### Text Classification

```bash
cd text-classification
createml text sentiment.csv -o SentimentClassifier.mlmodel
```

### Word Tagging

```bash
cd word-tagging
createml word-tag ner_data.json -o NERTagger.mlmodel
```

### Tabular Classification

```bash
cd tabular-classification
createml tabular iris.csv -o IrisClassifier.mlmodel -t species
```

### Tabular Regression

```bash
cd tabular-regression
createml tabular housing.csv -o HousePricePredictor.mlmodel -t price --type regressor
```

### Recommendation (Explicit Ratings)

```bash
cd recommendation
createml recommend movie_ratings.csv -o MovieRecommender.mlmodel --rating-column rating
```

### Recommendation (Implicit Feedback)

```bash
cd recommendation
createml recommend product_purchases.csv -o ProductRecommender.mlmodel
```

## Using Trained Models

After training, use [coreml-cli](https://github.com/schappim/coreml-cli) to inspect and run your models:

```bash
# Inspect model
coreml inspect SentimentClassifier.mlmodel

# Run prediction
coreml predict SentimentClassifier.mlmodel --input "Great product!"
```
