# Word Tagging Example

This example trains a Named Entity Recognition (NER) model to identify persons, organizations, and locations in text.

## Data Format

The training data is a JSON file with `tokens` and `labels` arrays. Each token must have a corresponding label.

```json
[
  {
    "tokens": ["Apple", "Inc.", "is", "in", "Cupertino", "."],
    "labels": ["ORG", "ORG", "O", "O", "LOC", "O"]
  }
]
```

### Common NER Labels

- `PERSON` - Names of people
- `ORG` - Organizations, companies
- `LOC` - Locations, places
- `O` - Other (not an entity)

## Training

```bash
createml word-tag ner_data.json -o NERTagger.mlmodel
```

### With Options

```bash
createml word-tag ner_data.json -o NERTagger.mlmodel \
  --token-column "tokens" \
  --label-column "labels" \
  --author "Your Name" \
  --description "Named Entity Recognition model"
```

## Output

```
Loading training data from ner_data.json...
Found 20 training examples...
Training word tagger...
Saving model to NERTagger.mlmodel...

Training Complete!
==================================================

Model saved to: NERTagger.mlmodel

Tags: LOC, O, ORG, PERSON

Metrics:
  Training accuracy:   98.50%
  Training duration:   1.25s
```

## Use Cases

- **Named Entity Recognition (NER)** - Identify people, places, organizations
- **Part-of-Speech Tagging** - Tag words as nouns, verbs, adjectives, etc.
- **Custom Token Classification** - Any token-level labeling task
