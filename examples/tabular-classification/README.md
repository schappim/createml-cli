# Tabular Classification Example

This example trains a classifier on the classic Iris dataset to predict flower species based on measurements.

## Data Format

The training data is a CSV file with feature columns and a target column:

```csv
sepal_length,sepal_width,petal_length,petal_width,species
5.1,3.5,1.4,0.2,setosa
7.0,3.2,4.7,1.4,versicolor
6.3,3.3,6.0,2.5,virginica
```

## Training

```bash
createml tabular iris.csv -o IrisClassifier.mlmodel -t species
```

### With Options

```bash
createml tabular iris.csv -o IrisClassifier.mlmodel \
  -t species \
  --type classifier \
  --algorithm boostedtree \
  --max-depth 10 \
  --max-iterations 100 \
  --author "Your Name"
```

## Algorithms

- `auto` (default) - Automatically select the best algorithm
- `randomforest` - Random Forest ensemble
- `boostedtree` - Gradient Boosted Trees
- `decisiontree` - Single Decision Tree
- `logistic` - Logistic Regression

## Output

```
Loading training data from iris.csv...
Found 60 training examples...
Training tabular classifier on 4 features...
Saving model to IrisClassifier.mlmodel...

Training Complete!
==================================================

Model saved to: IrisClassifier.mlmodel

Metrics:
  Training accuracy:   100.00%
  Validation accuracy: 96.67%
  Training duration:   0.35s
```
