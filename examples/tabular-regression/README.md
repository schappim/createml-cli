# Tabular Regression Example

This example trains a regression model to predict house prices based on property features.

## Data Format

The training data is a CSV file with feature columns and a numeric target column:

```csv
bedrooms,bathrooms,sqft,lot_size,year_built,garage,price
3,2,1500,5000,1990,2,350000
4,3,2200,7500,2005,2,485000
```

## Training

```bash
createml tabular housing.csv -o HousePricePredictor.mlmodel -t price --type regressor
```

### With Options

```bash
createml tabular housing.csv -o HousePricePredictor.mlmodel \
  -t price \
  --type regressor \
  --algorithm boostedtree \
  --max-depth 8 \
  --max-iterations 200 \
  --author "Your Name"
```

## Algorithms

- `auto` (default) - Automatically select the best algorithm
- `randomforest` - Random Forest ensemble
- `boostedtree` - Gradient Boosted Trees
- `decisiontree` - Single Decision Tree
- `linear` - Linear Regression

## Output

```
Loading training data from housing.csv...
Found 30 training examples...
Training tabular regressor on 6 features...
Saving model to HousePricePredictor.mlmodel...

Training Complete!
==================================================

Model saved to: HousePricePredictor.mlmodel

Metrics:
  Training RMSE:       15234.56
  Validation RMSE:     28456.78
  Training duration:   0.42s
```

## Metrics

- **RMSE (Root Mean Squared Error)** - Average prediction error in the same units as the target (dollars in this case)
