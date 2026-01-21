# Recommendation Example

This example trains collaborative filtering recommendation models for personalized recommendations.

## Data Formats

### Explicit Ratings

Use when you have user ratings (e.g., 1-5 stars):

```csv
user,item,rating
user_1,The Matrix,5
user_1,Inception,4
user_2,The Matrix,3
```

### Implicit Feedback

Use when you only have interaction data (purchases, views, clicks):

```csv
user,item
customer_1,laptop
customer_1,mouse
customer_2,laptop
```

## Training

### With Explicit Ratings

```bash
createml recommend movie_ratings.csv -o MovieRecommender.mlmodel --rating-column rating
```

### With Implicit Feedback

```bash
createml recommend product_purchases.csv -o ProductRecommender.mlmodel
```

### With Custom Column Names

```bash
createml recommend data.csv -o Recommender.mlmodel \
  --user-column "customer_id" \
  --item-column "product_id" \
  --rating-column "stars" \
  --author "Your Name"
```

## Output

```
Loading training data from movie_ratings.csv...
Found 50 interactions...
Training recommender model...
Saving model to MovieRecommender.mlmodel...

Training Complete!
==================================================

Model saved to: MovieRecommender.mlmodel

Metrics:
  Training duration:   0.85s
```

## How It Works

The model uses **collaborative filtering** to find patterns in user-item interactions:

1. **User-based** - Find users with similar preferences and recommend what they liked
2. **Item-based** - Find items similar to what the user has interacted with

## Use Cases

- **Movie/Music Recommendations** - Suggest content based on viewing/listening history
- **E-commerce** - Recommend products based on purchase history
- **Content Platforms** - Personalized article or video recommendations
