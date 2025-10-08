# Anime Recommender System
This project implements an item-based collaborative filtering recommendation system for anime. It uses a dataset of user ratings to find and recommend anime that are similar to a user's input, demonstrating a core technique in building recommendation engines.

# Features
- Collaborative Filtering: Implements item-based collaborative filtering using Cosine Similarity to measure similarity between anime based on user ratings.

- Data Processing: Uses pandas to load, merge, and process user ratings and anime information into a user-item matrix.

- Recommendation Engine: Provides a command-line interface to get recommendations for a given anime title.

# Dataset
This project uses the "Anime Recommendations Database" from Kaggle.

Please download the dataset from the link below and place anime.csv and rating.csv in a data/ directory:
https://www.kaggle.com/datasets/CooperUnion/anime-recommendations-database

# How to Run
1. Clone the repository and cd into it.

2. Create a data directory and place the dataset CSVs inside.

3. Install dependencies: pip install -r requirements.txt

4. Run the recommender: python recommender.py

5. Get recommendations: Enter a popular anime name like Death Note or Naruto when prompted. Type quit to exit.
