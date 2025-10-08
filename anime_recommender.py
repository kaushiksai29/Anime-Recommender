# -*- coding: utf-8 -*-
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import os

def main():
    anime_path = 'data/anime.csv'
    rating_path = 'data/rating.csv'

    if not os.path.exists(anime_path) or not os.path.exists(rating_path):
        print("Error: Dataset files not found.")
        print("Please download from: https://www.kaggle.com/datasets/CooperUnion/anime-recommendations-database")
        return

    anime_df = pd.read_csv(anime_path)
    rating_df = pd.read_csv(rating_path)

    df = pd.merge(rating_df, anime_df.drop('rating', axis=1), on='anime_id')
    df = df[df['rating'] != -1]

    # For performance on a local machine, let's filter for more popular anime and active users
    anime_counts = df['name'].value_counts()
    user_counts = df['user_id'].value_counts()

    popular_anime = anime_counts[anime_counts >= 1000].index
    active_users = user_counts[user_counts >= 100].index

    filtered_df = df[df['name'].isin(popular_anime) & df['user_id'].isin(active_users)]

    print("Creating user-item matrix on filtered data...")
    user_item_matrix = filtered_df.pivot_table(index='name', columns='user_id', values='rating').fillna(0)

    print("Calculating item-item similarity matrix...")
    item_similarity = cosine_similarity(user_item_matrix)
    item_similarity_df = pd.DataFrame(item_similarity, index=user_item_matrix.index, columns=user_item_matrix.index)
    print("Similarity matrix calculation complete.")

    def get_recommendations(anime_name, num_recs=5):
        if anime_name not in item_similarity_df:
            print(f"\nSorry, '{anime_name}' is not in our filtered dataset. Please try a more popular title.")
            return

        print(f"\nRecommendations for '{anime_name}':")
        similar_scores = item_similarity_df[anime_name].sort_values(ascending=False)
        recommendations = similar_scores[1:num_recs+1]

        for name, score in recommendations.items():
            print(f"- {name} (Similarity: {score:.2f})")

    print("\n--- Anime Recommendation System ---")
    while True:
        anime_input = input("Enter a popular anime name (e.g., 'Death Note', 'Naruto') or 'quit' to exit: ")
        if anime_input.lower() == 'quit':
            break
        get_recommendations(anime_input)

if __name__ == '__main__':
    main()
