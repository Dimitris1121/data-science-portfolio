import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity

# ---- SIMULATE GAMING DATA ----
np.random.seed(42)

# Game names - realistic for lottery/casual gaming company
games = [
    'Mega Millions', 'Powerball', 'Scratch & Win', 'Lucky 7',
    'Bingo Blast', 'Poker Stars', 'Slot Frenzy', 'Keno Classic',
    'Sports Bet', 'Horse Racing', 'Blackjack Pro', 'Roulette Royal',
    'Daily Draw', 'Instant Win', 'Cash Splash', 'Fortune Wheel',
    'Gold Rush', 'Diamond Strike', 'Lucky Clover', 'Jackpot City'
]

# Generate player IDs
players = [f'P{i:04d}' for i in range(1, 501)]

# Simulate player-game interactions
records = []
for player in players:
    n_games_played = np.random.randint(3, 9)
    games_played = np.random.choice(games, n_games_played, replace=False)

    for game in games_played:
        rating = np.random.choice([1, 2, 3, 4, 5],
                                   p=[0.05, 0.10, 0.20, 0.35, 0.30])
        records.append({
            'player_id': player,
            'game': game,
            'rating': rating
        })

df = pd.DataFrame(records)

print("Dataset shape:", df.shape)
print("\nFirst 10 rows:")
print(df.head(10))
print("\nRating distribution:")
print(df['rating'].value_counts().sort_index())
print("\nAverage rating per game:")
print(df.groupby('game')['rating'].mean().sort_values(ascending=False).round(2))


# ---- BUILD USER-ITEM MATRIX ----
# Pivot table - players as rows, games as columns, ratings as values
user_item_matrix = df.pivot_table(
    index='player_id',
    columns='game',
    values='rating',
    fill_value=0
)

print("User-Item Matrix Shape:", user_item_matrix.shape)
print("\nSample of matrix:")
print(user_item_matrix.iloc[:5, :5])


# ---- CALCULATE PLAYER SIMILARITY ----
# Calculate cosine similarity between all players
player_similarity = cosine_similarity(user_item_matrix)
player_similarity_df = pd.DataFrame(
    player_similarity,
    index=user_item_matrix.index,
    columns=user_item_matrix.index
)

print("Player similarity matrix shape:", player_similarity_df.shape)
print("\nSample similarities for P0001:")
print(player_similarity_df['P0001'].sort_values(ascending=False).head(6))

similar_players = player_similarity_df['P0001'].sort_values(ascending=False)[1:11]  # top 10 similar players (skip self)
print(similar_players.head(15))


# ---- RECOMMENDATION FUNCTION ----
def recommend_games(player_id, n_recommendations=5):

    # Step 1 - Find most similar players
    similar_players = player_similarity_df[player_id]\
        .sort_values(ascending=False)[1:11]  # top 10 similar players (skip self)

    # Step 2 - Get games current player already played
    played_games = set(
        df[df['player_id'] == player_id]['game'].values
    )

    # Step 3 - Get games similar players loved (rating >= 4)
    recommendations = {}

    for similar_player, similarity_score in similar_players.items():
        # Get highly rated games from this similar player
        similar_player_games = df[
            (df['player_id'] == similar_player) &
            (df['rating'] >= 4)
        ]

        for _, row in similar_player_games.iterrows():
            game = row['game']
            rating = row['rating']

            # Only recommend games the player hasn't played
            if game not in played_games:
                if game not in recommendations:
                    recommendations[game] = 0
                # Weight by similarity score
                recommendations[game] += similarity_score * rating

    # Step 4 - Sort by score and return top N
    recommendations = pd.Series(recommendations)\
        .sort_values(ascending=False)\
        .head(n_recommendations)

    return recommendations


# ---- TEST THE RECOMMENDER ----
test_player = 'P0001'

print(f"Games {test_player} has played:")
played = df[df['player_id'] == test_player][['game', 'rating']]
print(played.to_string(index=False))

print(f"\nTop 5 recommendations for {test_player}:")
recommendations = recommend_games(test_player)
print(recommendations.to_string())


# ---- VISUALIZE RECOMMENDATIONS ----
plt.figure(figsize=(10, 6))
recommendations.plot(kind='barh', color='steelblue')
plt.title(f'Top 5 Game Recommendations for {test_player}')
plt.xlabel('Recommendation Score')
plt.ylabel('Game')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

# ---- TEST WITH MULTIPLE PLAYERS ----
print("\n---- RECOMMENDATIONS FOR MULTIPLE PLAYERS ----")
for player in ['P0001', 'P0050', 'P0100', 'P0200']:
    print(f"\n{player}:")
    print(recommend_games(player, 3).to_string())
