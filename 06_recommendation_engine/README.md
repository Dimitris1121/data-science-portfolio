# 06 — Game Recommendation Engine

## Overview
Collaborative filtering recommendation engine suggesting games to players
based on similar players' preferences. Built to simulate a real-world
iGaming recommendation system similar to what Allwyn uses to personalize
player experience.

## Problem
With 20+ games in a catalog, how do you show each player the games
they're most likely to enjoy? A recommendation engine solves this by
learning from the behavior of similar players.

## Dataset
- **Type:** Synthetic — simulated to mimic real iGaming player data
- **Players:** 500
- **Games:** 20 (lottery, casino, sports betting, instant win)
- **Interactions:** ~2,500 player-game ratings
- **Ratings:** 1-5 stars with realistic positive bias

## Approach

### Collaborative Filtering
*"Players like you also played..."*

Recommends games based on what similar players enjoyed.
Does not require game content data — only player behavior.

### How It Works
1. Build user-item matrix (players × games × ratings)
2. Calculate cosine similarity between all players
3. Find 10 most similar players for target player
4. Collect games similar players rated 4+ stars
5. Score each game: similarity_score × rating
6. Return top N highest scoring unplayed games

### Cosine Similarity
Measures the angle between two players' rating vectors:
- Score = 1.0 → identical taste
- Score = 0.0 → completely different taste

Games loved by highly similar players receive higher scores
than games loved by barely similar players.

## Results

### Sample Recommendations

| Player | Top Recommendation | Score |
|---|---|---|
| P0001 | Diamond Strike | 5.83 |
| P0050 | Keno Classic | 3.91 |
| P0100 | Daily Draw | 6.12 |
| P0200 | Scratch & Win | 3.40 |

Every player receives unique personalized recommendations
based on their individual play history.

## Key Features
- Fully personalized — different recommendations per player
- Never recommends already played games
- Weighted scoring — similar players have more influence
- Configurable — adjust number of recommendations easily
- No external ML libraries — built with pandas and numpy only

## Business Application (Allwyn Context)
- Show personalized game suggestions on player homepage
- Email campaigns — *"Games you might love"*
- Post-session recommendations — *"Players like you also play..."*
- New game launches — target players most likely to enjoy it
- Player retention — keep players engaged with relevant content

## Tools
Python, pandas, numpy, scikit-learn, matplotlib

## How to Run
```bash
pip install -r requirements.txt
python recommendation_engine.py
```

## Files
- `recommendation_engine.py` — full recommendation engine
