import os
import json
from SoccerNet.utils import getListGames

def main():
    test_games = [g.replace('\\', '/') for g in getListGames(split='test') if g.startswith('england_epl')]
    valid_games = [g.replace('\\', '/') for g in getListGames(split='valid') if g.startswith('england_epl')]

    print(f"Total EPL games found in 'test' split: {len(test_games)}")
    print(f"Total EPL games found in 'valid' split: {len(valid_games)}")

    eval_games = []
    
    # 1. Include all 18 EPL test games
    for g in test_games:
        parts = g.split('/')
        eval_games.append({
            "game_id": len(eval_games),
            "relative_path": g,
            "league": parts[0],
            "season": parts[1],
            "match": parts[2],
            "split": "test",
            "halves": [1, 2]
        })

    # 2. Add 2 EPL games from valid split to round out to an exact 20 games (40 halves)
    for g in valid_games[:2]:
        parts = g.split('/')
        eval_games.append({
            "game_id": len(eval_games),
            "relative_path": g,
            "league": parts[0],
            "season": parts[1],
            "match": parts[2],
            "split": "valid",
            "halves": [1, 2]
        })

    output_path = os.path.join(os.path.dirname(__file__), "..", "eval_set.json")
    output_path = os.path.normpath(output_path)

    official_17_classes = [
        "Penalty",
        "Kick-off",
        "Goal",
        "Substitution",
        "Offside",
        "Shots on target",
        "Shots off target",
        "Clearance",
        "Ball out of play",
        "Throw-in",
        "Foul",
        "Indirect free-kick",
        "Direct free-kick",
        "Corner",
        "Yellow card",
        "Red card",
        "Yellow->red card"
    ]

    metadata = {
        "description": "Frozen evaluation set for SoccerNet VLM Benchmark",
        "total_games": len(eval_games),
        "total_halves": len(eval_games) * 2,
        "splits": {
            "test_games": len(test_games),
            "valid_games": 2
        },
        "resolution": "720p",
        "container": "mp4",
        "num_classes": 17,
        "classes": official_17_classes,
        "macro_chunk_duration_sec": 600,  # 10 minutes per macro-window
        "macro_chunk_overlap_sec": 15,
        "games": eval_games
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"Successfully froze {len(eval_games)} games ({len(eval_games)*2} halves) to {output_path}")

if __name__ == "__main__":
    main()
