import math
import glob
import re
import csv
from collections import defaultdict
from pathlib import Path

RECORDS_DIR = Path("records")

# Standard Elo calculation
def calculate_elo(rating_a, rating_b, score_a, k=32):
    expected_a = 1 / (1 + 10 ** ((rating_b - rating_a) / 400))
    new_rating_a = rating_a + k * (score_a - expected_a)
    return new_rating_a

def analyze():
    replays = glob.glob(str(RECORDS_DIR / "*.json"))
    print(f"Found {len(replays)} replays to analyze. Calculating local Elo...", flush=True)

    # Data structures
    stats_1v1 = defaultdict(lambda: {"elo": 1000.0, "matches": 0, "wins": 0})
    stats_4v4 = defaultdict(lambda: {"elo": 1000.0, "matches": 0, "wins": 0})

    team_regex = re.compile(r'"TeamNames"\s*:\s*\[(.*?)\]')
    reward_regex = re.compile(r'"rewards"\s*:\s*\[(.*?)\]')

    for i, path in enumerate(replays):
        if (i + 1) % 1000 == 0:
            print(f"Processed {i + 1} / {len(replays)} files...", flush=True)
            
        try:
            with open(path, 'r', encoding='utf-8') as f:
                content = f.read()
        except Exception:
            continue
            
        team_match = team_regex.search(content)
        reward_match = reward_regex.search(content)
        
        if not team_match or not reward_match:
            continue
            
        teams_str = team_match.group(1)
        rewards_str = reward_match.group(1)
        
        # Parse names: split by comma, strip spaces and quotes
        team_names = [t.strip().strip('"').strip("'") for t in teams_str.split(',')]
        
        # Parse rewards
        rewards = []
        for r in rewards_str.split(','):
            r = r.strip()
            if r == 'null':
                rewards.append(None)
            else:
                try:
                    rewards.append(float(r))
                except ValueError:
                    rewards.append(None)
        
        if not team_names or not rewards or None in rewards:
            continue

        num_players = len(team_names)
        max_r = max(rewards)
        
        # 1v1 Match
        if num_players == 2:
            name_a, name_b = team_names[0], team_names[1]
            r_a, r_b = rewards[0], rewards[1]
            
            if r_a == r_b:
                continue
                
            score_a = 1 if r_a > r_b else 0
            score_b = 1 if r_b > r_a else 0
            
            elo_a = stats_1v1[name_a]["elo"]
            elo_b = stats_1v1[name_b]["elo"]
            
            stats_1v1[name_a]["elo"] = calculate_elo(elo_a, elo_b, score_a)
            stats_1v1[name_b]["elo"] = calculate_elo(elo_b, elo_a, score_b)
            
            stats_1v1[name_a]["matches"] += 1
            stats_1v1[name_b]["matches"] += 1
            if score_a == 1: stats_1v1[name_a]["wins"] += 1
            if score_b == 1: stats_1v1[name_b]["wins"] += 1

        # 4v4 Match
        elif num_players == 4:
            for idx in range(4):
                stats_4v4[team_names[idx]]["matches"] += 1
                if rewards[idx] == max_r and rewards.count(max_r) == 1:
                    stats_4v4[team_names[idx]]["wins"] += 1
                    
            for idx in range(4):
                for j in range(idx + 1, 4):
                    name_a, name_b = team_names[idx], team_names[j]
                    r_a, r_b = rewards[idx], rewards[j]
                    
                    if r_a == r_b:
                        continue
                        
                    score_a = 1 if r_a > r_b else 0
                    score_b = 1 if r_b > r_a else 0
                    
                    elo_a = stats_4v4[name_a]["elo"]
                    elo_b = stats_4v4[name_b]["elo"]
                    
                    stats_4v4[name_a]["elo"] = calculate_elo(elo_a, elo_b, score_a, k=8)
                    stats_4v4[name_b]["elo"] = calculate_elo(elo_b, elo_a, score_b, k=8)

    # Export to CSV
    def export_csv(stats, filename):
        if not stats:
            return
        
        sorted_stats = sorted(stats.items(), key=lambda x: x[1]['elo'], reverse=True)
        
        with open(filename, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['Rank', 'Bot_Name', 'elo', 'matches', 'wins', 'Win_Rate'])
            
            for rank, (name, data) in enumerate(sorted_stats, 1):
                win_rate = f"{(data['wins'] / data['matches'] * 100):.2f}%" if data['matches'] > 0 else "0.00%"
                writer.writerow([rank, name, round(data['elo'], 1), data['matches'], data['wins'], win_rate])
                
        print(f"Saved {filename}", flush=True)

    export_csv(stats_1v1, "leaderboard_1v1.csv")
    export_csv(stats_4v4, "leaderboard_4v4.csv")
    print("Analysis complete!", flush=True)

if __name__ == "__main__":
    analyze()
