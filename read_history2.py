import json
with open('rl/transformer_history.json') as f:
    data = json.load(f)
    print(f"Total Rounds: {len(data)}")
    for d in data[-10:]:
        print(f"Round {d['round']}: WR={d.get('win_rate',0):.2%} Ent={d.get('entropy',0):.4f} Alpha={d.get('alpha',0):.4f} Evt={d.get('event_reward',0):.2f}")
