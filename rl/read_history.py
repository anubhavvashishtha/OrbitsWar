import json, math

h = json.loads(open('rl/transformer_history.json').read())

# --- What happens if we reset ENT_COEF ramp at round 206 ---
# ENT_COEF_START=0.05, END=0.04, RAMP=100
# ent_ramp_start = 206 (start_rnd on resume)
ENT_COEF_START = 0.05
ENT_COEF_END_CURR = 0.04
ENT_COEF_RAMP = 100

def ent_coef(rnd, start):
    t = min(1.0, max(0.0, (rnd - start) / max(1, ENT_COEF_RAMP)))
    return ENT_COEF_START + (ENT_COEF_END_CURR - ENT_COEF_START) * t

print("=== ENT_COEF schedule (current settings, RAMP=100) ===")
print(f"{'Abs Rnd':>8} {'Session Rnd':>11} {'ENT_COEF':>10}")
for s_rnd in [1, 5, 10, 20, 30, 50, 100, 142]:
    abs_rnd = 206 + s_rnd - 1
    print(f"{abs_rnd:>8} {s_rnd:>11} {ent_coef(abs_rnd, 206):>10.4f}")

print()

# --- What entropy do those ENT_COEFs produce, from historical data ---
# Empirically: ENT_COEF=0.013 → entropy~0.46
# Use linear regression from data
ecs = [x['ent_coef'] for x in h if x['ent_coef'] > 0]
ens = [x['entropy']  for x in h if x['ent_coef'] > 0]
# Simple correlation
n = len(ecs)
mean_ec = sum(ecs)/n; mean_en = sum(ens)/n
cov  = sum((ec-mean_ec)*(en-mean_en) for ec,en in zip(ecs,ens))
var  = sum((ec-mean_ec)**2 for ec in ecs)
slope = cov/var; intercept = mean_en - slope*mean_ec

print(f"Historical fit: entropy ~= {intercept:.4f} + {slope:.4f} * ENT_COEF")

print()

print("=== Predicted entropy at each ENT_COEF value ===")
for ec_val in [0.05, 0.04, 0.03, 0.02, 0.015, 0.013, 0.010]:
    pred_ent = intercept + slope * ec_val
    print(f"  ENT_COEF={ec_val:.3f}  →  predicted entropy ≈ {pred_ent:.4f}")

print()
print("=== Target: entropy 0.40-0.45 → ENT_COEF needed ===")
for target_ent in [0.40, 0.42, 0.44, 0.45]:
    needed_ec = (target_ent - intercept) / slope
    print(f"  entropy={target_ent:.2f}  →  ENT_COEF ≈ {needed_ec:.4f}")

print()
# --- Reward hacking analysis ---
print("=== Reward balance analysis ===")
# Per game avg events from last 20 rounds (on losses)
avg_events_per_game = sum(h2['events'] for h2 in h[-20:]) / (32*20)
print(f"Avg event reward per game (last 20 rounds): {avg_events_per_game:.3f}")

print("\nScenario: 1 WIN + 31 LOSSES per round, avg events=1.5/game")
for loss_scale in [1.0, 0.4]:
    win_term = 9.0; loss_term = -6.0
    terminal = 31*loss_term + 1*win_term
    events_w = 1.5 * 1           # 1 win
    events_l = 1.5 * 31 * loss_scale  # 31 losses scaled
    total = terminal + events_w + events_l
    print(f"  loss_scale={loss_scale}: terminal={terminal:.1f}  events={events_w+events_l:.1f}  total={total:.1f}")

print("\nScenario: 0 WINS + 32 LOSSES per round (100% loss)")
for loss_scale in [1.0, 0.4]:
    win_term = 9.0; loss_term = -6.0
    terminal = 32*loss_term
    events_l = 1.5 * 32 * loss_scale
    total = terminal + events_l
    print(f"  loss_scale={loss_scale}: terminal={terminal:.1f}  events={events_l:.1f}  total={total:.1f}")

print("\nMarginal value of 1 WIN vs 1 LOSS (with loss_scale=0.4):")
win_reward  = 9.0 + 1.5                # terminal + full events
loss_reward = -6.0 + 1.5*0.4          # terminal + scaled events
print(f"  1 win  reward = {win_reward:.2f}")
print(f"  1 loss reward = {loss_reward:.2f}")
print(f"  Delta         = {win_reward - loss_reward:.2f}  ← clear incentive to win")
