import json

with open('train_transformer_ppo.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'code' and any('CELL 6' in line for line in cell['source']):
        cell['source'] = [
            "# ---------------------------------------------------------\n",
            "# CELL 6 — Session configuration\n",
            "# <- CHANGE THESE each session\n",
            "# ---------------------------------------------------------\n",
            "import rl.transformer_trainer as TT\n",
            "\n",
            "N_ROUNDS            = 999      # run until Kaggle 12h limit\n",
            "START_TIER          = 1        # 0=easy  1=medium  2=hard\n",
            "START_TIER_PURE     = True     # True=only that tier  False=cumulative mix\n",
            "RESET_ENT_ON_RESUME = False    # False: Keep adaptive alpha from checkpoint\n",
            "\n",
            "_tier_names = [\"Tier1 only (easy)\", \"Tier2 only (medium)\", \"Tier3 only (hard)\"]\n",
            "_mode       = \"pure\" if START_TIER_PURE else \"cumulative\"\n",
            "print(f\"N_ROUNDS            : {N_ROUNDS}\")\n",
            "print(f\"START_TIER          : {START_TIER}  ({_tier_names[START_TIER]})  mode={_mode}\")\n",
            "print(f\"RESET_ENT_ON_RESUME : {RESET_ENT_ON_RESUME}\")\n",
            "print()\n",
            "print(\"Hyperparameters (from transformer_trainer.py):\")\n",
            "print(f\"  LR_ACTOR   : {TT.LR_ACTOR}\")\n",
            "print(f\"  LR_CRITIC  : {TT.LR_CRITIC}\")\n",
            "print(f\"  CLIP_EPS   : {TT.CLIP_EPS}\")\n",
            "print(f\"  PPO_EPOCHS : {TT.PPO_EPOCHS}\")\n",
            "print(f\"  TARGET_KL  : {TT.TARGET_KL}\")\n",
            "print(f\"  ENTROPY    : Target={TT.TARGET_ENTROPY} (Adaptive SAC-style)\")\n",
            "print(f\"  TERMINAL   : win={TT.TERMINAL_WIN}  loss={TT.TERMINAL_LOSS}  break-even={abs(TT.TERMINAL_LOSS)/(TT.TERMINAL_WIN+abs(TT.TERMINAL_LOSS)):.1%}\")\n",
            "print(f\"  EVENT_SCALE: {TT._E}\")\n",
            "print(f\"  WARMUP     : {TT.CRITIC_WARMUP_ROUNDS} rounds (skipped on resume)\")\n",
            "print()\n",
            "print(\"Curriculum tiers:\")\n",
            "for i, tier in enumerate(TT.CURRICULUM_TIERS_1V1):\n",
            "    print(f\"  Tier {i+1}: {tier}\")\n"
        ]

with open('train_transformer_ppo.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)
