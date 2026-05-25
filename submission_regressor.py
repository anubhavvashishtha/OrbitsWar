"""
submission_regressor.py
========================
Build regressor-only submission zips.

    python submission_regressor.py

Outputs (in submissions/):
    1v1_reg.zip   — main.py + 1v1 GBR weights
    4v4_reg.zip   — main.py + 4v4 GBR weights + quadrant agent

Train first:
    python regressor/train_from_replays.py         # 1v1 weights
    python regressor_4v4/train_from_replays.py     # 4v4 weights

Fine-tune vs bots (optional but recommended):
    python regressor/train_vs_bots.py
    python regressor_4v4/train_vs_bots.py
"""
from submission_script import create_zip


def main():
    print("=========================================")
    print("  OrbitsWar -- Regressor Submissions")
    print("=========================================")

    create_zip(
        "1v1_reg.zip",
        [
            "main.py",
            ("regressor/value_fn_1v1.py", "value_fn_1v1.py"),
        ],
        description="1v1 GBR  |  train: regressor/train_from_replays.py"
    )

    create_zip(
        "4v4_reg.zip",
        [
            "main.py",
            ("regressor_4v4/value_fn_4v4.py", "value_fn_4v4.py"),
            ("heuristic_4v4/agent_4v4.py",    "agent_4v4.py"),
        ],
        description="4v4 GBR  |  train: regressor_4v4/train_from_replays.py"
    )


if __name__ == "__main__":
    main()
