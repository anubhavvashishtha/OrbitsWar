"""
submission_hybrid.py
=====================
Build all hybrid submission zips (every 1v1 / 4v4 model combination).

    python submission_hybrid.py

Outputs (in submissions/):
    1v1_class_4v4_class.zip  — 1v1 GBC  + 4v4 GBC
    1v1_class_4v4_reg.zip    — 1v1 GBC  + 4v4 GBR
    1v1_reg_4v4_class.zip    — 1v1 GBR  + 4v4 GBC
    1v1_reg_4v4_reg.zip      — 1v1 GBR  + 4v4 GBR

Train first:
    python classifier/train_from_replays.py        # 1v1 GBC
    python heuristic_4v4/train_4v4.py              # 4v4 GBC
    python regressor/train_from_replays.py         # 1v1 GBR
    python regressor_4v4/train_from_replays.py     # 4v4 GBR
"""
from submission_script import create_zip


def main():
    print("=========================================")
    print("  OrbitsWar -- Hybrid Submissions")
    print("=========================================")

    create_zip(
        "1v1_class_4v4_class.zip",
        [
            "main.py",
            ("classifier/value_fn_1v1.py",    "value_fn_1v1.py"),
            ("heuristic_4v4/value_fn_4v4.py", "value_fn_4v4.py"),
            ("heuristic_4v4/agent_4v4.py",    "agent_4v4.py"),
        ],
        description="1v1 GBC + 4v4 GBC"
    )

    create_zip(
        "1v1_class_4v4_reg.zip",
        [
            "main.py",
            ("classifier/value_fn_1v1.py",     "value_fn_1v1.py"),
            ("regressor_4v4/value_fn_4v4.py",  "value_fn_4v4.py"),
            ("heuristic_4v4/agent_4v4.py",     "agent_4v4.py"),
        ],
        description="1v1 GBC + 4v4 GBR"
    )

    create_zip(
        "1v1_reg_4v4_class.zip",
        [
            "main.py",
            ("regressor/value_fn_1v1.py",     "value_fn_1v1.py"),
            ("heuristic_4v4/value_fn_4v4.py", "value_fn_4v4.py"),
            ("heuristic_4v4/agent_4v4.py",    "agent_4v4.py"),
        ],
        description="1v1 GBR + 4v4 GBC"
    )

    create_zip(
        "1v1_reg_4v4_reg.zip",
        [
            "main.py",
            ("regressor/value_fn_1v1.py",    "value_fn_1v1.py"),
            ("regressor_4v4/value_fn_4v4.py","value_fn_4v4.py"),
            ("heuristic_4v4/agent_4v4.py",   "agent_4v4.py"),
        ],
        description="1v1 GBR + 4v4 GBR"
    )


if __name__ == "__main__":
    main()
