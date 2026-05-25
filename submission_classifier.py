"""
submission_classifier.py
========================
Build classifier-only submission zips.

    python submission_classifier.py

Outputs (in submissions/):
    1v1_class.zip   — main.py + 1v1 GBC weights
    4v4_class.zip   — main.py + 4v4 GBC weights + quadrant agent

Train first:
    python classifier/train_from_replays.py       # 1v1 weights
    python heuristic_4v4/train_4v4.py             # 4v4 weights
"""
from submission_script import create_zip


def main():
    print("=========================================")
    print("  OrbitsWar -- Classifier Submissions")
    print("=========================================")

    create_zip(
        "1v1_class.zip",
        [
            "main.py",
            ("classifier/value_fn_1v1.py", "value_fn_1v1.py"),
        ],
        description="1v1 GBC  |  train: classifier/train_from_replays.py"
    )

    create_zip(
        "4v4_class.zip",
        [
            "main.py",
            ("heuristic_4v4/value_fn_4v4.py", "value_fn_4v4.py"),
            ("heuristic_4v4/agent_4v4.py",    "agent_4v4.py"),
        ],
        description="4v4 GBC  |  train: heuristic_4v4/train_4v4.py"
    )


if __name__ == "__main__":
    main()
