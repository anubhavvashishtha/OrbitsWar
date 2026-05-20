from kaggle_environments import make
import json

env = make("orbit_wars", debug=True)
env.run(["main.py", "submission.py"])
print("[BLUE] Your agent (main.py) = Player 0 | [ORANGE] Opponent (submission.py) = Player 1")

for i, s in enumerate(env.steps[-1]):
    print(f"Player {i}: reward={s.reward}  status={s.status}")

html_output = env.render(mode="html", width=900, height=700)

header = "<h2 style='text-align: center; font-family: sans-serif; padding: 10px; background-color: #f0f0f0;'>🔵 Your Agent (main.py) is Player 0 (Blue) | 🟠 Opponent is Player 1 (Orange)</h2>"
if "<body" in html_output:
    html_output = html_output.replace("<body>", f"<body>\n{header}")
else:
    html_output = header + html_output

with open("simulation.html", "w", encoding="utf-8") as f:
    f.write(html_output)
print("Simulation saved to simulation.html")
