from kaggle_environments import make
import json

env = make("orbit_wars", debug=True)
env.run(["main.py", "submission.py"])

for i, s in enumerate(env.steps[-1]):
    print(f"Player {i}: reward={s.reward}  status={s.status}")

html_output = env.render(mode="html", width=900, height=700)
with open("simulation.html", "w", encoding="utf-8") as f:
    f.write(html_output)
print("Simulation saved to simulation.html")
