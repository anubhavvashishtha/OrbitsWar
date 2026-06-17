import os
import sys
from kaggle_environments import make

def main():
    print("Initializing Orbit Wars environment...")
    env = make("orbit_wars", configuration={"seed": 42, "episodeSteps": 1000}, debug=True)
    
    # Extract the zip to a temporary folder and run main.py inside it to exactly simulate Kaggle
    import zipfile
    zip_path = "kaggle_upload/submission_ppo_best.zip"
    extract_dir = "tmp_submission"
    os.makedirs(extract_dir, exist_ok=True)
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)
        
    agent1 = os.path.abspath(os.path.join(extract_dir, "main.py"))
    agent2 = "main.py"
    
    print(f"Running Match: [PPO Neural Network] vs [Heuristic Bot]")
    env.run([agent1, agent2])
    
    # Print the final scores
    print("\nMatch Complete!")
    final_state = env.steps[-1]
    
    for i, state in enumerate(final_state):
        agent_name = "PPO Neural Network" if i == 0 else "Heuristic Bot"
        reward = state.reward if state.reward is not None else 0
        status = state.status
        print(f"Player {i+1} ({agent_name}): Reward = {reward}, Status = {status}")
        
    print("\nSaving simulation replay to replay.html...")
    html_content = env.render(mode="html")
    with open("replay.html", "w", encoding="utf-8") as f:
        f.write(html_content)
        
    print("Done! You can open 'replay.html' in your web browser to watch the match.")

if __name__ == "__main__":
    main()
