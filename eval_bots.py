import os
import sys
import glob
import random
import shutil
import time
from kaggle_environments import make

def force_reload_modules():
    """Forces Python to reload the main.py and value_fn.py modules for the new match."""
    modules_to_remove = []
    for mod in sys.modules:
        if 'value_fn' in mod or 'main' in mod:
            modules_to_remove.append(mod)
    for mod in modules_to_remove:
        del sys.modules[mod]

def run_evaluation(agent_name, value_fn_file, num_matches=20):
    print(f"\n==================================================")
    print(f"  Evaluating {agent_name} ({value_fn_file})")
    print(f"==================================================")

    if not os.path.exists(value_fn_file):
        print(f"Error: {value_fn_file} not found!")
        return

    # Copy the target brain to value_fn.py
    shutil.copy(value_fn_file, "value_fn.py")
    
    enemy_pool = glob.glob("enemy agents/*.py")
    if not enemy_pool:
        print("Error: No enemies found in 'enemy agents' folder!")
        return

    wins = 0
    ties = 0
    losses = 0

    for i in range(num_matches):
        force_reload_modules()
        
        # Pick 3 random enemies (with replacement, so they can be the same)
        opponents = [random.choice(enemy_pool) for _ in range(3)]
        
        # We are Player 0. Enemies are Players 1, 2, 3
        agents = ["main.py"] + opponents
        
        # Randomize seat order so we aren't always Player 0 (seat position matters!)
        seat_order = [0, 1, 2, 3]
        random.shuffle(seat_order)
        shuffled_agents = [agents[idx] for idx in seat_order]
        our_seat = seat_order.index(0) # Find where our agent went

        print(f"Match {i+1}/{num_matches} (Seat {our_seat}) against: {[os.path.basename(op).replace('.py','') for op in opponents]} ...", end=" ", flush=True)

        env = make("orbit_wars", configuration={"episodeSteps": 500})
        try:
            steps = env.run(shuffled_agents)
            last_step = steps[-1]
            
            # Extract rewards (Kaggle Orbit Wars uses final ship count as reward)
            rewards = [agent.reward if agent.reward is not None else 0 for agent in last_step]
            
            our_reward = rewards[our_seat]
            max_reward = max(rewards)
            
            if our_reward == max_reward:
                if rewards.count(max_reward) > 1:
                    print(f"TIE! (Ships: {our_reward})")
                    ties += 1
                else:
                    print(f"WIN! (Ships: {our_reward})")
                    wins += 1
            else:
                print(f"LOSS! (Our Ships: {our_reward}, Winner: {max_reward})")
                losses += 1
                
        except Exception as e:
            print(f"ERROR crashed: {e}")
            losses += 1

    print(f"\n--- {agent_name} Final Stats ---")
    print(f"Wins:   {wins}/{num_matches} ({wins/num_matches*100:.1f}%)")
    print(f"Ties:   {ties}/{num_matches} ({ties/num_matches*100:.1f}%)")
    print(f"Losses: {losses}/{num_matches} ({losses/num_matches*100:.1f}%)")
    print(f"Win/Tie Rate: {(wins+ties)/num_matches*100:.1f}%")

if __name__ == "__main__":
    # Test both agents for 20 matches each
    run_evaluation("Classifier Agent", "value_fn_classifier.py", num_matches=20)
    run_evaluation("TD Regressor Agent", "value_fn_regresoor.py", num_matches=20)
    
    print("\nEvaluation Complete!")
