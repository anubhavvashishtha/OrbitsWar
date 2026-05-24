import zipfile
import os

def create_zip(zip_name, files_to_zip):
    print(f"Creating {zip_name}...")
    
    missing = [f for f in files_to_zip if not os.path.exists(f)]
    if missing:
        print(f"  [ERROR] Missing files: {', '.join(missing)}")
        print(f"  Skipping {zip_name}\n")
        return False

    with zipfile.ZipFile(zip_name, 'w', zipfile.ZIP_DEFLATED) as zf:
        for f in files_to_zip:
            print(f"  Adding {f} ...")
            zf.write(f)
            
    size_kb = os.path.getsize(zip_name) / 1024
    print(f"  [SUCCESS] Created {zip_name} ({size_kb:.1f} KB)\n")
    return True

def main():
    print("=========================================")
    print("  Creating Kaggle Submissions")
    print("=========================================\n")
    
    # 1. Classifier Bot (Hybrid 1v1 and 4v4 models)
    create_zip("submission_classifier.zip", [
        "main.py",
        "classifier/__init__.py",
        "classifier/value_fn_1v1.py",
        "heuristic_4v4/__init__.py",
        "heuristic_4v4/value_fn_4v4.py",
        "heuristic_4v4/agent_4v4.py"
    ])
    
    # 2. RL PPO Actor-Critic Bot
    create_zip("submission_rl.zip", ["main.py", "rl_value_fn.py"])
    
    print("All done! You can now upload either zip file to Kaggle.")

if __name__ == "__main__":
    main()
