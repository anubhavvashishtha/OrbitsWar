import zipfile
import os

def create_zip(zip_name, files_to_zip, description=None):
    label = f"  ({description})" if description else ""
    print(f"Creating {zip_name}{label}...")

    paths = [f[0] if isinstance(f, tuple) else f for f in files_to_zip]
    missing = [p for p in paths if not os.path.exists(p)]
    if missing:
        print(f"  [ERROR] Missing files: {', '.join(missing)}")
        print(f"  Skipping {zip_name}\n")
        return False

    out = os.path.join("submissions", zip_name)
    os.makedirs("submissions", exist_ok=True)
    with zipfile.ZipFile(out, 'w', zipfile.ZIP_DEFLATED) as zf:
        for f in files_to_zip:
            if isinstance(f, tuple):
                src, arcname = f
            else:
                src, arcname = f, f
            print(f"  Adding {src} -> {arcname}")
            zf.write(src, arcname)

    size_kb = os.path.getsize(out) / 1024
    print(f"  [SUCCESS] {out} ({size_kb:.1f} KB)\n")
    return True

def main():
    print("=========================================")
    print("  Creating Kaggle Submissions")
    print("=========================================\n")
    
    # 1. Classifier Bot
    create_zip("submission_classifier.zip", ["main.py", "value_fn.py"])
    
    # 2. RL PPO Actor-Critic Bot
    create_zip("submission_rl.zip", ["main.py", "rl_value_fn.py"])
    
    print("All done! You can now upload either zip file to Kaggle.")

if __name__ == "__main__":
    main()
