import os
import zipfile

def create_clean_zip():
    zip_name = "OrbitsWar_CleanCodebase.zip"
    
    # The folders and files we explicitly need
    include_paths = [
        "records",
        "enemy agents",
        "rl",
        "classifier",
        "main.py",
        "value_fn_1v1.py",
        "value_fn_4v4.py"
    ]
    
    # What we explicitly want to ignore
    exclude_extensions = ['.pkl', '.pyc']
    exclude_dirs = ['__pycache__']
    
    print(f"Creating {zip_name}...")
    
    with zipfile.ZipFile(zip_name, 'w', zipfile.ZIP_DEFLATED) as zf:
        for path in include_paths:
            if not os.path.exists(path):
                print(f"  [WARN] Missing path: {path}")
                continue
                
            if os.path.isfile(path):
                # It's a single file
                if not any(path.endswith(ext) for ext in exclude_extensions):
                    print(f"  Adding file: {path}")
                    zf.write(path)
            else:
                # It's a directory
                for root, dirs, files in os.walk(path):
                    # Filter out excluded directories
                    dirs[:] = [d for d in dirs if d not in exclude_dirs]
                    
                    for file in files:
                        if any(file.endswith(ext) for ext in exclude_extensions):
                            continue # Skip .pkl and .pyc files
                            
                        file_path = os.path.join(root, file)
                        # Avoid printing every single record file, just print Python/key files
                        if not file.endswith('.json'):
                            print(f"  Adding: {file_path}")
                        zf.write(file_path)
                        
    size_mb = os.path.getsize(zip_name) / (1024 * 1024)
    print(f"\n[SUCCESS] Saved clean dataset as {zip_name} ({size_mb:.2f} MB)")
    print("This file contains NO .pkl caches and NO .venv bloat.")

if __name__ == "__main__":
    create_clean_zip()
