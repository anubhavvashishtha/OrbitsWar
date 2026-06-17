import zipfile
import tarfile
from pathlib import Path

ROOT = Path(__file__).parent

def create_training_zip():
    print("Creating kaggle_training_code.zip...")
    with zipfile.ZipFile(ROOT / 'kaggle_training_code.zip', 'w', zipfile.ZIP_DEFLATED) as zf:
        # Add physics engine
        zf.write(ROOT / 'physics.py', 'physics.py')
        
        # Add ML scripts
        for file in ['features.py', 'extract.py', 'model.py', 'train.py', 'agent.py']:
            path = ROOT / 'imitation' / file
            if path.exists():
                zf.write(path, f'imitation/{file}')
                
    print("Created kaggle_training_code.zip!")

def create_submission_tar():
    print("Creating submission.tar.gz...")
    with tarfile.open(ROOT / 'submission.tar.gz', 'w:gz') as tf:
        # Root entry point
        tf.add(ROOT / 'main.py', arcname='main.py')
        tf.add(ROOT / 'physics.py', arcname='physics.py')
        
        # Add imitation module (including checkpoints)
        imitation_dir = ROOT / 'imitation'
        for path in imitation_dir.rglob('*'):
            if '__pycache__' in path.parts:
                continue
            if path.suffix in ['.py', '.pt', '.npz'] and path.name != 'dataset.npz':
                tf.add(path, arcname=path.relative_to(ROOT))
                
    print("Created submission.tar.gz!")

if __name__ == "__main__":
    create_training_zip()
    create_submission_tar()
