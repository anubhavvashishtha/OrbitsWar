import sys
from pathlib import Path

# Add the imitation folder to the path so the agent can find its local imports
sys.path.insert(0, str(Path(__file__).parent / 'imitation'))

from imitation.agent import agent
