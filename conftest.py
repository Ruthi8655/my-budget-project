import sys
from pathlib import Path

# Add project root to sys.path exactly once so every import of shared.*
# resolves to the same module object — no duplicate registrations.
ROOT = Path(__file__).parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
