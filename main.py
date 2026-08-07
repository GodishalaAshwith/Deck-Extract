"""Top-level executable launcher for Video-to-Slides Extractor."""

import sys
from pathlib import Path

# Add src to python path for direct script execution
sys.path.insert(0, str(Path(__file__).parent / "src"))

from video_to_ppt.cli import main

if __name__ == "__main__":
    main()
