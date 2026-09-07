import os
import sys
import PyInstaller.__main__
from pathlib import Path

def build_app():
    project_root = Path(__file__).parent.absolute()
    main_script = project_root / "main.py"
    
    print(f"Building application from {main_script}")
    
    PyInstaller.__main__.run([
        str(main_script),
        '--name=deck-extract',
        '--windowed',
        '--noconfirm',
        '--clean',
        '--icon=app_icon.ico',
        '--add-data=src/video_to_ppt;video_to_ppt',
        '--add-data=app_icon.ico;.',
    ])

if __name__ == "__main__":
    build_app()
