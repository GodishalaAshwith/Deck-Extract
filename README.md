# Video-to-Slides Extractor

A lightweight command-line tool that automatically detects slide changes in recorded videos (e.g., lectures, webinars) and extracts each unique slide as a high-quality image. The extracted slides can also be exported to PDF, PPTX, and ZIP formats.

## Features
- Fast video frame extraction (using FFmpeg by default).
- Accurate slide transition detection using SSIM (Structural Similarity) and Perceptual Hashing (pHash).
- Automatic deduplication of identical or near-identical slides.
- Export to multiple formats: PNG, PDF, PowerPoint (PPTX), and ZIP archives.
- Configurable settings for frame sampling rate, detection sensitivity, and ignoring specific video regions (e.g., webcam overlays).

## Installation

### Prerequisites
- Python 3.9 or higher

### Steps

1. **Clone the repository:**
   ```bash
   git clone <repository_url>
   cd VideoToPPT
   ```

2. **Create and activate a virtual environment (Optional but recommended):**
   ```bash
   python -m venv venv
   
   # On Windows (Command Prompt):
   venv\Scripts\activate
   # On Windows (PowerShell):
   venv\Scripts\Activate.ps1
   # On Windows (Git Bash):
   source venv/Scripts/activate
   
   # On macOS/Linux:
   source venv/bin/activate
   ```

3. **Install the package and its dependencies:**
   ```bash
   pip install -e .
   ```
   *Alternatively, you can install the dependencies using `pip install -r requirements.txt`.*

## Usage

You can run the tool via the installed CLI command `video-to-ppt` or by directly running `main.py`.

### Basic Usage

```bash
video-to-ppt path/to/your/video.mp4
```
Or:
```bash
python main.py path/to/your/video.mp4
```

This will process `video.mp4` and save the extracted slides (PNGs, PDF, PPTX, and ZIP) in the default `./output` directory.

### Advanced Usage

```bash
video-to-ppt input_video.mp4 -o custom_output_dir/ -f 2.0 -t 0.85
```

### Common Options

- `video_path`: Path to the input video file (required).
- `-o`, `--output-dir`: Directory to save extracted slides and documents (default: `./output`).
- `-f`, `--fps`: Frame sampling rate in frames per second (default: `1.0`).
- `-t`, `--threshold`: SSIM similarity threshold below which a frame is considered a new slide (default: `0.90`).
- `--phash-threshold`: Hamming distance threshold for fast perceptual hash pre-filtering (default: `6`).
- `-d`, `--debounce`: Number of consecutive samples a candidate slide must persist to confirm transition (default: `2`).
- `--ignore-region`: Normalized region (0.0-1.0) to exclude from comparison (format: `x,y,width,height`).
- `--no-pdf`: Skip generating PDF document.
- `--no-pptx`: Skip generating PowerPoint PPTX presentation.
- `--no-zip`: Skip generating ZIP archive.
- `--no-png`: Skip saving individual PNG images.
- `-v`, `--verbose`: Enable verbose output logging.

For a full list of options, run:
```bash
video-to-ppt --help
```
