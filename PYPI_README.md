# deck-extract 🎞️ ➔ 📑

A highly-optimized, state-aware command-line tool that automatically detects slide changes in recorded videos (e.g., lectures, webinars) and extracts each unique slide as a high-quality image. 

Unlike basic extraction scripts that dump hundreds of messy transition frames and duplicate bullet points, `deck-extract` uses a dual-threshold state machine to **merge incremental build-ups** and **prevent backward-navigation duplicates**.

It automatically compiles the final slides into **PDF**, native **PowerPoint (PPTX)**, and **ZIP** formats.

---

## 🌟 Why `deck-extract` is different

1. **Incremental Build-Up Merging**: When a presenter adds 5 bullet points one by one, standard tools extract 5 separate slides. `deck-extract` intelligently recognizes these as incremental additions (using 85%-95% SSIM bounds) and merges them, outputting only the final, complete slide.
2. **Time-Travel Duplicate Prevention**: If a professor jumps back to Slide 2 to answer a question, basic tools extract it again as a new slide. `deck-extract` maintains a global Perceptual Hash (`pHash`) history to instantly recognize seen slides and block them from polluting the timeline.
3. **Picture-in-Picture Masking**: Does your video have a ticking clock or a webcam overlay in the corner? Use the `--ignore-region` flag to mask it out so moving hands don't trigger false slide changes.
4. **Clean Animations**: A built-in debounce algorithm ensures we wait for slide transitions (fades/swipes) to fully settle before locking in a crisp, clear frame.

## 🚀 Installation

Install globally via `pip`:

```bash
pip install deck-extract
```
*(Requires Python 3.9+)*

## 🛠️ Usage

Simply point the tool at any video file:

```bash
deck-extract path/to/your/video.mp4
```

This will process the video and automatically generate the following in a `./output` folder:
- `presentation.pdf` (Compiled slide deck)
- `presentation.pptx` (Native PowerPoint)
- `slides.zip` (Archived PNGs)
- `slides_metadata.json` (Timestamps and SSIM metrics)

### Advanced Usage

Process at 2 frames-per-second, ignore the bottom-right corner (for webcams), and save to a custom folder:

```bash
deck-extract input_video.mp4 -o my_lecture_slides/ -f 2.0 --ignore-region 0.8,0.8,0.2,0.2
```

## ⚙️ Full Configuration Options

| Option | Shortcut | Description | Default |
|--------|----------|-------------|---------|
| `--output-dir` | `-o` | Directory to save extracted artifacts. | `./output` |
| `--fps` | `-f` | Frame sampling rate. Higher = more CPU, but captures faster clicks. | `1.0` |
| `--threshold` | `-t` | SSIM similarity threshold. Below this = completely new slide. | `0.90` |
| `--build-up` | | SSIM threshold for merging incremental bullet points. | `0.85` |
| `--debounce` | `-d` | Seconds a slide must remain completely static before capturing. | `2` |
| `--ignore-region` | | Normalized region `x,y,w,h` (0.0 to 1.0) to ignore for webcam overlays. | None |

### Disabling specific exports:
Don't need a PowerPoint? You can skip specific exporters to save time:
- `--no-pdf`
- `--no-pptx`
- `--no-zip`
- `--no-png`

## 📝 License
Distributed under the MIT License.
