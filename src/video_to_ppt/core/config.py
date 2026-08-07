"""Configuration dataclasses for Video-to-Slides Extractor."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple


@dataclass
class IgnoreRegion:
    """Defines a bounding box (normalized 0.0 to 1.0) to exclude from comparison.

    Example: Webcam in top-right corner: x=0.75, y=0.0, width=0.25, height=0.3
    """
    x: float
    y: float
    width: float
    height: float

    def to_pixel_box(self, img_width: int, img_height: int) -> Tuple[int, int, int, int]:
        """Convert normalized (0.0-1.0) coordinates to (x1, y1, x2, y2) pixel slice."""
        x1 = int(max(0.0, min(1.0, self.x)) * img_width)
        y1 = int(max(0.0, min(1.0, self.y)) * img_height)
        x2 = int(max(0.0, min(1.0, self.x + self.width)) * img_width)
        y2 = int(max(0.0, min(1.0, self.y + self.height)) * img_height)
        return x1, y1, x2, y2


@dataclass
class ExtractionConfig:
    """Frame extraction configuration."""
    sampling_fps: float = 1.0
    compare_width: int = 320
    compare_height: int = 180
    use_ffmpeg: bool = True


@dataclass
class DetectionConfig:
    """Slide change detection configuration."""
    ssim_threshold: float = 0.90
    phash_threshold: int = 6
    debounce_samples: int = 2
    ignore_regions: List[IgnoreRegion] = field(default_factory=list)


@dataclass
class ExportConfig:
    """Slide export and packaging configuration."""
    output_dir: Path = field(default_factory=lambda: Path("./output"))
    save_png: bool = True
    save_pdf: bool = True
    save_pptx: bool = True
    save_zip: bool = True
    save_metadata: bool = True
    slide_prefix: str = "slide"
    deck_title: Optional[str] = None
