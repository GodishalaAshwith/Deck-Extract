"""Core modules for Video-to-Slides Extractor."""

from .config import ExtractionConfig, DetectionConfig, ExportConfig, IgnoreRegion
from .extractor import VideoProbeInfo, VideoFrameExtractor
from .detector import SlideDetector, DetectedSlide
from .exporter import SlideExporter, ExportResult
from .pipeline import SlideExtractionPipeline, PipelineProgress

__all__ = [
    "ExtractionConfig",
    "DetectionConfig",
    "ExportConfig",
    "IgnoreRegion",
    "VideoProbeInfo",
    "VideoFrameExtractor",
    "SlideDetector",
    "DetectedSlide",
    "SlideExporter",
    "ExportResult",
    "SlideExtractionPipeline",
    "PipelineProgress",
]
