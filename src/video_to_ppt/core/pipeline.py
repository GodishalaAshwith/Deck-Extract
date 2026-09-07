"""End-to-end slide extraction pipeline orchestrator."""

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional, Tuple

from .config import DetectionConfig, ExportConfig, ExtractionConfig
from .detector import DetectedSlide, SlideDetector
from .exporter import ExportResult, SlideExporter
from .extractor import VideoFrameExtractor, VideoProbeInfo


@dataclass
class PipelineProgress:
    """Snapshot of current pipeline execution progress."""
    current_timestamp: float
    total_duration: float
    progress_pct: float
    processed_samples: int
    estimated_total_samples: int
    detected_slides_count: int
    speed_factor: float  # e.g., 20.0 means 20x real-time


class SlideExtractionPipeline:
    """Orchestrates extraction, detection, export, and progress event dispatching."""

    def __init__(
        self,
        extraction_config: Optional[ExtractionConfig] = None,
        detection_config: Optional[DetectionConfig] = None,
        export_config: Optional[ExportConfig] = None,
    ):
        self.extraction_config = extraction_config or ExtractionConfig()
        self.detection_config = detection_config or DetectionConfig()
        self.export_config = export_config or ExportConfig()

        from .config import ProcessingMode
        if self.detection_config.mode == ProcessingMode.LIGHTWEIGHT:
            # Force lower FPS and disable heavy exports in lightweight mode
            self.extraction_config.sampling_fps = min(0.5, self.extraction_config.sampling_fps)
            self.export_config.save_pdf = False
            self.export_config.save_pptx = False

        self._cancelled = False

    def cancel(self):
        """Request graceful cancellation of the running pipeline."""
        self._cancelled = True

    def run(
        self,
        video_path: Path | str,
        on_progress: Optional[Callable[[PipelineProgress], None]] = None,
        on_slide_detected: Optional[Callable[[DetectedSlide], None]] = None,
        on_log: Optional[Callable[[str, str], None]] = None,
    ) -> Tuple[List[DetectedSlide], ExportResult, VideoProbeInfo]:
        """Execute the end-to-end extraction and export pipeline."""
        self._cancelled = False
        start_wall_time = time.perf_counter()

        def log(msg: str, level: str = "info"):
            if on_log:
                on_log(msg, level)

        log(f"Initializing video extractor for: {video_path}")
        extractor = VideoFrameExtractor(video_path)
        probe_info = extractor.probe_info
        log(
            f"Video probed successfully: {probe_info.width}x{probe_info.height} "
            f"at {probe_info.native_fps:.2f} fps, duration: {probe_info.formatted_duration}"
        )

        detector = SlideDetector(
            detection_config=self.detection_config,
            extraction_config=self.extraction_config,
        )

        total_duration = max(1.0, probe_info.duration_seconds)
        sampling_fps = self.extraction_config.sampling_fps
        estimated_samples = max(1, int(round(total_duration * sampling_fps)))

        detected_slides: List[DetectedSlide] = []
        sample_count = 0

        frame_gen = extractor.extract_frames(
            sampling_fps=sampling_fps,
            use_ffmpeg=self.extraction_config.use_ffmpeg
        )

        for sample_idx, timestamp_sec, frame_bgr in frame_gen:
            if self._cancelled:
                log("Pipeline cancelled by user.", "warning")
                break

            sample_count += 1
            new_slides = detector.process_frame(sample_idx, timestamp_sec, frame_bgr)

            for s in new_slides:
                detected_slides.append(s)
                log(f"Detected Slide #{s.index} at {s.formatted_timestamp} (SSIM: {s.ssim_score:.3f})")
                if on_slide_detected:
                    on_slide_detected(s)

            # Progress computation
            if on_progress:
                elapsed_real = max(0.001, time.perf_counter() - start_wall_time)
                speed_factor = timestamp_sec / elapsed_real if timestamp_sec > 0 else 1.0
                pct = min(100.0, (timestamp_sec / total_duration) * 100.0)

                progress = PipelineProgress(
                    current_timestamp=timestamp_sec,
                    total_duration=total_duration,
                    progress_pct=pct,
                    processed_samples=sample_count,
                    estimated_total_samples=estimated_samples,
                    detected_slides_count=len(detected_slides),
                    speed_factor=speed_factor,
                )
                on_progress(progress)

        # Finalize any pending debounced candidate at end of stream
        if not self._cancelled:
            tail_slides = detector.finalize()
            for s in tail_slides:
                detected_slides.append(s)
                log(f"Detected Slide #{s.index} (end of video) at {s.formatted_timestamp}")
                if on_slide_detected:
                    on_slide_detected(s)

        log(f"Extraction completed. Total unique slides found: {len(detected_slides)}")

        # Exporter
        log(f"Exporting artifacts to: {self.export_config.output_dir}")
        exporter = SlideExporter(self.export_config)
        export_result = exporter.export(detected_slides, probe_info)

        elapsed_total = time.perf_counter() - start_wall_time
        log(f"All tasks finished in {elapsed_total:.2f}s.")

        return detected_slides, export_result, probe_info
