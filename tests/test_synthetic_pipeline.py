"""Comprehensive synthetic test suite for Video-to-Slides Extractor."""

import json
import shutil
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from video_to_ppt.core.config import DetectionConfig, ExportConfig, ExtractionConfig, IgnoreRegion
from video_to_ppt.core.detector import DetectedSlide, SlideDetector
from video_to_ppt.core.exporter import SlideExporter
from video_to_ppt.core.extractor import VideoFrameExtractor, format_timestamp, format_timestamp_filename
from video_to_ppt.core.pipeline import SlideExtractionPipeline


def create_slide_frame(
    text: str,
    bg_color: tuple = (30, 30, 30),
    width: int = 1280,
    height: int = 720,
    with_cursor: bool = False,
    webcam_frame_count: int = 0,
) -> np.ndarray:
    """Generate a clean synthetic slide frame."""
    frame = np.full((height, width, 3), bg_color, dtype=np.uint8)

    # Add header bar
    cv2.rectangle(frame, (0, 0), (width, 80), (50, 50, 50), -1)
    cv2.putText(
        frame,
        "Video-to-Slides Presentation",
        (40, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (200, 200, 200),
        2,
        cv2.LINE_AA,
    )

    # Add main slide content text
    cv2.putText(
        frame,
        text,
        (100, 320),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.5,
        (255, 255, 255),
        3,
        cv2.LINE_AA,
    )

    # Optional minor cursor blink
    if with_cursor:
        cv2.rectangle(frame, (600, 300), (605, 340), (255, 255, 255), -1)

    # Simulated webcam in top-right corner (region: x=0.75, y=0.0, w=0.25, h=0.3)
    if webcam_frame_count > 0:
        cam_x1 = int(width * 0.78)
        cam_y1 = int(height * 0.05)
        cam_x2 = int(width * 0.98)
        cam_y2 = int(height * 0.32)
        # Moving/changing webcam content
        cam_color = (
            (webcam_frame_count * 25) % 255,
            (webcam_frame_count * 45) % 255,
            (webcam_frame_count * 65) % 255,
        )
        cv2.rectangle(frame, (cam_x1, cam_y1), (cam_x2, cam_y2), cam_color, -1)
        cv2.putText(
            frame,
            f"CAM {webcam_frame_count}",
            (cam_x1 + 10, cam_y1 + 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
        )

    return frame


def generate_test_video(video_path: Path, fps: int = 10, duration_sec: int = 12):
    """Create a 12-second test video with 3 distinct slides and noise."""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    width, height = 1280, 720
    out = cv2.VideoWriter(str(video_path), fourcc, fps, (width, height))

    total_frames = fps * duration_sec

    for frame_i in range(total_frames):
        t = frame_i / fps

        # Slide 1: 0.0s to 4.0s
        if t < 4.0:
            with_cursor = (frame_i % 10 == 0)  # cursor blink
            frame = create_slide_frame(
                "Slide 1: Introduction to AI",
                bg_color=(120, 40, 20),
                with_cursor=with_cursor,
                webcam_frame_count=frame_i,
            )
        # 1-frame momentary glitch at 4.0s (should be debounced!)
        elif frame_i == int(4.0 * fps):
            frame = create_slide_frame("MID-TRANSITION GLITCH", bg_color=(0, 0, 0))
        # Slide 2: 4.1s to 8.0s
        elif t < 8.0:
            frame = create_slide_frame(
                "Slide 2: System Architecture",
                bg_color=(20, 90, 40),
                webcam_frame_count=frame_i,
            )
        # Slide 3: 8.0s to 12.0s
        else:
            frame = create_slide_frame(
                "Slide 3: Experimental Results",
                bg_color=(40, 20, 110),
                webcam_frame_count=frame_i,
            )

        out.write(frame)

    out.release()


class TestVideoToSlidesExtractor(unittest.TestCase):
    """Test suite for VideoToSlides extractor core components and pipeline."""

    @classmethod
    def setUpClass(cls):
        cls.temp_dir = Path(tempfile.mkdtemp(prefix="v2ppt_test_"))
        cls.test_video_path = cls.temp_dir / "synthetic_lecture.mp4"
        generate_test_video(cls.test_video_path, fps=10, duration_sec=12)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.temp_dir, ignore_errors=True)

    def test_01_timestamp_formatters(self):
        """Test timestamp string formatting functions."""
        self.assertEqual(format_timestamp(0), "00:00:00")
        self.assertEqual(format_timestamp(65), "00:01:05")
        self.assertEqual(format_timestamp(3665), "01:01:05")
        self.assertEqual(format_timestamp_filename(3665), "01-01-05")

    def test_02_ignore_region_box(self):
        """Test normalized ignore region conversion to pixel box."""
        region = IgnoreRegion(x=0.5, y=0.5, width=0.25, height=0.25)
        x1, y1, x2, y2 = region.to_pixel_box(320, 180)
        self.assertEqual((x1, y1, x2, y2), (160, 90, 240, 135))

    def test_03_video_probe(self):
        """Test probing video metadata."""
        extractor = VideoFrameExtractor(self.test_video_path)
        probe = extractor.probe_info
        self.assertEqual(probe.width, 1280)
        self.assertEqual(probe.height, 720)
        self.assertAlmostEqual(probe.duration_seconds, 12.0, delta=0.5)
        self.assertEqual(probe.formatted_duration, "00:00:12")

    def test_04_end_to_end_pipeline(self):
        """Test full end-to-end pipeline execution with webcam ignore masking and debouncing."""
        out_dir = self.temp_dir / "pipeline_output"

        ext_cfg = ExtractionConfig(sampling_fps=1.0, use_ffmpeg=True)
        # Exclude the webcam region (top-right corner: x=0.75, y=0.0, w=0.25, h=0.35)
        det_cfg = DetectionConfig(
            ssim_threshold=0.90,
            phash_threshold=6,
            debounce_samples=2,
            ignore_regions=[IgnoreRegion(x=0.75, y=0.0, width=0.25, height=0.35)],
        )
        exp_cfg = ExportConfig(
            output_dir=out_dir,
            save_png=True,
            save_pdf=True,
            save_pptx=True,
            save_zip=True,
            save_metadata=True,
        )

        pipeline = SlideExtractionPipeline(
            extraction_config=ext_cfg,
            detection_config=det_cfg,
            export_config=exp_cfg,
        )

        progress_snapshots = []

        def on_progress(p):
            progress_snapshots.append(p)

        slides, export_res, probe_info = pipeline.run(
            video_path=self.test_video_path,
            on_progress=on_progress,
        )

        # We expect exactly 3 slides (Slide 1, Slide 2, Slide 3)
        self.assertEqual(len(slides), 3, f"Expected 3 slides, but got {len(slides)}")

        # Verify slide timestamps
        self.assertAlmostEqual(slides[0].timestamp_sec, 0.0, delta=1.0)
        self.assertAlmostEqual(slides[1].timestamp_sec, 4.0, delta=1.5)
        self.assertAlmostEqual(slides[2].timestamp_sec, 8.0, delta=1.5)

        # Verify exported artifacts
        self.assertTrue(export_res.pdf_path.exists(), "PDF file was not created")
        self.assertTrue(export_res.pptx_path.exists(), "PPTX file was not created")
        self.assertTrue(export_res.zip_path.exists(), "ZIP bundle was not created")
        self.assertTrue(export_res.metadata_path.exists(), "Metadata JSON was not created")

        # Verify saved PNG images
        self.assertEqual(len(export_res.image_paths), 3)
        for img_path in export_res.image_paths:
            self.assertTrue(img_path.exists())
            img = cv2.imread(str(img_path))
            self.assertEqual(img.shape, (720, 1280, 3))

        # Verify metadata JSON contents
        with open(export_res.metadata_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        self.assertEqual(meta["total_slides"], 3)
        self.assertEqual(len(meta["slides"]), 3)


if __name__ == "__main__":
    unittest.main()
