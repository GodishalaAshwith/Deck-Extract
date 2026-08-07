"""Video metadata probing and frame extraction."""

import math
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Generator, Optional, Tuple

import cv2
import numpy as np

try:
    import imageio_ffmpeg
    DEFAULT_FFMPEG_EXE = imageio_ffmpeg.get_ffmpeg_exe()
except Exception:
    DEFAULT_FFMPEG_EXE = shutil.which("ffmpeg")


@dataclass
class VideoProbeInfo:
    """Detailed metadata for an input video file."""
    file_path: Path
    duration_seconds: float
    total_frames: int
    native_fps: float
    width: int
    height: int
    codec_name: str = "unknown"

    @property
    def formatted_duration(self) -> str:
        """Return formatted duration as HH:MM:SS."""
        total_sec = int(round(self.duration_seconds))
        hours = total_sec // 3600
        minutes = (total_sec % 3600) // 60
        seconds = total_sec % 60
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def format_timestamp(seconds: float) -> str:
    """Format seconds into HH:MM:SS format."""
    total_sec = max(0, int(seconds))
    hours = total_sec // 3600
    minutes = (total_sec % 3600) // 60
    sec = total_sec % 60
    return f"{hours:02d}:{minutes:02d}:{sec:02d}"


def format_timestamp_filename(seconds: float) -> str:
    """Format seconds into HH-MM-SS format safe for filenames."""
    total_sec = max(0, int(seconds))
    hours = total_sec // 3600
    minutes = (total_sec % 3600) // 60
    sec = total_sec % 60
    return f"{hours:02d}-{minutes:02d}-{sec:02d}"


class VideoFrameExtractor:
    """Extracts frames from video files using FFmpeg streaming with OpenCV fallback."""

    SUPPORTED_EXTENSIONS = {".mp4", ".mkv", ".avi", ".mov", ".webm", ".m4v", ".flv", ".wmv"}

    def __init__(self, video_path: Path | str, ffmpeg_exe: Optional[str] = None):
        self.video_path = Path(video_path).resolve()
        self.ffmpeg_exe = ffmpeg_exe or DEFAULT_FFMPEG_EXE
        self.probe_info = self.probe()

    def probe(self) -> VideoProbeInfo:
        """Validate and probe video file using OpenCV and FFmpeg."""
        if not self.video_path.exists():
            raise FileNotFoundError(f"Video file does not exist: {self.video_path}")

        if not self.video_path.is_file():
            raise ValueError(f"Path is not a regular file: {self.video_path}")

        ext = self.video_path.suffix.lower()
        if ext not in self.SUPPORTED_EXTENSIONS:
            # Still attempt if user specified, but log warning
            pass

        cap = cv2.VideoCapture(str(self.video_path))
        if not cap.isOpened():
            raise ValueError(
                f"Failed to open video file: '{self.video_path.name}'. "
                f"The file may be corrupt or an unsupported format."
            )

        try:
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            native_fps = float(cap.get(cv2.CAP_PROP_FPS))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            if native_fps <= 0.0 or math.isnan(native_fps):
                native_fps = 25.0  # Fallback assumption

            duration = 0.0
            if total_frames > 0 and native_fps > 0:
                duration = total_frames / native_fps
            else:
                # Fallback duration measurement
                duration_msec = cap.get(cv2.CAP_PROP_POS_MSEC)
                if duration_msec > 0:
                    duration = duration_msec / 1000.0

            # Read first frame to ensure decodability
            ret, first_frame = cap.read()
            if not ret or first_frame is None:
                raise ValueError(
                    f"Video file '{self.video_path.name}' cannot be decoded (no valid frames found)."
                )

            # Refine width/height from actual frame if metadata is missing
            if width <= 0 or height <= 0:
                height, width = first_frame.shape[:2]

            # If total_frames was inaccurate (e.g. some webm/mkv formats), estimate or keep
            if duration <= 0:
                duration = 1.0  # minimum placeholder

            fourcc_int = int(cap.get(cv2.CAP_PROP_FOURCC))
            codec = "".join([chr((fourcc_int >> 8 * i) & 0xFF) for i in range(4)]).strip()
            if not codec:
                codec = "unknown"

            return VideoProbeInfo(
                file_path=self.video_path,
                duration_seconds=duration,
                total_frames=total_frames if total_frames > 0 else int(duration * native_fps),
                native_fps=native_fps,
                width=width,
                height=height,
                codec_name=codec,
            )
        finally:
            cap.release()

    def extract_frames(
        self,
        sampling_fps: float = 1.0,
        use_ffmpeg: bool = True
    ) -> Generator[Tuple[int, float, np.ndarray], None, None]:
        """Yield (sample_index, timestamp_seconds, frame_bgr) at the requested sampling rate."""
        if sampling_fps <= 0:
            raise ValueError("sampling_fps must be greater than 0")

        if use_ffmpeg and self.ffmpeg_exe and os.path.exists(self.ffmpeg_exe):
            try:
                yield from self._extract_frames_ffmpeg(sampling_fps)
                return
            except Exception:
                # If FFmpeg subprocess fails, fallback smoothly to OpenCV
                pass

        yield from self._extract_frames_opencv(sampling_fps)

    def _extract_frames_ffmpeg(
        self, sampling_fps: float
    ) -> Generator[Tuple[int, float, np.ndarray], None, None]:
        """Extract frames using FFmpeg rawvideo pipe."""
        width = self.probe_info.width
        height = self.probe_info.height
        frame_bytes = width * height * 3

        # Construct FFmpeg command
        cmd = [
            self.ffmpeg_exe,
            "-v", "error",
            "-i", str(self.video_path),
            "-vf", f"fps={sampling_fps}",
            "-f", "rawvideo",
            "-pix_fmt", "bgr24",
            "-an",
            "-sn",
            "-"
        ]

        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=10**7
        )

        sample_idx = 0
        time_step = 1.0 / sampling_fps

        try:
            while True:
                raw_frame = process.stdout.read(frame_bytes)
                if len(raw_frame) < frame_bytes:
                    break

                frame = np.frombuffer(raw_frame, dtype=np.uint8).reshape((height, width, 3))
                timestamp = sample_idx * time_step
                yield sample_idx, timestamp, frame
                sample_idx += 1
        finally:
            if process.stdout:
                process.stdout.close()
            if process.stderr:
                process.stderr.close()
            process.terminate()
            try:
                process.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                process.kill()

    def _extract_frames_opencv(
        self, sampling_fps: float
    ) -> Generator[Tuple[int, float, np.ndarray], None, None]:
        """Extract frames using OpenCV VideoCapture."""
        cap = cv2.VideoCapture(str(self.video_path))
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {self.video_path}")

        try:
            native_fps = self.probe_info.native_fps
            step_interval_sec = 1.0 / sampling_fps
            step_frames = max(1, int(round(native_fps * step_interval_sec)))

            sample_idx = 0
            current_frame_pos = 0

            while True:
                # Seek or read forward
                cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame_pos)
                ret, frame = cap.read()
                if not ret or frame is None:
                    break

                timestamp = current_frame_pos / native_fps
                yield sample_idx, timestamp, frame

                sample_idx += 1
                current_frame_pos += step_frames
        finally:
            cap.release()
