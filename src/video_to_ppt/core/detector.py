"""Classical Computer Vision slide-change detection engine."""

from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import imagehash
import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as ssim

from .config import DetectionConfig, ExtractionConfig, IgnoreRegion
from .extractor import format_timestamp


@dataclass
class DetectedSlide:
    """Represents a confirmed slide capture."""
    index: int
    timestamp_sec: float
    formatted_timestamp: str
    frame_bgr: np.ndarray
    ssim_score: float
    phash_distance: int


class SlideDetector:
    """Detects slide changes across sampled video frames using pHash + SSIM + Debouncing."""

    def __init__(
        self,
        detection_config: Optional[DetectionConfig] = None,
        extraction_config: Optional[ExtractionConfig] = None,
    ):
        self.det_config = detection_config or DetectionConfig()
        self.ext_config = extraction_config or ExtractionConfig()

        self._slide_count = 0
        self._reference_slide: Optional[DetectedSlide] = None
        self._ref_compare_bgr: Optional[np.ndarray] = None
        self._ref_phash: Optional[imagehash.ImageHash] = None

        # Debounce state tracking
        self._candidate_slide: Optional[DetectedSlide] = None
        self._candidate_compare_bgr: Optional[np.ndarray] = None
        self._candidate_phash: Optional[imagehash.ImageHash] = None
        self._candidate_persist_count = 0

    def reset(self):
        """Reset internal detector state."""
        self._slide_count = 0
        self._reference_slide = None
        self._ref_compare_bgr = None
        self._ref_phash = None
        self._candidate_slide = None
        self._candidate_compare_bgr = None
        self._candidate_phash = None
        self._candidate_persist_count = 0

    def _preprocess_frame(
        self, frame_bgr: np.ndarray
    ) -> Tuple[np.ndarray, imagehash.ImageHash]:
        """Downscale frame, apply ignore region masks, and compute pHash."""
        target_w = self.ext_config.compare_width
        target_h = self.ext_config.compare_height

        # Downscale for cheap comparison
        resized = cv2.resize(frame_bgr, (target_w, target_h), interpolation=cv2.INTER_AREA)

        # Mask out any ignored regions (webcam overlay, timers, etc.)
        for region in self.det_config.ignore_regions:
            x1, y1, x2, y2 = region.to_pixel_box(target_w, target_h)
            resized[y1:y2, x1:x2] = 0

        # Compute perceptual hash from RGB PIL image
        rgb_img = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb_img)
        hash_val = imagehash.phash(pil_img)

        return resized, hash_val

    def _compute_similarity(self, img1_bgr: np.ndarray, img2_bgr: np.ndarray) -> float:
        """Compute structural similarity across color channels."""
        return float(
            ssim(
                img1_bgr,
                img2_bgr,
                channel_axis=-1,
                data_range=255.0
            )
        )

    def process_frame(
        self,
        sample_index: int,
        timestamp_sec: float,
        frame_bgr: np.ndarray
    ) -> List[DetectedSlide]:
        """Process a single sampled frame. Returns a list of newly confirmed slides (if any)."""
        confirmed_slides: List[DetectedSlide] = []
        resized_bgr, phash_val = self._preprocess_frame(frame_bgr)

        # Case 1: First frame in the video -> immediately capture as Slide 1
        if self._reference_slide is None:
            self._slide_count += 1
            first_slide = DetectedSlide(
                index=self._slide_count,
                timestamp_sec=timestamp_sec,
                formatted_timestamp=format_timestamp(timestamp_sec),
                frame_bgr=frame_bgr.copy(),
                ssim_score=0.0,
                phash_distance=64,
            )
            self._reference_slide = first_slide
            self._ref_compare_bgr = resized_bgr
            self._ref_phash = phash_val
            confirmed_slides.append(first_slide)
            return confirmed_slides

        # Fast pre-filter: Perceptual Hash distance against active reference slide
        hash_dist = int(phash_val - self._ref_phash)

        # If hash distance is 0, frame is completely identical
        if hash_dist == 0:
            similarity = 1.0
        else:
            similarity = self._compute_similarity(self._ref_compare_bgr, resized_bgr)

        is_different_from_ref = similarity < self.det_config.ssim_threshold

        if not is_different_from_ref:
            # Frame matches reference slide -> reset any pending candidate
            self._candidate_slide = None
            self._candidate_persist_count = 0
            return confirmed_slides

        # Frame differs from reference slide -> handle debouncing
        if self.det_config.debounce_samples <= 1:
            # No debouncing requested -> confirm immediately
            self._slide_count += 1
            new_slide = DetectedSlide(
                index=self._slide_count,
                timestamp_sec=timestamp_sec,
                formatted_timestamp=format_timestamp(timestamp_sec),
                frame_bgr=frame_bgr.copy(),
                ssim_score=similarity,
                phash_distance=hash_dist,
            )
            self._reference_slide = new_slide
            self._ref_compare_bgr = resized_bgr
            self._ref_phash = phash_val
            confirmed_slides.append(new_slide)
            return confirmed_slides

        # Debouncing is active (debounce_samples >= 2)
        if self._candidate_slide is None:
            # First time seeing a different slide -> initiate candidate
            self._candidate_slide = DetectedSlide(
                index=self._slide_count + 1,
                timestamp_sec=timestamp_sec,
                formatted_timestamp=format_timestamp(timestamp_sec),
                frame_bgr=frame_bgr.copy(),
                ssim_score=similarity,
                phash_distance=hash_dist,
            )
            self._candidate_compare_bgr = resized_bgr
            self._candidate_phash = phash_val
            self._candidate_persist_count = 1
        else:
            # Check if this frame is consistent with the active candidate
            cand_hash_dist = int(phash_val - self._candidate_phash)
            if cand_hash_dist == 0:
                cand_sim = 1.0
            else:
                cand_sim = self._compute_similarity(self._candidate_compare_bgr, resized_bgr)

            if cand_sim >= self.det_config.ssim_threshold:
                # Frame is consistent with candidate -> increment persistence count
                self._candidate_persist_count += 1

                if self._candidate_persist_count >= self.det_config.debounce_samples:
                    # Candidate persisted for required samples -> confirm new slide!
                    self._slide_count += 1
                    confirmed_slide = self._candidate_slide
                    confirmed_slide.index = self._slide_count
                    
                    self._reference_slide = confirmed_slide
                    self._ref_compare_bgr = self._candidate_compare_bgr
                    self._ref_phash = self._candidate_phash

                    # Clear candidate
                    self._candidate_slide = None
                    self._candidate_compare_bgr = None
                    self._candidate_phash = None
                    self._candidate_persist_count = 0

                    confirmed_slides.append(confirmed_slide)
            else:
                # Frame changed yet again (transition or animation in progress) -> update candidate
                self._candidate_slide = DetectedSlide(
                    index=self._slide_count + 1,
                    timestamp_sec=timestamp_sec,
                    formatted_timestamp=format_timestamp(timestamp_sec),
                    frame_bgr=frame_bgr.copy(),
                    ssim_score=similarity,
                    phash_distance=hash_dist,
                )
                self._candidate_compare_bgr = resized_bgr
                self._candidate_phash = phash_val
                self._candidate_persist_count = 1

        return confirmed_slides

    def finalize(self) -> List[DetectedSlide]:
        """Finalize detection at end of video stream, flushing any valid pending candidate."""
        confirmed: List[DetectedSlide] = []
        if self._candidate_slide is not None:
            # Candidate was active at stream end -> confirm it
            self._slide_count += 1
            self._candidate_slide.index = self._slide_count
            confirmed.append(self._candidate_slide)
            self._candidate_slide = None
            self._candidate_persist_count = 0
        return confirmed
