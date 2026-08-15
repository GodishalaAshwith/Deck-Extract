# Video-to-PPT Slide Detection Engine: End-to-End Implementation Report

This document details the end-to-end architecture and implementation of the slide detection engine in the Video-to-PPT project. It specifically highlights the logic used to accurately detect distinct slides, merge build-up animations, and prevent duplicate outputs when presenters navigate backwards.

---

## 1. Frame Extraction & Pre-processing
The extraction pipeline (`extractor.py`) begins by reading the input video and sampling frames at a configured rate (default: `1.0 FPS`). This minimizes CPU overhead while ensuring we don't miss slide changes.

For each sampled frame, the `SlideDetector` (`detector.py`) performs the following pre-processing steps:
1. **Downscaling:** The frame is resized to `640x360`. This resolution strikes a balance between fast processing and preserving enough detail to detect minor text additions (e.g., a single line of text).
2. **Region Masking:** Any configured `ignore_regions` (such as a picture-in-picture webcam overlay or a clock) are blacked out to prevent movement in these areas from triggering false slide changes.
3. **Perceptual Hashing (pHash):** A perceptual hash is computed for the frame to enable ultra-fast, structural distance comparisons.

---

## 2. The Detection State Machine
The core of the `SlideDetector` relies on comparing the current frame against active reference states. It maintains three key properties:
* **`_reference_slide`:** The immediate previously confirmed slide.
* **`_candidate_slide`:** A temporary slide state used to verify that a visual change is persistent (debouncing) rather than a brief video artifact or a transition animation in progress.
* **`_history_slides`:** A buffer containing lightweight representations (pHash + downscaled BGR image) of *all* previously confirmed slides.

---

## 3. Core Detection Logic & Debouncing
When a new frame is processed, it is compared against the `_reference_slide`. 
* The system first checks the pHash distance for an immediate `0` match.
* If the pHash differs, it calculates the **Structural Similarity Index (SSIM)** between the two frames.

**The Trigger:**
If the SSIM falls below the strict `ssim_threshold` (configured to `0.95`), the system flags the frame as "different". 

**Debouncing (Stabilization):**
To avoid capturing messy transition animations (e.g., a fade or slide-in effect), the frame is placed into the `_candidate_slide` state. 
* The candidate must persist for a set number of consecutive frames (defined by `debounce_samples`, default `2`). 
* If the frame keeps changing (SSIM drops against the candidate), the candidate is reset. 
* Once the frame stabilizes for the required duration, it moves to the finalization phase.

---

## 4. Finalization: Build-up Merging & Duplicate Prevention
Before a stabilized candidate is blindly appended to the output list, it must pass through two advanced checks to ensure a clean, chronological output deck.

### A. Incremental Build-up Merging
When a presenter adds a single bullet point or a line of text, we want to capture the final completed slide, rather than outputting 5 separate slides for each bullet point.
* The system checks the SSIM of the candidate against the immediate `_reference_slide`.
* If the similarity is `>= build_up_threshold` (configured to `0.85`), the system classifies this as a **build-up slide**.
* **Action:** Instead of creating a new slide, the system **mutates the existing reference slide in-place**. It overwrites the old image frame and timestamp with the new ones, and updates the history buffer. The incomplete step is discarded, leaving only the final slide in the output.

### B. Duplicate Prevention (Navigating Backwards)
If the candidate is not a build-up, it might be a slide we've already seen (e.g., the presenter clicked "Back" to a previous slide).
* The system iterates backwards through the `_history_slides` buffer.
* It compares the candidate's pHash and SSIM against every previously extracted slide.
* **Action:** If it finds a match (`SSIM >= 0.95`), the system updates its active `_reference_slide` to this historical slide so it knows where it is in the presentation. However, it **deliberately skips appending this slide to the final output list**.

### C. Confirming a New Slide
If the candidate is neither a build-up nor a historical duplicate, it is finally stamped as a brand new slide. It is appended to the `confirmed_slides` list, added to the history buffer, and the active `_reference_slide` is updated.

---

## 5. Artifact Export
At the end of the video, or dynamically as slides are confirmed, the `confirmed_slides` list is passed to the `SlideExporter` (`exporter.py`).
Because of the in-place build-up merging and history buffer filtering, the exporter receives a clean, duplicate-free array of the most complete slide frames.

It then generates:
1. Individual high-resolution PNG images.
2. A consolidated PDF document.
3. A native PowerPoint (PPTX) presentation.
4. A JSON metadata file tracking timestamps, SSIM scores, and slide indexes.
5. A packaged ZIP archive containing the artifacts.
