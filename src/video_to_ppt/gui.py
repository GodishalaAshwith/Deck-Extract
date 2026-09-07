import threading
import time
from pathlib import Path

import customtkinter as ctk
from customtkinter import filedialog

from .core.config import DetectionConfig, ExportConfig, ExtractionConfig, ProcessingMode
from .core.pipeline import PipelineProgress, SlideExtractionPipeline
from .core.detector import DetectedSlide
from .core.extractor import format_timestamp

# macOS Dark Theme Palette
BG_COLOR = "#1c1c1e"          # System Gray 6 (Dark)
FRAME_COLOR = "#2c2c2e"       # System Gray 5 (Dark)
ACCENT_COLOR = "#0a84ff"      # System Blue
HOVER_COLOR = "#007aff"       # System Blue (Darker)
TEXT_COLOR = "#ffffff"        # White
SECONDARY_TEXT = "#8e8e93"    # System Gray

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

class DeckExtractApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        
        self.title("deck-extract")
        self.geometry("700x750")
        self.minsize(700, 750)
        self.resizable(True, True)
        self.configure(fg_color=BG_COLOR)
        
        # Setup window icon
        import os
        import sys
        if getattr(sys, 'frozen', False):
            icon_path = os.path.join(sys._MEIPASS, 'app_icon.ico')
        else:
            icon_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), 'app_icon.ico')
            
        if os.path.exists(icon_path):
            self.iconbitmap(icon_path)

        
        self.pipeline = None
        self.is_running = False
        
        self._build_ui()

    def _build_ui(self):
        # Header
        self.header_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.header_frame.pack(pady=(40, 20), fill="x")
        
        self.header_label = ctk.CTkLabel(
            self.header_frame, text="deck-extract", 
            font=ctk.CTkFont(family="SF Pro Display", size=32, weight="bold"),
            text_color=TEXT_COLOR
        )
        self.header_label.pack()
        
        self.sub_label = ctk.CTkLabel(
            self.header_frame, text="Video-to-Slides Extractor", 
            font=ctk.CTkFont(family="SF Pro Text", size=14),
            text_color=SECONDARY_TEXT
        )
        self.sub_label.pack()

        # Main Card Frame
        self.card_frame = ctk.CTkFrame(self, fg_color=FRAME_COLOR, corner_radius=16)
        self.card_frame.pack(padx=40, pady=10, fill="x")

        # Mode Selector
        self.mode_var = ctk.StringVar(value="Normal")
        self.mode_selector = ctk.CTkSegmentedButton(
            self.card_frame, 
            values=["Lightweight", "Normal"],
            variable=self.mode_var,
            selected_color=ACCENT_COLOR,
            selected_hover_color=HOVER_COLOR,
            unselected_color=BG_COLOR,
            unselected_hover_color="#3a3a3c",
            font=ctk.CTkFont(family="SF Pro Text", size=13, weight="bold")
        )
        self.mode_selector.pack(pady=(25, 15), padx=30, fill="x")
        
        self.mode_desc_label = ctk.CTkLabel(
            self.card_frame, text="Normal: High accuracy, all formats.", 
            font=ctk.CTkFont(family="SF Pro Text", size=12),
            text_color=SECONDARY_TEXT
        )
        self.mode_desc_label.pack(pady=(0, 20))
        self.mode_selector.configure(command=self._on_mode_change)

        # File Selection
        self.video_path_var = ctk.StringVar()
        self.video_btn = ctk.CTkButton(
            self.card_frame, text="🎬 Select Video Presentation", 
            command=self._browse_video, 
            fg_color=BG_COLOR, hover_color="#3a3a3c",
            text_color=TEXT_COLOR, font=ctk.CTkFont(family="SF Pro Text", size=14, weight="bold"),
            corner_radius=8, height=45
        )
        self.video_btn.pack(padx=30, pady=(0, 10), fill="x")
        
        self.video_label = ctk.CTkLabel(
            self.card_frame, textvariable=self.video_path_var,
            font=ctk.CTkFont(family="SF Pro Text", size=12),
            text_color=SECONDARY_TEXT,
            wraplength=500
        )
        self.video_label.pack(padx=30, pady=(0, 20))

        # Output Dir
        self.output_dir_var = ctk.StringVar(value="./output")
        self.out_btn = ctk.CTkButton(
            self.card_frame, text="📁 Set Output Directory", 
            command=self._browse_output, 
            fg_color=BG_COLOR, hover_color="#3a3a3c",
            text_color=TEXT_COLOR, font=ctk.CTkFont(family="SF Pro Text", size=14, weight="bold"),
            corner_radius=8, height=45
        )
        self.out_btn.pack(padx=30, pady=(0, 10), fill="x")

        self.out_label = ctk.CTkLabel(
            self.card_frame, textvariable=self.output_dir_var,
            font=ctk.CTkFont(family="SF Pro Text", size=12),
            text_color=SECONDARY_TEXT,
            wraplength=500
        )
        self.out_label.pack(padx=30, pady=(0, 25))

        # Action Button
        self.action_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.action_frame.pack(padx=40, pady=25, fill="x")

        self.run_btn = ctk.CTkButton(
            self.action_frame, text="Start Extraction", 
            command=self._toggle_extraction,
            font=ctk.CTkFont(family="SF Pro Text", size=16, weight="bold"),
            height=50, corner_radius=12,
            fg_color=ACCENT_COLOR, hover_color=HOVER_COLOR, text_color="#ffffff"
        )
        self.run_btn.pack(side="left", fill="x", expand=True, padx=(0, 5))

        self.open_out_btn = ctk.CTkButton(
            self.action_frame, text="📂 Open Output", 
            command=self._open_output_folder, state="disabled",
            font=ctk.CTkFont(family="SF Pro Text", size=16, weight="bold"),
            height=50, corner_radius=12, width=150,
            fg_color=FRAME_COLOR, text_color=SECONDARY_TEXT
        )
        self.open_out_btn.pack(side="right")
        
        # Progress & Status
        self.progress_bar = ctk.CTkProgressBar(self, progress_color=ACCENT_COLOR, fg_color=FRAME_COLOR, height=6)
        self.progress_bar.pack(padx=40, fill="x")
        self.progress_bar.set(0)
        
        self.status_label = ctk.CTkLabel(self, text="Ready", text_color=SECONDARY_TEXT, font=ctk.CTkFont(family="SF Pro Text", size=12))
        self.status_label.pack(pady=(10, 15))

        # Log Section
        self.log_frame = ctk.CTkFrame(self, fg_color=FRAME_COLOR, corner_radius=12)
        self.log_frame.pack(padx=40, pady=(0, 30), fill="both", expand=True)

        self.log_label = ctk.CTkLabel(
            self.log_frame, text="Extraction Logs", 
            font=ctk.CTkFont(family="SF Pro Text", size=13, weight="bold"),
            text_color=SECONDARY_TEXT
        )
        self.log_label.pack(anchor="w", padx=20, pady=(15, 5))

        self.log_box = ctk.CTkTextbox(
            self.log_frame, fg_color="#141415", text_color="#30d158", 
            corner_radius=8, border_color=FRAME_COLOR, border_width=1,
            font=ctk.CTkFont(family="Consolas", size=12)
        )
        self.log_box.pack(padx=20, pady=(0, 20), fill="both", expand=True)

    def _on_mode_change(self, value):
        if value == "Lightweight":
            self.mode_desc_label.configure(text="Lightweight: Blazing fast, PNG only, lower accuracy.")
        else:
            self.mode_desc_label.configure(text="Normal: High accuracy, all formats (PDF/PPTX/ZIP).")

    def _browse_video(self):
        file_path = filedialog.askopenfilename(
            title="Select Video",
            filetypes=[("Video Files", "*.mp4 *.mkv *.avi *.mov *.webm"), ("All Files", "*.*")]
        )
        if file_path:
            self.video_path_var.set(file_path)

    def _browse_output(self):
        dir_path = filedialog.askdirectory(title="Select Output Directory")
        if dir_path:
            self.output_dir_var.set(dir_path)

    def _log(self, message: str):
        def append():
            self.log_box.insert("end", message + "\n")
            self.log_box.see("end")
        self.after(0, append)

    def _open_output_folder(self):
        import os, subprocess, platform
        out_path = getattr(self, "last_output_dir", None)
        if not out_path or not os.path.exists(out_path):
            return
        if platform.system() == "Windows":
            os.startfile(out_path)
        elif platform.system() == "Darwin":
            subprocess.Popen(["open", out_path])
        else:
            subprocess.Popen(["xdg-open", out_path])

    def _update_progress(self, progress: PipelineProgress):
        def update():
            self.progress_bar.set(progress.progress_pct / 100.0)
            cur_ts = format_timestamp(progress.current_timestamp)
            tot_ts = format_timestamp(progress.total_duration)
            self.status_label.configure(
                text=f"Processing: {cur_ts} / {tot_ts} ({progress.detected_slides_count} slides) - {progress.speed_factor:.1f}x speed"
            )
        self.after(0, update)
        
    def _on_slide_detected(self, slide: DetectedSlide):
        self._log(f"Slide #{slide.index} detected at {slide.formatted_timestamp}")

    def _run_extraction(self):
        video_path = Path(self.video_path_var.get())
        if not video_path.exists() or not video_path.is_file():
            self._log("ERROR: Invalid video file selected.")
            self._set_ui_state("normal")
            return

        out_dir = Path(self.output_dir_var.get())
        
        mode = ProcessingMode.LIGHTWEIGHT if self.mode_var.get() == "Lightweight" else ProcessingMode.NORMAL
        
        ext_config = ExtractionConfig()
        det_config = DetectionConfig(mode=mode)
        exp_config = ExportConfig(output_dir=out_dir)
        
        self.pipeline = SlideExtractionPipeline(
            extraction_config=ext_config,
            detection_config=det_config,
            export_config=exp_config
        )
        
        self._log(f"Starting extraction for: {video_path.name}")
        start_time = time.time()
        
        try:
            slides, export_res, probe = self.pipeline.run(
                video_path=video_path,
                on_progress=self._update_progress,
                on_slide_detected=self._on_slide_detected,
                on_log=lambda msg, lvl: self._log(msg) if lvl != "debug" else None
            )
            elapsed = time.time() - start_time
            self._log(f"\n--- SUCCESS ---")
            self._log(f"Extracted {len(slides)} slides in {elapsed:.1f} seconds.")
            self._log(f"Output saved to: {export_res.output_dir}")
            self.last_output_dir = str(export_res.output_dir)
            self._set_ui_state("normal", success=True)
            return
        except Exception as e:
            if str(e) == "Pipeline cancelled":
                self._log("\nExtraction cancelled by user.")
            else:
                self._log(f"\nERROR: {str(e)}")
                
        self._set_ui_state("normal", success=False)

    def _set_ui_state(self, state: str, success: bool = False):
        def update():
            self.is_running = (state == "disabled")
            new_state = "disabled" if self.is_running else "normal"
            
            self.video_btn.configure(state=new_state)
            self.out_btn.configure(state=new_state)
            self.mode_selector.configure(state=new_state)
            
            
            if self.is_running:
                self.run_btn.configure(text="Stop", fg_color="#ff3b30", hover_color="#d70015")
                self.open_out_btn.configure(state="disabled", fg_color=FRAME_COLOR, text_color=SECONDARY_TEXT, hover_color=FRAME_COLOR)
            else:
                self.run_btn.configure(text="Start Extraction", fg_color=ACCENT_COLOR, hover_color=HOVER_COLOR)
                if success:
                    self.open_out_btn.configure(state="normal", fg_color="#34c759", hover_color="#30d158", text_color="#ffffff")
                else:
                    self.progress_bar.set(0)
                    self.status_label.configure(text="Ready")
                    self.open_out_btn.configure(state="disabled", fg_color=FRAME_COLOR, text_color=SECONDARY_TEXT, hover_color=FRAME_COLOR)
        self.after(0, update)

    def _toggle_extraction(self):
        if self.is_running:
            if self.pipeline:
                self.pipeline.cancel()
        else:
            if not self.video_path_var.get():
                self._log("Please select a video file first.")
                return
                
            self._set_ui_state("disabled")
            self.log_box.delete("0.0", "end")
            threading.Thread(target=self._run_extraction, daemon=True).start()

def launch_gui():
    app = DeckExtractApp()
    app.mainloop()

if __name__ == "__main__":
    launch_gui()
