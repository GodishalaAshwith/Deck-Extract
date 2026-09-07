"""Command-line interface for Video-to-Slides Extractor."""

import argparse
import sys
import time
from pathlib import Path
from typing import List

from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table
from rich.text import Text

from .core.config import DetectionConfig, ExportConfig, ExtractionConfig, IgnoreRegion
from .core.detector import DetectedSlide
from .core.extractor import format_timestamp
from .core.pipeline import PipelineProgress, SlideExtractionPipeline

console = Console()


def parse_ignore_regions(region_strings: List[str]) -> List[IgnoreRegion]:
    """Parse comma-separated 'x,y,w,h' strings into IgnoreRegion objects."""
    regions = []
    for s in region_strings:
        parts = [float(p.strip()) for p in s.split(",") if p.strip()]
        if len(parts) != 4:
            console.print(
                f"[yellow]Warning: Ignoring invalid region '{s}'. Expected format: x,y,width,height[/yellow]"
            )
            continue
        regions.append(IgnoreRegion(x=parts[0], y=parts[1], width=parts[2], height=parts[3]))
    return regions


def build_parser() -> argparse.ArgumentParser:
    """Construct command-line argument parser."""
    parser = argparse.ArgumentParser(
        prog="video-to-ppt",
        description="Extract unique slides from video presentations into PNG, PDF, PPTX, and ZIP.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "video_path",
        type=str,
        help="Path to the input video file (MP4, MKV, AVI, MOV, WEBM, etc.)",
    )
    parser.add_argument(
        "-o", "--output-dir",
        type=str,
        default="./output",
        help="Directory to save extracted slides and documents",
    )
    parser.add_argument(
        "-f", "--fps",
        type=float,
        default=1.0,
        help="Frame sampling rate in frames per second (e.g. 1.0 = 1 frame/sec)",
    )
    parser.add_argument(
        "-t", "--threshold",
        type=float,
        default=0.90,
        help="SSIM similarity threshold below which a frame is considered a new slide (0.0 - 1.0)",
    )
    parser.add_argument(
        "--phash-threshold",
        type=int,
        default=6,
        help="Hamming distance threshold for fast perceptual hash pre-filtering",
    )
    parser.add_argument(
        "-d", "--debounce",
        type=int,
        default=2,
        help="Number of consecutive samples a candidate slide must persist to confirm transition",
    )
    parser.add_argument(
        "--ignore-region",
        action="append",
        default=[],
        help="Normalized region (0.0-1.0) to exclude from comparison (format: x,y,width,height). Can be repeated.",
    )
    parser.add_argument(
        "--no-pdf",
        action="store_true",
        help="Skip generating PDF document",
    )
    parser.add_argument(
        "--no-pptx",
        action="store_true",
        help="Skip generating PowerPoint PPTX presentation",
    )
    parser.add_argument(
        "--no-zip",
        action="store_true",
        help="Skip generating ZIP archive",
    )
    parser.add_argument(
        "--no-png",
        action="store_true",
        help="Skip saving individual PNG images",
    )
    parser.add_argument(
        "--no-ffmpeg",
        action="store_true",
        help="Force OpenCV frame extraction instead of FFmpeg streaming",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose output logging",
    )

    return parser


def run_cli(args: argparse.Namespace) -> int:
    """Run CLI application."""
    video_path = Path(args.video_path)

    console.print(
        Panel.fit(
            "[bold cyan]Video-to-Slides Extractor[/bold cyan]\n"
            f"[dim]Input Video:[/dim] [white]{video_path.name}[/white]\n"
            f"[dim]Output Directory:[/dim] [yellow]{args.output_dir}[/yellow]",
            border_style="cyan",
        )
    )

    if not video_path.exists():
        console.print(f"[bold red]Error:[/bold red] Video file not found: {video_path}")
        return 1

    ignore_regions = parse_ignore_regions(args.ignore_region)

    ext_config = ExtractionConfig(
        sampling_fps=args.fps,
        use_ffmpeg=not args.no_ffmpeg,
    )
    det_config = DetectionConfig(
        ssim_threshold=args.threshold,
        phash_threshold=args.phash_threshold,
        debounce_samples=args.debounce,
        ignore_regions=ignore_regions,
    )
    exp_config = ExportConfig(
        output_dir=Path(args.output_dir),
        save_png=not args.no_png,
        save_pdf=not args.no_pdf,
        save_pptx=not args.no_pptx,
        save_zip=not args.no_zip,
        save_metadata=True,
    )

    pipeline = SlideExtractionPipeline(
        extraction_config=ext_config,
        detection_config=det_config,
        export_config=exp_config,
    )

    progress_bar = Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
    )

    main_task = progress_bar.add_task("[cyan]Processing video...", total=100.0)

    def on_progress_callback(p: PipelineProgress):
        cur_ts = format_timestamp(p.current_timestamp)
        tot_ts = format_timestamp(p.total_duration)
        desc = (
            f"[cyan]Processing:[/cyan] [bold]{cur_ts}[/bold] / {tot_ts} "
            f"([green]{p.detected_slides_count} slides[/green], [magenta]{p.speed_factor:.1f}x speed[/magenta])"
        )
        progress_bar.update(main_task, completed=p.progress_pct, description=desc)

    def on_slide_detected_callback(slide: DetectedSlide):
        if args.verbose:
            console.log(
                f"[bold green]Slide #{slide.index}[/bold green] detected at "
                f"[yellow]{slide.formatted_timestamp}[/yellow] (SSIM: {slide.ssim_score:.3f})"
            )

    def on_log_callback(msg: str, level: str):
        if args.verbose:
            style = "red" if level == "error" else ("yellow" if level == "warning" else "dim")
            console.log(f"[{style}]{msg}[/{style}]")

    start_time = time.time()

    try:
        with progress_bar:
            slides, export_res, probe_info = pipeline.run(
                video_path=video_path,
                on_progress=on_progress_callback,
                on_slide_detected=on_slide_detected_callback,
                on_log=on_log_callback,
            )
            progress_bar.update(main_task, completed=100.0, description="[green]Processing complete!")
    except KeyboardInterrupt:
        console.print("\n[bold yellow]Process interrupted by user.[/bold yellow]")
        pipeline.cancel()
        return 130
    except Exception as e:
        console.print(f"\n[bold red]Pipeline Error:[/bold red] {e}")
        if args.verbose:
            console.print_exception()
        return 1

    elapsed = time.time() - start_time

    # Results Table
    table = Table(title=f"Extracted Slides Summary ({len(slides)} total)", border_style="cyan")
    table.add_column("Slide #", justify="center", style="bold green")
    table.add_column("Timestamp", justify="center", style="yellow")
    table.add_column("Seconds", justify="right", style="dim")
    table.add_column("SSIM Score", justify="right", style="cyan")
    table.add_column("pHash Dist", justify="right", style="magenta")

    for s in slides:
        table.add_row(
            str(s.index),
            s.formatted_timestamp,
            f"{s.timestamp_sec:.1f}s",
            f"{s.ssim_score:.3f}" if s.index > 1 else "1.000 (Initial)",
            str(s.phash_distance) if s.index > 1 else "0",
        )

    console.print()
    console.print(table)

    # Export Summary
    summary_text = (
        f"[bold green]Slide Extraction Completed in {elapsed:.2f}s![/bold green]\n\n"
        f"[dim]Total Slides:[/dim] [bold]{len(slides)}[/bold]\n"
        f"[dim]Video Duration:[/dim] {probe_info.formatted_duration} ({probe_info.width}x{probe_info.height})\n"
        f"[dim]Output Folder:[/dim] {export_res.output_dir}\n"
    )

    if export_res.pdf_path:
        summary_text += f"• [bold]PDF Deck:[/bold] {export_res.pdf_path.name}\n"
    if export_res.pptx_path:
        summary_text += f"• [bold]PowerPoint:[/bold] {export_res.pptx_path.name}\n"
    if export_res.zip_path:
        summary_text += f"• [bold]ZIP Bundle:[/bold] {export_res.zip_path.name}\n"
    if export_res.metadata_path:
        summary_text += f"• [bold]Metadata:[/bold] {export_res.metadata_path.name}\n"

    console.print(Panel(summary_text, title="Export Summary", border_style="green"))
    return 0


def main():
    """Main CLI entrypoint."""
    if len(sys.argv) == 1:
        from .gui import launch_gui
        launch_gui()
        sys.exit(0)
        
    parser = build_parser()
    args = parser.parse_args()
    sys.exit(run_cli(args))


if __name__ == "__main__":
    main()
