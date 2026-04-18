from __future__ import annotations

import argparse
from pathlib import Path

from gui import launch_gui
from video_to_3d.pipeline import PipelineConfig, run_pipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Convert a handheld object video into a coarse 3D mesh.")
    parser.add_argument("--video", type=Path, help="Path to the input video.")
    parser.add_argument("--output", type=Path, help="Output directory. Defaults to outputs/<video-name>.")
    parser.add_argument("--target-frames", type=int, default=24, help="Number of selected frames for reconstruction.")
    parser.add_argument("--long-edge", type=int, default=512, help="Resize long edge for processing.")
    parser.add_argument("--mask-mode", choices=["full", "core"], default="core", help="Segmentation aggressiveness.")
    parser.add_argument("--mask-threshold", type=int, default=24, help="Background difference threshold.")
    parser.add_argument("--blur-floor", type=float, default=40.0, help="Minimum Laplacian variance to keep a frame.")
    parser.add_argument("--voxel-resolution", type=int, default=96, help="Voxel grid resolution.")
    parser.add_argument("--canvas-size", type=int, default=256, help="Normalized silhouette canvas size.")
    parser.add_argument("--size-mm", type=float, default=120.0, help="Longest exported dimension in millimeters.")
    parser.add_argument("--grabcut-iterations", type=int, default=2, help="GrabCut refinement iterations per selected frame.")
    parser.add_argument("--consensus-ratio", type=float, default=0.68, help="Fraction of silhouettes that must agree during carving.")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.video is None:
        launch_gui()
        return

    video_path = args.video.expanduser().resolve()
    if not video_path.exists():
        raise SystemExit(f"Video not found: {video_path}")

    output_dir = args.output
    if output_dir is None:
        output_dir = Path("outputs") / video_path.stem.replace(" ", "_")
    output_dir = output_dir.expanduser().resolve()

    config = PipelineConfig(
        target_frames=args.target_frames,
        long_edge=args.long_edge,
        mask_mode=args.mask_mode,
        mask_threshold=args.mask_threshold,
        blur_floor=args.blur_floor,
        voxel_resolution=args.voxel_resolution,
        canvas_size=args.canvas_size,
        size_mm=args.size_mm,
        grabcut_iterations=args.grabcut_iterations,
        consensus_ratio=args.consensus_ratio,
    )

    result = run_pipeline(video_path, output_dir, config=config, progress=print)
    print()
    print("Done")
    print(f"Output directory: {result.output_dir}")
    print(f"OBJ: {result.obj_path}")
    print(f"STL: {result.stl_path}")
    print(f"Preview: {result.preview_path}")


if __name__ == "__main__":
    main()
