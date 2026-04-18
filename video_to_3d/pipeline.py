from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy import ndimage as ndi
from skimage import measure


ProgressFn = Callable[[str], None]


@dataclass(slots=True)
class PipelineConfig:
    target_frames: int = 24
    long_edge: int = 512
    mask_mode: str = "core"
    mask_threshold: int = 24
    blur_floor: float = 40.0
    voxel_resolution: int = 96
    canvas_size: int = 256
    size_mm: float = 120.0
    background_samples: int = 64
    start_skip_ratio: float = 0.03
    end_skip_ratio: float = 0.02
    min_foreground_ratio: float = 0.01
    max_foreground_ratio: float = 0.60
    grabcut_iterations: int = 2
    consensus_ratio: float = 0.68


@dataclass(slots=True)
class PipelineResult:
    output_dir: Path
    obj_path: Path
    stl_path: Path
    preview_path: Path
    metrics_path: Path


@dataclass(slots=True)
class FrameRecord:
    index: int
    frame: np.ndarray
    sharpness: float
    mask: np.ndarray
    bbox: tuple[int, int, int, int]
    area_ratio: float


@dataclass(slots=True)
class FrameMetric:
    index: int
    sharpness: float
    presence_score: float


def run_pipeline(
    video_path: Path,
    output_dir: Path,
    *,
    config: PipelineConfig | None = None,
    progress: ProgressFn | None = None,
) -> PipelineResult:
    config = config or PipelineConfig()
    progress = progress or (lambda _: None)
    output_dir.mkdir(parents=True, exist_ok=True)

    progress("Loading video frames")
    frames = load_video(video_path, long_edge=config.long_edge)
    if len(frames) < 12:
        raise RuntimeError("Video is too short for reconstruction.")

    progress("Estimating background")
    background = estimate_background(frames, sample_count=config.background_samples)
    cv2.imwrite(str(output_dir / "background.png"), background)

    progress("Scoring frames")
    metrics = build_frame_metrics(frames)
    candidate_indices = filter_candidate_indices(metrics, config)
    if len(candidate_indices) < max(8, config.target_frames // 2):
        raise RuntimeError("Too few usable frames. Record with the object centered and visible for longer.")

    progress("Selecting frames")
    selected_indices = select_keyframe_indices(metrics, candidate_indices, config.target_frames)

    progress("Segmenting selected frames")
    selected = segment_selected_frames(frames, selected_indices, config)
    if len(selected) < max(6, config.target_frames // 3):
        raise RuntimeError("Segmentation kept too few frames. Try --mask-mode full or record with less hand coverage.")

    normalized_masks, normalized_frames = normalize_views(selected, config.canvas_size)
    normalized_masks, normalized_frames, selected = prune_duplicate_views(normalized_masks, normalized_frames, selected)

    save_contact_sheet(normalized_frames, output_dir / "selected_frames.png", columns=4)
    save_contact_sheet([mask_to_bgr(mask) for mask in normalized_masks], output_dir / "selected_masks.png", columns=4)
    save_contact_sheet(build_overlay_frames(normalized_frames, normalized_masks), output_dir / "selected_overlays.png", columns=4)

    progress("Building voxel hull")
    occupancy = carve_visual_hull(normalized_masks, config.voxel_resolution, consensus_ratio=config.consensus_ratio)
    occupancy = keep_largest_component(occupancy)
    filled = ndi.binary_fill_holes(occupancy)
    smoothed = ndi.gaussian_filter(filled.astype(np.float32), sigma=1.0)
    if np.count_nonzero(smoothed > 0.5) < 1000:
        raise RuntimeError("The voxel hull collapsed. Try a steadier orbit, less hand occlusion, or --mask-mode full.")

    progress("Extracting mesh")
    vertices, faces = marching_cubes_to_mesh(smoothed, level=0.5)
    depth_scale = infer_depth_scale(normalized_masks)
    vertices[:, 2] *= depth_scale
    vertices = scale_vertices(vertices, target_mm=config.size_mm)

    obj_path = output_dir / "mesh.obj"
    stl_path = output_dir / "mesh.stl"
    preview_path = output_dir / "mesh_preview.png"
    metrics_path = output_dir / "metrics.json"

    export_obj(vertices, faces, obj_path)
    export_stl_ascii(vertices, faces, stl_path)
    render_mesh_preview(vertices, faces, preview_path)

    metrics = {
        "video_path": str(video_path),
        "config": asdict(config),
        "frames_total": len(frames),
        "candidate_frames": len(candidate_indices),
        "selected_frames": len(selected),
        "selected_indices": [record.index for record in selected],
        "selected_sharpness_mean": float(np.mean([record.sharpness for record in selected])),
        "selected_area_ratio_mean": float(np.mean([record.area_ratio for record in selected])),
        "voxel_occupied": int(np.count_nonzero(filled)),
        "depth_scale": float(depth_scale),
        "mesh_vertices": int(vertices.shape[0]),
        "mesh_faces": int(faces.shape[0]),
        "notes": [
            "The current backend reconstructs an external visual hull, not true internal concavity.",
            "Hand occlusion usually removes or distorts the covered area.",
            "Best results come from a slow orbit with minimal tilt and a clean background.",
        ],
    }
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    write_report(output_dir / "report.txt", metrics)
    progress("Finished")

    return PipelineResult(
        output_dir=output_dir,
        obj_path=obj_path,
        stl_path=stl_path,
        preview_path=preview_path,
        metrics_path=metrics_path,
    )


def load_video(video_path: Path, *, long_edge: int) -> list[np.ndarray]:
    cap = cv2.VideoCapture(str(video_path))
    frames: list[np.ndarray] = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(resize_long_edge(frame, long_edge))
    cap.release()
    if not frames:
        raise RuntimeError(f"Could not read frames from {video_path}")
    return frames


def resize_long_edge(frame: np.ndarray, long_edge: int) -> np.ndarray:
    height, width = frame.shape[:2]
    scale = long_edge / max(height, width)
    if scale >= 1.0:
        return frame
    size = (max(1, int(round(width * scale))), max(1, int(round(height * scale))))
    return cv2.resize(frame, size, interpolation=cv2.INTER_AREA)


def estimate_background(frames: list[np.ndarray], *, sample_count: int) -> np.ndarray:
    count = min(len(frames), sample_count)
    indices = np.linspace(0, len(frames) - 1, count, dtype=int)
    stack = np.stack([frames[index] for index in indices], axis=0)
    return np.median(stack, axis=0).astype(np.uint8)


def build_frame_metrics(frames: list[np.ndarray]) -> list[FrameMetric]:
    records: list[FrameMetric] = []
    for index, frame in enumerate(frames):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        presence_score = estimate_presence(gray)
        records.append(FrameMetric(index=index, sharpness=sharpness, presence_score=presence_score))
    return records


def estimate_presence(gray: np.ndarray) -> float:
    height, width = gray.shape
    y0, y1 = int(height * 0.18), int(height * 0.82)
    x0, x1 = int(width * 0.18), int(width * 0.82)
    crop = gray[y0:y1, x0:x1]
    return float(crop.std())


def filter_candidate_indices(metrics: list[FrameMetric], config: PipelineConfig) -> list[int]:
    start = int(len(metrics) * config.start_skip_ratio)
    end = int(len(metrics) * (1.0 - config.end_skip_ratio))
    window = metrics[start:end]
    if not window:
        return []
    scores = np.array([metric.presence_score for metric in window], dtype=np.float32)
    threshold = max(12.0, float(scores.min() + (scores.max() - scores.min()) * 0.18))
    filtered = [metric.index for metric in window if metric.sharpness >= config.blur_floor and metric.presence_score >= threshold]
    return filtered


def select_keyframe_indices(metrics: list[FrameMetric], indices: list[int], target_frames: int) -> list[int]:
    if len(indices) <= target_frames:
        return indices

    selected: list[int] = []
    chosen = [metrics[index] for index in indices]
    edges = np.linspace(0, len(chosen), target_frames + 1, dtype=int)
    used: set[int] = set()
    for left, right in zip(edges[:-1], edges[1:]):
        chunk = chosen[left:right]
        if not chunk:
            continue
        best = max(chunk, key=lambda metric: metric.sharpness)
        if best.index in used:
            continue
        used.add(best.index)
        selected.append(best.index)
    return selected


def segment_selected_frames(frames: list[np.ndarray], indices: list[int], config: PipelineConfig) -> list[FrameRecord]:
    records: list[FrameRecord] = []
    for index in indices:
        frame = frames[index]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        mask = build_mask(frame, config)
        x, y, w, h = bounding_box(mask)
        area_ratio = float(np.count_nonzero(mask) / mask.size)
        if w <= 20 or h <= 20:
            continue
        if not (config.min_foreground_ratio <= area_ratio <= config.max_foreground_ratio):
            continue
        records.append(FrameRecord(index=index, frame=frame, sharpness=sharpness, mask=mask, bbox=(x, y, w, h), area_ratio=area_ratio))
    return records


def build_mask(frame: np.ndarray, config: PipelineConfig) -> np.ndarray:
    height, width = frame.shape[:2]
    rect = (int(width * 0.08), int(height * 0.08), int(width * 0.84), int(height * 0.84))
    gc_mask = np.zeros((height, width), np.uint8)
    bg_model = np.zeros((1, 65), np.float64)
    fg_model = np.zeros((1, 65), np.float64)
    cv2.grabCut(frame, gc_mask, rect, bg_model, fg_model, config.grabcut_iterations, cv2.GC_INIT_WITH_RECT)

    mask = np.where((gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))

    before_skin = mask.copy()
    mask = cut_border_skin(frame, mask)
    if np.count_nonzero(mask) < np.count_nonzero(before_skin) * 0.25:
        mask = before_skin

    mask = keep_best_component(mask)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    mask = fill_mask_holes(mask)

    if config.mask_mode == "full":
        return mask
    return extract_conservative_core(mask)


def keep_best_component(mask: np.ndarray) -> np.ndarray:
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)
    if count <= 1:
        return mask

    height, width = mask.shape[:2]
    image_center = np.array([width / 2.0, height / 2.0], dtype=np.float32)
    best_index = 0
    best_score = float("-inf")
    for label in range(1, count):
        area = float(stats[label, cv2.CC_STAT_AREA])
        if area < 1000:
            continue
        cx, cy = centroids[label]
        x, y, w, h, _ = stats[label]
        border_touch = int(x <= 0 or y <= 0 or x + w >= width - 1 or y + h >= height - 1)
        distance = float(np.linalg.norm(np.array([cx, cy], dtype=np.float32) - image_center))
        score = area - distance * 240.0 - border_touch * 5000.0
        if score > best_score:
            best_index = label
            best_score = score

    if best_index == 0:
        return np.zeros_like(mask)
    out = np.zeros_like(mask)
    out[labels == best_index] = 255
    return out


def cut_border_skin(frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
    if np.count_nonzero(mask) == 0:
        return mask

    ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
    skin = cv2.inRange(ycrcb, np.array([0, 133, 77], dtype=np.uint8), np.array([255, 173, 127], dtype=np.uint8))
    skin = cv2.morphologyEx(skin, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    skin = cv2.dilate(skin, np.ones((9, 9), np.uint8), iterations=1)

    height, width = skin.shape
    border = np.zeros_like(skin)
    margin = 8
    border[:margin, :] = skin[:margin, :]
    border[-margin:, :] = skin[-margin:, :]
    border[:, :margin] = skin[:, :margin]
    border[:, -margin:] = skin[:, -margin:]

    count, labels, _, _ = cv2.connectedComponentsWithStats(skin)
    cut = np.zeros_like(skin)
    for label in range(1, count):
        component = labels == label
        if np.any(border[component] > 0):
            cut[component] = 255

    cut = cv2.dilate(cut, np.ones((13, 13), np.uint8), iterations=1)
    cleaned = mask.copy()
    cleaned[cut > 0] = 0
    return cleaned


def extract_conservative_core(mask: np.ndarray) -> np.ndarray:
    if np.count_nonzero(mask) == 0:
        return mask

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
    eroded = cv2.erode(mask, kernel, iterations=1)
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(eroded)
    if count <= 1:
        return mask

    height, width = mask.shape[:2]
    center = np.array([width / 2.0, height / 2.0], dtype=np.float32)
    best_index = 0
    best_score = float("-inf")
    for label in range(1, count):
        x, y, w, h, area = stats[label]
        if area < 500:
            continue
        if x <= 2 or y <= 2 or x + w >= width - 2 or y + h >= height - 2:
            continue
        distance = float(np.linalg.norm(np.array(centroids[label], dtype=np.float32) - center))
        score = float(area) - distance * 180.0
        if score > best_score:
            best_index = label
            best_score = score

    if best_index == 0:
        return mask

    core = np.zeros_like(mask)
    core[labels == best_index] = 255
    growth_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    for _ in range(8):
        grown = cv2.dilate(core, growth_kernel, iterations=1)
        core = cv2.bitwise_and(grown, mask)

    if np.count_nonzero(core) < np.count_nonzero(mask) * 0.18:
        return mask
    return fill_mask_holes(core)


def fill_mask_holes(mask: np.ndarray) -> np.ndarray:
    filled = ndi.binary_fill_holes(mask > 0)
    return (filled.astype(np.uint8)) * 255


def bounding_box(mask: np.ndarray) -> tuple[int, int, int, int]:
    ys, xs = np.nonzero(mask > 0)
    if len(xs) == 0:
        return (0, 0, 0, 0)
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    return (x0, y0, x1 - x0 + 1, y1 - y0 + 1)


def normalize_views(records: list[FrameRecord], canvas_size: int) -> tuple[list[np.ndarray], list[np.ndarray]]:
    extents = [max(record.bbox[2], record.bbox[3]) for record in records if record.bbox[2] > 0 and record.bbox[3] > 0]
    if not extents:
        raise RuntimeError("No non-empty masks were selected.")
    target_extent = int(canvas_size * 0.74)
    reference_extent = float(np.median(extents))

    normalized_masks: list[np.ndarray] = []
    normalized_frames: list[np.ndarray] = []
    canvas_center = np.array([canvas_size / 2.0, canvas_size / 2.0], dtype=np.float32)

    for record in records:
        x, y, w, h = record.bbox
        if w == 0 or h == 0:
            continue
        ys, xs = np.nonzero(record.mask > 0)
        centroid = np.array([xs.mean(), ys.mean()], dtype=np.float32)
        scale = target_extent / max(reference_extent, 1.0)
        matrix = np.array(
            [
                [scale, 0.0, canvas_center[0] - centroid[0] * scale],
                [0.0, scale, canvas_center[1] - centroid[1] * scale],
            ],
            dtype=np.float32,
        )
        mask_canvas = cv2.warpAffine(record.mask, matrix, (canvas_size, canvas_size), flags=cv2.INTER_NEAREST)
        frame_canvas = cv2.warpAffine(record.frame, matrix, (canvas_size, canvas_size), flags=cv2.INTER_LINEAR)
        normalized_masks.append((mask_canvas > 0).astype(np.uint8) * 255)
        normalized_frames.append(frame_canvas)

    return normalized_masks, normalized_frames


def prune_duplicate_views(
    normalized_masks: list[np.ndarray],
    normalized_frames: list[np.ndarray],
    records: list[FrameRecord],
) -> tuple[list[np.ndarray], list[np.ndarray], list[FrameRecord]]:
    if len(normalized_masks) <= 8:
        return normalized_masks, normalized_frames, records

    kept_masks: list[np.ndarray] = []
    kept_frames: list[np.ndarray] = []
    kept_records: list[FrameRecord] = []
    previous_mask: np.ndarray | None = None
    for mask, frame, record in zip(normalized_masks, normalized_frames, records):
        if previous_mask is not None:
            intersection = np.count_nonzero((mask > 0) & (previous_mask > 0))
            union = np.count_nonzero((mask > 0) | (previous_mask > 0))
            iou = intersection / max(union, 1)
            if iou > 0.97:
                continue
        kept_masks.append(mask)
        kept_frames.append(frame)
        kept_records.append(record)
        previous_mask = mask
    return kept_masks, kept_frames, kept_records


def carve_visual_hull(masks: list[np.ndarray], resolution: int, *, consensus_ratio: float) -> np.ndarray:
    if len(masks) < 4:
        raise RuntimeError("At least four masks are needed for carving.")

    canvas_size = masks[0].shape[0]
    grid = np.linspace(-1.0, 1.0, resolution, dtype=np.float32)
    x, y, z = np.meshgrid(grid, grid, grid, indexing="xy")
    votes = np.zeros((resolution, resolution, resolution), dtype=np.uint16)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))

    for index, mask in enumerate(masks):
        dilated = cv2.dilate(mask, kernel, iterations=1)
        angle = 2.0 * np.pi * index / len(masks)
        xr = np.cos(angle) * x + np.sin(angle) * z
        yr = y

        u = np.clip(((xr + 1.0) * 0.5 * (canvas_size - 1)).astype(np.int32), 0, canvas_size - 1)
        v = np.clip(((1.0 - (yr + 1.0) * 0.5) * (canvas_size - 1)).astype(np.int32), 0, canvas_size - 1)
        votes += (dilated[v, u] > 0).astype(np.uint16)

    required = max(1, int(np.ceil(len(masks) * consensus_ratio)))
    return votes >= required


def keep_largest_component(volume: np.ndarray) -> np.ndarray:
    labels, count = ndi.label(volume)
    if count <= 1:
        return volume
    areas = ndi.sum(volume, labels, index=np.arange(1, count + 1))
    keep = int(np.argmax(areas) + 1)
    return labels == keep


def marching_cubes_to_mesh(volume: np.ndarray, *, level: float) -> tuple[np.ndarray, np.ndarray]:
    vertices, faces, _, _ = measure.marching_cubes(volume, level=level)
    vertices = vertices.astype(np.float32)
    vertices /= max(volume.shape[0] - 1, 1)
    vertices = vertices * 2.0 - 1.0
    return vertices, faces.astype(np.int32)


def infer_depth_scale(masks: list[np.ndarray]) -> float:
    if not masks:
        return 1.0

    minor_extents: list[int] = []
    major_extents: list[int] = []
    for mask in masks:
        x, y, w, h = bounding_box(mask)
        if w <= 0 or h <= 0:
            continue
        minor_extents.append(min(w, h))
        major_extents.append(max(w, h))

    if not minor_extents or not major_extents:
        return 1.0

    minor = float(np.percentile(minor_extents, 20))
    major = float(np.percentile(major_extents, 80))
    ratio = minor / max(major, 1.0)
    return float(np.clip(ratio, 0.35, 1.0))


def scale_vertices(vertices: np.ndarray, *, target_mm: float) -> np.ndarray:
    minimum = vertices.min(axis=0)
    maximum = vertices.max(axis=0)
    extent = maximum - minimum
    longest = float(extent.max())
    if longest <= 0.0:
        return vertices
    scaled = vertices / longest * target_mm
    minimum = scaled.min(axis=0)
    maximum = scaled.max(axis=0)
    centered = scaled - (minimum + maximum) / 2.0
    return centered


def export_obj(vertices: np.ndarray, faces: np.ndarray, path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write("# Generated by video-to-3d-model\n")
        for vertex in vertices:
            handle.write(f"v {vertex[0]:.6f} {vertex[1]:.6f} {vertex[2]:.6f}\n")
        for face in faces:
            handle.write(f"f {face[0] + 1} {face[1] + 1} {face[2] + 1}\n")


def export_stl_ascii(vertices: np.ndarray, faces: np.ndarray, path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write("solid video_to_3d_model\n")
        for face in faces:
            p0, p1, p2 = vertices[face]
            normal = np.cross(p1 - p0, p2 - p0)
            norm = float(np.linalg.norm(normal))
            if norm > 1e-8:
                normal /= norm
            else:
                normal = np.array([0.0, 0.0, 0.0], dtype=np.float32)
            handle.write(f"  facet normal {normal[0]:.6f} {normal[1]:.6f} {normal[2]:.6f}\n")
            handle.write("    outer loop\n")
            handle.write(f"      vertex {p0[0]:.6f} {p0[1]:.6f} {p0[2]:.6f}\n")
            handle.write(f"      vertex {p1[0]:.6f} {p1[1]:.6f} {p1[2]:.6f}\n")
            handle.write(f"      vertex {p2[0]:.6f} {p2[1]:.6f} {p2[2]:.6f}\n")
            handle.write("    endloop\n")
            handle.write("  endfacet\n")
        handle.write("endsolid video_to_3d_model\n")


def render_mesh_preview(vertices: np.ndarray, faces: np.ndarray, path: Path) -> None:
    figure = plt.figure(figsize=(8, 8), dpi=150)
    axis = figure.add_subplot(111, projection="3d")

    mesh = vertices[faces]
    depth = mesh[:, :, 2].mean(axis=1)
    color_values = (depth - depth.min()) / max(depth.max() - depth.min(), 1e-6)
    colors = cm.copper(color_values)

    collection = Poly3DCollection(mesh, facecolors=colors, linewidths=0.02, alpha=1.0)
    collection.set_edgecolor((0.08, 0.08, 0.08, 0.12))
    axis.add_collection3d(collection)

    mins = vertices.min(axis=0)
    maxs = vertices.max(axis=0)
    center = (mins + maxs) / 2.0
    radius = float((maxs - mins).max() * 0.56)
    axis.set_xlim(center[0] - radius, center[0] + radius)
    axis.set_ylim(center[1] - radius, center[1] + radius)
    axis.set_zlim(center[2] - radius, center[2] + radius)
    axis.view_init(elev=18, azim=36)
    axis.set_axis_off()
    figure.tight_layout(pad=0)
    figure.savefig(path, bbox_inches="tight", pad_inches=0)
    plt.close(figure)


def save_contact_sheet(images: list[np.ndarray], path: Path, *, columns: int) -> None:
    if not images:
        return
    height, width = images[0].shape[:2]
    rows = int(np.ceil(len(images) / columns))
    canvas = np.zeros((rows * height, columns * width, 3), dtype=np.uint8)
    for index, image in enumerate(images):
        row = index // columns
        col = index % columns
        canvas[row * height : (row + 1) * height, col * width : (col + 1) * width] = ensure_bgr(image)
    cv2.imwrite(str(path), canvas)


def ensure_bgr(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    return image


def mask_to_bgr(mask: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)


def build_overlay_frames(frames: list[np.ndarray], masks: list[np.ndarray]) -> list[np.ndarray]:
    overlays: list[np.ndarray] = []
    for frame, mask in zip(frames, masks):
        overlay = frame.copy()
        overlay[mask > 0] = (0.35 * overlay[mask > 0] + 0.65 * np.array([0, 255, 0])).astype(np.uint8)
        overlays.append(overlay)
    return overlays


def write_report(path: Path, metrics: dict) -> None:
    lines = [
        "Video To 3D run report",
        "",
        f"Video: {metrics['video_path']}",
        f"Frames total: {metrics['frames_total']}",
        f"Candidate frames: {metrics['candidate_frames']}",
        f"Selected frames: {metrics['selected_frames']}",
        f"Mesh vertices: {metrics['mesh_vertices']}",
        f"Mesh faces: {metrics['mesh_faces']}",
        "",
        "Current interpretation:",
        "- This mesh is an external silhouette hull, not a metrically exact reconstruction.",
        "- Hand contact, missing back views, and strong tilt will make the result blobbier.",
        "- Use selected_overlays.png first to judge whether the silhouette tracking is good enough.",
        "",
        "Recommended next recording pass:",
        "- Keep the object centered and fully visible for the whole clip.",
        "- Use a neutral stand or support instead of a hand if possible.",
        "- Rotate mainly around one axis before adding tilt shots.",
        "- Add matte texture if the object is smooth or reflective.",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
