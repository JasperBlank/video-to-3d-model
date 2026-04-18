# Video To 3D Model

This project turns a handheld object video into a coarse 3D printable mesh using only libraries that are available locally in this environment.

The current backend is silhouette based:

1. It scores the clip for sharp, object-heavy frames.
2. It segments the selected frames with GrabCut plus simple hand suppression.
3. It normalizes the silhouettes to a shared canvas.
4. It builds a conservative visual hull voxel volume.
5. It converts the volume to a mesh with marching cubes.
6. It exports `OBJ` and `STL`.

This works best when:

- the object stays roughly centered
- the background is static
- the camera or object performs a smooth orbit
- the object is solid and mostly convex
- the hand covers as little of the object as possible

Known limits:

- Smooth glossy objects still segment better than they reconstruct.
- Strong concavities are not fully recovered by a visual hull.
- A hand on the object will usually clip or distort the covered region.
- Random tilts are less reliable than a clean turntable style orbit.

## Install

```bash
cd video-to-3d-model
pip install -r requirements.txt
```

## CLI

```bash
python main.py --video "C:\path\to\object.mp4"
python main.py --video "C:\path\to\object.mp4" --output outputs\run1 --mask-mode core --voxel-resolution 96
```

Useful flags:

- `--mask-mode full`: keep more of the silhouette, including more risk from hands
- `--mask-mode core`: conservative mask, usually better for handheld videos
- `--target-frames 24`: number of frames used for reconstruction
- `--size-mm 120`: scale longest mesh dimension to a target print size
- `--grabcut-iterations 3`: spend more work on segmentation
- `--consensus-ratio 0.75`: require more silhouette agreement and get a stricter hull

## GUI

```bash
python main.py
```

The GUI is intentionally simple: choose a video, choose an output folder, and run the pipeline.

## Output

Each run writes:

- `background.png`
- `selected_frames.png`
- `selected_masks.png`
- `selected_overlays.png`
- `mesh_preview.png`
- `mesh.obj`
- `mesh.stl`
- `metrics.json`
