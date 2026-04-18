from __future__ import annotations

import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from video_to_3d.pipeline import PipelineConfig, run_pipeline


class App(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Video To 3D Model")
        self.geometry("760x420")

        self.video_var = tk.StringVar()
        self.output_var = tk.StringVar()
        self.frames_var = tk.IntVar(value=24)
        self.mask_mode_var = tk.StringVar(value="core")
        self.voxel_var = tk.IntVar(value=96)
        self.size_var = tk.DoubleVar(value=120.0)
        self.status_var = tk.StringVar(value="Choose a video to start.")

        self._build()

    def _build(self) -> None:
        root = ttk.Frame(self, padding=16)
        root.pack(fill="both", expand=True)
        root.columnconfigure(1, weight=1)

        ttk.Label(root, text="Video").grid(row=0, column=0, sticky="w", pady=4)
        ttk.Entry(root, textvariable=self.video_var).grid(row=0, column=1, sticky="ew", padx=8)
        ttk.Button(root, text="Browse", command=self._pick_video).grid(row=0, column=2, sticky="ew")

        ttk.Label(root, text="Output").grid(row=1, column=0, sticky="w", pady=4)
        ttk.Entry(root, textvariable=self.output_var).grid(row=1, column=1, sticky="ew", padx=8)
        ttk.Button(root, text="Browse", command=self._pick_output).grid(row=1, column=2, sticky="ew")

        ttk.Label(root, text="Frames").grid(row=2, column=0, sticky="w", pady=4)
        ttk.Spinbox(root, from_=8, to=64, textvariable=self.frames_var, width=10).grid(row=2, column=1, sticky="w", padx=8)

        ttk.Label(root, text="Mask mode").grid(row=3, column=0, sticky="w", pady=4)
        ttk.Combobox(root, textvariable=self.mask_mode_var, values=["core", "full"], state="readonly").grid(row=3, column=1, sticky="w", padx=8)

        ttk.Label(root, text="Voxel resolution").grid(row=4, column=0, sticky="w", pady=4)
        ttk.Spinbox(root, from_=48, to=160, textvariable=self.voxel_var, width=10).grid(row=4, column=1, sticky="w", padx=8)

        ttk.Label(root, text="Longest side (mm)").grid(row=5, column=0, sticky="w", pady=4)
        ttk.Entry(root, textvariable=self.size_var, width=12).grid(row=5, column=1, sticky="w", padx=8)

        ttk.Button(root, text="Run", command=self._run).grid(row=6, column=0, columnspan=3, sticky="ew", pady=(12, 12))

        status = ttk.Label(root, textvariable=self.status_var, justify="left", anchor="nw")
        status.grid(row=7, column=0, columnspan=3, sticky="nsew")
        root.rowconfigure(7, weight=1)

    def _pick_video(self) -> None:
        path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4;*.mov;*.avi;*.mkv"), ("All files", "*.*")])
        if not path:
            return
        self.video_var.set(path)
        if not self.output_var.get():
            self.output_var.set(str(Path(path).resolve().parent / "outputs" / Path(path).stem))

    def _pick_output(self) -> None:
        path = filedialog.askdirectory()
        if path:
            self.output_var.set(path)

    def _run(self) -> None:
        video = self.video_var.get().strip()
        output = self.output_var.get().strip()
        if not video:
            messagebox.showerror("Missing video", "Choose a video first.")
            return
        if not output:
            messagebox.showerror("Missing output", "Choose an output folder first.")
            return

        config = PipelineConfig(
            target_frames=self.frames_var.get(),
            mask_mode=self.mask_mode_var.get(),
            voxel_resolution=self.voxel_var.get(),
            size_mm=self.size_var.get(),
        )

        worker = threading.Thread(
            target=self._run_worker,
            args=(Path(video).expanduser().resolve(), Path(output).expanduser().resolve(), config),
            daemon=True,
        )
        worker.start()

    def _run_worker(self, video: Path, output: Path, config: PipelineConfig) -> None:
        def progress(message: str) -> None:
            self.after(0, lambda: self.status_var.set(message))

        try:
            result = run_pipeline(video, output, config=config, progress=progress)
        except Exception as exc:  # noqa: BLE001
            self.after(0, lambda: messagebox.showerror("Pipeline failed", str(exc)))
            self.after(0, lambda: self.status_var.set("Pipeline failed."))
            return

        self.after(0, lambda: self.status_var.set(f"Done. Preview: {result.preview_path}"))
        self.after(0, lambda: messagebox.showinfo("Finished", f"Output written to:\n{result.output_dir}"))


def launch_gui() -> None:
    app = App()
    app.mainloop()
