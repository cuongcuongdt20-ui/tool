import os
import re
import sys
import json
import math
import time
import queue
import argparse
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import numpy as np
import cv2
from PIL import Image, ImageTk

from zoom_canvas import ZoomPanView

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)
ARCHIVE_DIR = os.path.join(PROJECT_DIR, "archive")  # shared with the labeler
IMG_EXTS = (".jpg", ".jpeg", ".png")
VIDEO_TYPES = [
    ("Video Files", "*.mp4;*.avi;*.mkv;*.mov;*.webm;*.flv;*.ts;*.m4v"),
    ("MP4 Video", "*.mp4"),
    ("All Files", "*.*")
]
SEEK_THRESHOLD = 90  # frame gap above which seeking is cheaper than sequential grab()
HANDLE_PX = 7

ASPECT_OUTPUT = "Theo kích thước xuất"
ASPECT_FREE = "Tự do (Free)"
ASPECT_VALUES = {"1:1 (Vuông)": 1.0, "16:9": 16 / 9, "9:16": 9 / 16, "4:3": 4 / 3, "3:4": 3 / 4}
ASPECT_CHOICES = [ASPECT_OUTPUT, ASPECT_FREE] + list(ASPECT_VALUES)


# =====================================================================
# Core helpers (shared by GUI and CLI)
# =====================================================================
def sanitize_clip_name(name):
    """ASCII-only snippet id (safe for OpenCV paths and CSV)."""
    return re.sub(r"[^A-Za-z0-9_\-]+", "_", name.strip()).strip("_")


def clamp_roi(roi, vw, vh):
    x, y, w, h = roi
    x = int(round(max(0, min(vw - 1, x))))
    y = int(round(max(0, min(vh - 1, y))))
    w = int(round(max(1, min(vw - x, w))))
    h = int(round(max(1, min(vh - y, h))))
    return [x, y, w, h]


def zoom_roi(roi, factor, vw, vh, center=None):
    """Shrink the ROI by `factor` around its centre (or `center`) -> digital zoom."""
    x, y, w, h = roi
    cx, cy = center if center else (x + w / 2, y + h / 2)
    nw, nh = max(2.0, w / factor), max(2.0, h / factor)
    nx = max(0.0, min(vw - nw, cx - nw / 2))
    ny = max(0.0, min(vh - nh, cy - nh / 2))
    return [nx, ny, nw, nh]


def largest_rect(vw, vh, aspect=None, center=None):
    """Largest rect with the given aspect ratio that fits the frame, centred on `center`."""
    if not aspect:
        return [0, 0, vw, vh]
    w, h = (vh * aspect, vh) if vw / vh >= aspect else (vw, vw / aspect)
    cx, cy = center if center else (vw / 2, vh / 2)
    return [max(0.0, min(vw - w, cx - w / 2)), max(0.0, min(vh - h, cy - h / 2)), w, h]


def compute_frame_indices(start, end, mode="all", step=1, src_fps=30.0, target_fps=None):
    if end < start:
        return []
    if mode == "step":
        return list(range(start, end + 1, max(1, int(step))))
    if mode == "fps" and target_fps and src_fps > 0 and target_fps < src_fps:
        stride = src_fps / float(target_fps)
        n = int((end - start) / stride) + 1
        return sorted({start + int(round(i * stride)) for i in range(n) if start + int(round(i * stride)) <= end})
    return list(range(start, end + 1))


def render_crop(frame, roi, target_size=None, keep_aspect=False):
    """Crop ROI from a BGR frame and resize to target_size (stretch, or letterbox if keep_aspect)."""
    fh, fw = frame.shape[:2]
    x, y, w, h = clamp_roi(roi, fw, fh)
    crop = frame[y:y + h, x:x + w]
    if target_size is None:
        return crop
    tw, th = target_size
    if keep_aspect:
        s = min(tw / w, th / h)
        nw, nh = max(1, int(round(w * s))), max(1, int(round(h * s)))
        resized = cv2.resize(crop, (nw, nh), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC)
        out = np.zeros((th, tw, 3), dtype=frame.dtype)
        ox, oy = (tw - nw) // 2, (th - nh) // 2
        out[oy:oy + nh, ox:ox + nw] = resized
        return out
    downscale = tw <= w and th <= h
    return cv2.resize(crop, (tw, th), interpolation=cv2.INTER_AREA if downscale else cv2.INTER_CUBIC)


def imwrite_any(path, img, img_format="jpg", quality=95):
    """cv2.imwrite replacement that also works with non-ASCII Windows paths."""
    if img_format == "png":
        ok, buf = cv2.imencode(".png", img, [cv2.IMWRITE_PNG_COMPRESSION, 3])
    else:
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
    if ok:
        buf.tofile(path)
    return ok


def probe_video(path):
    """Return (fps, width, height, frame_count) or None. Counts frames if the container lies."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return None
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    if fps <= 0 or fps > 1000:
        fps = 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:  # some webm/stream files report 0 -> count manually
        total = 0
        while cap.grab():
            total += 1
    cap.release()
    return fps, w, h, total


def list_frame_files(folder):
    if not os.path.isdir(folder):
        return []
    return sorted(f for f in os.listdir(folder) if f.lower().endswith(IMG_EXTS))


def prepare_output_dir(output_dir, write_mode="overwrite"):
    """Create the folder; 'overwrite' removes old frames, 'append' continues numbering. Returns start number."""
    os.makedirs(output_dir, exist_ok=True)
    existing = list_frame_files(output_dir)
    if write_mode == "append":
        nums = [int(os.path.splitext(f)[0]) for f in existing if os.path.splitext(f)[0].isdigit()]
        return max(nums, default=0) + 1
    for f in existing:
        os.remove(os.path.join(output_dir, f))
    return 1


def extract_frames(video_path, output_dir, roi, frame_indices, target_size=None, keep_aspect=False,
                   img_format="jpg", quality=95, start_number=1, progress_cb=None, cancel_cb=None):
    """
    Read the video sequentially (grab() skips unwanted frames, seek only on large gaps),
    crop + resize and write 000001.jpg, 000002.jpg, ...
    Returns {"saved", "mapping": [(file_name, source_frame_idx)], "elapsed"}.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Không mở được video: {video_path}")

    indices = sorted(set(frame_indices))
    ext = "png" if img_format == "png" else "jpg"
    mapping = []
    pos = None
    t0 = time.time()
    total = len(indices)

    for i, idx in enumerate(indices):
        if cancel_cb and cancel_cb():
            break
        if pos is None or idx < pos or idx - pos > SEEK_THRESHOLD:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            pos = idx
        while pos < idx and cap.grab():
            pos += 1
        ok, frame = cap.read()
        pos += 1
        if not ok or frame is None:
            continue

        name = f"{start_number + len(mapping):06d}.{ext}"
        img = render_crop(frame, roi, target_size, keep_aspect)
        if imwrite_any(os.path.join(output_dir, name), img, img_format, quality):
            mapping.append((name, idx))
        if progress_cb:
            progress_cb(i + 1, total, len(mapping), time.time() - t0)

    cap.release()
    return {"saved": len(mapping), "mapping": mapping, "elapsed": time.time() - t0}


def register_unlabeled(clip_name, archive_dir=ARCHIVE_DIR):
    """Append clip_name to unlabeled.csv once (no duplicates, no blank lines)."""
    csv_path = os.path.join(archive_dir, "unlabeled.csv")
    existing = []
    if os.path.exists(csv_path):
        with open(csv_path, "r", encoding="utf-8") as f:
            existing = [line.strip() for line in f if line.strip()]
    if clip_name in existing:
        return False
    needs_newline = False
    if os.path.exists(csv_path) and os.path.getsize(csv_path) > 0:
        with open(csv_path, "rb") as f:
            f.seek(-1, os.SEEK_END)
            needs_newline = f.read(1) not in (b"\n", b"\r")
    with open(csv_path, "a", encoding="utf-8") as f:
        if needs_newline:
            f.write("\n")
        f.write(f"{clip_name}\n")
    return True


def write_extract_log(archive_dir, job, result):
    """Keep a record of how each clip was produced (source video, ROI, source frame per output file)."""
    log_dir = os.path.join(archive_dir, "extract_logs")
    os.makedirs(log_dir, exist_ok=True)
    path = os.path.join(log_dir, f"{job['clip_name']}.json")
    frames = []
    if job.get("write_mode") == "append" and os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                frames = json.load(f).get("frames", [])
        except (OSError, ValueError):
            frames = []
    info = {
        "clip_name": job["clip_name"],
        "video": os.path.abspath(job["video_path"]),
        "video_fps": job.get("video_fps"),
        "roi_xywh": [int(v) for v in job["roi"]],
        "output_size": list(job["target_size"]) if job["target_size"] else "original",
        "keep_aspect": job.get("keep_aspect", False),
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "frames": frames + [[name, idx] for name, idx in result["mapping"]],
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=1)


def run_job(job, progress_cb=None, cancel_cb=None, archive_dir=ARCHIVE_DIR):
    start_number = prepare_output_dir(job["output_dir"], job.get("write_mode", "overwrite"))
    result = extract_frames(
        job["video_path"], job["output_dir"], job["roi"], job["frame_indices"],
        target_size=job["target_size"], keep_aspect=job.get("keep_aspect", False),
        img_format=job.get("img_format", "jpg"), quality=job.get("quality", 95),
        start_number=start_number, progress_cb=progress_cb, cancel_cb=cancel_cb
    )
    result["cancelled"] = bool(cancel_cb and cancel_cb())
    result["registered"] = False
    if result["saved"] > 0:
        write_extract_log(archive_dir, job, result)
        if job.get("add_unlabeled") and not result["cancelled"]:
            result["registered"] = register_unlabeled(job["clip_name"], archive_dir)
    return result


# =====================================================================
# GUI
# =====================================================================
class ScrollFrame(tk.Frame):
    """Vertical scrollable container (settings panel)."""

    def __init__(self, parent, bg):
        super().__init__(parent, bg=bg)
        self.canvas = tk.Canvas(self, bg=bg, highlightthickness=0)
        sb = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.inner = tk.Frame(self.canvas, bg=bg)
        self._win = self.canvas.create_window(0, 0, window=self.inner, anchor="nw")
        self.canvas.configure(yscrollcommand=sb.set)
        self.canvas.pack(side="left", fill="both", expand=True)
        sb.pack(side="right", fill="y")
        self.inner.bind("<Configure>", lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
        self.canvas.bind("<Configure>", lambda e: self.canvas.itemconfigure(self._win, width=e.width))
        self.bind_all("<MouseWheel>", self._on_wheel, add="+")

    def _on_wheel(self, event):
        w = self.winfo_containing(event.x_root, event.y_root)
        if w is None or isinstance(w, tk.Listbox) or not str(w).startswith(str(self)):
            return
        self.canvas.yview_scroll(int(-event.delta / 120), "units")


class VideoFrameExtractorGUI:
    def __init__(self, root, initial_video=None):
        self.root = root
        self.root.title("🎬 Video Frame Extractor & Dataset Creator")
        self.root.geometry("1400x900")
        self.root.minsize(1150, 720)
        self.root.configure(bg="#0f172a")

        self.style = ttk.Style()
        self.style.theme_use("clam")
        self.setup_styles()

        # Video state
        self.video_path = None
        self.cap = None
        self._cap_next = -1  # frame index the capture will return on the next read()
        self.total_frames = 0
        self.video_fps = 30.0
        self.video_w = 0
        self.video_h = 0
        self.current_frame_idx = 0
        self.current_bgr = None
        self.current_pil = None

        # Playback
        self.is_playing = False
        self.play_job = None
        self._slider_lock = False
        self._seek_job = None
        self._pending_seek = 0

        # ROI [x, y, w, h] in video pixels (floats while dragging)
        self.roi = None
        self.drag = None
        self._cursor = "crosshair"
        self._preview_job = None
        self._preview_tk = None

        # Trim range
        self.start_frame = 0
        self.end_frame = 0

        # Queue + worker
        self.jobs = []
        self.is_extracting = False
        self.cancel_requested = False
        self.msgq = queue.Queue()

        self.create_ui()
        self.bind_shortcuts()

        if initial_video and os.path.exists(initial_video):
            self.load_video(initial_video)

    def setup_styles(self):
        self.style.configure(".", background="#0f172a", foreground="#f8fafc", font=("Inter", 9))
        self.style.configure("TFrame", background="#0f172a")
        self.style.configure("TLabel", background="#0f172a", foreground="#f8fafc", font=("Inter", 9))
        self.style.configure("Horizontal.TProgressbar", background="#06b6d4", troughcolor="#0f172a", thickness=10)

    # ------------------------------------------------------------------ widget helpers
    @staticmethod
    def btn(parent, text, cmd, bg="#334155", fg="#fff", bold=False, **kw):
        return tk.Button(parent, text=text, font=("Inter", 8, "bold" if bold else "normal"), bg=bg, fg=fg,
                         activebackground="#475569", activeforeground="#fff", relief="flat", command=cmd, **kw)

    @staticmethod
    def section(parent, title):
        sec = tk.LabelFrame(parent, text=f" {title} ", font=("Inter", 9, "bold"), fg="#38bdf8", bg="#1e293b", bd=1, relief="solid")
        sec.pack(fill="x", padx=10, pady=(8, 0))
        return sec

    @staticmethod
    def entry(parent, value, width=6, **kw):
        e = tk.Entry(parent, bg="#0f172a", fg="#fff", insertbackground="#fff", width=width, font=("Inter", 8), relief="flat", **kw)
        e.insert(0, str(value))
        return e

    def radio(self, parent, text, var, value, cmd=None):
        return tk.Radiobutton(parent, text=text, variable=var, value=value, bg="#1e293b", fg="#f8fafc", selectcolor="#0f172a",
                              activebackground="#1e293b", activeforeground="#fff", font=("Inter", 8), command=cmd)

    def check(self, parent, text, var, fg="#94a3b8", cmd=None):
        return tk.Checkbutton(parent, text=text, variable=var, bg="#1e293b", fg=fg, selectcolor="#0f172a",
                              activebackground="#1e293b", activeforeground="#fff", font=("Inter", 8), command=cmd)

    # ------------------------------------------------------------------ layout
    def create_ui(self):
        # 1. TOP TOOLBAR
        top_bar = tk.Frame(self.root, bg="#1e293b")
        top_bar.pack(side="top", fill="x", padx=12, pady=(10, 6))
        brand = tk.Frame(top_bar, bg="#1e293b")
        brand.pack(side="left", padx=12, pady=8)
        tk.Label(brand, text="🎬 Video Frame Extractor", font=("Inter", 12, "bold"), fg="#f8fafc", bg="#1e293b").pack(side="left")
        tk.Label(brand, text=" | Dataset ROI Creator", font=("Inter", 9), fg="#94a3b8", bg="#1e293b").pack(side="left", padx=4)
        tk.Button(top_bar, text="📁 Chọn Video...", font=("Inter", 9, "bold"), bg="#4f46e5", fg="#fff", relief="flat", padx=12, pady=4, command=self.browse_video).pack(side="left", padx=14, pady=8)
        info = tk.Frame(top_bar, bg="#1e293b")
        info.pack(side="right", padx=12, pady=8)
        self.lbl_video_name = tk.Label(info, text="Chưa mở video nào", font=("Inter", 9, "italic"), fg="#94a3b8", bg="#1e293b")
        self.lbl_video_name.pack(side="left", padx=6)
        self.lbl_video_meta = tk.Label(info, text="", font=("Inter", 9, "bold"), fg="#38bdf8", bg="#1e293b")
        self.lbl_video_meta.pack(side="left", padx=6)

        # 2. BODY
        body = tk.Frame(self.root, bg="#0f172a")
        body.pack(side="top", fill="both", expand=True, padx=12, pady=(0, 10))
        body.columnconfigure(0, weight=1)
        body.rowconfigure(0, weight=1)

        center = tk.Frame(body, bg="#0f172a")
        center.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
        center.rowconfigure(1, weight=1)
        center.columnconfigure(0, weight=1)

        # View toolbar
        vbar = tk.Frame(center, bg="#1e293b")
        vbar.grid(row=0, column=0, sticky="ew", pady=(0, 6))
        for txt, cmd in (("−", lambda: self.view.zoom_center(1 / 1.25)), ("+", lambda: self.view.zoom_center(1.25)),
                         ("Vừa khung (F)", self.fit_view), ("🔍 Soi vùng cắt (Z)", self.zoom_view_to_roi)):
            self.btn(vbar, txt, cmd, padx=8).pack(side="left", padx=(6 if txt == "−" else 1, 1), pady=5)
        tk.Label(vbar, text="Kéo chuột trái: vẽ / di chuyển / chỉnh vùng cắt • Double-click: đặt tâm vùng cắt • Lăn chuột: zoom • Chuột phải kéo: di chuyển ảnh",
                 font=("Inter", 7), fg="#64748b", bg="#1e293b").pack(side="left", padx=10)
        self.lbl_view_info = tk.Label(vbar, text="", font=("Inter", 8), fg="#94a3b8", bg="#1e293b")
        self.lbl_view_info.pack(side="right", padx=8)

        canvas_card = tk.Frame(center, bg="#020617", bd=1, relief="solid")
        canvas_card.grid(row=1, column=0, sticky="nsew")
        self.canvas = tk.Canvas(canvas_card, bg="#020617", highlightthickness=0, cursor="crosshair", takefocus=1)
        self.canvas.pack(fill="both", expand=True)
        self.view = ZoomPanView(self.canvas, on_change=self.render, on_shift_wheel=self.step_frame)
        self.canvas.bind("<Configure>", self.on_canvas_resize)
        self.canvas.bind("<ButtonPress-1>", self.on_canvas_press)
        self.canvas.bind("<B1-Motion>", self.on_canvas_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_canvas_release)
        self.canvas.bind("<Double-Button-1>", self.on_canvas_double)
        self.canvas.bind("<Motion>", self.on_canvas_motion)

        # Playbar
        playbar = tk.Frame(center, bg="#1e293b")
        playbar.grid(row=2, column=0, sticky="ew", pady=(6, 0))
        ctrl = tk.Frame(playbar, bg="#1e293b")
        ctrl.pack(side="left", padx=10, pady=8)
        self.btn(ctrl, "◀◀ -10", lambda: self.step_frame(-10), width=6).pack(side="left", padx=1)
        self.btn(ctrl, "◀ -1", lambda: self.step_frame(-1), width=5).pack(side="left", padx=1)
        self.btn_play = tk.Button(ctrl, text="▶ Phát", font=("Inter", 9, "bold"), bg="#4f46e5", fg="#fff", relief="flat", width=8, command=self.toggle_play)
        self.btn_play.pack(side="left", padx=4)
        self.btn(ctrl, "+1 ▶", lambda: self.step_frame(1), width=5).pack(side="left", padx=1)
        self.btn(ctrl, "+10 ▶▶", lambda: self.step_frame(10), width=6).pack(side="left", padx=1)

        tl = tk.Frame(playbar, bg="#1e293b")
        tl.pack(side="left", fill="x", expand=True, padx=12)
        self.slider = ttk.Scale(tl, from_=0, to=0, orient="horizontal", command=self.on_slider_move)
        self.slider.pack(fill="x")
        self.trim_bar = tk.Canvas(tl, height=6, bg="#0f172a", highlightthickness=0)
        self.trim_bar.pack(fill="x", pady=(2, 0))
        self.trim_bar.bind("<Configure>", lambda e: self.draw_trim_bar())
        sub = tk.Frame(tl, bg="#1e293b")
        sub.pack(fill="x", pady=(2, 0))
        self.lbl_current_frame = tk.Label(sub, text="Frame: 0 (00:00.00)", font=("Inter", 8), fg="#f8fafc", bg="#1e293b")
        self.lbl_current_frame.pack(side="left")
        self.lbl_trim_info = tk.Label(sub, text="Phạm vi: [0 -> 0]", font=("Inter", 8, "bold"), fg="#38bdf8", bg="#1e293b")
        self.lbl_trim_info.pack(side="right")

        trim = tk.Frame(playbar, bg="#1e293b")
        trim.pack(side="right", padx=10, pady=8)
        self.btn(trim, "[ Đặt Start", self.set_start_frame).pack(side="left", padx=1)
        self.btn(trim, "Đặt End ]", self.set_end_frame).pack(side="left", padx=1)
        self.btn(trim, "↺ Toàn bộ", self.reset_trim, fg="#94a3b8").pack(side="left", padx=1)

        # 3. RIGHT PANEL: preview (pinned) + scrollable settings + action (pinned)
        right = tk.Frame(body, bg="#1e293b", width=370)
        right.grid(row=0, column=1, sticky="nsew")
        right.pack_propagate(False)

        prev_card = tk.Frame(right, bg="#1e293b")
        prev_card.pack(side="top", fill="x", padx=10, pady=(10, 0))
        tk.Label(prev_card, text="👁 Xem trước ảnh xuất", font=("Inter", 9, "bold"), fg="#38bdf8", bg="#1e293b").pack(anchor="w")
        self.preview_canvas = tk.Canvas(prev_card, bg="#020617", height=190, highlightthickness=0)
        self.preview_canvas.pack(fill="x", pady=(4, 2))
        self.preview_canvas.bind("<Configure>", lambda e: self.schedule_preview())
        self.lbl_preview_info = tk.Label(prev_card, text="", font=("Inter", 8), fg="#94a3b8", bg="#1e293b", anchor="w", justify="left")
        self.lbl_preview_info.pack(fill="x")

        action = tk.Frame(right, bg="#1e293b")
        action.pack(side="bottom", fill="x", padx=10, pady=(6, 10))
        self.progress_bar = ttk.Progressbar(action, orient="horizontal", mode="determinate", maximum=100, style="Horizontal.TProgressbar")
        self.progress_bar.pack(fill="x", pady=(2, 4))
        self.lbl_progress_status = tk.Label(action, text="Sẵn sàng trích xuất", font=("Inter", 8), fg="#94a3b8", bg="#1e293b", anchor="w")
        self.lbl_progress_status.pack(fill="x")
        row = tk.Frame(action, bg="#1e293b")
        row.pack(fill="x", pady=(6, 0))
        self.btn_extract = tk.Button(row, text="🚀 BẮT ĐẦU TÁCH FRAME", font=("Inter", 10, "bold"), bg="#10b981", fg="#fff", relief="flat", pady=8, command=self.start_extraction)
        self.btn_extract.pack(side="left", fill="x", expand=True, padx=(0, 4))
        self.btn_cancel = tk.Button(row, text="Dừng", font=("Inter", 9), bg="#334155", fg="#ef4444", relief="flat", width=6, state="disabled", command=self.cancel_extraction)
        self.btn_cancel.pack(side="right")

        scroll = ScrollFrame(right, bg="#1e293b")
        scroll.pack(side="top", fill="both", expand=True, pady=(4, 0))
        self.build_settings_panel(scroll.inner)

    def build_settings_panel(self, parent):
        # --- 1. ROI ---
        sec = self.section(parent, "🎯 1. Vùng cắt / Phóng to (ROI)")
        grid = tk.Frame(sec, bg="#1e293b")
        grid.pack(fill="x", padx=8, pady=(6, 2))
        presets = [("Toàn khung", self.preset_full_frame), ("Giữa 640x640", self.preset_center_640),
                   ("Vuông lớn nhất", self.preset_center_square), ("Lớn nhất theo tỉ lệ", self.preset_max_aspect)]
        for i, (txt, cmd) in enumerate(presets):
            self.btn(grid, txt, cmd, bg="#4f46e5" if i == 1 else "#334155", bold=i == 1).grid(row=i // 2, column=i % 2, sticky="ew", padx=2, pady=2)
        grid.columnconfigure(0, weight=1)
        grid.columnconfigure(1, weight=1)

        zrow = tk.Frame(sec, bg="#1e293b")
        zrow.pack(fill="x", padx=8, pady=(4, 2))
        tk.Label(zrow, text="Phóng to vùng cắt:", font=("Inter", 8), fg="#94a3b8", bg="#1e293b").pack(side="left", padx=2)
        for f in (1.5, 2, 3, 4):
            self.btn(zrow, f"×{f:g}", lambda f=f: self.zoom_roi_by(f), width=4).pack(side="left", padx=1)
        self.btn(zrow, "Thu ×0.5", lambda: self.zoom_roi_by(0.5), fg="#94a3b8").pack(side="left", padx=1)

        lock = tk.Frame(sec, bg="#1e293b")
        lock.pack(fill="x", padx=8, pady=(4, 2))
        tk.Label(lock, text="Khoá tỉ lệ:", font=("Inter", 8), fg="#94a3b8", bg="#1e293b").pack(side="left", padx=2)
        self.aspect_lock_var = tk.StringVar(value=ASPECT_OUTPUT)
        cb = ttk.Combobox(lock, textvariable=self.aspect_lock_var, values=ASPECT_CHOICES, width=20, state="readonly")
        cb.pack(side="right", padx=2)
        cb.bind("<<ComboboxSelected>>", lambda e: (self.enforce_aspect(), self.canvas.focus_set()))

        nums = tk.Frame(sec, bg="#1e293b")
        nums.pack(fill="x", padx=8, pady=(4, 2))
        self.roi_entries = {}
        for key in ("x", "y", "w", "h"):
            tk.Label(nums, text=key, font=("Inter", 8), fg="#94a3b8", bg="#1e293b").pack(side="left", padx=(4, 1))
            e = self.entry(nums, 0, width=5)
            e.pack(side="left")
            e.bind("<Return>", lambda ev: self.apply_roi_entries())
            self.roi_entries[key] = e
        self.btn(nums, "Áp dụng", self.apply_roi_entries, bg="#0e7490").pack(side="right", padx=2)

        self.lbl_roi_coords = tk.Label(sec, text="Vùng chọn: Chưa chọn", font=("Inter", 8, "bold"), fg="#10b981", bg="#1e293b")
        self.lbl_roi_coords.pack(anchor="w", padx=10, pady=(2, 8))

        # --- 2. OUTPUT SIZE ---
        sec = self.section(parent, "📐 2. Kích thước xuất")
        self.res_choice_var = tk.StringVar(value="640x640")
        for txt, val in (("640 x 640 (Chuẩn YOLO / Detection)", "640x640"), ("224 x 224 (Chuẩn YOWO Dataset)", "224x224"),
                         ("Giữ nguyên kích thước vùng cắt", "original"), ("Tuỳ chỉnh (W x H):", "custom")):
            self.radio(sec, txt, self.res_choice_var, val, self.on_output_size_change).pack(anchor="w", padx=8, pady=1)
        cf = tk.Frame(sec, bg="#1e293b")
        cf.pack(fill="x", padx=26, pady=(0, 4))
        self.entry_custom_w = self.entry(cf, 640)
        self.entry_custom_w.pack(side="left")
        tk.Label(cf, text="x", bg="#1e293b", fg="#94a3b8", font=("Inter", 8)).pack(side="left", padx=4)
        self.entry_custom_h = self.entry(cf, 640)
        self.entry_custom_h.pack(side="left")
        for e in (self.entry_custom_w, self.entry_custom_h):
            e.bind("<KeyRelease>", lambda ev: self.on_output_size_change() if self.res_choice_var.get() == "custom" else None)
        self.keep_aspect_var = tk.BooleanVar(value=False)
        self.check(sec, "Giữ tỉ lệ ảnh (thêm viền đen, không bóp méo)", self.keep_aspect_var, cmd=self.schedule_preview).pack(anchor="w", padx=8, pady=(0, 6))

        # --- 3. SAMPLING ---
        sec = self.section(parent, "⏱️ 3. Tần suất trích xuất")
        self.sample_mode_var = tk.StringVar(value="all")
        self.radio(sec, "Tất cả các frame (1:1)", self.sample_mode_var, "all", self.update_sample_info).pack(anchor="w", padx=8, pady=1)
        f2 = tk.Frame(sec, bg="#1e293b")
        f2.pack(fill="x", padx=8, pady=1)
        self.radio(f2, "Mỗi N frame (Step):", self.sample_mode_var, "step", self.update_sample_info).pack(side="left")
        self.entry_step = self.entry(f2, 2, width=4)
        self.entry_step.pack(side="left", padx=6)
        self.entry_step.bind("<KeyRelease>", lambda e: self.update_sample_info())
        f3 = tk.Frame(sec, bg="#1e293b")
        f3.pack(fill="x", padx=8, pady=1)
        self.radio(f3, "Theo FPS đích:", self.sample_mode_var, "fps", self.update_sample_info).pack(side="left")
        self.target_fps_var = tk.StringVar(value="15")
        fps_box = ttk.Combobox(f3, textvariable=self.target_fps_var, values=["1", "2", "5", "10", "15", "20", "24", "30"], width=5, state="readonly")
        fps_box.pack(side="left", padx=6)
        fps_box.bind("<<ComboboxSelected>>", lambda e: (self.sample_mode_var.set("fps"), self.update_sample_info(), self.canvas.focus_set()))
        self.lbl_sample_info = tk.Label(sec, text="", font=("Inter", 8, "bold"), fg="#10b981", bg="#1e293b")
        self.lbl_sample_info.pack(anchor="w", padx=10, pady=(2, 6))

        # --- 4. OUTPUT ---
        sec = self.section(parent, "💾 4. Lưu vào dataset")
        tk.Label(sec, text="Tên Clip / Snippet ID (A-Z, 0-9, _ -):", font=("Inter", 8), fg="#94a3b8", bg="#1e293b").pack(anchor="w", padx=8, pady=(4, 1))
        self.entry_clip_name = tk.Entry(sec, bg="#0f172a", fg="#f8fafc", insertbackground="#fff", font=("Inter", 8, "bold"), relief="flat")
        self.entry_clip_name.insert(0, "new_clip_01")
        self.entry_clip_name.pack(fill="x", padx=8, pady=(0, 2), ipady=2)
        tk.Label(sec, text="Lưu tại: archive/frames/<clip_id>/000001.jpg ...", font=("Inter", 7), fg="#64748b", bg="#1e293b").pack(anchor="w", padx=8, pady=(0, 4))
        fmt = tk.Frame(sec, bg="#1e293b")
        fmt.pack(fill="x", padx=8, pady=2)
        tk.Label(fmt, text="Định dạng:", font=("Inter", 8), fg="#94a3b8", bg="#1e293b").pack(side="left")
        self.format_var = tk.StringVar(value="jpg")
        fcb = ttk.Combobox(fmt, textvariable=self.format_var, values=["jpg", "png"], width=5, state="readonly")
        fcb.pack(side="left", padx=4)
        fcb.bind("<<ComboboxSelected>>", lambda e: self.canvas.focus_set())
        tk.Label(fmt, text="Chất lượng JPG:", font=("Inter", 8), fg="#94a3b8", bg="#1e293b").pack(side="left", padx=(10, 2))
        self.entry_quality = self.entry(fmt, 95, width=4)
        self.entry_quality.pack(side="left")
        self.chk_unlabeled_var = tk.BooleanVar(value=True)
        self.check(sec, "Tự động thêm vào 'unlabeled.csv'", self.chk_unlabeled_var, fg="#38bdf8").pack(anchor="w", padx=8, pady=1)
        self.chk_open_folder_var = tk.BooleanVar(value=False)
        self.check(sec, "Mở thư mục khi hoàn tất", self.chk_open_folder_var).pack(anchor="w", padx=8, pady=(0, 6))

        # --- 5. QUEUE ---
        sec = self.section(parent, "📋 5. Hàng đợi (nhiều vùng cắt / đoạn / video)")
        tk.Label(sec, text="Chỉnh vùng cắt + phạm vi rồi bấm Thêm. Khi hàng đợi có job,\nnút Bắt đầu sẽ chạy toàn bộ hàng đợi.",
                 font=("Inter", 7), fg="#64748b", bg="#1e293b", justify="left").pack(anchor="w", padx=8, pady=(4, 2))
        qrow = tk.Frame(sec, bg="#1e293b")
        qrow.pack(fill="x", padx=8, pady=2)
        self.btn(qrow, "➕ Thêm vào hàng đợi", self.add_job, bg="#0e7490", bold=True).pack(side="left", fill="x", expand=True, padx=(0, 2))
        self.btn(qrow, "✖ Xoá", self.remove_job).pack(side="left", padx=1)
        self.btn(qrow, "Xoá hết", self.clear_jobs, fg="#94a3b8").pack(side="left", padx=1)
        self.jobs_listbox = tk.Listbox(sec, height=5, bg="#0f172a", fg="#f8fafc", selectbackground="#4f46e5", font=("Inter", 8), bd=0, relief="flat", exportselection=False)
        self.jobs_listbox.pack(fill="x", padx=8, pady=(2, 8))

    def bind_shortcuts(self):
        def bind(seqs, fn):
            def handler(event):
                if isinstance(self.root.focus_get(), (tk.Entry, ttk.Entry, tk.Spinbox)):
                    return None  # typing in a field
                fn()
                return "break"
            for s in seqs:
                self.root.bind(s, handler)

        bind(["<space>"], self.toggle_play)
        bind(["<Left>", "<Key-a>", "<Key-A>"], lambda: self.step_frame(-1))
        bind(["<Right>", "<Key-d>", "<Key-D>"], lambda: self.step_frame(1))
        bind(["<Shift-Left>", "<Shift-KeyPress-A>"], lambda: self.step_frame(-10))
        bind(["<Shift-Right>", "<Shift-KeyPress-D>"], lambda: self.step_frame(10))
        bind(["<Home>"], lambda: self.read_frame(0))
        bind(["<End>"], lambda: self.read_frame(self.total_frames - 1))
        bind(["<bracketleft>"], self.set_start_frame)
        bind(["<bracketright>"], self.set_end_frame)
        bind(["<Key-f>", "<Key-F>"], self.fit_view)
        bind(["<Key-z>", "<Key-Z>"], self.zoom_view_to_roi)
        bind(["<plus>", "<equal>", "<KP_Add>"], lambda: self.zoom_roi_by(1.25))
        bind(["<minus>", "<KP_Subtract>"], lambda: self.zoom_roi_by(0.8))

    # ------------------------------------------------------------------ video loading / reading
    def browse_video(self):
        path = filedialog.askopenfilename(title="Chọn Video Để Tách Frame", filetypes=VIDEO_TYPES)
        if path:
            self.load_video(path)

    def load_video(self, path):
        self.stop_play()
        info = probe_video(path)
        if info is None:
            messagebox.showerror("Lỗi", f"Không thể mở video:\n{path}\n\nNếu tên file / thư mục có dấu tiếng Việt, hãy thử đổi sang tên không dấu.")
            return
        if self.cap:
            self.cap.release()
        self.cap = cv2.VideoCapture(path)
        self._cap_next = 0

        prev_size = (self.video_w, self.video_h)
        self.video_path = path
        self.video_fps, self.video_w, self.video_h, self.total_frames = info

        duration = self.total_frames / self.video_fps
        vname = os.path.basename(path)
        self.lbl_video_name.config(text=vname)
        self.lbl_video_meta.config(text=f"{self.video_w}x{self.video_h} | {self.video_fps:.2f} fps | {self.total_frames}f ({int(duration // 60):02d}:{int(duration % 60):02d})")

        self.entry_clip_name.delete(0, tk.END)
        self.entry_clip_name.insert(0, sanitize_clip_name(os.path.splitext(vname)[0]) or "new_clip_01")

        self.start_frame = 0
        self.end_frame = max(0, self.total_frames - 1)
        self.update_trim_info()
        self.slider.config(to=max(0, self.total_frames - 1))

        self.view.set_image_size(self.video_w, self.video_h, reset=True)
        # Keep the ROI when the new video has the same resolution (batch of similar cameras)
        if self.roi is None or prev_size != (self.video_w, self.video_h):
            if self.video_w >= 640 and self.video_h >= 640:
                self.preset_center_640()
            else:
                self.preset_max_aspect()
        self.read_frame(0)

    def read_frame(self, idx):
        if not self.cap or self.total_frames <= 0:
            return False
        idx = max(0, min(self.total_frames - 1, idx))
        if idx != self._cap_next:  # sequential reads (playback, +1) skip the slow seek
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = self.cap.read()
        if not ok or frame is None:
            self._cap_next = -1
            return False
        self._cap_next = idx + 1
        self.current_bgr = frame
        self.current_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        self.current_frame_idx = idx

        sec = idx / self.video_fps
        self.lbl_current_frame.config(text=f"Frame: {idx + 1} / {self.total_frames} ({int(sec // 60):02d}:{sec % 60:05.2f})")
        self._slider_lock = True
        self.slider.set(idx)
        self._slider_lock = False
        self.render()
        self.draw_trim_bar()
        return True

    # ------------------------------------------------------------------ rendering
    def on_canvas_resize(self, event):
        self.view.on_resize()
        self.render()

    def render(self):
        self.view.render(self.current_pil)
        self.draw_roi_overlay()
        self.lbl_view_info.config(text=f"Zoom {self.view.scale * 100:.0f}%")

    def roi_canvas_rect(self):
        rx, ry, rw, rh = self.roi
        x1, y1 = self.view.to_canvas(rx, ry)
        x2, y2 = self.view.to_canvas(rx + rw, ry + rh)
        return x1, y1, x2, y2

    @staticmethod
    def handle_points(x1, y1, x2, y2):
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        return {"nw": (x1, y1), "n": (mx, y1), "ne": (x2, y1), "e": (x2, my),
                "se": (x2, y2), "s": (mx, y2), "sw": (x1, y2), "w": (x1, my)}

    def draw_roi_overlay(self):
        c = self.canvas
        c.delete("roi")
        if not self.roi or not self.video_w:
            return
        cx1, cy1, cx2, cy2 = self.roi_canvas_rect()
        ix1, iy1 = self.view.to_canvas(0, 0)
        ix2, iy2 = self.view.to_canvas(self.video_w, self.video_h)
        # Clip to the visible canvas so huge zoom levels don't create giant items
        cw, ch = self.view.canvas_size()
        ix1, iy1, ix2, iy2 = max(ix1, -2), max(iy1, -2), min(ix2, cw + 2), min(iy2, ch + 2)

        # Dim everything outside the ROI
        dim = dict(fill="#000000", stipple="gray50", outline="", tags="roi")
        c.create_rectangle(ix1, iy1, ix2, max(iy1, cy1), **dim)
        c.create_rectangle(ix1, min(iy2, cy2), ix2, iy2, **dim)
        c.create_rectangle(ix1, max(iy1, cy1), max(ix1, cx1), min(iy2, cy2), **dim)
        c.create_rectangle(min(ix2, cx2), max(iy1, cy1), ix2, min(iy2, cy2), **dim)

        c.create_rectangle(cx1, cy1, cx2, cy2, outline="#06b6d4", width=2, tags="roi")
        # Rule-of-thirds guides help framing the zoomed crop
        for k in (1, 2):
            gx = cx1 + (cx2 - cx1) * k / 3
            gy = cy1 + (cy2 - cy1) * k / 3
            c.create_line(gx, cy1, gx, cy2, fill="#334155", dash=(2, 4), tags="roi")
            c.create_line(cx1, gy, cx2, gy, fill="#334155", dash=(2, 4), tags="roi")
        ccx, ccy = (cx1 + cx2) / 2, (cy1 + cy2) / 2
        c.create_line(ccx - 6, ccy, ccx + 6, ccy, fill="#e2e8f0", tags="roi")
        c.create_line(ccx, ccy - 6, ccx, ccy + 6, fill="#e2e8f0", tags="roi")
        for px, py in self.handle_points(cx1, cy1, cx2, cy2).values():
            c.create_rectangle(px - 4, py - 4, px + 4, py + 4, fill="#10b981", outline="#022c22", tags="roi")

        rx, ry, rw, rh = clamp_roi(self.roi, self.video_w, self.video_h)
        text = f"CROP {rw} x {rh}"
        ty = cy1 - 20 if cy1 >= 20 else cy1 + 2
        c.create_rectangle(cx1, ty, cx1 + len(text) * 7 + 10, ty + 18, fill="#06b6d4", outline="#06b6d4", tags="roi")
        c.create_text(cx1 + 5, ty + 9, text=text, fill="#000000", font=("Inter", 8, "bold"), anchor="w", tags="roi")

        self.lbl_roi_coords.config(text=f"Vùng chọn: x={rx}, y={ry}, w={rw}, h={rh}")
        self.sync_roi_entries()
        self.schedule_preview()

    def sync_roi_entries(self):
        if self.root.focus_get() in self.roi_entries.values():
            return  # don't overwrite what the user is typing
        vals = dict(zip("xywh", clamp_roi(self.roi, self.video_w, self.video_h)))
        for k, e in self.roi_entries.items():
            e.delete(0, tk.END)
            e.insert(0, str(vals[k]))

    def schedule_preview(self):
        if self._preview_job is None:
            self._preview_job = self.root.after(40, self.update_preview)

    def update_preview(self):
        self._preview_job = None
        pc = self.preview_canvas
        pc.delete("all")
        if self.current_bgr is None or not self.roi:
            return
        try:
            target = self.get_target_size()
        except ValueError as ex:
            self.lbl_preview_info.config(text=str(ex), fg="#f59e0b")
            return
        keep = self.keep_aspect_var.get()
        out = render_crop(self.current_bgr, self.roi, target, keep)
        oh, ow = out.shape[:2]
        pw, ph = max(1, pc.winfo_width()), max(1, pc.winfo_height())
        s = min(pw / ow, ph / oh)
        disp = cv2.resize(out, (max(1, int(ow * s)), max(1, int(oh * s))), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_NEAREST)
        self._preview_tk = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(disp, cv2.COLOR_BGR2RGB)))
        pc.create_image(pw // 2, ph // 2, image=self._preview_tk)

        _, _, rw, rh = clamp_roi(self.roi, self.video_w, self.video_h)
        if target:
            zoom = min(target[0] / rw, target[1] / rh) if keep else max(target[0] / rw, target[1] / rh)
            distort = not keep and abs((target[0] / target[1]) - (rw / rh)) > 0.02
        else:
            zoom, distort = 1.0, False
        msg = f"Xuất {ow}x{oh} • phóng ×{zoom:.2f}"
        color = "#94a3b8"
        if zoom > 2.5:
            msg += "  ⚠ phóng quá lớn, ảnh sẽ mờ"
            color = "#f59e0b"
        if distort:
            msg += "\n⚠ Tỉ lệ vùng cắt khác kích thước xuất → ảnh bị méo (bật 'Giữ tỉ lệ' hoặc khoá tỉ lệ)"
            color = "#f59e0b"
        self.lbl_preview_info.config(text=msg, fg=color)

    # ------------------------------------------------------------------ ROI geometry
    def current_aspect(self):
        choice = self.aspect_lock_var.get()
        if choice == ASPECT_OUTPUT:
            try:
                ts = self.get_target_size()
            except ValueError:
                return None
            return ts[0] / ts[1] if ts else None
        return ASPECT_VALUES.get(choice)

    def rect_from_anchor(self, ax, ay, px, py, aspect):
        """Rect with one corner at (ax, ay) growing towards (px, py), respecting aspect and frame bounds."""
        dx, dy = px - ax, py - ay
        w, h = abs(dx), abs(dy)
        maxw = (self.video_w - ax) if dx >= 0 else ax
        maxh = (self.video_h - ay) if dy >= 0 else ay
        if aspect:
            if h * aspect < w:
                h = w / aspect
            else:
                w = h * aspect
            if w > maxw:
                w, h = maxw, maxw / aspect
            if h > maxh:
                h, w = maxh, maxh * aspect
        else:
            w, h = min(w, maxw), min(h, maxh)
        return [ax if dx >= 0 else ax - w, ay if dy >= 0 else ay - h, w, h]

    def set_roi(self, roi):
        self.roi = [float(v) for v in roi]
        self.draw_roi_overlay()

    def enforce_aspect(self):
        """Re-shape the current ROI to the locked aspect ratio, keeping its centre and area."""
        aspect = self.current_aspect()
        if not self.roi or not aspect:
            self.schedule_preview()
            return
        x, y, w, h = self.roi
        cx, cy = x + w / 2, y + h / 2
        nw = math.sqrt(w * h * aspect)
        nh = nw / aspect
        if nw > self.video_w:
            nw, nh = self.video_w, self.video_w / aspect
        if nh > self.video_h:
            nh, nw = self.video_h, self.video_h * aspect
        self.set_roi([max(0, min(self.video_w - nw, cx - nw / 2)), max(0, min(self.video_h - nh, cy - nh / 2)), nw, nh])

    def on_output_size_change(self):
        if self.aspect_lock_var.get() == ASPECT_OUTPUT:
            self.enforce_aspect()
        else:
            self.schedule_preview()

    # ROI presets
    def preset_full_frame(self):
        if self.video_w:
            self.set_roi([0, 0, self.video_w, self.video_h])

    def preset_center_640(self):
        if self.video_w:
            w, h = min(640, self.video_w), min(640, self.video_h)
            self.set_roi([(self.video_w - w) // 2, (self.video_h - h) // 2, w, h])

    def preset_center_square(self):
        if self.video_w:
            self.set_roi(largest_rect(self.video_w, self.video_h, 1.0))

    def preset_max_aspect(self):
        if self.video_w:
            center = (self.roi[0] + self.roi[2] / 2, self.roi[1] + self.roi[3] / 2) if self.roi else None
            self.set_roi(largest_rect(self.video_w, self.video_h, self.current_aspect(), center))

    def zoom_roi_by(self, factor):
        if not self.roi:
            return
        x, y, w, h = self.roi
        if factor < 1:  # zoom out: grow but stay inside the frame
            grow = 1 / factor
            w2, h2 = w * grow, h * grow
            s = min(1.0, self.video_w / w2, self.video_h / h2)
            factor = 1 / (grow * s)
        self.set_roi(zoom_roi(self.roi, factor, self.video_w, self.video_h))

    def apply_roi_entries(self):
        if not self.video_w:
            return
        try:
            vals = [float(self.roi_entries[k].get()) for k in "xywh"]
        except ValueError:
            messagebox.showwarning("ROI", "x, y, w, h phải là số")
            return
        self.canvas.focus_set()
        self.set_roi(clamp_roi(vals, self.video_w, self.video_h))

    # ------------------------------------------------------------------ mouse
    def roi_hit(self, cx, cy):
        if not self.roi:
            return None
        x1, y1, x2, y2 = self.roi_canvas_rect()
        for name, (px, py) in self.handle_points(x1, y1, x2, y2).items():
            if abs(cx - px) <= HANDLE_PX and abs(cy - py) <= HANDLE_PX:
                return name
        if x1 <= cx <= x2 and y1 <= cy <= y2:
            return "move"
        return None

    def on_canvas_press(self, event):
        self.canvas.focus_set()
        if not self.cap:
            return
        self.stop_play()
        ix, iy = self.view.clamp_to_image(*self.view.to_image(event.x, event.y))
        hit = self.roi_hit(event.x, event.y)
        self.drag = {"hit": hit, "start": (ix, iy), "orig": list(self.roi) if self.roi else None}

    def on_canvas_drag(self, event):
        d = self.drag
        if not d:
            return
        ix, iy = self.view.clamp_to_image(*self.view.to_image(event.x, event.y))
        sx, sy = d["start"]
        aspect = self.current_aspect()
        hit = d["hit"]

        if hit is None:  # draw a new ROI
            roi = self.rect_from_anchor(sx, sy, ix, iy, aspect)
            if roi[2] < 2 or roi[3] < 2:
                return
        elif hit == "move":
            ox, oy, ow, oh = d["orig"]
            roi = [max(0, min(self.video_w - ow, ox + ix - sx)), max(0, min(self.video_h - oh, oy + iy - sy)), ow, oh]
        elif len(hit) == 2:  # corner: anchor at the opposite corner
            ox, oy, ow, oh = d["orig"]
            ax = ox + ow if "w" in hit else ox
            ay = oy + oh if "n" in hit else oy
            roi = self.rect_from_anchor(ax, ay, ix, iy, aspect)
            if roi[2] < 2 or roi[3] < 2:
                return
        else:  # edge
            roi = self.resize_edge(d["orig"], hit, ix, iy, aspect)
        self.set_roi(roi)

    def resize_edge(self, orig, edge, ix, iy, aspect):
        x, y, w, h = orig
        x2, y2 = x + w, y + h
        if edge in ("e", "w"):
            if edge == "e":
                x2 = max(x + 2, min(self.video_w, ix))
            else:
                x = min(x2 - 2, max(0, ix))
            nw = x2 - x
            if not aspect:
                return [x, y, nw, h]
            nh = nw / aspect
            if nh > self.video_h:
                nh = self.video_h
                nw = nh * aspect
                x = x if edge == "e" else x2 - nw
            cy = y + h / 2
            return [x, max(0, min(self.video_h - nh, cy - nh / 2)), nw, nh]
        if edge == "s":
            y2 = max(y + 2, min(self.video_h, iy))
        else:
            y = min(y2 - 2, max(0, iy))
        nh = y2 - y
        if not aspect:
            return [x, y, w, nh]
        nw = nh * aspect
        if nw > self.video_w:
            nw = self.video_w
            nh = nw / aspect
            y = y if edge == "s" else y2 - nh
        cx = x + w / 2
        return [max(0, min(self.video_w - nw, cx - nw / 2)), y, nw, nh]

    def on_canvas_release(self, event):
        if self.drag and self.roi:
            self.set_roi(clamp_roi(self.roi, self.video_w, self.video_h))
        self.drag = None

    def on_canvas_double(self, event):
        """Double-click: move the crop window so it is centred on the clicked point."""
        if not self.roi or not self.cap:
            return
        ix, iy = self.view.clamp_to_image(*self.view.to_image(event.x, event.y))
        _, _, w, h = self.roi
        self.set_roi([max(0, min(self.video_w - w, ix - w / 2)), max(0, min(self.video_h - h, iy - h / 2)), w, h])

    def on_canvas_motion(self, event):
        if self.drag is not None or not self.cap:
            return
        hit = self.roi_hit(event.x, event.y)
        cursor = {"nw": "size_nw_se", "se": "size_nw_se", "ne": "size_ne_sw", "sw": "size_ne_sw",
                  "n": "sb_v_double_arrow", "s": "sb_v_double_arrow", "e": "sb_h_double_arrow",
                  "w": "sb_h_double_arrow", "move": "fleur"}.get(hit, "crosshair")
        if cursor != self._cursor:
            try:
                self.canvas.config(cursor=cursor)
            except tk.TclError:
                self.canvas.config(cursor="crosshair")
            self._cursor = cursor
        ix, iy = self.view.to_image(event.x, event.y)
        self.lbl_view_info.config(text=f"x={ix:.0f} y={iy:.0f}  |  Zoom {self.view.scale * 100:.0f}%")

    # ------------------------------------------------------------------ view
    def fit_view(self):
        self.view.fit()
        self.render()

    def zoom_view_to_roi(self):
        if self.roi:
            self.view.zoom_to_rect(*self.roi, pad=0.08)

    # ------------------------------------------------------------------ playback
    def toggle_play(self):
        if self.is_playing:
            self.stop_play()
        elif self.cap:
            self.is_playing = True
            self.btn_play.config(text="⏸ Dừng", bg="#ef4444")
            self.schedule_next_frame()

    def stop_play(self):
        self.is_playing = False
        if hasattr(self, "btn_play"):
            self.btn_play.config(text="▶ Phát", bg="#4f46e5")
        if self.play_job:
            self.root.after_cancel(self.play_job)
            self.play_job = None

    def schedule_next_frame(self):
        if not self.is_playing:
            return
        cur = self.current_frame_idx
        limit = self.end_frame if self.start_frame <= cur < self.end_frame else self.total_frames - 1
        if cur >= limit:
            self.stop_play()
            return
        t0 = time.perf_counter()
        if not self.read_frame(cur + 1):
            self.stop_play()
            return
        spent = (time.perf_counter() - t0) * 1000
        fps = max(1.0, min(60.0, self.video_fps))
        self.play_job = self.root.after(max(1, int(1000 / fps - spent)), self.schedule_next_frame)

    def step_frame(self, delta):
        if self.cap:
            self.read_frame(self.current_frame_idx + delta)

    def on_slider_move(self, val):
        if self._slider_lock or not self.cap:
            return
        self.stop_play()
        self._pending_seek = int(float(val))
        if self._seek_job is None:  # coalesce fast slider drags into one seek
            self._seek_job = self.root.after(30, self._do_seek)

    def _do_seek(self):
        self._seek_job = None
        if self._pending_seek != self.current_frame_idx:
            self.read_frame(self._pending_seek)

    # ------------------------------------------------------------------ trimming
    def set_start_frame(self):
        if self.cap:
            self.start_frame = self.current_frame_idx
            self.end_frame = max(self.end_frame, self.start_frame)
            self.update_trim_info()

    def set_end_frame(self):
        if self.cap:
            self.end_frame = self.current_frame_idx
            self.start_frame = min(self.start_frame, self.end_frame)
            self.update_trim_info()

    def reset_trim(self):
        if self.cap:
            self.start_frame = 0
            self.end_frame = max(0, self.total_frames - 1)
            self.update_trim_info()

    def update_trim_info(self):
        count = max(0, self.end_frame - self.start_frame + 1)
        dur = count / self.video_fps if self.video_fps else 0
        self.lbl_trim_info.config(text=f"Phạm vi: [{self.start_frame + 1} -> {self.end_frame + 1}] ({count} frames, {dur:.1f}s)")
        self.draw_trim_bar()
        self.update_sample_info()

    def draw_trim_bar(self):
        tb = self.trim_bar
        tb.delete("all")
        if self.total_frames <= 0:
            return
        w = max(1, tb.winfo_width())
        denom = max(1, self.total_frames - 1)
        x1, x2 = self.start_frame * (w - 1) / denom, self.end_frame * (w - 1) / denom
        tb.create_rectangle(x1, 0, max(x1 + 2, x2), 6, fill="#0e7490", outline="")
        cx = self.current_frame_idx * (w - 1) / denom
        tb.create_rectangle(cx - 1, 0, cx + 1, 6, fill="#f8fafc", outline="")

    # ------------------------------------------------------------------ settings -> job
    def get_target_size(self):
        choice = self.res_choice_var.get()
        if choice == "640x640":
            return 640, 640
        if choice == "224x224":
            return 224, 224
        if choice == "original":
            return None
        try:
            tw, th = int(self.entry_custom_w.get()), int(self.entry_custom_h.get())
        except ValueError:
            raise ValueError("Kích thước tuỳ chỉnh W x H phải là số nguyên")
        if not (16 <= tw <= 8192 and 16 <= th <= 8192):
            raise ValueError("Kích thước tuỳ chỉnh phải trong khoảng 16 – 8192 px")
        return tw, th

    def current_frame_indices(self):
        mode = self.sample_mode_var.get()
        try:
            step = max(1, int(self.entry_step.get()))
        except ValueError:
            step = 1
        try:
            tfps = float(self.target_fps_var.get())
        except ValueError:
            tfps = None
        return compute_frame_indices(self.start_frame, self.end_frame, mode, step, self.video_fps, tfps)

    def update_sample_info(self):
        if not self.cap:
            self.lbl_sample_info.config(text="")
            return
        n = len(self.current_frame_indices())
        self.lbl_sample_info.config(text=f"→ Sẽ xuất {n} frame")

    def build_job(self):
        if not self.video_path or not self.cap:
            messagebox.showwarning("Cảnh báo", "Vui lòng chọn video trước!")
            return None
        raw = self.entry_clip_name.get()
        clip_name = sanitize_clip_name(raw)
        if not clip_name:
            messagebox.showwarning("Cảnh báo", "Vui lòng nhập Tên Clip / Snippet ID (A-Z, 0-9, _ -)!")
            return None
        if clip_name != raw:
            self.entry_clip_name.delete(0, tk.END)
            self.entry_clip_name.insert(0, clip_name)
        if any(j["clip_name"] == clip_name for j in self.jobs):
            messagebox.showwarning("Trùng tên", f"'{clip_name}' đã có trong hàng đợi. Hãy đặt tên khác.")
            return None
        try:
            target_size = self.get_target_size()
        except ValueError as ex:
            messagebox.showwarning("Cảnh báo", str(ex))
            return None
        try:
            quality = max(50, min(100, int(self.entry_quality.get())))
        except ValueError:
            messagebox.showwarning("Cảnh báo", "Chất lượng JPG phải là số 50-100")
            return None
        if not self.roi or self.roi[2] < 4 or self.roi[3] < 4:
            self.preset_full_frame()
        indices = self.current_frame_indices()
        if not indices:
            messagebox.showwarning("Cảnh báo", "Không có frame nào trong phạm vi đã chọn!")
            return None

        output_dir = os.path.join(ARCHIVE_DIR, "frames", clip_name)
        write_mode = "overwrite"
        existing = len(list_frame_files(output_dir))
        if existing:
            msg = f"Thư mục 'archive/frames/{clip_name}' đã có {existing} ảnh.\n\nYes = Xoá ảnh cũ, ghi mới từ 000001\nNo = Ghi tiếp (đánh số nối tiếp)\nCancel = Huỷ"
            if os.path.exists(os.path.join(ARCHIVE_DIR, "labels", f"{clip_name}.xml")):
                msg += "\n\n⚠ Clip này đã có nhãn — nhãn cũ có thể không còn khớp với frame mới!"
            ans = messagebox.askyesnocancel("Thư mục đã tồn tại", msg)
            if ans is None:
                return None
            write_mode = "overwrite" if ans else "append"

        return {
            "video_path": self.video_path,
            "video_fps": self.video_fps,
            "clip_name": clip_name,
            "output_dir": output_dir,
            "roi": clamp_roi(self.roi, self.video_w, self.video_h),
            "frame_indices": indices,
            "target_size": target_size,
            "keep_aspect": self.keep_aspect_var.get(),
            "img_format": self.format_var.get(),
            "quality": quality,
            "write_mode": write_mode,
            "add_unlabeled": self.chk_unlabeled_var.get(),
        }

    @staticmethod
    def next_clip_name(name):
        m = re.match(r"^(.*?)(\d+)$", name)
        if m:
            return f"{m.group(1)}{int(m.group(2)) + 1:0{len(m.group(2))}d}"
        return f"{name}_02"

    # ------------------------------------------------------------------ queue
    def add_job(self):
        job = self.build_job()
        if not job:
            return
        self.jobs.append(job)
        self.refresh_jobs_list()
        # Suggest the next name so several crops from one video can be queued quickly
        self.entry_clip_name.delete(0, tk.END)
        self.entry_clip_name.insert(0, self.next_clip_name(job["clip_name"]))
        self.lbl_progress_status.config(text=f"Đã thêm '{job['clip_name']}' vào hàng đợi ({len(self.jobs)} job)")

    def remove_job(self):
        sel = self.jobs_listbox.curselection()
        if sel:
            self.jobs.pop(sel[0])
            self.refresh_jobs_list()

    def clear_jobs(self):
        self.jobs.clear()
        self.refresh_jobs_list()

    def refresh_jobs_list(self):
        self.jobs_listbox.delete(0, tk.END)
        for j in self.jobs:
            size = f"{j['target_size'][0]}x{j['target_size'][1]}" if j["target_size"] else "gốc"
            _, _, w, h = j["roi"]
            self.jobs_listbox.insert(tk.END, f"{j['clip_name']} | {len(j['frame_indices'])}f | {w}x{h}→{size} | {j['write_mode']}")
        self.btn_extract.config(text=f"🚀 CHẠY HÀNG ĐỢI ({len(self.jobs)} job)" if self.jobs else "🚀 BẮT ĐẦU TÁCH FRAME")

    # ------------------------------------------------------------------ extraction
    def start_extraction(self):
        if self.is_extracting:
            return
        jobs = list(self.jobs)
        if not jobs:
            job = self.build_job()
            if not job:
                return
            jobs = [job]

        self.is_extracting = True
        self.cancel_requested = False
        self.btn_extract.config(state="disabled", bg="#64748b")
        self.btn_cancel.config(state="normal", bg="#ef4444", fg="#fff")
        self.progress_bar.config(value=0)
        threading.Thread(target=self._extraction_worker, args=(jobs,), daemon=True).start()
        self.root.after(100, self._poll_worker)

    def cancel_extraction(self):
        if self.is_extracting:
            self.cancel_requested = True
            self.lbl_progress_status.config(text="Đang dừng tiến trình...")

    def _extraction_worker(self, jobs):
        # Runs in a background thread: never touch Tk widgets here, only post to self.msgq
        results = []
        for n, job in enumerate(jobs, 1):
            if self.cancel_requested:
                break

            def progress(done, total, saved, elapsed, n=n, job=job):
                self.msgq.put(("progress", n, len(jobs), job["clip_name"], done, total, saved, elapsed))

            try:
                res = run_job(job, progress_cb=progress, cancel_cb=lambda: self.cancel_requested)
            except Exception as ex:
                res = {"saved": 0, "error": str(ex), "cancelled": False, "registered": False}
            res["job"] = job
            results.append(res)
        self.msgq.put(("done", results))

    def _poll_worker(self):
        last_progress, done_msg = None, None
        while True:
            try:
                msg = self.msgq.get_nowait()
            except queue.Empty:
                break
            if msg[0] == "progress":
                last_progress = msg
            else:
                done_msg = msg
        if last_progress:
            _, n, n_jobs, clip, done, total, saved, elapsed = last_progress
            pct = done / total * 100 if total else 100
            self.progress_bar.config(value=((n - 1) + done / max(1, total)) / n_jobs * 100)
            rate = saved / elapsed if elapsed > 0 else 0
            eta = (total - done) / rate if rate > 0 else 0
            self.lbl_progress_status.config(text=f"[{n}/{n_jobs}] {clip}: {done}/{total} ({pct:.0f}%) • {rate:.1f} fps • còn ~{eta:.0f}s")
        if done_msg:
            self._finish_extraction(done_msg[1])
        else:
            self.root.after(100, self._poll_worker)

    def _finish_extraction(self, results):
        self.is_extracting = False
        self.btn_extract.config(state="normal", bg="#10b981")
        self.btn_cancel.config(state="disabled", bg="#334155", fg="#ef4444")

        lines, finished = [], set()
        for r in results:
            clip = r["job"]["clip_name"]
            if r.get("error"):
                lines.append(f"❌ {clip}: {r['error']}")
                continue
            line = f"{'⏹' if r['cancelled'] else '✅'} {clip}: {r['saved']} frame"
            if r["registered"]:
                line += " (+ unlabeled.csv)"
            lines.append(line)
            if not r["cancelled"]:
                finished.add(id(r["job"]))

        # Drop finished jobs from the queue, keep failed / cancelled ones to retry
        self.jobs = [j for j in self.jobs if id(j) not in finished]
        self.refresh_jobs_list()
        self.progress_bar.config(value=100 if not self.cancel_requested else self.progress_bar["value"])
        self.lbl_progress_status.config(text="Đã dừng!" if self.cancel_requested else "Hoàn tất!")

        title = "Đã dừng" if self.cancel_requested else "Thành công! 🎉"
        summary = "\n".join(lines) or "Không có job nào được chạy."
        messagebox.showinfo(title, summary + "\n\nẢnh lưu tại archive/frames/<clip>/ — bấm 🔄 trong Frame Labeler để thấy clip mới.")

        if self.chk_open_folder_var.get() and results:
            out = results[-1]["job"]["output_dir"]
            if os.path.isdir(out):
                try:
                    os.startfile(out)
                except (AttributeError, OSError):
                    pass


# =====================================================================
# CLI
# =====================================================================
def parse_size(text):
    if not text or text == "original":
        return None
    tw, th = map(int, text.lower().split("x"))
    return tw, th


def run_cli_extraction(args):
    video_path = args.video
    info = probe_video(video_path) if os.path.exists(video_path) else None
    if info is None:
        print(f"Error: cannot open video: {video_path}")
        return 1
    fps, vw, vh, total = info
    print(f"Loaded: {video_path} ({vw}x{vh}, {fps:.2f} fps, {total} frames)")

    clip_name = sanitize_clip_name(args.clip_name or os.path.splitext(os.path.basename(video_path))[0])
    output_dir = args.output_dir or os.path.join(ARCHIVE_DIR, "frames", clip_name)
    target_size = parse_size(args.size)

    if args.crop:
        roi = [float(v) for v in args.crop.split(",")]
    elif args.center_640:
        rw, rh = min(640, vw), min(640, vh)
        roi = [(vw - rw) // 2, (vh - rh) // 2, rw, rh]
    elif target_size and not args.keep_aspect:
        roi = largest_rect(vw, vh, target_size[0] / target_size[1])  # avoid stretching
    else:
        roi = [0, 0, vw, vh]
    if args.zoom and args.zoom > 1:
        roi = zoom_roi(roi, args.zoom, vw, vh)
    roi = clamp_roi(roi, vw, vh)

    start = max(0, int(round(args.start * fps))) if args.start else 0
    end = min(total - 1, int(round(args.end * fps))) if args.end else total - 1
    mode = "fps" if args.fps else ("step" if args.step > 1 else "all")
    indices = compute_frame_indices(start, end, mode, args.step, fps, args.fps)

    existing = len(list_frame_files(output_dir))
    if existing and not (args.overwrite or args.append):
        print(f"Error: {output_dir} already has {existing} images. Use --overwrite or --append.")
        return 1

    job = {
        "video_path": video_path, "video_fps": fps, "clip_name": clip_name, "output_dir": output_dir,
        "roi": roi, "frame_indices": indices, "target_size": target_size, "keep_aspect": args.keep_aspect,
        "img_format": args.format, "quality": args.quality,
        "write_mode": "append" if args.append else "overwrite", "add_unlabeled": args.add_unlabeled,
    }
    print(f"Extracting {len(indices)} frames -> {output_dir} (ROI x,y,w,h={roi}, size={target_size or 'original'})")

    def progress(done, total_n, saved, elapsed):
        if done % 50 == 0 or done == total_n:
            print(f"  {done}/{total_n} ({saved} saved, {saved / max(elapsed, 1e-6):.1f} fps)")

    res = run_job(job, progress_cb=progress)
    print(f"\nDone! Extracted {res['saved']} frames to {output_dir}")
    if res["registered"]:
        print(f"Registered '{clip_name}' in archive/unlabeled.csv")
    return 0


def main():
    p = argparse.ArgumentParser(description="Video Frame Extractor & Dataset Creator")
    p.add_argument("--video", type=str, help="Input video file (triggers CLI mode)")
    p.add_argument("--clip_name", type=str, help="Output clip / snippet id")
    p.add_argument("--output_dir", type=str, help="Directory to save frames (default archive/frames/<clip_name>)")
    p.add_argument("--crop", type=str, help="Crop ROI 'x,y,w,h' in video pixels")
    p.add_argument("--center_640", action="store_true", help="Crop the centre 640x640 box")
    p.add_argument("--zoom", type=float, default=0, help="Digital zoom: shrink the ROI around its centre by this factor (e.g. 2)")
    p.add_argument("--size", type=str, default="640x640", help="Output size 'WxH' or 'original'")
    p.add_argument("--keep_aspect", action="store_true", help="Letterbox instead of stretching when ROI aspect != output aspect")
    p.add_argument("--step", type=int, default=1, help="Take every N-th frame")
    p.add_argument("--fps", type=float, default=None, help="Target sampling FPS (overrides --step)")
    p.add_argument("--start", type=float, default=None, help="Start time in seconds")
    p.add_argument("--end", type=float, default=None, help="End time in seconds")
    p.add_argument("--format", choices=["jpg", "png"], default="jpg")
    p.add_argument("--quality", type=int, default=95, help="JPEG quality")
    p.add_argument("--overwrite", action="store_true", help="Delete existing frames in the output folder first")
    p.add_argument("--append", action="store_true", help="Continue numbering after existing frames")
    p.add_argument("--add_unlabeled", action="store_true", help="Append clip_name to archive/unlabeled.csv")
    args = p.parse_args()

    if args.video:
        sys.exit(run_cli_extraction(args))
    root = tk.Tk()
    VideoFrameExtractorGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
