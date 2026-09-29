import os
import sys
import copy
import time
import subprocess
from collections import OrderedDict
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image

from annotation_io import (
    CLASS_MAP,
    delete_snippet,
    snippet_paths,
    load_dataset_snippets,
    parse_cvat_xml,
    parse_yolo_txt,
    save_cvat_xml,
    save_yolo_txt
)
from zoom_canvas import ZoomPanView

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)  # archive/ is shared with the frame extractor
EXTRACTOR_SCRIPT = os.path.join(PROJECT_DIR, "frame_extractor", "video_frame_extractor.py")
IMG_EXTS = (".jpg", ".jpeg", ".png")
CLASS_COLORS = {"swim": "#ff00cc", "drown": "#fafa37"}
HANDLE_PX = 6        # handle hit radius / half size (canvas px)
MIN_BOX = 3          # minimum box side (image px)
UNDO_LIMIT = 100
CACHE_SIZE = 96      # decoded frames kept in memory

HANDLE_CURSORS = {
    "nw": "size_nw_se", "se": "size_nw_se",
    "ne": "size_ne_sw", "sw": "size_ne_sw",
    "n": "sb_v_double_arrow", "s": "sb_v_double_arrow",
    "e": "sb_h_double_arrow", "w": "sb_h_double_arrow",
    "move": "fleur",
}


class FrameCache:
    """Small LRU cache of decoded frames so stepping / playback does not re-decode JPEGs."""

    def __init__(self, maxsize):
        self.maxsize = maxsize
        self._items = OrderedDict()

    def get(self, path):
        img = self._items.get(path)
        if img is not None:
            self._items.move_to_end(path)
            return img
        with Image.open(path) as im:
            img = im.convert("RGB")
        self._items[path] = img
        if len(self._items) > self.maxsize:
            self._items.popitem(last=False)
        return img

    def clear(self):
        self._items.clear()


def box_center(b):
    return (b["xtl"] + b["xbr"]) / 2.0, (b["ytl"] + b["ybr"]) / 2.0


class DesktopFrameLabeler:
    def __init__(self, root, archive_dir=None):
        self.root = root
        self.root.title("Lifeguard Frame Labeler - Desktop Application")
        self.root.geometry("1360x860")
        self.root.minsize(1050, 700)
        self.root.configure(bg="#0f172a")

        # Styling
        self.style = ttk.Style()
        self.style.theme_use("clam")
        self.setup_styles()

        # Dataset state
        self.archive_dir = archive_dir or os.path.join(PROJECT_DIR, "archive")
        self.snippets = []
        self.filtered_snippets = []
        self.current_tab = "unlabeled"
        self.current_snippet = None
        self.frames_dir = None
        self.frame_files = []
        self.total_frames = 0
        self.current_frame_idx = 0
        self.img_w = 224
        self.img_h = 224
        self.annotations = {}  # frame_idx -> list of box dicts
        self.active_class = "swim"
        self.selected_box_idx = -1

        # Rendering
        self.cache = FrameCache(CACHE_SIZE)
        self.current_pil = None
        self._cursor = "crosshair"

        # Editing state
        self.drag = None
        self._drag_snap = None
        self.undo_stack = []
        self.redo_stack = []
        self.dirty = False

        # Playback state
        self.is_playing = False
        self.play_job = None
        self.fps = 15

        self._slider_lock = False
        self._timeline_job = None

        self.create_ui()
        self.bind_shortcuts()
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.load_dataset()

    def setup_styles(self):
        self.style.configure(".", background="#0f172a", foreground="#f8fafc", font=("Inter", 9))
        self.style.configure("TFrame", background="#0f172a")
        self.style.configure("TLabel", background="#0f172a", foreground="#f8fafc", font=("Inter", 9))
        self.style.configure("Horizontal.TProgressbar", background="#06b6d4", troughcolor="#1e293b", thickness=8)

    # ================================================================== UI
    def create_ui(self):
        self.root.columnconfigure(1, weight=1)
        self.root.rowconfigure(0, weight=1)

        # 1. LEFT SIDEBAR (Clip List)
        sidebar = tk.Frame(self.root, bg="#1e293b", width=300)
        sidebar.grid(row=0, column=0, sticky="nsew", padx=(10, 5), pady=10)
        sidebar.pack_propagate(False)

        brand_frame = tk.Frame(sidebar, bg="#1e293b")
        brand_frame.pack(fill="x", padx=12, pady=(10, 6))
        brand_top = tk.Frame(brand_frame, bg="#1e293b")
        brand_top.pack(fill="x")
        tk.Label(brand_top, text="🛟 Frame Labeler", font=("Inter", 13, "bold"), fg="#f8fafc", bg="#1e293b").pack(side="left")
        tk.Button(brand_top, text="🔄", font=("Inter", 8), bg="#334155", fg="#38bdf8", relief="flat", padx=4, command=self.load_dataset).pack(side="right")
        tk.Label(brand_frame, text="Lifeguard Drowning Dataset", font=("Inter", 8), fg="#94a3b8", bg="#1e293b").pack(anchor="w")

        tk.Button(sidebar, text="🎬 Tách Frame Từ Video...", font=("Inter", 8, "bold"), bg="#06b6d4", fg="#020617", relief="flat", pady=3, command=self.open_frame_extractor).pack(fill="x", padx=12, pady=(0, 8))

        stats_frame = tk.Frame(sidebar, bg="#0f172a", bd=1, relief="solid")
        stats_frame.pack(fill="x", padx=12, pady=(0, 10))
        self.stats_lbl = tk.Label(stats_frame, text="Unlabeled: 0 / 0 (0%)", font=("Inter", 8, "bold"), fg="#38bdf8", bg="#0f172a")
        self.stats_lbl.pack(anchor="w", padx=8, pady=(6, 2))
        self.progress_bar = ttk.Progressbar(stats_frame, orient="horizontal", mode="determinate", style="Horizontal.TProgressbar")
        self.progress_bar.pack(fill="x", padx=8, pady=(0, 8))

        self.search_entry = tk.Entry(sidebar, bg="#0f172a", fg="#f8fafc", insertbackground="#fff", relief="flat", font=("Inter", 9))
        self.search_entry.pack(fill="x", ipady=4, padx=13, pady=(0, 6))
        self.search_entry.bind("<KeyRelease>", lambda e: self.render_snippet_list())

        tab_frame = tk.Frame(sidebar, bg="#1e293b")
        tab_frame.pack(fill="x", padx=12, pady=(0, 6))
        self.tab_unlabeled_btn = tk.Button(tab_frame, text="Unlabeled", font=("Inter", 8, "bold"), bg="#4f46e5", fg="#fff", relief="flat", command=lambda: self.switch_tab("unlabeled"))
        self.tab_unlabeled_btn.pack(side="left", expand=True, fill="x", padx=(0, 2))
        self.tab_all_btn = tk.Button(tab_frame, text="All Clips", font=("Inter", 8), bg="#334155", fg="#94a3b8", relief="flat", command=lambda: self.switch_tab("all"))
        self.tab_all_btn.pack(side="right", expand=True, fill="x", padx=(2, 0))

        tk.Button(sidebar, text="🗑️ Xoá bộ frame đang chọn...", font=("Inter", 8), bg="#7f1d1d", fg="#fecaca", relief="flat", pady=2, command=self.delete_selected_snippet).pack(side="bottom", fill="x", padx=12, pady=(0, 10))

        list_container = tk.Frame(sidebar, bg="#0f172a")
        list_container.pack(fill="both", expand=True, padx=12, pady=(0, 6))
        scrollbar = tk.Scrollbar(list_container, orient="vertical")
        self.snippet_listbox = tk.Listbox(list_container, yscrollcommand=scrollbar.set, bg="#0f172a", fg="#f8fafc", selectbackground="#4f46e5", selectforeground="#fff", font=("Inter", 8), bd=0, relief="flat", exportselection=False)
        scrollbar.config(command=self.snippet_listbox.yview)
        scrollbar.pack(side="right", fill="y")
        self.snippet_listbox.pack(side="left", fill="both", expand=True)
        self.snippet_listbox.bind("<<ListboxSelect>>", self.on_snippet_select)
        self.snippet_listbox.bind("<Button-3>", self.on_snippet_right_click)
        self.snippet_menu = tk.Menu(self.root, tearoff=0)
        self.snippet_menu.add_command(label="📂 Mở thư mục frame", command=lambda: self.open_snippet_folder(self._menu_snippet))
        self.snippet_menu.add_separator()
        self.snippet_menu.add_command(label="🗑️ Xoá bộ frame...", command=lambda: self.confirm_delete_snippet(self._menu_snippet))
        self._menu_snippet = None

        # 2. CENTER CANVAS VIEWPORT
        center_frame = tk.Frame(self.root, bg="#0f172a")
        center_frame.grid(row=0, column=1, sticky="nsew", padx=5, pady=10)
        center_frame.rowconfigure(1, weight=1)
        center_frame.columnconfigure(0, weight=1)

        top_bar = tk.Frame(center_frame, bg="#1e293b", height=50)
        top_bar.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        self.clip_title_lbl = tk.Label(top_bar, text="Select a clip to begin", font=("Inter", 11, "bold"), fg="#f8fafc", bg="#1e293b")
        self.clip_title_lbl.pack(side="left", padx=16, pady=8)
        self.frame_counter_lbl = tk.Label(top_bar, text="Frame: 0 / 0", font=("Inter", 9), fg="#94a3b8", bg="#1e293b")
        self.frame_counter_lbl.pack(side="right", padx=16, pady=8)
        zoom_box = tk.Frame(top_bar, bg="#1e293b")
        zoom_box.pack(side="right", padx=4)
        for txt, cmd in (("−", lambda: self.view.zoom_center(1 / 1.25)), ("+", lambda: self.view.zoom_center(1.25)),
                         ("Vừa khung (F)", self.fit_view), ("Zoom box (Z)", self.zoom_to_selected)):
            tk.Button(zoom_box, text=txt, font=("Inter", 8), bg="#334155", fg="#f8fafc", relief="flat", padx=6, command=cmd).pack(side="left", padx=1)

        canvas_card = tk.Frame(center_frame, bg="#020617", bd=1, relief="solid")
        canvas_card.grid(row=1, column=0, sticky="nsew")

        self.canvas = tk.Canvas(canvas_card, bg="#020617", highlightthickness=0, cursor="crosshair", takefocus=1)
        self.canvas.pack(fill="both", expand=True)
        self.view = ZoomPanView(self.canvas, on_change=self.render_frame, on_shift_wheel=self.step_frame)
        self.canvas.bind("<Configure>", self.on_canvas_resize)
        self.canvas.bind("<ButtonPress-1>", self.on_canvas_press)
        self.canvas.bind("<B1-Motion>", self.on_canvas_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_canvas_release)
        self.canvas.bind("<Motion>", self.on_canvas_motion)

        # 3. RIGHT PANEL
        right_panel = tk.Frame(self.root, bg="#1e293b", width=270)
        right_panel.grid(row=0, column=2, sticky="nsew", padx=(5, 10), pady=10)
        right_panel.pack_propagate(False)

        tk.Label(right_panel, text="🏷️ Action Class", font=("Inter", 10, "bold"), fg="#f8fafc", bg="#1e293b").pack(anchor="w", padx=14, pady=(12, 6))
        class_btn_frame = tk.Frame(right_panel, bg="#1e293b")
        class_btn_frame.pack(fill="x", padx=14, pady=(0, 4))
        self.swim_btn = tk.Button(class_btn_frame, text="Swim (1)", font=("Inter", 9, "bold"), bg="#ff00cc", fg="#000", relief="flat", command=lambda: self.set_class("swim"))
        self.swim_btn.pack(side="left", expand=True, fill="x", padx=(0, 4), ipady=4)
        self.drown_btn = tk.Button(class_btn_frame, text="Drown (2)", font=("Inter", 9), bg="#334155", fg="#fff", relief="flat", command=lambda: self.set_class("drown"))
        self.drown_btn.pack(side="right", expand=True, fill="x", padx=(4, 0), ipady=4)
        tk.Label(right_panel, text="Chọn box rồi bấm 1/2 để đổi class", font=("Inter", 7), fg="#64748b", bg="#1e293b").pack(anchor="w", padx=14, pady=(0, 6))

        boxes_hdr = tk.Frame(right_panel, bg="#1e293b")
        boxes_hdr.pack(fill="x", padx=14, pady=(6, 4))
        tk.Label(boxes_hdr, text="📦 Boxes in Frame", font=("Inter", 9, "bold"), fg="#f8fafc", bg="#1e293b").pack(side="left")
        self.box_count_lbl = tk.Label(boxes_hdr, text="0", font=("Inter", 8, "bold"), fg="#38bdf8", bg="#1e293b")
        self.box_count_lbl.pack(side="right")

        boxes_box_frame = tk.Frame(right_panel, bg="#0f172a")
        boxes_box_frame.pack(fill="both", expand=True, padx=14, pady=4)
        self.boxes_listbox = tk.Listbox(boxes_box_frame, bg="#0f172a", fg="#f8fafc", selectbackground="#4f46e5", selectforeground="#fff", font=("Inter", 8), bd=0, relief="flat", exportselection=False)
        self.boxes_listbox.pack(fill="both", expand=True)
        self.boxes_listbox.bind("<<ListboxSelect>>", self.on_box_select)

        actions_frame = tk.Frame(right_panel, bg="#1e293b")
        actions_frame.pack(fill="x", padx=14, pady=10)

        def action_btn(text, cmd, bg="#334155", bold=False):
            b = tk.Button(actions_frame, text=text, font=("Inter", 8, "bold" if bold else "normal"), bg=bg, fg="#fff", relief="flat", command=cmd)
            b.pack(fill="x", pady=2, ipady=3)
            return b

        action_btn("📋 Copy sang frame kế (C)", self.copy_boxes_to_next)
        action_btn("⚡ Nội suy tới keyframe kế (I)", self.interpolate_range)
        action_btn("⚡ Nội suy toàn clip (Shift+I)", self.interpolate_all)
        undo_row = tk.Frame(actions_frame, bg="#1e293b")
        undo_row.pack(fill="x", pady=2)
        tk.Button(undo_row, text="↶ Undo (Ctrl+Z)", font=("Inter", 8), bg="#334155", fg="#fff", relief="flat", command=self.undo).pack(side="left", expand=True, fill="x", padx=(0, 2), ipady=3)
        tk.Button(undo_row, text="↷ Redo (Ctrl+Y)", font=("Inter", 8), bg="#334155", fg="#fff", relief="flat", command=self.redo).pack(side="right", expand=True, fill="x", padx=(2, 0), ipady=3)
        action_btn("🗑️ Xoá box đang chọn (Del)", self.delete_selected_box, bg="#ef4444")
        action_btn("🧹 Xoá tất cả nhãn frame (X)", self.clear_current_frame_boxes, bg="#b91c1c", bold=True)
        save_btn = tk.Button(actions_frame, text="💾 Save Labels (Ctrl+S)", font=("Inter", 10, "bold"), bg="#10b981", fg="#fff", relief="flat", command=self.save_current_annotations)
        save_btn.pack(fill="x", pady=(8, 2), ipady=6)

        help_txt = ("Chuột trái: vẽ / chọn / kéo / resize box\n"
                    "Ctrl + kéo: vẽ box đè lên box khác\n"
                    "Lăn chuột: zoom • Chuột phải kéo: di chuyển ảnh\n"
                    "Shift + lăn: đổi frame • Shift+A/D: ±10 frame\n"
                    "Tab: chọn box kế • Esc: bỏ chọn")
        tk.Label(right_panel, text=help_txt, justify="left", font=("Inter", 7), fg="#64748b", bg="#1e293b").pack(anchor="w", padx=14, pady=(0, 10))

        # 4. BOTTOM PLAYBAR
        bottom_bar = tk.Frame(self.root, bg="#1e293b", height=70)
        bottom_bar.grid(row=1, column=0, columnspan=3, sticky="ew", padx=10, pady=(0, 4))

        ctrl_frame = tk.Frame(bottom_bar, bg="#1e293b")
        ctrl_frame.pack(side="left", padx=16, pady=8)
        tk.Button(ctrl_frame, text="◀ (A)", font=("Inter", 9), bg="#334155", fg="#fff", relief="flat", width=6, command=lambda: self.step_frame(-1)).pack(side="left", padx=2)
        self.play_btn = tk.Button(ctrl_frame, text="▶ Play (Space)", font=("Inter", 9, "bold"), bg="#4f46e5", fg="#fff", relief="flat", width=12, command=self.toggle_play)
        self.play_btn.pack(side="left", padx=4)
        tk.Button(ctrl_frame, text="(D) ▶", font=("Inter", 9), bg="#334155", fg="#fff", relief="flat", width=6, command=lambda: self.step_frame(1)).pack(side="left", padx=2)

        slider_frame = tk.Frame(bottom_bar, bg="#1e293b")
        slider_frame.pack(side="left", fill="x", expand=True, padx=16, pady=6)
        self.slider = ttk.Scale(slider_frame, from_=0, to=0, orient="horizontal", command=self.on_slider_move)
        self.slider.pack(fill="x")
        # Timeline strip: which frames already have boxes (tall = manual keyframe, short = interpolated)
        self.timeline = tk.Canvas(slider_frame, height=12, bg="#0f172a", highlightthickness=0, cursor="hand2")
        self.timeline.pack(fill="x", pady=(3, 0))
        self.timeline.bind("<Configure>", lambda e: self.schedule_timeline_redraw())
        self.timeline.bind("<Button-1>", self.on_timeline_click)
        self.timeline.bind("<B1-Motion>", self.on_timeline_click)

        fps_frame = tk.Frame(bottom_bar, bg="#1e293b")
        fps_frame.pack(side="right", padx=16)
        tk.Label(fps_frame, text="Speed:", font=("Inter", 8), fg="#94a3b8", bg="#1e293b").pack(side="left", padx=4)
        self.fps_var = tk.StringVar(value="15")
        fps_menu = ttk.Combobox(fps_frame, textvariable=self.fps_var, values=["5", "10", "15", "24", "30"], width=4, state="readonly")
        fps_menu.pack(side="left")
        fps_menu.bind("<<ComboboxSelected>>", lambda e: (self.set_fps(int(self.fps_var.get())), self.canvas.focus_set()))

        # 5. STATUS BAR
        status_bar = tk.Frame(self.root, bg="#0f172a")
        status_bar.grid(row=2, column=0, columnspan=3, sticky="ew", padx=12, pady=(0, 6))
        self.status_lbl = tk.Label(status_bar, text="Sẵn sàng", font=("Inter", 8), fg="#94a3b8", bg="#0f172a", anchor="w")
        self.status_lbl.pack(side="left", fill="x", expand=True)
        self.info_lbl = tk.Label(status_bar, text="", font=("Inter", 8), fg="#64748b", bg="#0f172a")
        self.info_lbl.pack(side="right")

    def bind_shortcuts(self):
        def bind(seqs, fn):
            def handler(event):
                # Don't hijack keys while the user is typing in an entry / combobox
                if isinstance(self.root.focus_get(), (tk.Entry, ttk.Entry, tk.Spinbox)):
                    return None
                fn()
                return "break"
            for s in seqs:
                self.root.bind(s, handler)

        bind(["<Key-1>"], lambda: self.set_class("swim"))
        bind(["<Key-2>"], lambda: self.set_class("drown"))
        bind(["<Key-a>", "<Key-A>", "<Left>"], lambda: self.step_frame(-1))
        bind(["<Key-d>", "<Key-D>", "<Right>"], lambda: self.step_frame(1))
        bind(["<Shift-Left>", "<Shift-KeyPress-A>"], lambda: self.step_frame(-10))
        bind(["<Shift-Right>", "<Shift-KeyPress-D>"], lambda: self.step_frame(10))
        bind(["<Home>"], lambda: self.load_frame(0))
        bind(["<End>"], lambda: self.load_frame(self.total_frames - 1))
        bind(["<space>"], self.toggle_play)
        bind(["<Key-c>", "<Key-C>"], self.copy_boxes_to_next)
        bind(["<Key-i>", "<Key-I>"], self.interpolate_range)
        bind(["<Shift-KeyPress-I>"], self.interpolate_all)
        bind(["<Delete>", "<BackSpace>"], self.delete_selected_box)
        bind(["<Key-x>", "<Key-X>", "<Shift-Delete>", "<Shift-BackSpace>"], self.clear_current_frame_boxes)
        bind(["<Control-s>", "<Control-S>"], self.save_current_annotations)
        bind(["<Control-z>", "<Control-Z>"], self.undo)
        bind(["<Control-y>", "<Control-Y>", "<Control-Shift-Z>"], self.redo)
        bind(["<Key-f>", "<Key-F>"], self.fit_view)
        bind(["<Key-z>", "<Key-Z>"], self.zoom_to_selected)
        bind(["<Tab>"], self.select_next_box)
        bind(["<Escape>"], self.deselect_box)

    # ================================================================== dataset
    def load_dataset(self):
        self.snippets = load_dataset_snippets(self.archive_dir)
        self.update_stats()
        self.render_snippet_list()

    def update_stats(self):
        unlabeled = [s for s in self.snippets if s["split"] == "unlabeled"]
        total = len(unlabeled)
        completed = sum(1 for s in unlabeled if s["is_labeled"])
        pct = int(completed / total * 100) if total > 0 else 0
        self.stats_lbl.config(text=f"Unlabeled: {completed} / {total} ({pct}%)")
        self.progress_bar.config(value=pct)

    def switch_tab(self, tab):
        self.current_tab = tab
        on = dict(bg="#4f46e5", fg="#fff", font=("Inter", 8, "bold"))
        off = dict(bg="#334155", fg="#94a3b8", font=("Inter", 8))
        self.tab_unlabeled_btn.config(**(on if tab == "unlabeled" else off))
        self.tab_all_btn.config(**(on if tab == "all" else off))
        self.render_snippet_list()

    def render_snippet_list(self):
        self.snippet_listbox.delete(0, tk.END)
        query = self.search_entry.get().lower()
        self.filtered_snippets = [
            s for s in self.snippets
            if (self.current_tab == "all" or s["split"] == "unlabeled") and query in s["snippet_id"].lower()
        ]
        for i, s in enumerate(self.filtered_snippets):
            status = "✅" if s["is_labeled"] else "⏳"
            self.snippet_listbox.insert(tk.END, f"{status} {s['snippet_id']} ({s['frame_count']}f)")
            if s["snippet_id"] == self.current_snippet:
                self.snippet_listbox.selection_set(i)
                self.snippet_listbox.see(i)

    def on_snippet_select(self, event):
        sel = self.snippet_listbox.curselection()
        if not sel:
            return
        snippet_id = self.filtered_snippets[sel[0]]["snippet_id"]
        if snippet_id == self.current_snippet:
            return
        if not self.confirm_discard():
            self.render_snippet_list()  # restore the previous selection
            return
        self.load_snippet(snippet_id)

    def load_snippet(self, snippet_id):
        frames_dir = os.path.join(self.archive_dir, "frames", snippet_id)
        if not os.path.isdir(frames_dir):
            messagebox.showerror("Error", f"Frames folder not found: {frames_dir}")
            return
        files = sorted(f for f in os.listdir(frames_dir) if f.lower().endswith(IMG_EXTS))
        if not files:
            messagebox.showerror("Error", f"Không có ảnh nào trong: {frames_dir}")
            return

        self.stop_play()
        self.current_snippet = snippet_id
        self.frames_dir = frames_dir
        self.frame_files = files
        self.total_frames = len(files)
        self.current_frame_idx = 0
        self.cache.clear()

        # Real frame size (224 for the original dataset, 640 / custom for extracted videos)
        first = self.cache.get(os.path.join(frames_dir, files[0]))
        self.img_w, self.img_h = first.size
        self.annotations = self.load_annotations(snippet_id)

        self.undo_stack.clear()
        self.redo_stack.clear()
        self.set_dirty(False)

        self.clip_title_lbl.config(text=f"Clip: {snippet_id}")
        self.slider.config(to=max(0, self.total_frames - 1))
        self.view.set_image_size(self.img_w, self.img_h, reset=True)
        self.load_frame(0)
        self.schedule_timeline_redraw()
        self.canvas.focus_set()
        n_labeled = sum(1 for v in self.annotations.values() if v)
        self.set_status(f"Đã mở {snippet_id}: {self.total_frames} frame, {self.img_w}x{self.img_h}, {n_labeled} frame có nhãn")

    def load_annotations(self, snippet_id):
        xml_p = os.path.join(self.archive_dir, "labels", f"{snippet_id}.xml")
        txt_d = os.path.join(self.archive_dir, "labels", snippet_id)
        anns = {}
        if os.path.exists(xml_p):
            parsed = parse_cvat_xml(xml_p)
            if parsed:
                anns = parsed["annotations"]
                # Rescale if the XML was made for a different frame size
                meta = parsed.get("meta", {})
                sx = self.img_w / float(meta.get("width") or self.img_w)
                sy = self.img_h / float(meta.get("height") or self.img_h)
                if abs(sx - 1) > 1e-6 or abs(sy - 1) > 1e-6:
                    for boxes in anns.values():
                        for b in boxes:
                            b["xtl"], b["xbr"] = round(b["xtl"] * sx, 2), round(b["xbr"] * sx, 2)
                            b["ytl"], b["ybr"] = round(b["ytl"] * sy, 2), round(b["ybr"] * sy, 2)
        elif os.path.isdir(txt_d) and any(f.endswith(".txt") for f in os.listdir(txt_d)):
            parsed = parse_yolo_txt(txt_d, self.img_w, self.img_h)
            if parsed:
                anns = parsed["annotations"]
        return {int(k): v for k, v in anns.items() if v}

    # ================================================================== frames & drawing
    def load_frame(self, frame_idx):
        if not self.current_snippet or frame_idx < 0 or frame_idx >= self.total_frames:
            return
        if self.drag:  # frame changed mid-drag (keyboard) -> cancel the drag
            self.drag = None
            self.canvas.delete("temp")
        self.current_frame_idx = frame_idx
        self.selected_box_idx = -1
        self.current_pil = self.cache.get(os.path.join(self.frames_dir, self.frame_files[frame_idx]))
        self.render_frame()

        boxes = self.annotations.get(frame_idx, [])
        kind = ""
        if boxes:
            kind = "  • keyframe" if any(b.get("source") != "interpolation" for b in boxes) else "  • nội suy"
        self.frame_counter_lbl.config(text=f"Frame: {frame_idx + 1} / {self.total_frames}{kind}")
        self._slider_lock = True
        self.slider.set(frame_idx)
        self._slider_lock = False
        self.update_boxes_listbox()
        self.draw_timeline_cursor()

    def on_canvas_resize(self, event):
        self.view.on_resize()
        self.render_frame()

    def render_frame(self):
        self.view.render(self.current_pil)
        self.draw_boxes()
        self.info_lbl.config(text=f"{self.img_w}x{self.img_h}  |  Zoom {self.view.scale * 100:.0f}%")

    def box_canvas_rect(self, b):
        x1, y1 = self.view.to_canvas(b["xtl"], b["ytl"])
        x2, y2 = self.view.to_canvas(b["xbr"], b["ybr"])
        return x1, y1, x2, y2

    @staticmethod
    def handle_points(x1, y1, x2, y2):
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        return {"nw": (x1, y1), "n": (mx, y1), "ne": (x2, y1), "e": (x2, my),
                "se": (x2, y2), "s": (mx, y2), "sw": (x1, y2), "w": (x1, my)}

    def draw_boxes(self):
        self.canvas.delete("box")
        for i, b in enumerate(self.annotations.get(self.current_frame_idx, [])):
            is_sel = (i == self.selected_box_idx)
            color = CLASS_COLORS.get(b["label"], "#ff00cc")
            x1, y1, x2, y2 = self.box_canvas_rect(b)
            dash = (5, 3) if b.get("source") == "interpolation" else None
            self.canvas.create_rectangle(x1, y1, x2, y2, outline=color, width=3 if is_sel else 2, dash=dash, tags="box")

            text = f"{i + 1}:{b['label'].upper()}"
            ty = y1 - 16 if y1 >= 16 else y1
            self.canvas.create_rectangle(x1, ty, x1 + len(text) * 7 + 8, ty + 16, fill=color, outline=color, tags="box")
            self.canvas.create_text(x1 + 4, ty + 8, text=text, fill="#000", font=("Inter", 8, "bold"), anchor="w", tags="box")

            if is_sel:
                for px, py in self.handle_points(x1, y1, x2, y2).values():
                    self.canvas.create_rectangle(px - 4, py - 4, px + 4, py + 4, fill="#ffffff", outline=color, tags="box")

    # ================================================================== mouse editing
    def hit_test(self, cx, cy):
        """Return (box_idx, 'nw'|'n'|...|'move') for the canvas point, or (-1, None)."""
        boxes = self.annotations.get(self.current_frame_idx, [])
        if 0 <= self.selected_box_idx < len(boxes):
            pts = self.handle_points(*self.box_canvas_rect(boxes[self.selected_box_idx]))
            for name, (px, py) in pts.items():
                if abs(cx - px) <= HANDLE_PX and abs(cy - py) <= HANDLE_PX:
                    return self.selected_box_idx, name
        # Prefer the selected box, then the top-most (last drawn) box
        order = list(range(len(boxes) - 1, -1, -1))
        if 0 <= self.selected_box_idx < len(boxes):
            order.remove(self.selected_box_idx)
            order.insert(0, self.selected_box_idx)
        for i in order:
            x1, y1, x2, y2 = self.box_canvas_rect(boxes[i])
            if x1 <= cx <= x2 and y1 <= cy <= y2:
                return i, "move"
        return -1, None

    def event_image_xy(self, event):
        return self.view.clamp_to_image(*self.view.to_image(event.x, event.y))

    def on_canvas_press(self, event):
        self.canvas.focus_set()
        if not self.current_snippet:
            return
        self.stop_play()
        ix, iy = self.event_image_xy(event)
        force_draw = bool(event.state & 0x0004)  # Ctrl held
        idx, handle = (-1, None) if force_draw else self.hit_test(event.x, event.y)
        self._drag_snap = self.snapshot([self.current_frame_idx])

        if idx >= 0:
            self.selected_box_idx = idx
            box = self.annotations[self.current_frame_idx][idx]
            self.drag = {"mode": "move" if handle == "move" else "resize", "handle": handle,
                         "idx": idx, "start": (ix, iy), "orig": dict(box)}
        else:
            self.selected_box_idx = -1
            self.drag = {"mode": "draw", "start": (ix, iy)}
        self.draw_boxes()
        self.update_boxes_listbox()

    def on_canvas_drag(self, event):
        d = self.drag
        if not d:
            return
        ix, iy = self.event_image_xy(event)
        sx, sy = d["start"]

        if d["mode"] == "draw":
            self.canvas.delete("temp")
            color = CLASS_COLORS[self.active_class]
            (x1, y1), (x2, y2) = self.view.to_canvas(sx, sy), self.view.to_canvas(ix, iy)
            self.canvas.create_rectangle(x1, y1, x2, y2, outline=color, width=2, dash=(4, 4), tags="temp")
            self.show_cursor_info(ix, iy, extra=f"  box {abs(ix - sx):.0f}x{abs(iy - sy):.0f}")
            return

        box = self.annotations[self.current_frame_idx][d["idx"]]
        o = d["orig"]
        if d["mode"] == "move":
            w, h = o["xbr"] - o["xtl"], o["ybr"] - o["ytl"]
            nx = max(0.0, min(self.img_w - w, o["xtl"] + ix - sx))
            ny = max(0.0, min(self.img_h - h, o["ytl"] + iy - sy))
            x1, y1, x2, y2 = nx, ny, nx + w, ny + h
        else:
            x1, y1, x2, y2 = o["xtl"], o["ytl"], o["xbr"], o["ybr"]
            h = d["handle"]
            if "w" in h:
                x1 = ix
            if "e" in h:
                x2 = ix
            if "n" in h:
                y1 = iy
            if "s" in h:
                y2 = iy
            x1, x2 = min(x1, x2), max(x1, x2)
            y1, y2 = min(y1, y2), max(y1, y2)
        box.update(xtl=round(x1, 2), ytl=round(y1, 2), xbr=round(x2, 2), ybr=round(y2, 2), source="manual")
        self.draw_boxes()
        self.show_cursor_info(ix, iy, extra=f"  box {x2 - x1:.0f}x{y2 - y1:.0f}")

    def on_canvas_release(self, event):
        d = self.drag
        if not d:
            return
        self.drag = None
        f = self.current_frame_idx

        if d["mode"] == "draw":
            self.canvas.delete("temp")
            (sx, sy), (ix, iy) = d["start"], self.event_image_xy(event)
            x1, y1, x2, y2 = min(sx, ix), min(sy, iy), max(sx, ix), max(sy, iy)
            if (x2 - x1) >= MIN_BOX and (y2 - y1) >= MIN_BOX:
                self.annotations.setdefault(f, []).append({
                    "label": self.active_class,
                    "class_id": CLASS_MAP[self.active_class],
                    "xtl": round(x1, 2), "ytl": round(y1, 2),
                    "xbr": round(x2, 2), "ybr": round(y2, 2),
                    "occluded": 0, "source": "manual", "z_order": 0
                })
                self.selected_box_idx = len(self.annotations[f]) - 1
        else:
            box = self.annotations[f][d["idx"]]
            if (box["xbr"] - box["xtl"]) < MIN_BOX or (box["ybr"] - box["ytl"]) < MIN_BOX:
                box.update(d["orig"])  # too small -> revert

        if self.annotations.get(f, []) != self._drag_snap[f]:
            self.push_undo(self._drag_snap)
        self.draw_boxes()
        self.update_boxes_listbox()
        self.schedule_timeline_redraw()

    def on_canvas_motion(self, event):
        if not self.current_snippet:
            return
        if self.drag is None:
            _, handle = self.hit_test(event.x, event.y)
            cursor = HANDLE_CURSORS.get(handle, "crosshair")
            if cursor != self._cursor:
                try:
                    self.canvas.config(cursor=cursor)
                except tk.TclError:  # cursor name not available on this platform
                    self.canvas.config(cursor="crosshair")
                self._cursor = cursor
        self.show_cursor_info(*self.view.to_image(event.x, event.y))

    def show_cursor_info(self, ix, iy, extra=""):
        self.info_lbl.config(text=f"x={ix:.0f} y={iy:.0f}{extra}  |  {self.img_w}x{self.img_h}  |  Zoom {self.view.scale * 100:.0f}%")

    # ================================================================== view
    def fit_view(self):
        self.view.fit()
        self.render_frame()

    def zoom_to_selected(self):
        boxes = self.annotations.get(self.current_frame_idx, [])
        if 0 <= self.selected_box_idx < len(boxes):
            b = boxes[self.selected_box_idx]
            self.view.zoom_to_rect(b["xtl"], b["ytl"], b["xbr"] - b["xtl"], b["ybr"] - b["ytl"], pad=0.6)
        else:
            self.view.zoom_center(2.0)

    # ================================================================== box list / selection
    def set_class(self, cls):
        self.active_class = cls
        if cls == "swim":
            self.swim_btn.config(bg=CLASS_COLORS["swim"], fg="#000", font=("Inter", 9, "bold"))
            self.drown_btn.config(bg="#334155", fg="#fff", font=("Inter", 9))
        else:
            self.swim_btn.config(bg="#334155", fg="#fff", font=("Inter", 9))
            self.drown_btn.config(bg=CLASS_COLORS["drown"], fg="#000", font=("Inter", 9, "bold"))

        # Re-label the selected box
        boxes = self.annotations.get(self.current_frame_idx, [])
        if 0 <= self.selected_box_idx < len(boxes) and boxes[self.selected_box_idx]["label"] != cls:
            snap = self.snapshot([self.current_frame_idx])
            boxes[self.selected_box_idx].update(label=cls, class_id=CLASS_MAP[cls])
            self.push_undo(snap)
            self.draw_boxes()
            self.update_boxes_listbox()
            self.schedule_timeline_redraw()

    def update_boxes_listbox(self):
        self.boxes_listbox.delete(0, tk.END)
        boxes = self.annotations.get(self.current_frame_idx, [])
        self.box_count_lbl.config(text=str(len(boxes)))
        for i, b in enumerate(boxes):
            src = "~" if b.get("source") == "interpolation" else " "
            self.boxes_listbox.insert(tk.END, f"{i + 1}.{src}{b['label'].upper()} ({int(b['xtl'])},{int(b['ytl'])})-({int(b['xbr'])},{int(b['ybr'])})")
        if 0 <= self.selected_box_idx < len(boxes):
            self.boxes_listbox.selection_set(self.selected_box_idx)

    def on_box_select(self, event):
        sel = self.boxes_listbox.curselection()
        if not sel:
            return
        self.selected_box_idx = sel[0]
        self.draw_boxes()
        self.canvas.focus_set()

    def select_next_box(self):
        boxes = self.annotations.get(self.current_frame_idx, [])
        if boxes:
            self.selected_box_idx = (self.selected_box_idx + 1) % len(boxes)
            self.draw_boxes()
            self.update_boxes_listbox()

    def deselect_box(self):
        self.selected_box_idx = -1
        self.draw_boxes()
        self.update_boxes_listbox()

    # ================================================================== edit commands
    def delete_selected_box(self):
        boxes = self.annotations.get(self.current_frame_idx, [])
        if 0 <= self.selected_box_idx < len(boxes):
            snap = self.snapshot([self.current_frame_idx])
            boxes.pop(self.selected_box_idx)
            if not boxes:
                self.annotations.pop(self.current_frame_idx, None)
            self.selected_box_idx = -1
            self.push_undo(snap)
            self.after_edit()

    def clear_current_frame_boxes(self):
        if self.annotations.get(self.current_frame_idx):
            snap = self.snapshot([self.current_frame_idx])
            self.annotations.pop(self.current_frame_idx, None)
            self.selected_box_idx = -1
            self.push_undo(snap)
            self.after_edit()

    def copy_boxes_to_next(self):
        boxes = self.annotations.get(self.current_frame_idx, [])
        if not boxes or self.current_frame_idx >= self.total_frames - 1:
            return
        nxt = self.current_frame_idx + 1
        snap = self.snapshot([nxt])
        self.annotations[nxt] = [dict(b, source="manual") for b in copy.deepcopy(boxes)]
        self.push_undo(snap)
        self.schedule_timeline_redraw()
        self.step_frame(1)

    @staticmethod
    def match_boxes(start_boxes, end_boxes):
        """Pair boxes between two keyframes: by track_id if present, else greedy nearest centers."""
        if all(b.get("track_id") is not None for b in start_boxes + end_boxes):
            end_by_id = {b["track_id"]: b for b in end_boxes}
            return [(s, end_by_id[s["track_id"]]) for s in start_boxes if s["track_id"] in end_by_id]
        cands = []
        for i, s in enumerate(start_boxes):
            scx, scy = box_center(s)
            for j, e in enumerate(end_boxes):
                ecx, ecy = box_center(e)
                penalty = 0 if s["label"] == e["label"] else 1e6
                cands.append(((scx - ecx) ** 2 + (scy - ecy) ** 2 + penalty, i, j))
        used_s, used_e, pairs = set(), set(), {}
        for _, i, j in sorted(cands):
            if i not in used_s and j not in used_e:
                used_s.add(i)
                used_e.add(j)
                pairs[i] = j
        return [(start_boxes[i], end_boxes[pairs[i]]) for i in sorted(pairs)]

    def _interpolate_between(self, a, b):
        pairs = self.match_boxes(self.annotations[a], self.annotations[b])
        if not pairs:
            return 0
        steps = b - a
        for f in range(a + 1, b):
            t = (f - a) / float(steps)
            self.annotations[f] = [
                dict(s, **{k: round(s[k] + t * (e[k] - s[k]), 2) for k in ("xtl", "ytl", "xbr", "ybr")},
                     source="interpolation")
                for s, e in pairs
            ]
        return steps - 1

    def interpolate_range(self):
        cur = self.current_frame_idx
        if not self.annotations.get(cur):
            self.set_status("⚠ Vẽ box ở keyframe bắt đầu trước khi nội suy", warn=True)
            return
        end_idx = next((f for f in range(cur + 1, self.total_frames) if self.annotations.get(f)), -1)
        if end_idx == -1:
            self.set_status("⚠ Không tìm thấy keyframe phía sau để nội suy tới", warn=True)
            return
        if end_idx == cur + 1:
            self.set_status("Frame kế tiếp đã có nhãn, không có gì để nội suy")
            return
        snap = self.snapshot(range(cur + 1, end_idx))
        n = self._interpolate_between(cur, end_idx)
        self.push_undo(snap)
        self.after_edit()
        self.set_status(f"⚡ Đã nội suy {n} frame ({cur + 1} → {end_idx + 1})")

    def interpolate_all(self):
        keys = sorted(f for f, b in self.annotations.items() if b)
        gaps = [(a, b) for a, b in zip(keys, keys[1:]) if b - a > 1]
        if not gaps:
            self.set_status("Không có khoảng trống nào giữa các keyframe để nội suy")
            return
        snap = self.snapshot([f for a, b in gaps for f in range(a + 1, b)])
        n = sum(self._interpolate_between(a, b) for a, b in gaps)
        self.push_undo(snap)
        self.after_edit()
        self.set_status(f"⚡ Đã nội suy {n} frame trong {len(gaps)} đoạn")

    def after_edit(self):
        self.draw_boxes()
        self.update_boxes_listbox()
        self.schedule_timeline_redraw()

    # ================================================================== undo / redo
    def snapshot(self, frames):
        return {f: copy.deepcopy(self.annotations.get(f, [])) for f in frames}

    def _apply_snapshot(self, snap):
        for f, boxes in snap.items():
            if boxes:
                self.annotations[f] = copy.deepcopy(boxes)
            else:
                self.annotations.pop(f, None)

    def push_undo(self, snap):
        self.undo_stack.append(snap)
        if len(self.undo_stack) > UNDO_LIMIT:
            self.undo_stack.pop(0)
        self.redo_stack.clear()
        self.set_dirty(True)

    def _undo_redo(self, src, dst, name):
        if not src:
            return
        snap = src.pop()
        dst.append(self.snapshot(snap.keys()))
        self._apply_snapshot(snap)
        self.set_dirty(True)
        if self.current_frame_idx not in snap:
            self.load_frame(min(snap))  # jump to where the change happened
        else:
            self.selected_box_idx = -1
            self.after_edit()
        self.schedule_timeline_redraw()
        self.set_status(f"{name} ({len(src)} bước còn lại)")

    def undo(self):
        self._undo_redo(self.undo_stack, self.redo_stack, "↶ Undo")

    def redo(self):
        self._undo_redo(self.redo_stack, self.undo_stack, "↷ Redo")

    # ================================================================== timeline strip
    def schedule_timeline_redraw(self):
        if self._timeline_job is None:
            self._timeline_job = self.root.after_idle(self._redraw_timeline)

    def _redraw_timeline(self):
        self._timeline_job = None
        tl = self.timeline
        tl.delete("all")
        if self.total_frames <= 0:
            return
        w = max(1, tl.winfo_width())
        denom = max(1, self.total_frames - 1)
        cols = {}  # pixel x -> (is_drown, is_keyframe)
        for f, boxes in self.annotations.items():
            if not boxes:
                continue
            x = int(f * (w - 1) / denom)
            drown = any(b["label"] == "drown" for b in boxes)
            key = any(b.get("source") != "interpolation" for b in boxes)
            pd, pk = cols.get(x, (False, False))
            cols[x] = (drown or pd, key or pk)
        for x, (drown, key) in cols.items():
            tl.create_line(x, 12 if key else 6, x, 12, fill=CLASS_COLORS["drown" if drown else "swim"])
        self.draw_timeline_cursor()

    def draw_timeline_cursor(self):
        tl = self.timeline
        tl.delete("cursor")
        if self.total_frames <= 0:
            return
        x = int(self.current_frame_idx * (max(1, tl.winfo_width()) - 1) / max(1, self.total_frames - 1))
        tl.create_rectangle(x - 1, 0, x + 1, 12, fill="#38bdf8", outline="", tags="cursor")

    def on_timeline_click(self, event):
        if self.total_frames <= 0:
            return
        w = max(1, self.timeline.winfo_width())
        f = int(round(event.x / max(1, w - 1) * (self.total_frames - 1)))
        self.load_frame(max(0, min(self.total_frames - 1, f)))

    # ================================================================== playback
    def step_frame(self, delta):
        if self.total_frames <= 0:
            return
        new_idx = max(0, min(self.total_frames - 1, self.current_frame_idx + delta))
        if new_idx != self.current_frame_idx:
            self.load_frame(new_idx)

    def on_slider_move(self, val):
        if self._slider_lock:
            return
        target = int(float(val))
        if target != self.current_frame_idx:
            self.load_frame(target)

    def toggle_play(self):
        if self.is_playing:
            self.stop_play()
            return
        if not self.current_snippet:
            return
        self.is_playing = True
        self.play_btn.config(text="⏸ Pause (Space)", bg="#ef4444")
        self.schedule_next_frame()

    def stop_play(self):
        self.is_playing = False
        self.play_btn.config(text="▶ Play (Space)", bg="#4f46e5")
        if self.play_job:
            self.root.after_cancel(self.play_job)
            self.play_job = None

    def schedule_next_frame(self):
        if not self.is_playing:
            return
        if self.current_frame_idx >= self.total_frames - 1:
            self.stop_play()
            return
        t0 = time.perf_counter()
        self.step_frame(1)
        spent_ms = (time.perf_counter() - t0) * 1000
        delay = max(1, int(1000 / self.fps - spent_ms))
        self.play_job = self.root.after(delay, self.schedule_next_frame)

    def set_fps(self, fps):
        self.fps = fps

    # ================================================================== save / status
    def set_dirty(self, dirty):
        self.dirty = dirty
        base = "Lifeguard Frame Labeler - Desktop Application"
        if self.current_snippet:
            base += f"  [{self.current_snippet}{' *' if dirty else ''}]"
        self.root.title(base)

    def set_status(self, text, warn=False):
        self.status_lbl.config(text=text, fg="#f59e0b" if warn else "#94a3b8")

    def save_current_annotations(self):
        if not self.current_snippet:
            return False
        labels_dir = os.path.join(self.archive_dir, "labels")
        try:
            xml_path = save_cvat_xml(
                self.current_snippet, self.annotations,
                width=self.img_w, height=self.img_h, total_frames=self.total_frames,
                output_path=os.path.join(labels_dir, f"{self.current_snippet}.xml")
            )
            txt_dir = save_yolo_txt(
                self.current_snippet, self.annotations,
                width=self.img_w, height=self.img_h, total_frames=self.total_frames,
                output_dir=os.path.join(labels_dir, self.current_snippet)
            )
        except Exception as ex:
            messagebox.showerror("Lỗi lưu nhãn", str(ex))
            return False

        self.set_dirty(False)
        for s in self.snippets:
            if s["snippet_id"] == self.current_snippet:
                s["is_labeled"] = True
        self.update_stats()
        self.render_snippet_list()
        self.set_status(f"💾 Đã lưu lúc {time.strftime('%H:%M:%S')} → {self.short_path(xml_path)} + {self.short_path(txt_dir)}/")
        return True

    @staticmethod
    def short_path(path):
        try:
            return os.path.relpath(path, PROJECT_DIR)
        except ValueError:  # different drive on Windows
            return path

    def confirm_discard(self):
        """Ask to save unsaved changes. Returns False if the user cancelled."""
        if not self.dirty:
            return True
        ans = messagebox.askyesnocancel("Chưa lưu", f"Clip '{self.current_snippet}' có thay đổi chưa lưu.\nLưu trước khi tiếp tục?")
        if ans is None:
            return False
        if ans:
            return self.save_current_annotations()
        return True

    def on_close(self):
        if self.confirm_discard():
            self.root.destroy()

    # ================================================================== delete clip
    def on_snippet_right_click(self, event):
        idx = self.snippet_listbox.nearest(event.y)
        if idx < 0 or idx >= len(self.filtered_snippets):
            return
        bbox = self.snippet_listbox.bbox(idx)
        if not bbox or not (bbox[1] <= event.y <= bbox[1] + bbox[3]):
            return  # clicked below the last item
        self._menu_snippet = self.filtered_snippets[idx]["snippet_id"]
        self.snippet_menu.tk_popup(event.x_root, event.y_root)
        self.snippet_menu.grab_release()

    def open_snippet_folder(self, snippet_id):
        folder = os.path.join(self.archive_dir, "frames", snippet_id or "")
        if snippet_id and os.path.isdir(folder):
            try:
                os.startfile(folder)
            except (AttributeError, OSError):
                pass

    def delete_selected_snippet(self):
        sel = self.snippet_listbox.curselection()
        if not sel:
            messagebox.showinfo("Xoá bộ frame", "Hãy chọn một clip trong danh sách trước (hoặc chuột phải vào clip).")
            return
        self.confirm_delete_snippet(self.filtered_snippets[sel[0]]["snippet_id"])

    def confirm_delete_snippet(self, snippet_id):
        if not snippet_id:
            return
        meta = next((s for s in self.snippets if s["snippet_id"] == snippet_id), None)
        paths = snippet_paths(snippet_id, self.archive_dir)
        splits = sorted({s["split"] for s in self.snippets if s["snippet_id"] == snippet_id})

        dlg = tk.Toplevel(self.root)
        dlg.title("Xoá bộ frame")
        dlg.configure(bg="#1e293b")
        dlg.resizable(False, False)
        dlg.transient(self.root)

        tk.Label(dlg, text="🗑️ Xoá bộ frame", font=("Inter", 11, "bold"), fg="#fca5a5", bg="#1e293b").pack(anchor="w", padx=16, pady=(14, 4))
        tk.Label(dlg, text=snippet_id, font=("Inter", 9, "bold"), fg="#f8fafc", bg="#1e293b", wraplength=440, justify="left").pack(anchor="w", padx=16)
        info = [f"• {meta['frame_count'] if meta else 0} frame trong archive/frames/{snippet_id}/"]
        if "xml" in paths or "txt" in paths:
            info.append("• Có nhãn: " + " + ".join(k.upper() for k in ("xml", "txt") if k in paths))
        if splits:
            info.append(f"• Sẽ bị gỡ khỏi: {', '.join(sp + '.csv' for sp in splits)}")
        if {"train", "valid", "test"} & set(splits):
            info.append("⚠ Clip này thuộc tập train/valid/test của dataset gốc!")
        tk.Label(dlg, text="\n".join(info), font=("Inter", 8), fg="#cbd5e1", bg="#1e293b", justify="left").pack(anchor="w", padx=16, pady=(6, 8))

        labels_var = tk.BooleanVar(value=True)
        permanent_var = tk.BooleanVar(value=False)
        opts = dict(bg="#1e293b", fg="#f8fafc", selectcolor="#0f172a", activebackground="#1e293b", activeforeground="#fff", font=("Inter", 8))
        tk.Checkbutton(dlg, text="Xoá cả nhãn (XML + YOLO TXT)", variable=labels_var, **opts).pack(anchor="w", padx=12)
        tk.Checkbutton(dlg, text="Xoá vĩnh viễn (không chuyển vào archive/_trash, không khôi phục được)", variable=permanent_var, **opts).pack(anchor="w", padx=12)

        result = {"ok": False}

        def do_delete():
            result["ok"] = True
            dlg.destroy()

        row = tk.Frame(dlg, bg="#1e293b")
        row.pack(fill="x", padx=16, pady=(12, 14))
        tk.Button(row, text="Huỷ", font=("Inter", 9), bg="#334155", fg="#fff", relief="flat", width=10, command=dlg.destroy).pack(side="right")
        tk.Button(row, text="Xoá", font=("Inter", 9, "bold"), bg="#dc2626", fg="#fff", relief="flat", width=10, command=do_delete).pack(side="right", padx=6)
        dlg.bind("<Escape>", lambda e: dlg.destroy())

        dlg.update_idletasks()
        x = self.root.winfo_rootx() + (self.root.winfo_width() - dlg.winfo_width()) // 2
        y = self.root.winfo_rooty() + (self.root.winfo_height() - dlg.winfo_height()) // 3
        dlg.geometry(f"+{max(0, x)}+{max(0, y)}")
        dlg.grab_set()
        dlg.focus_set()
        self.root.wait_window(dlg)
        if not result["ok"]:
            return

        if snippet_id == self.current_snippet:
            self.close_snippet()
        try:
            trash_dir, removed, edited = delete_snippet(
                snippet_id, self.archive_dir,
                delete_labels=labels_var.get(), permanent=permanent_var.get()
            )
        except Exception as ex:
            messagebox.showerror("Lỗi xoá", f"Không xoá được '{snippet_id}':\n{ex}")
            self.load_dataset()
            return

        self.load_dataset()
        where = f"đã chuyển vào {self.short_path(trash_dir)}" if trash_dir else "đã xoá vĩnh viễn"
        self.set_status(f"🗑️ {snippet_id}: {len(removed)} mục {where}" + (f", gỡ khỏi {', '.join(edited)}" if edited else ""))

    def close_snippet(self):
        """Unload the current clip (used before deleting it)."""
        self.stop_play()
        self.current_snippet = None
        self.frames_dir = None
        self.frame_files = []
        self.total_frames = 0
        self.current_frame_idx = 0
        self.annotations = {}
        self.selected_box_idx = -1
        self.current_pil = None
        self.drag = None
        self.cache.clear()
        self.undo_stack.clear()
        self.redo_stack.clear()
        self.set_dirty(False)
        self.canvas.delete("all")
        self.clip_title_lbl.config(text="Select a clip to begin")
        self.frame_counter_lbl.config(text="Frame: 0 / 0")
        self._slider_lock = True
        self.slider.config(to=0)
        self.slider.set(0)
        self._slider_lock = False
        self.update_boxes_listbox()
        self.schedule_timeline_redraw()

    def open_frame_extractor(self):
        if not os.path.exists(EXTRACTOR_SCRIPT):
            messagebox.showerror("Không tìm thấy", f"Không tìm thấy tool tách frame:\n{EXTRACTOR_SCRIPT}")
            return
        subprocess.Popen([sys.executable, EXTRACTOR_SCRIPT], cwd=os.path.dirname(EXTRACTOR_SCRIPT))
        self.set_status("Đã mở tool tách frame. Bấm 🔄 sau khi tách xong để cập nhật danh sách clip.")


if __name__ == "__main__":
    root = tk.Tk()
    app = DesktopFrameLabeler(root)
    root.mainloop()
