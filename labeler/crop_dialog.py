"""Progress and cancellation while an inline crop is applied."""
import queue
import threading
import tkinter as tk
from tkinter import messagebox, ttk

from clip_crop import CropCancelled


class CropProgressDialog:
    def __init__(self, parent, task, on_complete):
        self.busy = True
        self.cancelled = threading.Event()
        self.messages = queue.Queue()
        self.on_complete = on_complete
        self.window = tk.Toplevel(parent)
        self.window.title("Đang crop và thay thế clip")
        self.window.configure(bg="#1e293b")
        self.window.transient(parent)
        self.window.resizable(False, False)
        self.window.protocol("WM_DELETE_WINDOW", self.close)
        self.status = tk.StringVar(master=self.window, value="Đang chuẩn bị bản crop…")
        tk.Label(self.window, textvariable=self.status, bg="#1e293b", fg="#f8fafc",
                 padx=20, pady=14).pack()
        self.progress = ttk.Progressbar(self.window, length=420, mode="determinate")
        self.progress.pack(padx=20, pady=5)
        tk.Button(self.window, text="Huỷ", command=self.close).pack(pady=12)
        self.window.grab_set()
        self.window.focus_set()

        def worker():
            try:
                result = task(lambda d, t, stage: self.messages.put(("progress", (d, t, stage))),
                              self.cancelled.is_set)
                self.messages.put(("done", result))
            except Exception as error:
                self.messages.put(("error", error))

        threading.Thread(target=worker, daemon=True).start()
        self.window.after(100, self.poll)

    def close(self):
        if self.busy:
            self.cancelled.set()
            self.status.set("Đang huỷ và dọn bản crop chưa hoàn tất…")
        else:
            self.window.destroy()

    def poll(self):
        last = None
        while True:
            try:
                kind, value = self.messages.get_nowait()
            except queue.Empty:
                break
            if kind == "progress":
                last = value
                continue
            self.busy = False
            self.window.destroy()
            if kind == "done":
                self.on_complete(value)
            elif not isinstance(value, CropCancelled):
                messagebox.showerror("Không crop được clip", str(value))
            return
        if last:
            done, total, stage = last
            self.progress["value"] = done / max(1, total) * 100
            self.status.set(f"Đang crop frame: {done}/{total}" if stage == "frames" else "Đang crop MP4 và giữ âm thanh…")
        self.window.after(100, self.poll)
