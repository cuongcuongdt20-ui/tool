"""
Zoom / pan helper for a Tkinter Canvas that displays one image.

- Mouse wheel        : zoom in/out around the cursor
- Right / middle drag: pan
- Only the visible part of the image is resized each redraw, so zooming
  deep into a large frame stays fast.
"""
import math
from PIL import Image, ImageTk


class ZoomPanView:
    def __init__(self, canvas, on_change=None, on_shift_wheel=None, max_scale=20.0):
        self.canvas = canvas
        self.on_change = on_change            # called after zoom/pan -> app re-renders
        self.on_shift_wheel = on_shift_wheel  # optional: Shift+wheel handler(direction)
        self.max_scale = max_scale

        self.img_w = 0
        self.img_h = 0
        self.scale = 1.0
        self.ox = 0.0  # canvas x of image origin
        self.oy = 0.0  # canvas y of image origin
        self.fit_mode = True

        self._pan_last = None
        self._tk_img = None

        canvas.bind("<MouseWheel>", self._on_wheel)
        canvas.bind("<Button-4>", lambda e: self._on_wheel(e, 120))   # Linux
        canvas.bind("<Button-5>", lambda e: self._on_wheel(e, -120))  # Linux
        for btn in ("2", "3"):
            canvas.bind(f"<ButtonPress-{btn}>", self._pan_begin)
            canvas.bind(f"<B{btn}-Motion>", self._pan_move)
            canvas.bind(f"<ButtonRelease-{btn}>", self._pan_end)

    # ------------------------------------------------------------------ geometry
    def canvas_size(self):
        return max(1, self.canvas.winfo_width()), max(1, self.canvas.winfo_height())

    def set_image_size(self, w, h, reset=False):
        changed = (w, h) != (self.img_w, self.img_h)
        self.img_w, self.img_h = w, h
        if changed or reset or self.fit_mode:
            self.fit()

    def fit_scale(self):
        if not self.img_w or not self.img_h:
            return 1.0
        cw, ch = self.canvas_size()
        return min(cw / self.img_w, ch / self.img_h)

    def fit(self):
        if not self.img_w:
            return
        cw, ch = self.canvas_size()
        self.scale = self.fit_scale()
        self.ox = (cw - self.img_w * self.scale) / 2
        self.oy = (ch - self.img_h * self.scale) / 2
        self.fit_mode = True

    def on_resize(self):
        if self.fit_mode:
            self.fit()
        else:
            self._clamp_pan()

    def zoom_at(self, cx, cy, factor):
        if not self.img_w:
            return
        min_scale = self.fit_scale()
        new_scale = max(min_scale, min(self.max_scale, self.scale * factor))
        if new_scale <= min_scale + 1e-9:
            self.fit()
        else:
            f = new_scale / self.scale
            self.ox = cx - (cx - self.ox) * f
            self.oy = cy - (cy - self.oy) * f
            self.scale = new_scale
            self.fit_mode = False
            self._clamp_pan()
        self._notify()

    def zoom_center(self, factor):
        cw, ch = self.canvas_size()
        self.zoom_at(cw / 2, ch / 2, factor)

    def zoom_to_rect(self, x, y, w, h, pad=0.15):
        """Zoom so that image rect (x, y, w, h) fills the canvas."""
        if not self.img_w or w <= 0 or h <= 0:
            return
        cw, ch = self.canvas_size()
        s = min(cw / (w * (1 + 2 * pad)), ch / (h * (1 + 2 * pad)))
        self.scale = max(self.fit_scale(), min(self.max_scale, s))
        self.ox = cw / 2 - (x + w / 2) * self.scale
        self.oy = ch / 2 - (y + h / 2) * self.scale
        self.fit_mode = False
        self._clamp_pan()
        self._notify()

    def _clamp_pan(self):
        cw, ch = self.canvas_size()
        iw, ih = self.img_w * self.scale, self.img_h * self.scale
        self.ox = (cw - iw) / 2 if iw <= cw else min(0.0, max(cw - iw, self.ox))
        self.oy = (ch - ih) / 2 if ih <= ch else min(0.0, max(ch - ih, self.oy))

    def to_canvas(self, x, y):
        return self.ox + x * self.scale, self.oy + y * self.scale

    def to_image(self, cx, cy):
        return (cx - self.ox) / self.scale, (cy - self.oy) / self.scale

    def clamp_to_image(self, x, y):
        return max(0.0, min(float(self.img_w), x)), max(0.0, min(float(self.img_h), y))

    # ------------------------------------------------------------------ render
    def render(self, pil_img, tag="img"):
        """Draw the visible region of pil_img on the canvas (below everything else)."""
        self.canvas.delete(tag)
        if pil_img is None or not self.img_w:
            return
        cw, ch = self.canvas_size()
        x0, y0 = self.to_image(0, 0)
        x1, y1 = self.to_image(cw, ch)
        x0 = max(0, int(math.floor(x0)))
        y0 = max(0, int(math.floor(y0)))
        x1 = min(pil_img.width, int(math.ceil(x1)))
        y1 = min(pil_img.height, int(math.ceil(y1)))
        if x1 <= x0 or y1 <= y0:
            return
        dw = max(1, int(round((x1 - x0) * self.scale)))
        dh = max(1, int(round((y1 - y0) * self.scale)))
        # Nearest-neighbour when zoomed in a lot so single pixels stay sharp for precise boxes
        resample = Image.Resampling.NEAREST if self.scale >= 4 else Image.Resampling.BILINEAR
        region = pil_img.crop((x0, y0, x1, y1)).resize((dw, dh), resample)
        self._tk_img = ImageTk.PhotoImage(region)
        px, py = self.to_canvas(x0, y0)
        self.canvas.create_image(int(round(px)), int(round(py)), image=self._tk_img, anchor="nw", tags=tag)
        self.canvas.tag_lower(tag)

    # ------------------------------------------------------------------ events
    def _notify(self):
        if self.on_change:
            self.on_change()

    def _on_wheel(self, event, delta=None):
        delta = event.delta if delta is None else delta
        if delta == 0:
            return "break"
        if (event.state & 0x0001) and self.on_shift_wheel:  # Shift held
            self.on_shift_wheel(-1 if delta > 0 else 1)
            return "break"
        self.zoom_at(event.x, event.y, 1.25 ** (delta / 120.0))
        return "break"

    def _pan_begin(self, event):
        self._pan_last = (event.x, event.y)
        self.canvas.config(cursor="fleur")

    def _pan_move(self, event):
        if self._pan_last is None or self.fit_mode:
            return
        dx, dy = event.x - self._pan_last[0], event.y - self._pan_last[1]
        self._pan_last = (event.x, event.y)
        self.ox += dx
        self.oy += dy
        self._clamp_pan()
        self._notify()

    def _pan_end(self, event):
        self._pan_last = None
        self.canvas.config(cursor="crosshair")
