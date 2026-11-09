"""
Pre-label (auto-label) the clips of a labeler dataset with an Ultralytics YOLO model, so a human
only has to fix the boxes in desktop_labeler.py instead of drawing them all.

    python labeler/prelabel.py --dataset val_yolo            # model: labeler/models/best-yolo.pt
    python labeler/prelabel.py --dataset val_yolo --model other.pt --clips my_clip --conf 0.2

- Reads   <dataset>/frames/<clip>/*.jpg
- Writes  <dataset>/labels/<clip>.xml        (CVAT XML 1.1, same format the labeler saves)
          <dataset>/labels/<clip>/000001.txt (YOLO TXT; skipped for repo layout = datalist.csv)
- Model class names are mapped by name, not by id: drowning/drown -> drown, not_drowning/swim -> swim.
  Names / ids written follow <dataset>/classes.txt (see annotation_io.load_yolo_classes).
- Clips that already have labels/<clip>.xml are SKIPPED, so finished labels are never overwritten.
  Use --overwrite to redo them (the old XML is kept as <clip>.xml.bak).
"""
import argparse
import collections
import os
import shutil
import sys

from annotation_io import (CLASS_ALIASES, canonical_label, is_repo_layout, load_label_names,
                           load_yolo_classes, save_cvat_xml, save_yolo_txt)

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
DEFAULT_MODEL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "best-yolo.pt")
KNOWN_NAMES = {a for aliases in CLASS_ALIASES.values() for a in aliases}


def iou(a, b):
    ix = max(0.0, min(a["xbr"], b["xbr"]) - max(a["xtl"], b["xtl"]))
    iy = max(0.0, min(a["ybr"], b["ybr"]) - max(a["ytl"], b["ytl"]))
    inter = ix * iy
    area = lambda r: (r["xbr"] - r["xtl"]) * (r["ybr"] - r["ytl"])  # noqa: E731
    return inter / (area(a) + area(b) - inter + 1e-9)


def drop_swim_on_drown(boxes, thr):
    """One person detected as both classes: keep the drown box (missing a drowning is worse)."""
    drown = [b for b in boxes if b["label"] == "drown"]
    return [b for b in boxes if b["label"] == "drown" or all(iou(b, d) < thr for d in drown)]


def prelabel_clip(model, dataset, clip, args, label_names, class_map, write_txt):
    frames_dir = os.path.join(dataset, "frames", clip)
    files = sorted(f for f in os.listdir(frames_dir) if f.lower().endswith(IMG_EXTS))
    if not files:
        print(f"skip {clip}: no images")
        return
    anns, stats = {}, collections.Counter()
    width = height = None
    for start in range(0, len(files), args.batch):
        batch = [os.path.join(frames_dir, f) for f in files[start:start + args.batch]]
        results = model.predict(batch, conf=args.conf, iou=args.iou, imgsz=args.imgsz,
                                device=args.device, verbose=False)
        for j, r in enumerate(results):
            height, width = r.orig_shape
            boxes = []
            for (x1, y1, x2, y2), c in zip(r.boxes.xyxy.tolist(), r.boxes.cls.tolist()):
                boxes.append({"label": canonical_label(model.names[int(c)]),
                              "xtl": round(x1, 2), "ytl": round(y1, 2), "xbr": round(x2, 2), "ybr": round(y2, 2),
                              "occluded": 0, "source": "manual", "z_order": 0})
            if args.overlap > 0:
                boxes = drop_swim_on_drown(boxes, args.overlap)
            stats.update(label_names[b["label"]] for b in boxes)
            anns[start + j] = boxes
        print(f"  {clip}: {min(start + args.batch, len(files))}/{len(files)}", end="\r", flush=True)

    labels_dir = os.path.join(dataset, "labels")
    os.makedirs(labels_dir, exist_ok=True)
    xml_path = os.path.join(labels_dir, f"{clip}.xml")
    if os.path.exists(xml_path):
        shutil.copy2(xml_path, xml_path + ".bak")
    save_cvat_xml(clip, anns, width, height, len(files), xml_path, label_names=label_names)
    if write_txt:
        save_yolo_txt(clip, anns, width, height, len(files), os.path.join(labels_dir, clip), class_map=class_map)
    with_box = sum(1 for b in anns.values() if b)
    print(f"{clip}: {len(files)} frames, {with_box} with boxes, {dict(stats)}" + " " * 10)


def main():
    p = argparse.ArgumentParser(description="Auto-label clips with a YOLO model (then fix them in the labeler)")
    p.add_argument("--dataset", required=True, help="Labeler dataset folder (contains frames/)")
    p.add_argument("--model", default=DEFAULT_MODEL,
                   help="Ultralytics YOLO weights (.pt), default labeler/models/best-yolo.pt")
    p.add_argument("--clips", nargs="*", help="Only these clips (default: every clip in frames/)")
    p.add_argument("--overwrite", action="store_true", help="Also redo clips that already have labels/<clip>.xml")
    p.add_argument("--conf", type=float, default=0.25, help="Confidence threshold (default 0.25)")
    p.add_argument("--iou", type=float, default=0.5, help="NMS IoU threshold (default 0.5)")
    p.add_argument("--imgsz", type=int, default=640, help="Inference size (default 640)")
    p.add_argument("--overlap", type=float, default=0.4,
                   help="Drop a swim box overlapping a drown box at IoU >= this (0 = off, default 0.4)")
    p.add_argument("--batch", type=int, default=16, help="Images per predict call (default 16)")
    p.add_argument("--device", default=None, help="cpu, 0, ... (default: auto)")
    args = p.parse_args()

    try:
        from ultralytics import YOLO
    except ImportError:
        sys.exit("ultralytics is not installed: pip install ultralytics")

    frames_root = os.path.join(args.dataset, "frames")
    if not os.path.isdir(frames_root):
        sys.exit(f"{frames_root} not found")
    class_map, _ = load_yolo_classes(args.dataset)
    label_names = load_label_names(args.dataset)
    write_txt = not is_repo_layout(args.dataset)

    model = YOLO(args.model)
    unknown = [n for n in model.names.values() if n.lower() not in KNOWN_NAMES]
    if unknown:
        sys.exit(f"Model classes {unknown} cannot be mapped to swim/drown (known: {sorted(KNOWN_NAMES)})")
    print(f"model classes {model.names} -> dataset ids {dict((label_names[k], v) for k, v in class_map.items())}")

    clips = args.clips or sorted(d for d in os.listdir(frames_root)
                                 if os.path.isdir(os.path.join(frames_root, d)) and not d.endswith(".importing"))
    for clip in clips:
        if not os.path.isdir(os.path.join(frames_root, clip)):
            print(f"skip {clip}: not in {frames_root}")
        elif os.path.exists(os.path.join(args.dataset, "labels", f"{clip}.xml")) and not args.overwrite:
            print(f"skip {clip}: already labeled (use --overwrite to redo)")
        else:
            prelabel_clip(model, args.dataset, clip, args, label_names, class_map, write_txt)


if __name__ == "__main__":
    main()
