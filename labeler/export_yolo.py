"""
Export labeled clips of a labeler dataset (frames/<clip>/ + labels/<clip>.xml) to a
standalone Ultralytics YOLO dataset:

    <out>/images/<split>/<clip>_000001.jpg
    <out>/labels/<split>/<clip>_000001.txt   (class cx cy w h, normalized)
    <out>/data.yaml

Class ids / names follow <dataset>/classes.txt when it exists (e.g. 0 = drowning, 1 = not_drowning),
otherwise the labeler default 0 = swim, 1 = drown.

Only clips that have labels/<clip>.xml are exported. Frames without boxes get an empty .txt
(background images).

    python labeler/export_yolo.py --dataset val_yolo --out val_yolo_export --split val
"""
import argparse
import os
import shutil

from annotation_io import load_yolo_classes, parse_cvat_xml

IMG_EXTS = (".jpg", ".jpeg", ".png")


def yolo_lines(boxes, width, height, class_map):
    lines = []
    for b in boxes:
        x1, x2 = sorted((max(0.0, min(width, b["xtl"])), max(0.0, min(width, b["xbr"]))))
        y1, y2 = sorted((max(0.0, min(height, b["ytl"])), max(0.0, min(height, b["ybr"]))))
        if x2 - x1 < 1 or y2 - y1 < 1:
            continue
        cls = class_map.get(b["label"], 0)
        lines.append(f"{cls} {(x1 + x2) / 2 / width:.6f} {(y1 + y2) / 2 / height:.6f} "
                     f"{(x2 - x1) / width:.6f} {(y2 - y1) / height:.6f}")
    return lines


def export(dataset, out, split, clips=None):
    class_map, names = load_yolo_classes(dataset)
    img_dir = os.path.join(out, "images", split)
    lbl_dir = os.path.join(out, "labels", split)
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(lbl_dir, exist_ok=True)

    frames_root = os.path.join(dataset, "frames")
    for clip in sorted(clips or os.listdir(frames_root)):
        xml_path = os.path.join(dataset, "labels", f"{clip}.xml")
        frames_dir = os.path.join(frames_root, clip)
        if not os.path.isdir(frames_dir) or not os.path.exists(xml_path):
            print(f"skip {clip}: no frames or no labels/{clip}.xml")
            continue
        parsed = parse_cvat_xml(xml_path)
        width = parsed["meta"].get("width", 224)
        height = parsed["meta"].get("height", 224)
        files = sorted(f for f in os.listdir(frames_dir) if f.lower().endswith(IMG_EXTS))
        n_boxes = 0
        for idx, name in enumerate(files):
            stem = f"{clip}_{os.path.splitext(name)[0]}"
            shutil.copy2(os.path.join(frames_dir, name), os.path.join(img_dir, stem + os.path.splitext(name)[1]))
            lines = yolo_lines(parsed["annotations"].get(idx, []), width, height, class_map)
            n_boxes += len(lines)
            with open(os.path.join(lbl_dir, stem + ".txt"), "w", encoding="utf-8") as f:
                f.write("\n".join(lines) + ("\n" if lines else ""))
        print(f"{clip}: {len(files)} images, {n_boxes} boxes")

    names = "\n".join(f"  {i}: {n}" for i, n in enumerate(names))
    splits = sorted(d for d in os.listdir(os.path.join(out, "images")) if os.path.isdir(os.path.join(out, "images", d)))
    with open(os.path.join(out, "data.yaml"), "w", encoding="utf-8") as f:
        # no "path:" key -> Ultralytics uses the folder of data.yaml (works on Kaggle too)
        keys = {("val" if s == "valid" else s): s for s in splits}
        if "train" not in keys and splits:  # Ultralytics requires a train key even for eval-only sets
            f.write(f"train: images/{splits[0]}  # no train split exported yet\n")
        for key, s in keys.items():
            f.write(f"{key}: images/{s}\n")
        f.write(f"names:\n{names}\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Export labeled clips to a YOLO dataset folder")
    p.add_argument("--dataset", required=True, help="Labeler dataset folder (contains frames/ and labels/)")
    p.add_argument("--out", required=True, help="Output YOLO dataset folder")
    p.add_argument("--split", default="val", help="Split sub-folder name (train / val / test)")
    p.add_argument("--clips", nargs="*", help="Only these clips (default: every labeled clip)")
    a = p.parse_args()
    export(a.dataset, a.out, a.split, a.clips)
