import os
import re
import shutil
import xml.etree.ElementTree as ET
from xml.dom import minidom
from datetime import datetime, timezone

CLASS_MAP = {
    "swim": 0,
    "drown": 1
}
REVERSE_CLASS_MAP = {v: k for k, v in CLASS_MAP.items()}

# Names a YOLO classes.txt may use for the labeler's two classes
CLASS_ALIASES = {
    "swim": ("swim", "not_drowning", "not_drown", "swimming"),
    "drown": ("drown", "drowning"),
}


def load_yolo_classes(dataset_dir):
    """
    YOLO class order for a dataset: <dataset>/classes.txt (one name per line, line index = class id),
    e.g. "drowning" then "not_drowning". Returns (class_map {label: id}, names [id -> name]).
    Without classes.txt the labeler's default order is used (0 = swim, 1 = drown).
    """
    path = os.path.join(dataset_dir, "classes.txt")
    if not os.path.exists(path):
        return dict(CLASS_MAP), [REVERSE_CLASS_MAP[i] for i in sorted(REVERSE_CLASS_MAP)]
    with open(path, "r", encoding="utf-8") as f:
        names = [l.strip() for l in f if l.strip()]
    class_map = {}
    for label, aliases in CLASS_ALIASES.items():
        ids = [i for i, n in enumerate(names) if n.lower() in aliases]
        if len(ids) != 1:
            raise ValueError(f"{path}: cần đúng một dòng cho class '{label}' (một trong {', '.join(aliases)})")
        class_map[label] = ids[0]
    return class_map, names


def canonical_label(name):
    """'drowning' -> 'drown', 'not_drowning' -> 'swim' (the labeler's internal class keys)."""
    name = (name or "").lower()
    for label, aliases in CLASS_ALIASES.items():
        if name in aliases:
            return label
    return "swim"


def load_label_names(dataset_dir):
    """
    Names written to the XML / shown in the UI, ordered by YOLO class id:
    {"drown": "drowning", "swim": "not_drowning"} with classes.txt, else {"swim": "swim", "drown": "drown"}.
    """
    class_map, names = load_yolo_classes(dataset_dir)
    return {label: names[i] for label, i in sorted(class_map.items(), key=lambda kv: kv[1])}

def load_dataset_snippets(archive_dir="archive"):
    """
    Loads snippet lists from unlabeled.csv, train.csv, valid.csv, test.csv
    Returns a dict with snippet metadata and overall statistics.
    """
    splits = {
        "unlabeled": "unlabeled.csv",
        "train": "train.csv",
        "valid": "valid.csv",
        "test": "test.csv"
    }
    
    snippets = []
    
    for split_name, filename in splits.items():
        file_path = os.path.join(archive_dir, filename)
        if not os.path.exists(file_path):
            continue
            
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = [l.strip() for l in f if l.strip()]
            
        for snippet_id in lines:
            frames_dir = os.path.join(archive_dir, 'frames', snippet_id)
            xml_label_path = os.path.join(archive_dir, 'labels', f"{snippet_id}.xml")
            txt_label_dir = os.path.join(archive_dir, 'labels', snippet_id)
            
            frame_count = 0
            if os.path.exists(frames_dir):
                frame_count = len([f for f in os.listdir(frames_dir) if f.endswith(('.jpg', '.png', '.jpeg'))])
                
            has_xml = os.path.exists(xml_label_path)
            has_txt = os.path.exists(txt_label_dir) and len(os.listdir(txt_label_dir)) > 0
            
            is_labeled = has_xml or has_txt
            
            snippets.append({
                "snippet_id": snippet_id,
                "split": split_name,
                "frame_count": frame_count,
                "is_labeled": is_labeled,
                "has_xml": has_xml,
                "has_txt": has_txt,
                "xml_path": xml_label_path if has_xml else None
            })

    # Clips that are in frames/ but in no csv (the dataset_lifeguard layout of the
    # drowning-detection repo has no split csv at all): "labeled" if labels/<clip>.xml exists.
    frames_root = os.path.join(archive_dir, 'frames')
    listed = {s["snippet_id"] for s in snippets}
    if os.path.isdir(frames_root):
        for snippet_id in sorted(os.listdir(frames_root), key=str.lower):
            frames_dir = os.path.join(frames_root, snippet_id)
            if snippet_id in listed or snippet_id.endswith(".importing") or not os.path.isdir(frames_dir):
                continue
            xml_label_path = os.path.join(archive_dir, 'labels', f"{snippet_id}.xml")
            has_xml = os.path.exists(xml_label_path)
            snippets.append({
                "snippet_id": snippet_id,
                "split": "labeled" if has_xml else "unlabeled",
                "frame_count": len([f for f in os.listdir(frames_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]),
                "is_labeled": has_xml,
                "has_xml": has_xml,
                "has_txt": False,
                "xml_path": xml_label_path if has_xml else None
            })

    return snippets


def is_repo_layout(archive_dir="archive"):
    """
    True for a dataset laid out like dataset_lifeguard in the drowning-detection repo
    (frames/ + labels/<clip>.xml + datalist.csv): labels there are CVAT XML only.
    """
    return os.path.exists(os.path.join(archive_dir, "datalist.csv"))

def parse_cvat_xml(xml_path):
    """
    Parses a CVAT XML 1.1 annotation file and returns a dictionary of frame annotations.
    """
    if not os.path.exists(xml_path):
        return None
        
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    task_meta = {}
    meta_elem = root.find("meta")
    if meta_elem is not None:
        task_elem = meta_elem.find("task")
        if task_elem is not None:
            task_meta = {
                "id": task_elem.findtext("id", "0"),
                "name": task_elem.findtext("name", ""),
                "size": task_elem.findtext("size", "0"),
                "mode": task_elem.findtext("mode", "interpolation"),
            }
            orig_size = task_elem.find("original_size")
            if orig_size is not None:
                task_meta["width"] = int(orig_size.findtext("width", "224"))
                task_meta["height"] = int(orig_size.findtext("height", "224"))

    annotations = {}  # frame_idx (int) -> list of box dicts
    
    for image_elem in root.findall("image"):
        img_id = int(image_elem.get("id", "0"))
        width = float(image_elem.get("width", "224"))
        height = float(image_elem.get("height", "224"))
        
        boxes = []
        for box_elem in image_elem.findall("box"):
            label = canonical_label(box_elem.get("label", "swim"))
            xtl = float(box_elem.get("xtl", "0"))
            ytl = float(box_elem.get("ytl", "0"))
            xbr = float(box_elem.get("xbr", "0"))
            ybr = float(box_elem.get("ybr", "0"))
            occluded = int(box_elem.get("occluded", "0"))
            source = box_elem.get("source", "manual")
            z_order = int(box_elem.get("z_order", "0"))
            track_id = int(box_elem.get("track_id", "0")) if "track_id" in box_elem.attrib else None
            
            boxes.append({
                "label": label,
                "class_id": CLASS_MAP.get(label, 0),
                "xtl": xtl,
                "ytl": ytl,
                "xbr": xbr,
                "ybr": ybr,
                "occluded": occluded,
                "source": source,
                "z_order": z_order,
                "track_id": track_id
            })
            
        annotations[img_id] = boxes
        
    return {
        "meta": task_meta,
        "annotations": annotations
    }

def parse_yolo_txt(txt_dir, img_width=224, img_height=224, class_map=None):
    """
    Parses YOLO formatted .txt files in a directory for a snippet.
    Frame index is derived from filename e.g. 000050.txt -> frame index 49 (or 50).
    """
    if not os.path.exists(txt_dir):
        return None
    reverse_map = {v: k for k, v in (class_map or CLASS_MAP).items()}
        
    annotations = {}
    files = sorted(os.listdir(txt_dir))
    
    for f in files:
        if not f.endswith('.txt'):
            continue
        try:
            # 000001.txt -> 0 or 1 depending on index
            frame_num = int(f.replace('.txt', ''))
            # 1-indexed to 0-indexed frame
            frame_idx = frame_num - 1 if frame_num > 0 else frame_num
        except ValueError:
            continue
            
        file_path = os.path.join(txt_dir, f)
        boxes = []
        with open(file_path, 'r', encoding='utf-8') as tf:
            for line in tf:
                parts = line.strip().split()
                if len(parts) >= 5:
                    cls_id = int(parts[0])
                    cx, cy, w, h = map(float, parts[1:5])
                    
                    # Convert normalized YOLO center/size to xtl, ytl, xbr, ybr
                    box_w = w * img_width
                    box_h = h * img_height
                    xtl = (cx * img_width) - (box_w / 2.0)
                    ytl = (cy * img_height) - (box_h / 2.0)
                    xbr = xtl + box_w
                    ybr = ytl + box_h
                    
                    label_name = reverse_map.get(cls_id, "swim")
                    boxes.append({
                        "label": label_name,
                        "class_id": cls_id,
                        "xtl": round(xtl, 2),
                        "ytl": round(ytl, 2),
                        "xbr": round(xbr, 2),
                        "ybr": round(ybr, 2),
                        "occluded": 0,
                        "source": "manual",
                        "z_order": 0
                    })
                    
        annotations[frame_idx] = boxes
        
    return {
        "meta": {"width": img_width, "height": img_height},
        "annotations": annotations
    }

def save_cvat_xml(snippet_id, annotations, width=224, height=224, total_frames=0, output_path=None,
                  label_names=None):
    """
    Saves annotations dictionary to CVAT XML 1.1 format matching existing files.
    annotations: dict mapping frame_idx (int) -> list of box dicts
    label_names: {"swim": <name>, "drown": <name>} written to the XML, in <labels> order
    (default "swim", "drown"; see load_label_names).
    """
    label_names = label_names or {"swim": "swim", "drown": "drown"}
    if output_path is None:
        output_path = os.path.join("archive", "labels", f"{snippet_id}.xml")
        
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S.%f+00:00")
    
    root = ET.Element("annotations")
    
    version_elem = ET.SubElement(root, "version")
    version_elem.text = "1.1"
    
    meta_elem = ET.SubElement(root, "meta")
    task_elem = ET.SubElement(meta_elem, "task")
    
    ET.SubElement(task_elem, "id").text = "1"
    ET.SubElement(task_elem, "name").text = f"{snippet_id}.webm"
    ET.SubElement(task_elem, "size").text = str(total_frames)
    ET.SubElement(task_elem, "mode").text = "interpolation"
    ET.SubElement(task_elem, "overlap").text = "5"
    ET.SubElement(task_elem, "bugtracker").text = ""
    ET.SubElement(task_elem, "created").text = now_str
    ET.SubElement(task_elem, "updated").text = now_str
    ET.SubElement(task_elem, "subset").text = "default"
    ET.SubElement(task_elem, "start_frame").text = "0"
    ET.SubElement(task_elem, "stop_frame").text = str(max(0, total_frames - 1))
    ET.SubElement(task_elem, "frame_filter").text = ""
    
    segments_elem = ET.SubElement(task_elem, "segments")
    segment_elem = ET.SubElement(segments_elem, "segment")
    ET.SubElement(segment_elem, "id").text = "1"
    ET.SubElement(segment_elem, "start").text = "0"
    ET.SubElement(segment_elem, "stop").text = str(max(0, total_frames - 1))
    ET.SubElement(segment_elem, "url").text = "http://localhost:8080/api/jobs/1"
    
    owner_elem = ET.SubElement(task_elem, "owner")
    ET.SubElement(owner_elem, "username").text = "annotator"
    ET.SubElement(owner_elem, "email").text = "annotator@lifeguard.local"
    ET.SubElement(task_elem, "assignee").text = ""
    
    labels_elem = ET.SubElement(task_elem, "labels")
    
    label_colors = {"swim": "#ff00cc", "drown": "#fafa37"}
    for label, name in label_names.items():
        label_elem = ET.SubElement(labels_elem, "label")
        ET.SubElement(label_elem, "name").text = name
        ET.SubElement(label_elem, "color").text = label_colors[label]
        ET.SubElement(label_elem, "type").text = "any"
        ET.SubElement(label_elem, "attributes")
    
    orig_size_elem = ET.SubElement(task_elem, "original_size")
    ET.SubElement(orig_size_elem, "width").text = str(int(width))
    ET.SubElement(orig_size_elem, "height").text = str(int(height))
    
    ET.SubElement(task_elem, "source").text = f"{snippet_id}.webm"
    ET.SubElement(meta_elem, "dumped").text = now_str
    
    # Sort frame indices
    for frame_idx in range(total_frames):
        img_elem = ET.SubElement(root, "image", {
            "id": str(frame_idx),
            "name": f"frame_{frame_idx:06d}",
            "width": str(int(width)),
            "height": str(int(height))
        })
        
        frame_boxes = annotations.get(frame_idx, annotations.get(str(frame_idx), []))
        for box in frame_boxes:
            box_attrs = {
                "label": label_names[canonical_label(box.get("label", "swim"))],
                "source": str(box.get("source", "manual")),
                "occluded": str(int(box.get("occluded", 0))),
                "xtl": f"{float(box.get('xtl', 0)):.2f}",
                "ytl": f"{float(box.get('ytl', 0)):.2f}",
                "xbr": f"{float(box.get('xbr', 0)):.2f}",
                "ybr": f"{float(box.get('ybr', 0)):.2f}",
                "z_order": str(int(box.get("z_order", 0)))
            }
            if "track_id" in box and box["track_id"] is not None:
                box_attrs["track_id"] = str(box["track_id"])
                
            ET.SubElement(img_elem, "box", box_attrs)
            
    # Convert to pretty XML string
    rough_string = ET.tostring(root, 'utf-8')
    reparsed = minidom.parseString(rough_string)
    pretty_xml = reparsed.toprettyxml(indent="  ")
    
    # Standardize xml declaration header to match CVAT
    lines = pretty_xml.splitlines()
    if lines and 'xml version' in lines[0]:
        lines[0] = '<?xml version="1.0" encoding="utf-8"?>'

    # CVAT never writes self-closing tags: <bugtracker></bugtracker>, and <image> / <box> close on their own line
    for i, line in enumerate(lines):
        m = re.fullmatch(r"(\s*)<(\w+)([^>]*?)/>", line)
        if m:
            indent, tag, attrs = m.groups()
            sep = "\n" + indent if tag in ("image", "box", "attributes") else ""
            lines[i] = f"{indent}<{tag}{attrs}>{sep}</{tag}>"

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(lines) + "\n")
        
    return output_path

def save_yolo_txt(snippet_id, annotations, width=224, height=224, total_frames=0, output_dir=None, class_map=None):
    """
    Saves annotations to YOLO .txt files format in archive/labels/<snippet_id>/
    Filename format: 000001.txt, 000002.txt ... (1-indexed matching frames)
    """
    if output_dir is None:
        output_dir = os.path.join("archive", "labels", snippet_id)
        
    os.makedirs(output_dir, exist_ok=True)
    
    for frame_idx in range(total_frames):
        frame_num = frame_idx + 1  # 1-indexed to match 000001.jpg
        txt_path = os.path.join(output_dir, f"{frame_num:06d}.txt")
        
        frame_boxes = annotations.get(frame_idx, annotations.get(str(frame_idx), []))
        lines = []
        for box in frame_boxes:
            label = box.get("label", "swim")
            cls_id = (class_map or CLASS_MAP).get(label, 0)
            
            xtl = float(box.get("xtl", 0))
            ytl = float(box.get("ytl", 0))
            xbr = float(box.get("xbr", 0))
            ybr = float(box.get("ybr", 0))
            
            bw = xbr - xtl
            bh = ybr - ytl
            cx = xtl + (bw / 2.0)
            cy = ytl + (bh / 2.0)
            
            # Normalize coordinates [0..1]
            ncx = max(0.0, min(1.0, cx / width))
            ncy = max(0.0, min(1.0, cy / height))
            nbw = max(0.0, min(1.0, bw / width))
            nbh = max(0.0, min(1.0, bh / height))
            
            lines.append(f"{cls_id} {ncx:.6f} {ncy:.6f} {nbw:.6f} {nbh:.6f}")
            
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write("\n".join(lines) + ("\n" if lines else ""))
            
    return output_dir


SPLIT_FILES = ("unlabeled.csv", "train.csv", "valid.csv", "test.csv")


def snippet_paths(snippet_id, archive_dir="archive"):
    """All files / folders that belong to one clip (only those that exist)."""
    candidates = {
        "frames": os.path.join(archive_dir, "frames", snippet_id),
        "xml": os.path.join(archive_dir, "labels", f"{snippet_id}.xml"),
        "txt": os.path.join(archive_dir, "labels", snippet_id),
        "log": os.path.join(archive_dir, "extract_logs", f"{snippet_id}.json"),
    }
    return {k: p for k, p in candidates.items() if os.path.exists(p)}


def delete_snippet(snippet_id, archive_dir="archive", delete_labels=True, permanent=False):
    """
    Remove a clip: its frames folder (and optionally its labels / extract log),
    and its line in unlabeled/train/valid/test.csv.
    By default everything is moved to archive/_trash/<snippet_id>_<timestamp>/ so it can be restored.
    Returns (trash_dir or None, list of removed paths, list of csv files edited).
    """
    if not snippet_id or os.sep in snippet_id or "/" in snippet_id or snippet_id in (".", ".."):
        raise ValueError(f"Invalid snippet id: {snippet_id!r}")

    paths = snippet_paths(snippet_id, archive_dir)
    if not delete_labels:
        paths = {k: p for k, p in paths.items() if k in ("frames", "log")}

    trash_dir = None
    if not permanent and paths:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        trash_dir = os.path.join(archive_dir, "_trash", f"{snippet_id}_{stamp}")
        os.makedirs(trash_dir, exist_ok=True)

    removed = []
    for kind, path in paths.items():
        if permanent:
            if os.path.isdir(path):
                shutil.rmtree(path)
            else:
                os.remove(path)
        else:
            dest_name = {"frames": "frames", "txt": "labels_txt"}.get(kind, os.path.basename(path))
            shutil.move(path, os.path.join(trash_dir, dest_name))
        removed.append(path)

    edited_csv = []
    for name in SPLIT_FILES:
        csv_path = os.path.join(archive_dir, name)
        if not os.path.exists(csv_path):
            continue
        with open(csv_path, "r", encoding="utf-8") as f:
            lines = [l.strip() for l in f if l.strip()]
        kept = [l for l in lines if l != snippet_id]
        if len(kept) != len(lines):
            with open(csv_path, "w", encoding="utf-8") as f:
                f.write("\n".join(kept) + ("\n" if kept else ""))
            edited_csv.append(name)

    return trash_dir, removed, edited_csv


IMPORT_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def natural_key(name):
    """Sort 'frame_2.jpg' before 'frame_10.jpg'."""
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", name)]


def list_image_files(folder):
    return sorted((f for f in os.listdir(folder) if f.lower().endswith(IMPORT_EXTS)), key=natural_key)


def add_to_split(snippet_id, split="unlabeled", archive_dir="archive"):
    """Append snippet_id to <split>.csv once (no duplicates, no blank lines)."""
    csv_path = os.path.join(archive_dir, f"{split}.csv")
    existing = []
    if os.path.exists(csv_path):
        with open(csv_path, "r", encoding="utf-8") as f:
            existing = [l.strip() for l in f if l.strip()]
    if snippet_id in existing:
        return False
    existing.append(snippet_id)
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("\n".join(existing) + "\n")
    return True


def import_frame_folder(src_dir, snippet_id, archive_dir="archive", target_size=(224, 224),
                        split="unlabeled", quality=95, progress=None):
    """
    Copy a folder of images into archive/frames/<snippet_id>/ as 000001.jpg, 000002.jpg, ...
    (natural sort order), optionally resized to target_size so it matches the original
    224x224 dataset, and register the clip in <split>.csv (split=None: no csv).
    progress(done, total) is called after each frame. Returns the number of frames written.
    """
    from PIL import Image

    if not snippet_id or any(c in snippet_id for c in '\\/:*?"<>|') or snippet_id in (".", ".."):
        raise ValueError(f"Tên bộ frame không hợp lệ: {snippet_id!r}")
    files = list_image_files(src_dir)
    if not files:
        raise ValueError(f"Không có ảnh nào trong: {src_dir}")
    dest = os.path.join(archive_dir, "frames", snippet_id)
    if os.path.exists(dest) or os.path.exists(os.path.join(archive_dir, "labels", f"{snippet_id}.xml")):
        raise FileExistsError(f"Bộ frame '{snippet_id}' đã tồn tại")

    tmp = dest + ".importing"
    if os.path.exists(tmp):
        shutil.rmtree(tmp)
    os.makedirs(tmp)
    try:
        for i, name in enumerate(files, 1):
            with Image.open(os.path.join(src_dir, name)) as img:
                img = img.convert("RGB")
                if target_size and img.size != tuple(target_size):
                    img = img.resize(tuple(target_size), Image.BICUBIC)
                img.save(os.path.join(tmp, f"{i:06d}.jpg"), quality=int(quality))
            if progress:
                progress(i, len(files))
        os.replace(tmp, dest)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    if split:
        add_to_split(snippet_id, split, archive_dir)
    return len(files)
