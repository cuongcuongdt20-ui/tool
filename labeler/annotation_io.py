import os
import shutil
import xml.etree.ElementTree as ET
from xml.dom import minidom
from datetime import datetime, timezone

CLASS_MAP = {
    "swim": 0,
    "drown": 1
}
REVERSE_CLASS_MAP = {v: k for k, v in CLASS_MAP.items()}

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
            
    return snippets

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
            label = box_elem.get("label", "swim")
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

def parse_yolo_txt(txt_dir, img_width=224, img_height=224):
    """
    Parses YOLO formatted .txt files in a directory for a snippet.
    Frame index is derived from filename e.g. 000050.txt -> frame index 49 (or 50).
    """
    if not os.path.exists(txt_dir):
        return None
        
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
                    
                    label_name = REVERSE_CLASS_MAP.get(cls_id, "swim")
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

def save_cvat_xml(snippet_id, annotations, width=224, height=224, total_frames=0, output_path=None):
    """
    Saves annotations dictionary to CVAT XML 1.1 format matching existing files.
    annotations: dict mapping frame_idx (int) -> list of box dicts
    """
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
    
    label_swim = ET.SubElement(labels_elem, "label")
    ET.SubElement(label_swim, "name").text = "swim"
    ET.SubElement(label_swim, "color").text = "#ff00cc"
    ET.SubElement(label_swim, "type").text = "any"
    ET.SubElement(label_swim, "attributes")
    
    label_drown = ET.SubElement(labels_elem, "label")
    ET.SubElement(label_drown, "name").text = "drown"
    ET.SubElement(label_drown, "color").text = "#fafa37"
    ET.SubElement(label_drown, "type").text = "any"
    ET.SubElement(label_drown, "attributes")
    
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
                "label": str(box.get("label", "swim")),
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
        
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(lines))
        
    return output_path

def save_yolo_txt(snippet_id, annotations, width=224, height=224, total_frames=0, output_dir=None):
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
            cls_id = CLASS_MAP.get(label, 0)
            
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
