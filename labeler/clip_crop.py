"""Crop labeled frames/video, with transactional replacement and backups."""
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import uuid
from datetime import datetime, timezone

from PIL import Image

from annotation_io import is_repo_layout, load_label_names, load_yolo_classes, save_cvat_xml, save_yolo_txt


class CropCancelled(Exception):
    pass


def normalize_crop_roi(roi, width, height):
    """Clamp x/y/w/h and use even dimensions compatible with MP4 encoders."""
    if len(roi) != 4 or not all(math.isfinite(float(v)) for v in roi):
        raise ValueError("Vùng cắt phải có bốn số hữu hạn: x, y, rộng, cao")
    x, y, w, h = map(float, roi)
    if w < 2 or h < 2 or width < 2 or height < 2:
        raise ValueError("Vùng cắt phải rộng và cao ít nhất 2 pixel")
    x = max(0, min(width - 2, int(round(x))))
    y = max(0, min(height - 2, int(round(y))))
    w = min(width - x, int(round(w)))
    h = min(height - y, int(round(h)))
    w, h = w - w % 2, h - h % 2
    if w < 2 or h < 2:
        raise ValueError("Vùng cắt nằm ngoài ảnh hoặc quá nhỏ")
    return x, y, w, h


def crop_annotations(annotations, roi, total_frames):
    """Keep only whole boxes; discard boxes cut by any crop boundary."""
    x, y, w, h = roi
    result, stats = {}, dict(kept=0, partial_removed=0, removed=0)
    for frame in range(total_frames):
        boxes = []
        for box in annotations.get(frame, []):
            coords = [float(box[k]) for k in ("xtl", "ytl", "xbr", "ybr")]
            if not all(math.isfinite(v) for v in coords) or box["label"] not in ("swim", "drown"):
                raise ValueError(f"Nhãn không hợp lệ ở frame {frame + 1}")
            left, top, right, bottom = coords
            intersects = min(x + w, right) > max(x, left) and min(y + h, bottom) > max(y, top)
            contained = x <= left < right <= x + w and y <= top < bottom <= y + h
            if not contained:
                stats["removed"] += 1
                stats["partial_removed"] += int(intersects)
                continue
            stats["kept"] += 1
            boxes.append(dict(box, xtl=left - x, ytl=top - y, xbr=right - x, ybr=bottom - y))
        result[frame] = boxes
    return result, stats


def _check_cancel(cancelled):
    if cancelled and cancelled():
        raise CropCancelled("Đã huỷ cắt clip")


def _crop_video(ffmpeg, source, output, roi, frame_size, cancelled):
    x, y, w, h = roi
    fw, fh = frame_size
    # Match the labeled frame geometry, including old frame sets resized from HD video.
    cmd = [ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin", "-y", "-i", str(source),
           "-map", "0:v:0", "-map", "0:a?", "-vf", f"scale={fw}:{fh},crop={w}:{h}:{x}:{y}:exact=1",
           "-c:v", "libx264", "-crf", "18", "-preset", "fast", "-c:a", "aac",
           "-movflags", "+faststart", str(output)]
    process = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                               text=True, encoding="utf-8", errors="replace",
                               creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    try:
        while True:
            _check_cancel(cancelled)
            try:
                _, error = process.communicate(timeout=.2)
                break
            except subprocess.TimeoutExpired:
                continue
        if process.returncode:
            raise RuntimeError(f"Không xuất được video: {error[-2000:]}")
        if not output.is_file() or output.stat().st_size == 0:
            raise RuntimeError("Video xuất bị rỗng")
    finally:
        if process.poll() is None:
            process.kill()
            process.communicate()


def crop_labeled_clip(dataset_dir, source_clip, new_clip, annotations, roi, *,
                      video_path=None, progress=None, cancelled=None, replace=False):
    """Crop a new clip, or replace the current one transactionally with a backup."""
    root = Path(dataset_dir).resolve()
    for name in (source_clip, new_clip):
        if not re.fullmatch(r"[A-Za-z0-9_-]+", name or ""):
            raise ValueError("Tên clip chỉ được chứa chữ không dấu, số, dấu _ và -")
    if replace and new_clip != source_clip:
        raise ValueError("Khi thay thế, tên clip phải giữ nguyên")
    source = root / "frames" / source_clip
    files = sorted((p for p in source.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png") and p.is_file()),
                   key=lambda p: p.name)
    if not files:
        raise ValueError("Clip không có frame")
    with Image.open(files[0]) as image:
        frame_size = image.size
    roi = normalize_crop_roi(roi, *frame_size)
    x, y, w, h = roi
    cropped, stats = crop_annotations(annotations, roi, len(files))
    class_map, _ = load_yolo_classes(str(root))
    label_names = load_label_names(str(root))
    targets = dict(xml=root / "labels" / f"{new_clip}.xml",
                   log=root / "extract_logs" / f"{new_clip}.json",
                   frames=root / "frames" / new_clip)
    if not is_repo_layout(str(root)):
        targets["txt"] = root / "labels" / new_clip
    # Check all known destinations, including outputs not enabled in this operation.
    reserved = list(targets.values()) + [root / "labels" / new_clip, root / "videos" / f"{new_clip}.mp4"]
    for path in reserved:
        if not path.resolve().is_relative_to(root):
            raise ValueError(f"Đường dẫn đích nằm ngoài dataset: {path}")
        if os.path.lexists(path) and not replace:
            raise FileExistsError(f"Đích đã tồn tại, hãy đặt tên clip khác: {path}")
    video = Path(video_path).expanduser().resolve() if video_path else None
    ffmpeg = shutil.which("ffmpeg") if video else None
    if video:
        if not video.is_file():
            raise FileNotFoundError(f"Không tìm thấy video: {video}")
        if not ffmpeg:
            raise RuntimeError("Cần FFmpeg trong PATH để xuất MP4; hoặc bỏ chọn xuất video")
        if replace and video.suffix.lower() != ".mp4":
            raise ValueError("Chọn file MP4 để crop và thay thế trực tiếp")
        targets["video"] = video if replace else root / "videos" / f"{new_clip}.mp4"
    _check_cancel(cancelled)
    stage = Path(tempfile.mkdtemp(prefix=".crop-", dir=root))
    committed = []
    output_names = []
    try:
        (stage / "frames").mkdir()
        for index, path in enumerate(files, 1):
            _check_cancel(cancelled)
            with Image.open(path) as image:
                if image.size != frame_size:
                    raise ValueError(f"Frame khác kích thước: {path.name}")
                name = path.name if replace else f"{index:06d}.jpg"
                output_names.append(name)
                image.convert("RGB").crop((x, y, x + w, y + h)).save(
                    stage / "frames" / name, **({} if name.lower().endswith(".png") else {"quality": 95}))
            if progress:
                progress(index, len(files), "frames")
        save_cvat_xml(new_clip, cropped, w, h, len(files), str(stage / "xml"), label_names=label_names)
        if "txt" in targets:
            save_yolo_txt(new_clip, cropped, w, h, len(files), str(stage / "txt"), class_map=class_map)
        if video:
            if progress:
                progress(len(files), len(files), "video")
            _crop_video(ffmpeg, video, stage / "video.mp4", roi, frame_size, cancelled)
        log = dict(clip_name=new_clip, source_clip=source_clip, roi_xywh=roi,
                   original_size=frame_size, output_size=[w, h], frames=len(files),
                   frame_mapping=[[name, p.name] for name, p in zip(output_names, files)],
                   source_video=str(video) if video else None, video_keeps_source_fps=bool(video),
                   video_audio="AAC from source when available" if video else None, **stats)
        (stage / "log").write_text(json.dumps(log, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        _check_cancel(cancelled)
        if replace:
            backup = _replace_outputs(root, targets, stage, source_clip, annotations,
                                      frame_size, label_names, len(files), cancelled)
            return dict(clip=new_clip, frames=len(files), roi=roi, size=(w, h),
                        video=str(video) if video else None, backup=str(backup), **stats)
        # Publish frames last: the labeler never sees a clip with unfinished labels.
        for key in ("xml", "txt", "log", "video", "frames"):
            if key not in targets:
                continue
            target = targets[key]
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.resolve().is_relative_to(root) or os.path.lexists(target):
                raise FileExistsError(f"Không ghi đè đích: {target}")
            staged = stage / ("video.mp4" if key == "video" else key)
            if staged.is_dir():
                os.rename(staged, target)
                committed.append((key, target))
            else:
                with target.open("xb") as out:
                    committed.append((key, target))
                    with staged.open("rb") as inp:
                        shutil.copyfileobj(inp, out)
        return dict(clip=new_clip, frames=len(files), roi=roi, size=(w, h),
                    video=str(targets["video"]) if video else None, **stats)
    except BaseException:
        for index, (_, target) in enumerate(reversed(committed)):
            if not target.resolve().is_relative_to(root):
                raise RuntimeError(f"Không di chuyển đường dẫn ngoài dataset: {target}")
            os.rename(target, stage / f"rollback-{index}")
        raise
    finally:
        if stage.parent.resolve() != root or not stage.name.startswith(".crop-"):
            raise RuntimeError("Thư mục tạm không hợp lệ")
        shutil.rmtree(stage)


def _local_video_copy(source, target):
    """Stage beside the selected MP4 so os.replace is atomic across dataset drives."""
    descriptor, name = tempfile.mkstemp(prefix=".crop-", suffix=".mp4", dir=target.parent)
    os.close(descriptor)
    temporary = Path(name)
    try:
        shutil.copy2(source, temporary)
        os.replace(temporary, target)
    finally:
        if temporary.parent.resolve() != target.parent.resolve():
            raise RuntimeError("Đường dẫn video tạm không hợp lệ")
        temporary.unlink(missing_ok=True)


def _replace_outputs(root, targets, stage, clip, annotations, frame_size, label_names, total, cancelled):
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
    backup = root / "_crop_backups" / clip / stamp
    if not backup.resolve().is_relative_to(root):
        raise ValueError("Đường dẫn backup nằm ngoài dataset")
    backup.mkdir(parents=True, exist_ok=False)
    save_cvat_xml(clip, annotations, *frame_size, total,
                  str(backup / "annotations_before_crop.xml"), label_names=label_names)
    originals, published = [], []
    video = targets.get("video")
    video_replaced = False
    manifest = dict(clip=clip, created_utc=stamp, status="preparing", original_paths={
        key: str(path) for key, path in targets.items()})
    manifest_path = backup / "backup.json"

    def write_status(status):
        manifest["status"] = status
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    try:
        if video:
            shutil.copy2(video, backup / "video.mp4")
            if (backup / "video.mp4").stat().st_size != video.stat().st_size:
                raise OSError("Không sao lưu được toàn bộ video")
        write_status("backed_up")
        _check_cancel(cancelled)
        for key in ("xml", "txt", "log", "frames"):
            if key not in targets:
                continue
            target = targets[key]
            if not target.resolve().is_relative_to(root):
                raise ValueError(f"Đích nằm ngoài dataset: {target}")
            target.parent.mkdir(parents=True, exist_ok=True)
            if os.path.lexists(target):
                os.rename(target, backup / key)
                originals.append((key, target))
        log = json.loads((stage / "log").read_text(encoding="utf-8"))
        log.update(operation="replace_crop", backup=str(backup))
        if video:
            log["current_video"] = str(video)
        (stage / "log").write_text(json.dumps(log, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        for key in ("xml", "txt", "log", "frames"):
            if key not in targets:
                continue
            os.rename(stage / key, targets[key])
            published.append((key, targets[key]))
        if video:
            _local_video_copy(stage / "video.mp4", video)
            video_replaced = True
        write_status("complete")
        return backup
    except BaseException:
        if video_replaced:
            _local_video_copy(backup / "video.mp4", video)
        for key, target in reversed(published):
            if not target.resolve().is_relative_to(root):
                raise RuntimeError("Đường dẫn rollback không hợp lệ")
            os.rename(target, stage / f"failed-{key}")
        for key, target in reversed(originals):
            os.rename(backup / key, target)
        write_status("rolled_back")
        raise
