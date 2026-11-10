"""Conservative box association for a user-reviewed multi-frame deletion."""
import math


def _geometry(box):
    try:
        left, top, right, bottom = (float(box[k]) for k in ("xtl", "ytl", "xbr", "ybr"))
    except (KeyError, ValueError, TypeError):
        return None
    if not all(math.isfinite(v) for v in (left, top, right, bottom)) or right <= left or bottom <= top:
        return None
    return left, top, right, bottom


def _score(previous, candidate):
    if previous.get("label") != candidate.get("label"):
        return None
    a, b = _geometry(previous), _geometry(candidate)
    if a is None or b is None:
        return None
    aw, ah, bw, bh = a[2] - a[0], a[3] - a[1], b[2] - b[0], b[3] - b[1]
    if not (.5 <= bw / aw <= 2 and .5 <= bh / ah <= 2 and .5 <= bw * bh / (aw * ah) <= 2):
        return None
    inter = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
    iou = inter / (aw * ah + bw * bh - inter)
    distance = math.hypot((a[0] + a[2] - b[0] - b[2]) / 2,
                          (a[1] + a[3] - b[1] - b[3]) / 2) / math.hypot(aw, ah)
    if iou < .25 or distance > .65:
        return None
    return .7 * iou + .3 * (1 - distance)


def propose_box_sequence(annotations, start_frame, selected_index, end_frame):
    """Follow consecutive frames, stopping at a gap or ambiguous association.

    This proposes box indexes, never edits labels. Without track_id it is only
    a geometric guess and must be visually reviewed before deletion.
    """
    boxes = annotations.get(start_frame, [])
    if not 0 <= selected_index < len(boxes) or end_frame < start_frame:
        raise ValueError("Chọn một box và khoảng frame hợp lệ")
    if _geometry(boxes[selected_index]) is None:
        raise ValueError("Box đã chọn không hợp lệ")
    matches = [(start_frame, selected_index)]
    track_id = boxes[selected_index].get("track_id")
    mode = "track_id" if track_id is not None else "geometry"
    previous_frame, previous_index = start_frame, selected_index

    def finish(reason, stop_frame=None):
        return dict(matches=matches, mode=mode, reason=reason, stop_frame=stop_frame)

    for frame in range(start_frame + 1, end_frame + 1):
        current = annotations.get(frame, [])
        if not current:
            return finish("frame không có box; không nối qua khoảng trống", frame)
        if track_id is not None:
            candidates = [i for i, box in enumerate(current)
                          if box.get("track_id") is not None and str(box["track_id"]) == str(track_id)]
            if len(candidates) != 1:
                return finish("không tìm thấy duy nhất box cùng track_id", frame)
            index = candidates[0]
        else:
            previous = annotations[previous_frame][previous_index]
            candidates = [(score, i) for i, box in enumerate(current)
                          if (score := _score(previous, box)) is not None]
            candidates.sort(reverse=True)
            if not candidates:
                return finish("không có box cùng lớp và vị trí đủ gần", frame)
            score, index = candidates[0]
            if len(candidates) > 1 and score - candidates[1][0] < .12:
                return finish("nhiều box gần giống nhau, dễ nhầm người", frame)
            # Refuse a current box that also matches another previous person well.
            for other_index, other in enumerate(annotations[previous_frame]):
                if other_index == previous_index:
                    continue
                other_score = _score(other, current[index])
                if other_score is not None and other_score >= score - .1:
                    return finish("box có thể thuộc người khác ở frame trước", frame)
        matches.append((frame, index))
        previous_frame, previous_index = frame, index
    return finish("đã tới frame giới hạn")
