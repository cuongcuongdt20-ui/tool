import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image

from annotation_io import parse_cvat_xml, save_cvat_xml
from clip_crop import CropCancelled, crop_annotations, crop_labeled_clip, normalize_crop_roi


class ClipCropTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.frames = self.root / "frames" / "original"
        self.frames.mkdir(parents=True)
        image = Image.new("RGB", (96, 64))
        image.putdata([(x * 2, y * 3, 50) for y in range(64) for x in range(96)])
        for i in (1, 2):
            image.save(self.frames / f"{i:06d}.png")
        (self.root / "classes.txt").write_text("not_drowning\ndrowning\n", encoding="utf-8")
        boxes = [dict(label="swim", xtl=20., ytl=12., xbr=30., ybr=22., source="manual"),
                 dict(label="drown", xtl=75., ytl=40., xbr=90., ybr=60., source="manual"),
                 dict(label="drown", xtl=55., ytl=35., xbr=75., ybr=58., source="interpolation")]
        self.annotations = {0: boxes, 1: [dict(b) for b in boxes]}
        save_cvat_xml("original", self.annotations, 96, 64, 2,
                      str(self.root / "labels" / "original.xml"))
        self.roi = (16, 8, 48, 40)

    def snapshot(self):
        return {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in self.root.rglob("*") if p.is_file()}

    def assert_preserved(self, snapshot):
        for p, expected in snapshot.items():
            self.assertEqual(hashlib.sha256(p.read_bytes()).hexdigest(), expected, str(p))

    def test_geometry_removes_partial_boxes_entirely(self):
        cropped, stats = crop_annotations(self.annotations, self.roi, 2)
        self.assertEqual(stats, dict(kept=2, partial_removed=2, removed=4))
        self.assertEqual([b["label"] for b in cropped[0]], ["swim"])
        self.assertEqual([cropped[0][0][k] for k in ("xtl", "ytl", "xbr", "ybr")], [4, 4, 14, 14])
        self.assertEqual(self.annotations[0][0]["xtl"], 20.)

    def test_export_all_frames_and_both_label_formats(self):
        original = self.snapshot()
        result = crop_labeled_clip(self.root, "original", "cropped", self.annotations, self.roi)
        self.assertEqual(result["frames"], 2)
        xml = parse_cvat_xml(str(self.root / "labels" / "cropped.xml"))
        self.assertEqual((xml["meta"]["width"], xml["meta"]["height"]), (48, 40))
        for i in (1, 2):
            with Image.open(self.root / "frames" / "cropped" / f"{i:06d}.jpg") as image:
                self.assertEqual(image.size, (48, 40))
                self.assertTrue(all(abs(a - b) <= 5 for a, b in zip(image.getpixel((0, 0)), (32, 24, 50))))
            lines = (self.root / "labels" / "cropped" / f"{i:06d}.txt").read_text().splitlines()
            self.assertEqual([int(s.split()[0]) for s in lines], [0])
            self.assertEqual(len(xml["annotations"][i - 1]), 1)
        self.assertEqual(json.loads((self.root / "extract_logs" / "cropped.json").read_text())["removed"], 4)
        self.assert_preserved(original)
        self.assertFalse(list(self.root.glob(".crop-*")))
        with self.assertRaises(FileExistsError):
            crop_labeled_clip(self.root, "original", "cropped", self.annotations, self.roi)

    def test_box_exactly_on_crop_boundary_is_kept(self):
        box = dict(label="drown", xtl=16., ytl=8., xbr=64., ybr=48.)
        cropped, stats = crop_annotations({0: [box]}, self.roi, 1)
        self.assertEqual(len(cropped[0]), 1)
        self.assertEqual(stats["removed"], 0)

    def test_any_partial_crossing_is_removed_on_all_four_edges(self):
        boxes = [dict(label="swim", xtl=15.9, ytl=10., xbr=30., ybr=20.),
                 dict(label="swim", xtl=20., ytl=7.9, xbr=30., ybr=20.),
                 dict(label="drown", xtl=20., ytl=10., xbr=64.1, ybr=20.),
                 dict(label="drown", xtl=20., ytl=10., xbr=30., ybr=48.1)]
        cropped, stats = crop_annotations({0: boxes}, self.roi, 1)
        self.assertEqual(cropped[0], [])
        self.assertEqual(stats, dict(kept=0, partial_removed=4, removed=4))

    def test_cancel_leaves_no_partial_clip(self):
        original = self.snapshot()
        flag = [False]
        with self.assertRaises(CropCancelled):
            crop_labeled_clip(self.root, "original", "cancelled", self.annotations, self.roi,
                              progress=lambda *args: flag.__setitem__(0, True), cancelled=lambda: flag[0])
        self.assertFalse((self.root / "frames" / "cancelled").exists())
        self.assertFalse((self.root / "labels" / "cancelled.xml").exists())
        self.assertFalse(list(self.root.glob(".crop-*")))
        self.assert_preserved(original)

    def test_mixed_frame_sizes_fail_without_publishing(self):
        Image.new("RGB", (100, 64)).save(self.frames / "000002.png")
        original = self.snapshot()
        with self.assertRaises(ValueError):
            crop_labeled_clip(self.root, "original", "bad", self.annotations, self.roi)
        self.assertFalse((self.root / "frames" / "bad").exists())
        self.assertFalse(list(self.root.glob(".crop-*")))
        self.assert_preserved(original)

    def test_replace_keeps_clip_name_and_backs_up_latest_edits(self):
        original = self.snapshot()
        self.annotations[0][0]["xtl"] = 22.
        result = crop_labeled_clip(self.root, "original", "original", self.annotations, self.roi, replace=True)
        self.assertEqual([p.name for p in (self.root / "frames").iterdir()], ["original"])
        self.assertTrue((self.root / "frames" / "original" / "000001.png").is_file())
        current = parse_cvat_xml(str(self.root / "labels" / "original.xml"))
        self.assertEqual((current["meta"]["width"], current["meta"]["height"]), (48, 40))
        self.assertEqual(current["annotations"][0][0]["xtl"], 6.)
        backup = Path(result["backup"])
        self.assertEqual(hashlib.sha256((backup / "xml").read_bytes()).hexdigest(), original[self.root / "labels" / "original.xml"])
        self.assertEqual(hashlib.sha256((backup / "frames" / "000001.png").read_bytes()).hexdigest(), original[self.frames / "000001.png"])
        latest = parse_cvat_xml(str(backup / "annotations_before_crop.xml"))
        self.assertEqual(latest["annotations"][0][0]["xtl"], 22.)
        self.assertEqual(json.loads((backup / "backup.json").read_text())["status"], "complete")

    def test_replace_failure_restores_original_assets(self):
        original = self.snapshot()
        rename = __import__("os").rename

        def fail_final_publish(src, dst):
            if Path(src).parent.name.startswith(".crop-") and Path(src).name == "frames":
                raise OSError("simulated replacement error")
            return rename(src, dst)

        with patch("clip_crop.os.rename", side_effect=fail_final_publish):
            with self.assertRaises(OSError):
                crop_labeled_clip(self.root, "original", "original", self.annotations, self.roi, replace=True)
        self.assert_preserved(original)
        self.assertFalse((self.root / "labels" / "original").exists())
        self.assertFalse(list(self.root.glob(".crop-*")))

    def test_frame_order_matches_labeler_for_mixed_case_names(self):
        (self.frames / "000001.png").rename(self.frames / "a.png")
        (self.frames / "000002.png").rename(self.frames / "B.png")
        crop_labeled_clip(self.root, "original", "original", self.annotations, self.roi, replace=True)
        log = json.loads((self.root / "extract_logs" / "original.json").read_text())
        self.assertEqual(log["frame_mapping"], [["B.png", "B.png"], ["a.png", "a.png"]])

    def test_reject_paths_and_normalize_dimensions(self):
        for name in ("../outside", "..", "C:\\outside", ""):
            with self.assertRaises(ValueError):
                crop_labeled_clip(self.root, "original", name, self.annotations, self.roi)
        with self.assertRaises(FileExistsError):
            crop_labeled_clip(self.root, "original", "original", self.annotations, self.roi)
        self.assertEqual(normalize_crop_roi((0, 0, 15, 13), 96, 64), (0, 0, 14, 12))
        with self.assertRaises(ValueError):
            normalize_crop_roi((0, 0, float("nan"), 10), 96, 64)

    def test_publish_failure_rolls_back_new_outputs(self):
        original = self.snapshot()
        rename = __import__("os").rename

        def fail_final_publish(src, dst):
            if Path(dst) == self.root / "frames" / "failure":
                raise OSError("simulated publish error")
            return rename(src, dst)

        with patch("clip_crop.os.rename", side_effect=fail_final_publish):
            with self.assertRaises(OSError):
                crop_labeled_clip(self.root, "original", "failure", self.annotations, self.roi)
        for path in ("frames/failure", "labels/failure", "labels/failure.xml", "extract_logs/failure.json"):
            self.assertFalse((self.root / path).exists())
        self.assertFalse(list(self.root.glob(".crop-*")))
        self.assert_preserved(original)

    def test_crop_preserves_linked_source_and_local_videos(self):
        source = self.root / "demo" / "source.mp4"
        local = self.root / "videos" / "original.mp4"
        for video in (source, local):
            video.parent.mkdir(parents=True, exist_ok=True)
            video.write_bytes(b"original video bytes: " + video.name.encode())
        log = self.root / "extract_logs" / "original.json"
        log.parent.mkdir()
        log.write_text(json.dumps(dict(source_video=str(source), current_video=str(local))))
        video_hashes = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in (source, local)}

        # A frames-only crop must work even with no video encoder or decoder installed.
        with patch.dict("os.environ", {"PATH": ""}):
            result = crop_labeled_clip(self.root, "original", "original", self.annotations,
                                       self.roi, replace=True)
        self.assert_preserved(video_hashes)
        with Image.open(self.frames / "000001.png") as image:
            self.assertEqual(image.size, (48, 40))
        current = parse_cvat_xml(str(self.root / "labels" / "original.xml"))
        self.assertEqual([b["label"] for b in current["annotations"][0]], ["swim"])
        self.assertEqual(json.loads(log.read_text())["crop_scope"], "frames_and_labels")
        backup = Path(result["backup"])
        self.assertFalse((backup / "video.mp4").exists())
        self.assertEqual(json.loads((backup / "log").read_text())["source_video"], str(source))


if __name__ == "__main__":
    unittest.main()
