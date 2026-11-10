import copy
import unittest

from sequence_delete import propose_box_sequence


def box(x, *, label="swim", width=20, track_id=None):
    result = dict(label=label, xtl=x, ytl=10, xbr=x + width, ybr=50)
    if track_id is not None:
        result["track_id"] = track_id
    return result


class SequenceDeleteTest(unittest.TestCase):
    def test_motion_and_changed_box_order_without_mutation(self):
        annotations = {0: [box(10), box(100)], 1: [box(100), box(12)], 2: [box(14), box(100)]}
        before = copy.deepcopy(annotations)
        proposal = propose_box_sequence(annotations, 0, 0, 2)
        self.assertEqual(proposal["matches"], [(0, 0), (1, 1), (2, 0)])
        self.assertEqual(proposal["mode"], "geometry")
        self.assertIsNone(proposal["stop_frame"])
        self.assertEqual(annotations, before)

    def test_does_not_bridge_missing_frames(self):
        proposal = propose_box_sequence({0: [box(10)], 2: [box(12)]}, 0, 0, 2)
        self.assertEqual(proposal["matches"], [(0, 0)])
        self.assertEqual(proposal["stop_frame"], 1)

    def test_does_not_guess_class_changes(self):
        proposal = propose_box_sequence({0: [box(10)], 1: [box(10, label="drown")]}, 0, 0, 1)
        self.assertEqual(proposal["matches"], [(0, 0)])

    def test_stops_when_next_frame_has_similar_candidates(self):
        proposal = propose_box_sequence({0: [box(10)], 1: [box(11), box(12)]}, 0, 0, 1)
        self.assertEqual(proposal["matches"], [(0, 0)])
        self.assertIn("nhiều box", proposal["reason"])

    def test_stops_when_another_previous_person_competes(self):
        proposal = propose_box_sequence({0: [box(10), box(11)], 1: [box(10.5)]}, 0, 0, 1)
        self.assertEqual(proposal["matches"], [(0, 0)])
        self.assertIn("người khác", proposal["reason"])

    def test_rejects_large_motion_or_size_jump(self):
        for candidate in (box(100), box(10, width=100)):
            with self.subTest(candidate=candidate):
                proposal = propose_box_sequence({0: [box(10)], 1: [candidate]}, 0, 0, 1)
                self.assertEqual(proposal["matches"], [(0, 0)])

    def test_track_id_takes_precedence_over_geometry_and_class(self):
        annotations = {0: [box(10, track_id=5)],
                       1: [box(10, track_id=6), box(150, track_id="5", label="drown")]}
        proposal = propose_box_sequence(annotations, 0, 0, 1)
        self.assertEqual(proposal["matches"], [(0, 0), (1, 1)])
        self.assertEqual(proposal["mode"], "track_id")

    def test_duplicate_or_missing_track_id_stops(self):
        for candidates in ([box(10, track_id=6)], [box(10, track_id=5), box(12, track_id=5)]):
            with self.subTest(candidates=candidates):
                proposal = propose_box_sequence({0: [box(10, track_id=5)], 1: candidates}, 0, 0, 1)
                self.assertEqual(proposal["matches"], [(0, 0)])

    def test_end_frame_is_respected(self):
        annotations = {i: [box(10 + i)] for i in range(5)}
        self.assertEqual(propose_box_sequence(annotations, 2, 0, 3)["matches"], [(2, 0), (3, 0)])

    def test_invalid_selection_is_rejected(self):
        for start, index, end in ((0, 1, 2), (1, 0, 0)):
            with self.assertRaises(ValueError):
                propose_box_sequence({0: [box(10)]}, start, index, end)


if __name__ == "__main__":
    unittest.main()
