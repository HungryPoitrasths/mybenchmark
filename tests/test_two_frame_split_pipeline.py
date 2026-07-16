import unittest

import numpy as np

import scripts.run_pipeline as run_pipeline_module
from src.utils.colmap_loader import CameraIntrinsics, CameraPose


def make_intrinsics() -> CameraIntrinsics:
    return CameraIntrinsics(width=320, height=240, fx=200.0, fy=200.0, cx=160.0, cy=120.0)


def make_pose(name: str, position_x: float) -> CameraPose:
    return CameraPose(
        image_name=name,
        rotation=np.eye(3, dtype=np.float64),
        translation=np.array([-position_x, 0.0, 0.0], dtype=np.float64),
    )


def make_object(obj_id: int, x: float, z: float = 2.0) -> dict:
    return {
        "id": obj_id,
        "label": f"obj{obj_id}",
        "center": [x, 0.0, z],
        "bbox_min": [x - 0.1, -0.1, z - 0.1],
        "bbox_max": [x + 0.1, 0.1, z + 0.1],
    }


# Bridge positions verified (see tests/test_two_frame_split.py) to give a real
# route-continuity chain from x=0 to x=8 at z=2 depth with this camera model.
_BRIDGE_POSITIONS = [1.0, 2.5, 4.0, 5.0, 6.5]
_EXPECTED_CHAIN = [f"bridge_{pos}.jpg" for pos in _BRIDGE_POSITIONS]


class TestApplyTwoFrameSplit(unittest.TestCase):
    def setUp(self):
        self.color_intrinsics = make_intrinsics()
        self.cam_orig = make_pose("orig.jpg", position_x=0.0)
        self.cam_far = make_pose("far.jpg", position_x=8.0)
        self.all_poses = {"orig.jpg": self.cam_orig, "far.jpg": self.cam_far}
        for pos in _BRIDGE_POSITIONS:
            name = f"bridge_{pos}.jpg"
            self.all_poses[name] = make_pose(name, position_x=pos)
        # obj 1/2 near x=0 (visible from orig), obj 3 near x=8 (only visible from
        # far) -- far enough apart that neither frame also fully frames the other
        # group, satisfying the mutual-exclusivity requirement.
        self.objects_by_id = {
            1: make_object(1, x=0.0),
            2: make_object(2, x=0.1),
            3: make_object(3, x=8.0),
        }
        # frame_a/frame_b are now required to come from the referability cache (a
        # positive gate, not just an optional post-hoc safeguard) -- this fixture
        # covers both real frames for every group used across the tests below so
        # existing single-call tests don't need per-test scene_frames wiring.
        self.scene_frames = {
            "orig.jpg": {
                "referable_object_ids": [1, 2],
                "attachment_referable_object_ids": [1, 2],
            },
            "far.jpg": {"referable_object_ids": [3]},
        }

    def _apply(self, q: dict) -> dict:
        q = dict(q)
        run_pipeline_module._apply_two_frame_split(
            q,
            objects_by_id=self.objects_by_id,
            all_poses=self.all_poses,
            color_intrinsics=self.color_intrinsics,
            camera_pose=self.cam_orig,
            scene_frames=self.scene_frames,
        )
        return q

    def _apply_kept(self, q: dict, *, scene_frames: dict | None = None) -> tuple[dict, bool]:
        q = dict(q)
        kept = run_pipeline_module._apply_two_frame_split(
            q,
            objects_by_id=self.objects_by_id,
            all_poses=self.all_poses,
            color_intrinsics=self.color_intrinsics,
            camera_pose=self.cam_orig,
            scene_frames=scene_frames,
        )
        return q, kept

    def test_object_move_agent_splits_move_pair_from_ref(self):
        q = {
            "type": "object_move_agent",
            "image_name": "orig.jpg",
            "moved_obj_id": 1,
            "query_obj_id": 1,
            "obj_c_id": 3,
        }
        result = self._apply(q)
        self.assertEqual(result["image_name"], "orig.jpg")
        self.assertEqual(result["reasoning_frame_2"], "far.jpg")
        self.assertEqual(result["auxiliary_image_names"], _EXPECTED_CHAIN)
        self.assertEqual(result["object_frame_groups"], {"frame_1": [1], "frame_2": [3]})

    def test_note_names_every_object_when_a_group_has_more_than_one(self):
        # moved_obj_id=1 and query_obj_id=2 are two DISTINCT objects both landing
        # in group_a -- the note must name both, not silently drop one under the
        # assumption that a "frame" always shows exactly one object.
        q = {
            "type": "object_move_agent",
            "image_name": "orig.jpg",
            "question": "What is the relative position of obj2 to obj3?",
            "moved_obj_id": 1,
            "query_obj_id": 2,
            "obj_c_id": 3,
        }
        result = self._apply(q)
        self.assertIn("obj1", result["question"])
        self.assertIn("obj2", result["question"])
        self.assertIn("obj3", result["question"])
        self.assertIn("the obj1 and the obj2", result["question"])
        self.assertEqual(result["object_frame_groups"], {"frame_1": [1, 2], "frame_2": [3]})

    def test_object_move_agent_prepends_multi_frame_note_to_question_text(self):
        q = {
            "type": "object_move_agent",
            "image_name": "orig.jpg",
            "question": "What is the relative position of obj1 to obj3?",
            "moved_obj_id": 1,
            "query_obj_id": 1,
            "obj_c_id": 3,
        }
        result = self._apply(q)
        self.assertTrue(result["question"].endswith(q["question"]))
        self.assertTrue(result["question"].startswith("A sequence of views follows"))
        self.assertIn("obj1", result["question"])
        self.assertIn("obj3", result["question"])

    def test_coordinate_rotation_agent_also_gets_the_note(self):
        q = {
            "type": "coordinate_rotation_agent",
            "image_name": "orig.jpg",
            "question": "In which direction is obj1 relative to obj3?",
            "obj_a_id": 1,
            "obj_b_id": 3,
        }
        result = self._apply(q)
        # Its own template only names the rotation-pivot camera, not which object
        # ends up in which frame, so it needs the note just like every other type.
        self.assertTrue(result["question"].startswith("A sequence of views follows"))
        self.assertTrue(result["question"].endswith(q["question"]))
        self.assertEqual(result["reasoning_frame_2"], "far.jpg")

    def test_attachment_move_pairs_root_and_query_against_ref(self):
        q = {
            "type": "attachment_move",
            "image_name": "orig.jpg",
            "root_id": 1,
            "query_obj_id": 2,
            "obj_ref_id": 3,
        }
        result = self._apply(q)
        self.assertEqual(result["reasoning_frame_2"], "far.jpg")
        self.assertEqual(sorted(result["object_frame_groups"]["frame_1"]), [1, 2])
        self.assertEqual(result["object_frame_groups"]["frame_2"], [3])

    def test_attachment_move_split_uses_role_specific_frame_pools(self):
        q = {
            "type": "attachment_move",
            "image_name": "orig.jpg",
            "root_id": 1,
            "query_obj_id": 2,
            "obj_ref_id": 3,
        }
        scene_frames = {
            "orig.jpg": {
                "referable_object_ids": [],
                "attachment_referable_object_ids": [1, 2],
            },
            "far.jpg": {
                "referable_object_ids": [3],
                "attachment_referable_object_ids": [],
            },
        }

        result, kept = self._apply_kept(q, scene_frames=scene_frames)

        self.assertTrue(kept)
        self.assertEqual(result["reasoning_frame_2"], "far.jpg")

    def test_attachment_move_split_rejects_attachment_only_ordinary_reference(self):
        q = {
            "type": "attachment_move",
            "image_name": "orig.jpg",
            "root_id": 1,
            "query_obj_id": 2,
            "obj_ref_id": 3,
        }
        scene_frames = {
            "orig.jpg": {"attachment_referable_object_ids": [1, 2]},
            "far.jpg": {
                "referable_object_ids": [],
                "attachment_referable_object_ids": [3],
            },
        }

        result, kept = self._apply_kept(q, scene_frames=scene_frames)

        self.assertFalse(kept)
        self.assertNotIn("reasoning_frame_2", result)

    def test_object_remove_is_excluded_and_stays_single_frame(self):
        q = {
            "type": "object_remove",
            "image_name": "orig.jpg",
            "removed_obj_id": 1,
            "obj_b_id": 3,
        }
        result, kept = self._apply_kept(q)
        # object_remove was decided to stay single-frame like attachment_chain,
        # even though removed_obj/obj_b are both real objects that could split.
        self.assertTrue(kept)
        self.assertEqual(result, q)
        self.assertNotIn("reasoning_frame_2", result)

    def test_unmapped_type_is_a_no_op(self):
        q = {"type": "L1_occlusion", "image_name": "orig.jpg"}
        result, kept = self._apply_kept(q)
        self.assertTrue(kept)
        self.assertEqual(result, {"type": "L1_occlusion", "image_name": "orig.jpg"})
        self.assertNotIn("reasoning_frame_2", result)

    def test_no_valid_split_drops_the_question(self):
        q = {
            "type": "object_move_agent",
            "image_name": "orig.jpg",
            "moved_obj_id": 1,
            "query_obj_id": 1,
            "obj_c_id": 999,  # unresolvable id -> group_b resolves empty
        }
        result, kept = self._apply_kept(q)
        self.assertFalse(kept)
        self.assertEqual(result["image_name"], "orig.jpg")
        self.assertNotIn("reasoning_frame_2", result)
        self.assertNotIn("object_frame_groups", result)

    def test_scene_frames_with_no_entry_for_either_frame_rejects_the_split(self):
        # scene_frames is present but has no data for orig.jpg/far.jpg (referability
        # was only run on some other frame) -- frame_a/frame_b can no longer be
        # confirmed referable, so the split must be rejected even though the
        # geometric check alone would have accepted it. This is a fail-closed
        # requirement, not an optional safeguard: a bbox can project fully in-frame
        # while the object is actually occluded or the frame is too blurry to make
        # out (the real-world bug this guards against), so absence of referability
        # data means absence of proof, not proof of absence.
        q = {
            "type": "object_move_agent",
            "image_name": "orig.jpg",
            "moved_obj_id": 1,
            "query_obj_id": 1,
            "obj_c_id": 3,
        }
        result, kept = self._apply_kept(
            q, scene_frames={"some_other.jpg": {"referable_object_ids": [1]}}
        )
        self.assertFalse(kept)
        self.assertNotIn("reasoning_frame_2", result)

    def test_scene_frames_covering_both_groups_allows_the_split(self):
        # Happy-path precedent: when scene_frames properly covers both frame_a and
        # frame_b with each frame's own group referable there, the split proceeds.
        q = {
            "type": "object_move_agent",
            "image_name": "orig.jpg",
            "moved_obj_id": 1,
            "query_obj_id": 1,
            "obj_c_id": 3,
        }
        result, kept = self._apply_kept(
            q,
            scene_frames={
                "orig.jpg": {"referable_object_ids": [1]},
                "far.jpg": {"referable_object_ids": [3]},
            },
        )
        self.assertTrue(kept)
        self.assertEqual(result["reasoning_frame_2"], "far.jpg")

    def test_referability_cache_rejects_split_where_frame_b_also_refers_group_a(self):
        # Geometrically far.jpg is a valid frame_b (obj1 isn't visible there at all),
        # but VLM referability data says obj1 IS referable in far.jpg anyway (e.g. a
        # duplicate/lookalike was judged unique there) -- the question's "first photo
        # shows obj1" premise would still be contradicted, so this must be rejected
        # even though the geometric check alone would have accepted it.
        q = {
            "type": "object_move_agent",
            "image_name": "orig.jpg",
            "moved_obj_id": 1,
            "query_obj_id": 1,
            "obj_c_id": 3,
        }
        result, kept = self._apply_kept(
            q,
            scene_frames={
                "orig.jpg": {"referable_object_ids": [1]},
                "far.jpg": {"referable_object_ids": [1], "attachment_referable_object_ids": []},
            },
        )
        self.assertFalse(kept)
        self.assertNotIn("reasoning_frame_2", result)

    def test_referability_rejection_uses_attachment_referable_ids_for_attachment_types(self):
        # attachment_move is a pure attachment type; obj1/obj2 are only in the relaxed
        # attachment_referable_object_ids set here (referable_object_ids is empty) --
        # the rejection check must still catch it via that set.
        q = {
            "type": "attachment_move",
            "image_name": "orig.jpg",
            "root_id": 1,
            "query_obj_id": 2,
            "obj_ref_id": 3,
        }
        result, kept = self._apply_kept(
            q,
            scene_frames={
                "orig.jpg": {"referable_object_ids": [], "attachment_referable_object_ids": [1, 2]},
                "far.jpg": {"referable_object_ids": [], "attachment_referable_object_ids": [1, 2]},
            },
        )
        self.assertFalse(kept)
        self.assertNotIn("reasoning_frame_2", result)

    def test_referability_rejection_unions_both_id_sets_regardless_of_question_type(self):
        # object_move_agent is NOT an attachment type, but the rejection check must
        # still catch obj1 via attachment_referable_object_ids (not just
        # referable_object_ids) -- an object can pass the looser attachment threshold
        # without being in the plain set (or vice versa, e.g. failing the human
        # review filter attachment_referable_object_ids applies), and either signal
        # is enough to prove it's identifiable in that frame, regardless of which
        # question type is asking.
        q = {
            "type": "object_move_agent",
            "image_name": "orig.jpg",
            "moved_obj_id": 1,
            "query_obj_id": 1,
            "obj_c_id": 3,
        }
        result, kept = self._apply_kept(
            q,
            scene_frames={
                "orig.jpg": {"referable_object_ids": [], "attachment_referable_object_ids": [1]},
                "far.jpg": {"referable_object_ids": [], "attachment_referable_object_ids": [1]},
            },
        )
        self.assertFalse(kept)
        self.assertNotIn("reasoning_frame_2", result)


if __name__ == "__main__":
    unittest.main()
