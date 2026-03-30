import unittest

from src.qa_generator import _default_templates, _normalize_template_aliases


EXPECTED_L2_OBJECT_CENTRIC_ORBIT_TEMPLATE = (
    "Imagine you are {obj_query} and facing toward {obj_face}. "
    "If {obj_move_source} were moved along a {angle}-degree {rotation_direction} "
    "(viewed from above) orbit around the center of {obj_face}, without changing "
    "its own facing direction, from your perspective, in which direction would "
    "{obj_ref} be?"
)


class QuestionTemplateTests(unittest.TestCase):
    def test_default_object_centric_rotation_templates_use_explicit_orbit_wording(self) -> None:
        templates = _default_templates()

        self.assertEqual(
            templates["L2_object_rotate_object_centric"],
            [EXPECTED_L2_OBJECT_CENTRIC_ORBIT_TEMPLATE],
        )
        self.assertEqual(
            templates["L2_object_move_object_centric"],
            [EXPECTED_L2_OBJECT_CENTRIC_ORBIT_TEMPLATE],
        )

    def test_normalize_template_aliases_backfills_missing_sibling_key(self) -> None:
        with self.subTest("canonical_present_alias_missing"):
            normalized = _normalize_template_aliases(
                {
                    "L2_object_rotate_object_centric": [
                        EXPECTED_L2_OBJECT_CENTRIC_ORBIT_TEMPLATE
                    ]
                }
            )
            self.assertEqual(
                normalized["L2_object_move_object_centric"],
                [EXPECTED_L2_OBJECT_CENTRIC_ORBIT_TEMPLATE],
            )

        with self.subTest("alias_present_canonical_missing"):
            normalized = _normalize_template_aliases(
                {
                    "L2_object_move_object_centric": [
                        EXPECTED_L2_OBJECT_CENTRIC_ORBIT_TEMPLATE
                    ]
                }
            )
            self.assertEqual(
                normalized["L2_object_rotate_object_centric"],
                [EXPECTED_L2_OBJECT_CENTRIC_ORBIT_TEMPLATE],
            )


if __name__ == "__main__":
    unittest.main()
