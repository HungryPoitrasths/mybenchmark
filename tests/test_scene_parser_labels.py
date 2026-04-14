import unittest

from src.alias_groups import get_alias_group_risk_level, resolve_alias_metadata
import src.scene_parser as scene_parser


class SceneParserLabelNormalizationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.original_map = dict(scene_parser._SCANNET_LABEL_MAP)
        self.original_attempted = scene_parser._SCANNET_LABEL_MAP_LOAD_ATTEMPTED
        self.original_default_path = scene_parser._DEFAULT_SCANNET_LABEL_MAP_PATH

    def tearDown(self) -> None:
        scene_parser._SCANNET_LABEL_MAP = dict(self.original_map)
        scene_parser._SCANNET_LABEL_MAP_LOAD_ATTEMPTED = self.original_attempted
        scene_parser._DEFAULT_SCANNET_LABEL_MAP_PATH = self.original_default_path
        scene_parser._refresh_excluded_labels()

    def test_normalize_label_uses_bundled_scannet_label_map_by_default(self) -> None:
        scene_parser._SCANNET_LABEL_MAP = {}
        scene_parser._SCANNET_LABEL_MAP_LOAD_ATTEMPTED = False
        scene_parser._refresh_excluded_labels()

        self.assertEqual(scene_parser.normalize_label("couch"), "sofa")
        self.assertEqual(scene_parser.normalize_label("nightstand"), "night stand")
        self.assertEqual(scene_parser.normalize_label("tv"), "television")
        self.assertEqual(scene_parser.normalize_label("books"), "book")
        self.assertEqual(scene_parser.normalize_label("backpack"), "bag")
        self.assertEqual(scene_parser.normalize_label("washing machines"), "washing machine")
        self.assertEqual(scene_parser.normalize_label("clothes dryers"), "clothes dryer")
        self.assertEqual(scene_parser.normalize_label("keyboard piano"), "piano")
        self.assertEqual(scene_parser.normalize_label("folded ladder"), "ladder")
        self.assertEqual(scene_parser.normalize_label("stepladder"), "ladder")
        self.assertEqual(scene_parser.normalize_label("water cooler"), "water cooler")
        self.assertEqual(scene_parser.normalize_label("ironing board"), "ironing board")
        self.assertEqual(scene_parser.normalize_label("compost bin"), "trash can")
        self.assertEqual(scene_parser.normalize_label("wardrobe closet"), "wardrobe")
        self.assertEqual(scene_parser.normalize_label("wardrobe cabinet"), "wardrobe")
        self.assertEqual(scene_parser.normalize_label("closet wardrobe"), "wardrobe")
    def test_normalize_label_keeps_selected_otherfurniture_labels_specific(self) -> None:
        scene_parser._SCANNET_LABEL_MAP = {
            "radiator": "otherfurniture",
            "trash can": "otherfurniture",
            "washing machine": "otherfurniture",
            "washing machines": "otherfurniture",
            "clothes dryer": "otherfurniture",
            "clothes dryers": "otherfurniture",
            "piano": "otherfurniture",
            "keyboard piano": "otherfurniture",
            "ladder": "otherfurniture",
            "folded ladder": "otherfurniture",
            "stepladder": "otherfurniture",
            "water cooler": "otherfurniture",
            "ironing board": "otherfurniture",
            "compost bin": "otherfurniture",
            "wardrobe": "otherfurniture",
            "wardrobe closet": "otherfurniture",
            "wardrobe cabinet": "otherfurniture",
            "closet wardrobe": "otherfurniture",
        }
        scene_parser._SCANNET_LABEL_MAP_LOAD_ATTEMPTED = True
        scene_parser._refresh_excluded_labels()

        self.assertEqual(scene_parser.normalize_label("radiator"), "radiator")
        self.assertEqual(scene_parser.normalize_label("trash can"), "trash can")
        self.assertEqual(scene_parser.normalize_label("washing machine"), "washing machine")
        self.assertEqual(scene_parser.normalize_label("washing machines"), "washing machine")
        self.assertEqual(scene_parser.normalize_label("clothes dryer"), "clothes dryer")
        self.assertEqual(scene_parser.normalize_label("clothes dryers"), "clothes dryer")
        self.assertEqual(scene_parser.normalize_label("piano"), "piano")
        self.assertEqual(scene_parser.normalize_label("keyboard piano"), "piano")
        self.assertEqual(scene_parser.normalize_label("ladder"), "ladder")
        self.assertEqual(scene_parser.normalize_label("folded ladder"), "ladder")
        self.assertEqual(scene_parser.normalize_label("stepladder"), "ladder")
        self.assertEqual(scene_parser.normalize_label("water cooler"), "water cooler")
        self.assertEqual(scene_parser.normalize_label("ironing board"), "ironing board")
        self.assertEqual(scene_parser.normalize_label("compost bin"), "trash can")
        self.assertEqual(scene_parser.normalize_label("wardrobe"), "wardrobe")
        self.assertEqual(scene_parser.normalize_label("wardrobe closet"), "wardrobe")
        self.assertEqual(scene_parser.normalize_label("wardrobe cabinet"), "wardrobe")
        self.assertEqual(scene_parser.normalize_label("closet wardrobe"), "wardrobe")

    def test_alias_metadata_groups_bedside_table_family(self) -> None:
        alias = resolve_alias_metadata(
            raw_label="nightstand",
            canonical_label="night stand",
        )

        self.assertEqual(alias.alias_group, "bedside_table_family")
        self.assertEqual(alias.alias_source, "explicit")
        self.assertIn("night stand", alias.alias_variants)
        self.assertIn("bedside table", alias.alias_variants)

    def test_alias_metadata_falls_back_to_singleton_group(self) -> None:
        alias = resolve_alias_metadata(
            raw_label="ottoman stool",
            canonical_label="ottoman stool",
        )

        self.assertEqual(alias.alias_group, "ottoman_stool_family")
        self.assertEqual(alias.alias_source, "singleton_fallback")
        self.assertEqual(alias.alias_variants, ("ottoman stool",))

    def test_alias_metadata_uses_explicit_wardrobe_family(self) -> None:
        alias = resolve_alias_metadata(
            raw_label="wardrobe closet",
            canonical_label="wardrobe",
        )

        self.assertEqual(alias.alias_group, "wardrobe_family")
        self.assertEqual(alias.alias_source, "explicit")
        self.assertIn("wardrobe closet", alias.alias_variants)
        self.assertIn("wardrobe", alias.alias_variants)

    def test_alias_metadata_keeps_raw_variant_for_explicit_group(self) -> None:
        alias = resolve_alias_metadata(
            raw_label="step ladder",
            canonical_label="ladder",
        )

        self.assertEqual(alias.alias_group, "ladder_family")
        self.assertEqual(alias.alias_source, "explicit")
        self.assertIn("step ladder", alias.alias_variants)
        self.assertIn("stepladder", alias.alias_variants)

    def test_alias_group_risk_level_marks_review_needed_families(self) -> None:
        self.assertEqual(get_alias_group_risk_level("chair_family"), "review_needed")
        self.assertEqual(get_alias_group_risk_level("bedside_table_family"), "low_risk")
        self.assertEqual(get_alias_group_risk_level("unknown_family"), "singleton")


if __name__ == "__main__":
    unittest.main()
