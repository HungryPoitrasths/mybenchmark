import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from scripts.build_full_benchmark_json import build_full_benchmark


class BuildFullBenchmarkJsonTests(unittest.TestCase):
    def test_build_full_benchmark_dedupes_across_roots_and_recomputes_stats(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            scannetpp_root = root / "output" / "scannetpp_polit" / "0-9"
            pilot_root = root / "output" / "pilot" / "0-9"
            l3_root = root / "output" / "l3_attachment_move_scannetpp"
            scannetpp_root.mkdir(parents=True)
            pilot_root.mkdir(parents=True)
            l3_root.mkdir(parents=True)

            duplicate_question = {
                "scene_id": "0d2ee665be",
                "image_name": "frame_001200.jpg",
                "level": "L2",
                "type": "object_move_distance",
                "question": "If the table moves right, what is the distance to the shelf?",
                "options": ["A", "B", "C", "D"],
                "answer": "B",
            }
            loose_duplicate = dict(duplicate_question)
            loose_duplicate["options"] = ["B", "A", "C", "D"]

            (scannetpp_root / "benchmark.json").write_text(
                json.dumps({"questions": [duplicate_question]}, ensure_ascii=False),
                encoding="utf-8",
            )
            (pilot_root / "benchmark.json").write_text(
                json.dumps({"questions": [dict(duplicate_question), loose_duplicate]}, ensure_ascii=False),
                encoding="utf-8",
            )
            (l3_root / "benchmark.json").write_text(
                json.dumps(
                    {
                        "questions": [
                            {
                                "scene_id": "5942004064",
                                "image_name": "frame_010470.jpg",
                                "level": "L3",
                                "type": "attachment_move",
                                "question": "If the counter moves backward, where is the bag?",
                                "options": ["left", "right", "front", "back"],
                                "answer": "D",
                            }
                        ]
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            payload, metadata = build_full_benchmark([root / "output" / "scannetpp_polit", root / "output" / "pilot", l3_root])

        self.assertEqual(metadata["source_file_count"], 3)
        self.assertEqual(metadata["input_question_count"], 4)
        self.assertEqual(metadata["deduped_question_count"], 2)
        self.assertEqual(metadata["duplicate_question_count"], 2)
        self.assertEqual(metadata["duplicate_exact_uid_count"], 1)
        self.assertEqual(metadata["duplicate_same_prompt_count"], 1)
        self.assertEqual(payload["statistics"]["total"], 2)
        self.assertEqual(payload["statistics"]["by_level"], {"L2": 1, "L3": 1})
        self.assertEqual(payload["statistics"]["by_type"], {"attachment_move": 1, "object_move_distance": 1})
        self.assertEqual(payload["questions"][0]["_dataset"], "scannetpp")
        self.assertEqual(payload["questions"][1]["_dataset"], "scannetpp")


if __name__ == "__main__":
    unittest.main()
