import unittest
from pathlib import Path
import shutil
import uuid

import cv2
import numpy as np

import scripts.filter_image_quality as quality_module


class _StubBrisqueScorer:
    def __init__(self, scores):
        self._scores = list(scores)

    def score(self, image_bgr: np.ndarray) -> float:
        self.last_shape = image_bgr.shape
        if not self._scores:
            raise AssertionError("BRISQUE scorer called more times than expected")
        return float(self._scores.pop(0))


class FilterImageQualityTests(unittest.TestCase):
    def test_select_image_paths_in_order_returns_sequential_slice(self) -> None:
        image_paths = [Path(f"{index:06d}.jpg") for index in range(6)]

        selected = quality_module.select_image_paths_in_order(
            image_paths,
            max_images=3,
            sample_start=2,
        )

        self.assertEqual(selected, image_paths[2:5])

    def test_select_image_paths_in_order_rejects_non_positive_max_images(self) -> None:
        with self.assertRaisesRegex(ValueError, "max_images must be > 0"):
            quality_module.select_image_paths_in_order(
                [Path("000000.jpg")],
                max_images=0,
                sample_start=0,
            )

    def test_sharp_metrics_rank_sharp_image_above_blurred_image(self) -> None:
        base = np.zeros((128, 128), dtype=np.uint8)
        cv2.rectangle(base, (24, 24), (104, 104), 255, thickness=-1)
        blurred = cv2.GaussianBlur(base, (15, 15), sigmaX=4.0)

        sharp_laplacian = quality_module.compute_laplacian_variance(base)
        blur_laplacian = quality_module.compute_laplacian_variance(blurred)
        sharp_tenengrad = quality_module.compute_tenengrad(base)
        blur_tenengrad = quality_module.compute_tenengrad(blurred)

        self.assertGreater(sharp_laplacian, blur_laplacian)
        self.assertGreater(sharp_tenengrad, blur_tenengrad)

    def test_resize_for_brisque_limits_longest_side(self) -> None:
        image = np.zeros((1000, 800, 3), dtype=np.uint8)

        resized = quality_module.resize_for_brisque(image, max_side=512)

        self.assertEqual(resized.shape[:2], (512, 410))

    def test_resize_for_brisque_can_be_disabled(self) -> None:
        image = np.zeros((1000, 800, 3), dtype=np.uint8)

        resized = quality_module.resize_for_brisque(image, max_side=0)

        self.assertIs(resized, image)

    def test_apply_brisque_filter_only_scores_stage1_survivors(self) -> None:
        image = np.zeros((32, 32, 3), dtype=np.uint8)
        records = [
            (
                quality_module.ImageQualityRecord(
                    image_path=Path("a.jpg"),
                    width=32,
                    height=32,
                    laplacian_variance=200.0,
                    tenengrad=40.0,
                    stage1_pass=True,
                ),
                image,
            ),
            (
                quality_module.ImageQualityRecord(
                    image_path=Path("b.jpg"),
                    width=32,
                    height=32,
                    laplacian_variance=10.0,
                    tenengrad=3.0,
                    stage1_pass=False,
                ),
                image,
            ),
            (
                quality_module.ImageQualityRecord(
                    image_path=Path("c.jpg"),
                    width=32,
                    height=32,
                    laplacian_variance=220.0,
                    tenengrad=50.0,
                    stage1_pass=True,
                ),
                image,
            ),
        ]

        results = quality_module.apply_brisque_filter(
            records,
            brisque_threshold=35.0,
            brisque_max_side=16,
            scorer=_StubBrisqueScorer([22.0, 41.0]),
        )

        self.assertEqual([item.final_pass for item in results], [True, False, False])
        self.assertEqual(results[0].brisque_score, 22.0)
        self.assertEqual(results[0].brisque_input_width, 16)
        self.assertEqual(results[0].brisque_input_height, 16)
        self.assertIsNone(results[1].brisque_score)
        self.assertEqual(results[2].brisque_score, 41.0)
        self.assertEqual(results[1].stage2_pass, False)

    def test_build_html_report_lists_selected_images(self) -> None:
        output_dir = Path("output/report_test")
        records = [
            quality_module.ImageQualityRecord(
                image_path=Path("images/000002_keep.jpg"),
                width=640,
                height=480,
                laplacian_variance=180.0,
                tenengrad=25.0,
                stage1_pass=True,
                brisque_score=19.5,
                brisque_input_width=640,
                brisque_input_height=480,
                stage2_pass=True,
                final_pass=True,
            ),
            quality_module.ImageQualityRecord(
                image_path=Path("images/000001_drop.jpg"),
                width=640,
                height=480,
                laplacian_variance=20.0,
                tenengrad=4.0,
                stage1_pass=False,
                brisque_score=None,
                brisque_input_width=None,
                brisque_input_height=None,
                stage2_pass=False,
                final_pass=False,
            ),
        ]

        html = quality_module.build_html_report(
            records=records,
            output_dir=output_dir,
            title="Quality Report",
            summary={"total_images": 2, "stage1_pass": 1, "final_selected": 1},
            thresholds={
                "laplacian_threshold": 120.0,
                "tenengrad_threshold": 15.0,
                "brisque_threshold": 35.0,
            },
        )

        self.assertIn("000001_drop.jpg", html)
        self.assertIn("000002_keep.jpg", html)
        self.assertIn("BRISQUE: 19.50", html)
        self.assertIn("BRISQUE: -", html)
        self.assertIn("filtered out", html)
        self.assertIn("kept", html)
        self.assertLess(html.index("000001_drop.jpg"), html.index("000002_keep.jpg"))

    def test_evaluate_stage1_reads_image_and_applies_thresholds(self) -> None:
        root = Path(__file__).resolve().parent / "_tmp" / f"quality_{uuid.uuid4().hex}"
        root.mkdir(parents=True, exist_ok=False)
        self.addCleanup(shutil.rmtree, root, True)
        image_path = root / "checker.png"
        image = np.zeros((128, 128, 3), dtype=np.uint8)
        cv2.rectangle(image, (24, 24), (104, 104), (255, 255, 255), thickness=-1)
        ok, encoded = cv2.imencode(".png", image)
        self.assertTrue(ok)
        encoded.tofile(str(image_path))

        record, _loaded = quality_module.evaluate_stage1(
            image_path,
            laplacian_threshold=10.0,
            tenengrad_threshold=5.0,
        )

        self.assertTrue(record.stage1_pass)
        self.assertGreater(record.laplacian_variance, 10.0)
        self.assertGreater(record.tenengrad, 5.0)


if __name__ == "__main__":
    unittest.main()
