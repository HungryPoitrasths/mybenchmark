from __future__ import annotations

from typing import Any

import cv2
import numpy as np

DEFAULT_BRISQUE_MAX_SIDE = 0


class BrisqueScorer:
    def __init__(self) -> None:
        try:
            from brisque import BRISQUE
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "BRISQUE scorer is unavailable. Install it with `python -m pip install brisque` and rerun."
            ) from exc

        try:
            self._model = BRISQUE(url=False)
        except TypeError:
            self._model = BRISQUE()

    def score(self, image_bgr: np.ndarray) -> float:
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        try:
            score = self._model.score(image_rgb)
        except TypeError:
            try:
                score = self._model.score(img=image_rgb)
            except TypeError:
                score = self._model.score(image=image_rgb)
        if score is None:
            raise RuntimeError("BRISQUE scorer returned no score")
        return float(score)


def resize_for_brisque(image_bgr: np.ndarray, *, max_side: int | None) -> np.ndarray:
    if max_side is None or int(max_side) <= 0:
        return image_bgr

    height, width = image_bgr.shape[:2]
    longest_side = max(int(width), int(height))
    if longest_side <= int(max_side):
        return image_bgr

    scale = float(max_side) / float(longest_side)
    resized_width = max(1, int(round(width * scale)))
    resized_height = max(1, int(round(height * scale)))
    return cv2.resize(
        image_bgr,
        (resized_width, resized_height),
        interpolation=cv2.INTER_AREA,
    )


def compute_brisque_score(
    image_bgr: np.ndarray,
    *,
    scorer: BrisqueScorer | None = None,
    max_side: int | None = DEFAULT_BRISQUE_MAX_SIDE,
) -> dict[str, Any]:
    brisque_scorer = scorer if scorer is not None else BrisqueScorer()
    brisque_image = resize_for_brisque(image_bgr, max_side=max_side)
    return {
        "brisque_score": float(brisque_scorer.score(brisque_image)),
        "brisque_input_width": int(brisque_image.shape[1]),
        "brisque_input_height": int(brisque_image.shape[0]),
    }
