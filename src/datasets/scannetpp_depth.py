"""Random access to ScanNet++ iPhone depth streams."""

from __future__ import annotations

from collections import OrderedDict
import json
from pathlib import Path
import re
import threading
import zlib

import numpy as np

from .base import DepthFrame
from ..utils.colmap_loader import CameraIntrinsics


DEPTH_HEIGHT = 192
DEPTH_WIDTH = 256
DEPTH_PIXELS = DEPTH_HEIGHT * DEPTH_WIDTH
RGB_HEIGHT = 1440
RGB_WIDTH = 1920
_FRAME_NAME_RE = re.compile(r"^frame_(\d{6})(?:\.[^.]+)?$")


class ScanNetPPDepthReader:
    """Decode length-prefixed ScanNet++ iPhone depth frames on demand."""

    def __init__(
        self,
        depth_path: str | Path,
        metadata_path: str | Path,
        *,
        cache_size: int = 32,
    ) -> None:
        if cache_size <= 0:
            raise ValueError("cache_size must be positive")
        self.depth_path = Path(depth_path)
        self.metadata_path = Path(metadata_path)
        self.cache_size = int(cache_size)
        self._offsets: tuple[tuple[int, int], ...] | None = None
        self._intrinsics: dict[int, CameraIntrinsics] | None = None
        self._cache: OrderedDict[int, DepthFrame] = OrderedDict()
        self._lock = threading.RLock()

    @property
    def frame_count(self) -> int:
        self._ensure_index()
        assert self._offsets is not None
        return len(self._offsets)

    def load(self, image_name: str) -> DepthFrame | None:
        match = _FRAME_NAME_RE.match(Path(image_name).name)
        if match is None:
            return None
        frame_index = int(match.group(1))
        with self._lock:
            self._ensure_index()
            assert self._offsets is not None
            assert self._intrinsics is not None
            if frame_index >= len(self._offsets) or frame_index not in self._intrinsics:
                return None
            cached = self._cache.pop(frame_index, None)
            if cached is not None:
                self._cache[frame_index] = cached
                return cached
            offset, payload_length = self._offsets[frame_index]
            with self.depth_path.open("rb") as stream:
                stream.seek(offset)
                payload = stream.read(payload_length)
            if len(payload) != payload_length:
                raise ValueError(
                    f"Truncated ScanNet++ depth payload for frame {frame_index}: "
                    f"expected {payload_length} bytes, got {len(payload)}"
                )
            image_m, source = self._decode_payload(payload)
            frame = DepthFrame(
                image_m=image_m,
                intrinsics=self._intrinsics[frame_index],
                valid_ratio=float(
                    np.count_nonzero(np.isfinite(image_m) & (image_m > 0.0))
                    / max(image_m.size, 1)
                ),
                source=source,
            )
            self._cache[frame_index] = frame
            while len(self._cache) > self.cache_size:
                self._cache.popitem(last=False)
            return frame

    def _ensure_index(self) -> None:
        if self._offsets is not None:
            return
        if not self.depth_path.is_file():
            raise FileNotFoundError(f"Missing ScanNet++ depth stream: {self.depth_path}")
        if not self.metadata_path.is_file():
            raise FileNotFoundError(
                f"Missing ScanNet++ pose/intrinsic metadata: {self.metadata_path}"
            )
        file_size = self.depth_path.stat().st_size
        offsets: list[tuple[int, int]] = []
        with self.depth_path.open("rb") as stream:
            while stream.tell() < file_size:
                raw_length = stream.read(4)
                if len(raw_length) != 4:
                    raise ValueError(
                        "Unsupported or truncated ScanNet++ depth stream; expected "
                        "length-prefixed per-frame payloads"
                    )
                payload_length = int.from_bytes(raw_length, byteorder="little")
                payload_offset = stream.tell()
                if payload_length <= 0 or payload_offset + payload_length > file_size:
                    raise ValueError(
                        "Unsupported or corrupt ScanNet++ depth stream; expected "
                        "length-prefixed per-frame payloads"
                    )
                offsets.append((payload_offset, payload_length))
                stream.seek(payload_length, 1)
            final_position = stream.tell()
        if final_position != file_size:
            raise ValueError("ScanNet++ depth stream does not end on a frame boundary")

        metadata = json.loads(self.metadata_path.read_text(encoding="utf-8"))
        if not isinstance(metadata, dict):
            raise ValueError("ScanNet++ pose_intrinsic_imu.json must contain a frame object")
        if len(metadata) != len(offsets):
            raise ValueError(
                f"ScanNet++ depth/metadata frame count mismatch: "
                f"{len(offsets)} depth frames versus {len(metadata)} metadata frames"
            )
        intrinsics: dict[int, CameraIntrinsics] = {}
        for frame_index in range(len(offsets)):
            key = f"frame_{frame_index:06d}"
            record = metadata.get(key)
            if not isinstance(record, dict):
                raise ValueError(f"Missing ScanNet++ depth metadata entry: {key}")
            matrix = np.asarray(record.get("intrinsic"), dtype=np.float64)
            if matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)):
                raise ValueError(f"Invalid ScanNet++ intrinsic matrix for {key}")
            scale_x = DEPTH_WIDTH / RGB_WIDTH
            scale_y = DEPTH_HEIGHT / RGB_HEIGHT
            intrinsics[frame_index] = CameraIntrinsics(
                width=DEPTH_WIDTH,
                height=DEPTH_HEIGHT,
                fx=float(matrix[0, 0] * scale_x),
                fy=float(matrix[1, 1] * scale_y),
                cx=float(matrix[0, 2] * scale_x),
                cy=float(matrix[1, 2] * scale_y),
            )
        self._offsets = tuple(offsets)
        self._intrinsics = intrinsics

    @staticmethod
    def _decode_payload(payload: bytes) -> tuple[np.ndarray, str]:
        try:
            import lz4.block as lz4_block
        except ImportError:
            lz4_block = None
        if lz4_block is not None:
            try:
                raw = lz4_block.decompress(
                    payload,
                    uncompressed_size=DEPTH_PIXELS * np.dtype(np.uint16).itemsize,
                )
                if len(raw) != DEPTH_PIXELS * np.dtype(np.uint16).itemsize:
                    raise ValueError("unexpected LZ4 depth payload size")
                image_m = (
                    np.frombuffer(raw, dtype=np.uint16)
                    .reshape(DEPTH_HEIGHT, DEPTH_WIDTH)
                    .astype(np.float32)
                    / 1000.0
                )
                return image_m, "scannetpp_depth_bin_lz4"
            except (lz4_block.LZ4BlockError, RuntimeError, ValueError):
                pass
        try:
            raw = zlib.decompress(payload, wbits=-zlib.MAX_WBITS)
        except zlib.error as exc:
            raise ValueError("Unsupported ScanNet++ depth payload encoding") from exc
        if len(raw) != DEPTH_PIXELS * np.dtype(np.float32).itemsize:
            raise ValueError("Unexpected zlib ScanNet++ depth payload size")
        image_m = np.frombuffer(raw, dtype=np.float32).reshape(
            DEPTH_HEIGHT, DEPTH_WIDTH
        ).copy()
        image_m[~np.isfinite(image_m) | (image_m < 0.0)] = 0.0
        return image_m, "scannetpp_depth_bin_zlib"
