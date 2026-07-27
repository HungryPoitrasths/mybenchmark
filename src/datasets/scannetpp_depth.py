"""Random access to ScanNet++ iPhone depth streams."""

from __future__ import annotations

from collections import OrderedDict
import json
import logging
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
DEFAULT_DEPTH_CACHE_SIZE = 1280

logger = logging.getLogger(__name__)


class ScanNetPPDepthReader:
    """Decode length-prefixed ScanNet++ iPhone depth frames on demand."""

    def __init__(
        self,
        depth_path: str | Path,
        metadata_path: str | Path,
        *,
        cache_size: int = DEFAULT_DEPTH_CACHE_SIZE,
    ) -> None:
        if cache_size <= 0:
            raise ValueError("cache_size must be positive")
        self.depth_path = Path(depth_path)
        self.metadata_path = Path(metadata_path)
        self.cache_size = int(cache_size)
        self._offsets: tuple[tuple[int, int], ...] | None = None
        self._intrinsics: dict[int, CameraIntrinsics] | None = None
        self._cache: OrderedDict[int, DepthFrame] = OrderedDict()
        self._unreadable_frames: set[int] = set()
        self._payload_encoding: str | None = None
        self.cache_hits = 0
        self.cache_misses = 0
        self.decode_count = 0
        self.decoder_redetections = 0
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
            if frame_index in self._unreadable_frames:
                return None
            cached = self._cache.pop(frame_index, None)
            if cached is not None:
                self.cache_hits += 1
                self._cache[frame_index] = cached
                return cached
            self.cache_misses += 1
            offset, payload_length = self._offsets[frame_index]
            with self.depth_path.open("rb") as stream:
                stream.seek(offset)
                payload = stream.read(payload_length)
            if len(payload) != payload_length:
                raise ValueError(
                    f"Truncated ScanNet++ depth payload for frame {frame_index}: "
                    f"expected {payload_length} bytes, got {len(payload)}"
                )
            try:
                image_m, source = self._decode_payload(payload)
            except ValueError as exc:
                self._unreadable_frames.add(frame_index)
                logger.warning(
                    "Skipping unreadable ScanNet++ depth frame %d in %s: %s",
                    frame_index,
                    self.depth_path,
                    exc,
                )
                return None
            self.decode_count += 1
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

    def diagnostics(self) -> dict[str, int | str | None]:
        with self._lock:
            return {
                "cache_size_limit": self.cache_size,
                "cached_frame_count": len(self._cache),
                "cache_hits": self.cache_hits,
                "cache_misses": self.cache_misses,
                "decode_count": self.decode_count,
                "decoder_redetections": self.decoder_redetections,
                "unreadable_frame_count": len(self._unreadable_frames),
                "payload_encoding": self._payload_encoding,
            }

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
    def _decoded_image(raw: bytes, source: str) -> tuple[np.ndarray, str]:
        uint16_size = DEPTH_PIXELS * np.dtype(np.uint16).itemsize
        float32_size = DEPTH_PIXELS * np.dtype(np.float32).itemsize
        if len(raw) == uint16_size:
            image_m = (
                np.frombuffer(raw, dtype="<u2")
                .reshape(DEPTH_HEIGHT, DEPTH_WIDTH)
                .astype(np.float32)
                / 1000.0
            )
        elif len(raw) == float32_size:
            image_m = np.frombuffer(raw, dtype="<f4").reshape(
                DEPTH_HEIGHT, DEPTH_WIDTH
            ).copy()
            image_m[~np.isfinite(image_m) | (image_m < 0.0)] = 0.0
        else:
            raise ValueError(
                f"unexpected decompressed size {len(raw)} bytes "
                f"(expected {uint16_size} or {float32_size})"
            )
        return image_m, source

    @classmethod
    def _decode_payload_as(
        cls,
        payload: bytes,
        encoding: str,
    ) -> tuple[np.ndarray, str]:
        if encoding == "lz4":
            try:
                import lz4.block as lz4_block
            except ImportError as exc:
                raise RuntimeError(
                    "Cannot decode this ScanNet++ depth payload because the lz4 "
                    "package is not installed; install it with `python -m pip install lz4`"
                ) from exc
            sizes = (
                DEPTH_PIXELS * np.dtype(np.float32).itemsize,
                DEPTH_PIXELS * np.dtype(np.uint16).itemsize,
            )
            last_error: Exception | None = None
            for uncompressed_size in sizes:
                try:
                    raw = lz4_block.decompress(
                        payload,
                        uncompressed_size=uncompressed_size,
                    )
                    return cls._decoded_image(raw, "scannetpp_depth_bin_lz4")
                except (lz4_block.LZ4BlockError, RuntimeError, ValueError) as exc:
                    last_error = exc
            raise ValueError("invalid LZ4 depth payload") from last_error
        if encoding == "zlib":
            raw = zlib.decompress(payload, wbits=zlib.MAX_WBITS | 32)
            return cls._decoded_image(raw, "scannetpp_depth_bin_zlib")
        if encoding == "deflate":
            raw = zlib.decompress(payload, wbits=-zlib.MAX_WBITS)
            return cls._decoded_image(raw, "scannetpp_depth_bin_deflate")
        raise ValueError(f"unknown ScanNet++ depth encoding: {encoding}")

    def _decode_payload(self, payload: bytes) -> tuple[np.ndarray, str]:
        if self._payload_encoding is not None:
            try:
                return self._decode_payload_as(payload, self._payload_encoding)
            except (RuntimeError, ValueError, zlib.error):
                self.decoder_redetections += 1
                self._payload_encoding = None

        lz4_error: RuntimeError | None = None
        for encoding in ("lz4", "zlib", "deflate"):
            try:
                decoded = self._decode_payload_as(payload, encoding)
            except RuntimeError as exc:
                lz4_error = exc
                continue
            except (ValueError, zlib.error):
                continue
            self._payload_encoding = encoding
            return decoded

        if lz4_error is not None:
            raise lz4_error
        raise ValueError(
            "unsupported or corrupt payload (not an LZ4 block, zlib stream, "
            "or raw DEFLATE stream with a recognized depth image size)"
        )
