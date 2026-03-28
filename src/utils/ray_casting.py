"""Ray casting utilities for occlusion detection using trimesh."""

from __future__ import annotations

import logging
from typing import Optional, TYPE_CHECKING
import zlib

import numpy as np

if TYPE_CHECKING:
    import trimesh as _trimesh

logger = logging.getLogger(__name__)
MAX_RELIABLE_RETRY_RAYS = 10
MIN_NUMPY2_TRIMESH_VERSION = (4, 6, 13)


def _parse_version_tuple(version: str) -> tuple[int, ...]:
    parts: list[int] = []
    for token in version.split("."):
        digits = "".join(ch for ch in token if ch.isdigit())
        if not digits:
            break
        parts.append(int(digits))
    return tuple(parts)


def _ensure_trimesh_numpy_compat(trimesh_module) -> None:
    """Reject known-bad trimesh/NumPy combinations up front.

    Older trimesh releases call ``ndarray.ptp()``, which raises under NumPy 2.x.
    Use a conservative lower bound here so the ray backends fail fast instead of
    silently degrading to a non-ray fallback later in the pipeline.
    """
    if np.lib.NumpyVersion(np.__version__) < "2.0.0":
        return

    version_str = str(getattr(trimesh_module, "__version__", "0"))
    version_tuple = _parse_version_tuple(version_str)
    if version_tuple and version_tuple >= MIN_NUMPY2_TRIMESH_VERSION:
        return

    required = ".".join(str(x) for x in MIN_NUMPY2_TRIMESH_VERSION)
    raise RuntimeError(
        "Ray backends require a NumPy-2-compatible trimesh build. "
        f"Detected numpy {np.__version__} and trimesh {version_str}. "
        f"Upgrade trimesh to >= {required} or downgrade numpy to < 2.0."
    )


class RayCaster:
    """Wraps a trimesh scene for batched ray-intersection queries."""

    def __init__(self, mesh):
        self.mesh = mesh
        self.has_embree = False
        self._reliable_intersector = None
        self._warned_slow_mesh_visibility = False
        self._warned_retry_cap = False
        # Build a ray-mesh intersector (uses embree if available, else slow fallback)
        try:
            import pyembree  # noqa: F401
            import trimesh
            self.intersector = trimesh.ray.ray_pyembree.RayMeshIntersector(mesh)
            self.has_embree = True
            try:
                self._reliable_intersector = trimesh.ray.ray_triangle.RayMeshIntersector(mesh)
            except Exception:
                self._reliable_intersector = self.intersector
        except (ImportError, AttributeError):
            self.intersector = mesh.ray
            self._reliable_intersector = self.intersector

    @classmethod
    def from_ply(cls, ply_path: str, axis_alignment: Optional[np.ndarray] = None) -> "RayCaster":
        import trimesh
        _ensure_trimesh_numpy_compat(trimesh)
        mesh = trimesh.load(ply_path, process=False)
        if isinstance(mesh, trimesh.Scene):
            mesh = mesh.dump(concatenate=True)
        # Apply axis alignment so the mesh lives in the same coordinate frame
        # as the object centres and camera poses (which are already aligned).
        if axis_alignment is not None and not np.allclose(axis_alignment, np.eye(4)):
            mesh.apply_transform(axis_alignment)
        return cls(mesh)

    def cast_ray(
        self,
        origin: np.ndarray,
        direction: np.ndarray,
    ) -> list[tuple[np.ndarray, int, float]]:
        """Cast a single ray and return sorted hit list.

        When Embree is active, retries a no-hit result with the reliable
        triangle intersector to reduce backend-specific misses.

        Returns list of (hit_point, triangle_index, distance) sorted by distance.
        """
        direction = np.asarray(direction, dtype=np.float64)
        norm = float(np.linalg.norm(direction))
        if not np.isfinite(norm) or norm <= 1e-12:
            return []
        direction = direction / norm
        ray_origin = np.asarray(origin, dtype=np.float64).reshape(1, 3)
        ray_direction = direction.reshape(1, 3)

        locations, index_ray, index_tri = self.intersector.intersects_location(
            ray_origins=ray_origin,
            ray_directions=ray_direction,
            multiple_hits=True,
        )
        if (
            len(locations) == 0
            and self.has_embree
            and self._reliable_intersector is not None
            and self._reliable_intersector is not self.intersector
        ):
            locations, index_ray, index_tri = self._reliable_intersector.intersects_location(
                ray_origins=ray_origin,
                ray_directions=ray_direction,
                multiple_hits=True,
            )
        if len(locations) == 0:
            return []

        distances = np.linalg.norm(locations - ray_origin[0], axis=1)
        order = np.argsort(distances)
        return [
            (locations[i], int(index_tri[i]), float(distances[i])) for i in order
        ]

    def first_visible_hit(
        self,
        origin: np.ndarray,
        direction: np.ndarray,
        ignored_tri_ids: Optional[set[int]] = None,
    ) -> tuple[np.ndarray, int, float] | None:
        """Return the first hit not masked by *ignored_tri_ids*."""
        hits = self.cast_ray(origin, direction)
        if not hits:
            return None

        if not ignored_tri_ids:
            return hits[0]

        for hit_point, tri_id, dist in hits:
            if tri_id not in ignored_tri_ids:
                return hit_point, tri_id, dist
        return None

    def first_hit_for_triangles(
        self,
        origin: np.ndarray,
        direction: np.ndarray,
        target_tri_ids: set[int],
        ignored_tri_ids: Optional[set[int]] = None,
    ) -> tuple[np.ndarray, int, float] | None:
        """Return the nearest hit whose triangle belongs to *target_tri_ids*."""
        if not target_tri_ids:
            return None
        hits = self.cast_ray(origin, direction)
        if not hits:
            return None
        for hit_point, tri_id, dist in hits:
            if ignored_tri_ids and tri_id in ignored_tri_ids:
                continue
            if tri_id in target_tri_ids:
                return hit_point, tri_id, dist
        return None

    def first_hits_for_triangles(
        self,
        origins: np.ndarray,
        directions: np.ndarray,
        target_tri_ids: set[int],
        ignored_tri_ids: Optional[set[int]] = None,
    ) -> dict[int, tuple[np.ndarray, int, float]]:
        """Return the nearest target-triangle hit for each ray in a batch."""
        if len(origins) == 0 or not target_tri_ids:
            return {}

        directions = np.asarray(directions, dtype=np.float64)
        dir_norms = np.linalg.norm(directions, axis=1)
        valid_mask = np.isfinite(dir_norms) & (dir_norms > 1e-12)
        if not np.any(valid_mask):
            return {}

        valid_ray_indices = np.flatnonzero(valid_mask)
        query_origins = np.asarray(origins, dtype=np.float64)[valid_mask]
        query_directions = directions[valid_mask] / dir_norms[valid_mask][:, None]

        locations, index_ray, index_tri = self.intersector.intersects_location(
            ray_origins=query_origins,
            ray_directions=query_directions,
            multiple_hits=True,
        )
        if (
            self.has_embree
            and self._reliable_intersector is not None
            and self._reliable_intersector is not self.intersector
        ):
            hit_query_indices = (
                np.unique(index_ray.astype(np.int64))
                if len(index_ray) > 0
                else np.empty(0, dtype=np.int64)
            )
            missing_query_indices = np.setdiff1d(
                np.arange(len(query_origins), dtype=np.int64),
                hit_query_indices,
                assume_unique=False,
            )
            if len(missing_query_indices) > 0:
                retry_locations, retry_index_ray, retry_index_tri = self._reliable_intersector.intersects_location(
                    ray_origins=query_origins[missing_query_indices],
                    ray_directions=query_directions[missing_query_indices],
                    multiple_hits=True,
                )
                if len(retry_locations) > 0:
                    retry_index_ray = missing_query_indices[np.asarray(retry_index_ray, dtype=np.int64)]
                    if len(locations) == 0:
                        locations = retry_locations
                        index_ray = retry_index_ray
                        index_tri = retry_index_tri
                    else:
                        locations = np.concatenate([locations, retry_locations], axis=0)
                        index_ray = np.concatenate([index_ray, retry_index_ray], axis=0)
                        index_tri = np.concatenate([index_tri, retry_index_tri], axis=0)

        if len(locations) == 0:
            return {}

        distances = np.linalg.norm(locations - query_origins[index_ray], axis=1)
        order = np.lexsort((distances, index_ray))
        first_hits: dict[int, tuple[np.ndarray, int, float]] = {}
        for idx in order:
            tri_id = int(index_tri[idx])
            if ignored_tri_ids and tri_id in ignored_tri_ids:
                continue
            if tri_id not in target_tri_ids:
                continue
            ray_idx = int(valid_ray_indices[int(index_ray[idx])])
            if ray_idx not in first_hits:
                first_hits[ray_idx] = (
                    np.asarray(locations[idx], dtype=np.float64),
                    tri_id,
                    float(distances[idx]),
                )
        return first_hits

    def _first_non_ignored_hits(
        self,
        origins: np.ndarray,
        directions: np.ndarray,
        ignored_tri_ids: Optional[set[int]] = None,
    ) -> tuple[dict[int, tuple[int, float]], np.ndarray, np.ndarray]:
        """Return nearest non-ignored hit per ray plus hit and forced-block masks.

        Any non-finite or zero-length direction is treated as a miss. Valid
        directions are normalized internally so callers do not need to do it.
        """
        n_rays = len(origins)
        first_hits: dict[int, tuple[int, float]] = {}
        has_any_hit = np.zeros(n_rays, dtype=bool)
        has_non_ignored_hit = np.zeros(n_rays, dtype=bool)
        forced_blocked = np.zeros(n_rays, dtype=bool)

        directions = np.asarray(directions, dtype=np.float64)
        dir_norms = np.linalg.norm(directions, axis=1)
        valid_mask = np.isfinite(dir_norms) & (dir_norms > 1e-12)
        if not np.any(valid_mask):
            return first_hits, has_any_hit, forced_blocked

        valid_ray_indices = np.flatnonzero(valid_mask)
        valid_index_lookup = {
            int(ray_idx): pos for pos, ray_idx in enumerate(valid_ray_indices)
        }
        query_origins = origins[valid_mask]
        query_directions = directions[valid_mask] / dir_norms[valid_mask][:, None]

        locations, index_ray, index_tri = self.intersector.intersects_location(
            ray_origins=query_origins,
            ray_directions=query_directions,
            multiple_hits=True,
        )
        if len(locations) > 0:
            distances = np.linalg.norm(locations - query_origins[index_ray], axis=1)
            order = np.lexsort((distances, index_ray))
            for idx in order:
                ray_idx = int(valid_ray_indices[int(index_ray[idx])])
                tri_id = int(index_tri[idx])
                has_any_hit[ray_idx] = True
                if ignored_tri_ids and tri_id in ignored_tri_ids:
                    continue
                if ray_idx not in first_hits:
                    first_hits[ray_idx] = (tri_id, float(distances[idx]))
                    has_non_ignored_hit[ray_idx] = True

        if (
            ignored_tri_ids
            and self.has_embree
            and self._reliable_intersector is not None
            and self._reliable_intersector is not self.intersector
        ):
            ignored_only = np.flatnonzero(has_any_hit & ~has_non_ignored_hit)
            if len(ignored_only) > 0:
                if len(ignored_only) > MAX_RELIABLE_RETRY_RAYS:
                    forced_blocked[ignored_only] = True
                    if not self._warned_retry_cap:
                        logger.warning(
                            "Skipping reliable ray retry for %d rays; capping at %d to avoid slow fallback",
                            len(ignored_only),
                            MAX_RELIABLE_RETRY_RAYS,
                        )
                        self._warned_retry_cap = True
                else:
                    retry_origins = origins[ignored_only]
                    retry_directions = np.asarray(
                        [query_directions[valid_index_lookup[int(ray_idx)]] for ray_idx in ignored_only],
                        dtype=np.float64,
                    )
                    retry_locations, retry_index_ray, retry_index_tri = self._reliable_intersector.intersects_location(
                        ray_origins=retry_origins,
                        ray_directions=retry_directions,
                        multiple_hits=True,
                    )
                    if len(retry_locations) > 0:
                        retry_distances = np.linalg.norm(
                            retry_locations - retry_origins[retry_index_ray],
                            axis=1,
                        )
                        retry_order = np.lexsort((retry_distances, retry_index_ray))
                        for idx in retry_order:
                            retry_ray_idx = int(retry_index_ray[idx])
                            ray_idx = int(ignored_only[retry_ray_idx])
                            tri_id = int(retry_index_tri[idx])
                            if tri_id in ignored_tri_ids:
                                continue
                            if ray_idx not in first_hits:
                                first_hits[ray_idx] = (tri_id, float(retry_distances[idx]))
                                has_non_ignored_hit[ray_idx] = True

        return first_hits, has_any_hit, forced_blocked

    def mesh_visibility_stats(
        self,
        camera_pos: np.ndarray,
        target_points: np.ndarray,
        target_tri_ids: set[int],
        ignored_tri_ids: Optional[set[int]] = None,
        hit_epsilon: float = 0.05,
    ) -> tuple[int, int]:
        """Return visible and valid counts for sampled target surface points.

        Visibility is inferred from point hits within ``hit_epsilon`` of the
        sampled point distance. Very thin objects whose back-facing samples sit
        within that epsilon can still be slightly over-counted as visible.
        ``ignored_tri_ids`` is a general ignore-mask for triangles that should
        not block visibility, such as removed objects or the target itself.
        """
        if len(target_points) == 0 or not target_tri_ids:
            return 0, 0

        sampled_points = np.asarray(target_points, dtype=np.float64)
        if not self.has_embree and len(sampled_points) > 128:
            if not self._warned_slow_mesh_visibility:
                logger.warning(
                    "Embree unavailable; mesh visibility with %d rays may be slow",
                    len(sampled_points),
                )
                self._warned_slow_mesh_visibility = True

        directions = sampled_points - np.asarray(camera_pos, dtype=np.float64)
        expected_dists = np.linalg.norm(directions, axis=1)
        valid_mask = np.isfinite(expected_dists) & (expected_dists > 1e-6)
        if not np.any(valid_mask):
            return 0, 0

        directions = directions[valid_mask]
        expected_dists = expected_dists[valid_mask]
        origins = np.broadcast_to(np.asarray(camera_pos, dtype=np.float64), directions.shape).copy()

        first_hits, _has_any_hit, forced_blocked = self._first_non_ignored_hits(
            origins=origins,
            directions=directions,
            ignored_tri_ids=ignored_tri_ids,
        )

        visible = 0
        valid = 0
        for ray_idx, expected_dist in enumerate(expected_dists):
            if forced_blocked[ray_idx]:
                continue
            hit = first_hits.get(ray_idx)
            if hit is None:
                continue
            valid += 1
            tri_id, hit_dist = hit
            if tri_id in target_tri_ids and abs(hit_dist - float(expected_dist)) <= hit_epsilon:
                visible += 1

        return visible, valid

    def mesh_visibility_ratio(
        self,
        camera_pos: np.ndarray,
        target_points: np.ndarray,
        target_tri_ids: set[int],
        ignored_tri_ids: Optional[set[int]] = None,
        hit_epsilon: float = 0.05,
    ) -> float:
        """Return the visible fraction of valid sampled target surface points."""
        visible, valid = self.mesh_visibility_stats(
            camera_pos=camera_pos,
            target_points=target_points,
            target_tri_ids=target_tri_ids,
            ignored_tri_ids=ignored_tri_ids,
            hit_epsilon=hit_epsilon,
        )
        if valid <= 0:
            return 0.0
        return float(visible / valid)

    def check_occlusion(
        self,
        camera_pos: np.ndarray,
        target_center: np.ndarray,
        blocker_tri_ids: Optional[set[int]] = None,
    ) -> str:
        """Determine occlusion status of *target* as seen from *camera_pos*.

        If *blocker_tri_ids* is provided, only hits on those triangles count as
        blocking.  Otherwise any hit closer than the target counts.

        Returns one of: "fully_visible" or "fully_occluded".

        This is a single-center-ray heuristic; it cannot distinguish partial
        occlusion. Use ``multi_ray_occlusion`` for three-way grading.
        """
        direction = target_center - camera_pos
        dist_to_target = np.linalg.norm(direction)
        if not np.isfinite(dist_to_target) or dist_to_target <= 1e-12:
            return "fully_visible"
        direction_norm = direction / dist_to_target

        hits = self.cast_ray(camera_pos, direction_norm)
        if not hits:
            return "fully_visible"

        first_hit_dist = hits[0][2]

        # If the first hit is (nearly) at the target distance, it's the target itself
        if abs(first_hit_dist - dist_to_target) < 0.05:
            return "fully_visible"

        # Something is in front of the target
        if first_hit_dist < dist_to_target - 0.05:
            if blocker_tri_ids is not None:
                if hits[0][1] in blocker_tri_ids:
                    return "fully_occluded"
                return "fully_visible"
            return "fully_occluded"

        return "fully_visible"

    def multi_ray_occlusion(
        self,
        camera_pos: np.ndarray,
        target_bbox_min: np.ndarray,
        target_bbox_max: np.ndarray,
        n_samples: int = 8,
    ) -> str:
        """Sample multiple rays toward the target bbox for finer occlusion grading.

        Returns: "fully_visible", "partially_occluded", or "fully_occluded".
        """
        target_center = (target_bbox_min + target_bbox_max) / 2
        half_extents = (target_bbox_max - target_bbox_min) / 2

        visible_count = 0
        # Keep sampling deterministic per camera/target pair while avoiding the
        # same fixed offsets for every object in every scene.
        seed_bytes = np.concatenate(
            [
                np.asarray(camera_pos, dtype=np.float32),
                np.asarray(target_bbox_min, dtype=np.float32),
                np.asarray(target_bbox_max, dtype=np.float32),
            ]
        ).tobytes()
        rng = np.random.RandomState(zlib.crc32(seed_bytes) & 0xFFFFFFFF)
        for _ in range(n_samples):
            offset = rng.uniform(-1, 1, size=3) * half_extents * 0.8
            sample_point = target_center + offset
            direction = sample_point - camera_pos
            dist = np.linalg.norm(direction)
            if not np.isfinite(dist) or dist <= 1e-12:
                visible_count += 1
                continue

            hits = self.cast_ray(camera_pos, direction / dist)
            if not hits or hits[0][2] >= dist - 0.05:
                visible_count += 1

        ratio = visible_count / n_samples
        if ratio > 0.8:
            return "fully_visible"
        elif ratio > 0.2:
            return "partially_occluded"
        else:
            return "fully_occluded"

    def remove_triangles(self, tri_ids_to_remove: set[int]) -> "RayCaster":
        """Return a new RayCaster with specified triangles removed."""
        import trimesh
        mask = np.ones(len(self.mesh.faces), dtype=bool)
        for tid in tri_ids_to_remove:
            if 0 <= tid < len(mask):
                mask[tid] = False
        kept_faces = self.mesh.faces[mask]
        reduced_mesh = trimesh.Trimesh(vertices=self.mesh.vertices, faces=kept_faces, process=False)
        return RayCaster(reduced_mesh)
