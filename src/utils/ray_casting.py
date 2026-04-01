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
_HIT_PATH_MERGE_EPS = 1e-3
_LOCAL_BOUNDARY_RESAMPLE_COUNT = 12
_LOCAL_BOUNDARY_BLEND = 0.2


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


def _compress_hit_path(
    hits: list[tuple[int, float]],
    target_tri_ids: set[int],
) -> list[tuple[bool, float]]:
    """Collapse numerically duplicated hits into a target/non-target path."""
    compressed: list[tuple[bool, float]] = []
    for tri_id, dist in hits:
        is_target = tri_id in target_tri_ids
        if (
            compressed
            and compressed[-1][0] == is_target
            and abs(compressed[-1][1] - float(dist)) <= _HIT_PATH_MERGE_EPS
        ):
            continue
        compressed.append((is_target, float(dist)))
    return compressed


def _classify_hit_path(
    hits: list[tuple[int, float]],
    expected_dist: float,
    target_tri_ids: set[int],
    hit_epsilon: float,
) -> str:
    """Classify one sample ray from ordered hit categories."""
    path = _compress_hit_path(hits, target_tri_ids)
    sample_hit_idx = next(
        (
            idx for idx, (is_target, dist) in enumerate(path)
            if is_target and abs(dist - float(expected_dist)) <= hit_epsilon
        ),
        None,
    )
    if sample_hit_idx is None:
        return "invalid"

    prior_hits = path[:sample_hit_idx]
    if not prior_hits:
        return "visible"
    if not prior_hits[0][0]:
        return "externally_occluded"
    if all(is_target for is_target, _ in prior_hits):
        return "self_occluded"
    return "mixed_boundary"


def _local_triangle_resamples(
    triangle_vertices: np.ndarray,
    barycentric: np.ndarray,
    triangle_id: int,
    n_samples: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate deterministic local surface samples near one source barycentric."""
    tri_vertices = np.asarray(triangle_vertices, dtype=np.float64)
    bary = np.asarray(barycentric, dtype=np.float64)
    if tri_vertices.shape != (3, 3) or bary.shape != (3,) or n_samples <= 0:
        return (
            np.empty((0, 3), dtype=np.float64),
            np.empty((0, 3), dtype=np.float64),
        )
    edge_a = tri_vertices[1] - tri_vertices[0]
    edge_b = tri_vertices[2] - tri_vertices[0]
    if float(np.linalg.norm(np.cross(edge_a, edge_b))) <= 1e-12:
        return (
            np.empty((0, 3), dtype=np.float64),
            np.empty((0, 3), dtype=np.float64),
        )

    bary_sum = float(np.sum(bary))
    if not np.isfinite(bary_sum) or bary_sum <= 1e-12:
        return (
            np.empty((0, 3), dtype=np.float64),
            np.empty((0, 3), dtype=np.float64),
        )
    bary = np.clip(bary / bary_sum, 0.0, 1.0)
    bary = bary / max(float(np.sum(bary)), 1e-12)

    bary_seed = np.round(bary * 1_000_000.0).astype(np.int64)
    seed = zlib.crc32(
        np.asarray([int(triangle_id), *bary_seed.tolist()], dtype=np.int64).tobytes(),
    ) & 0xFFFFFFFF
    rng = np.random.RandomState(seed)
    random_barys = rng.dirichlet(np.ones(3, dtype=np.float64), size=n_samples)
    local_barys = (
        (1.0 - _LOCAL_BOUNDARY_BLEND) * bary[None, :]
        + _LOCAL_BOUNDARY_BLEND * random_barys
    )
    local_barys = local_barys / np.maximum(local_barys.sum(axis=1, keepdims=True), 1e-12)
    local_points = local_barys @ tri_vertices
    return np.asarray(local_points, dtype=np.float64), np.asarray(local_barys, dtype=np.float64)


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

    def _hits_up_to_distance(
        self,
        origin: np.ndarray,
        direction: np.ndarray,
        max_distance: float,
        ignored_tri_ids: Optional[set[int]] = None,
    ) -> list[tuple[int, float]]:
        """Return ordered non-ignored hits up to *max_distance* along one ray."""
        if not np.isfinite(max_distance) or max_distance <= 1e-12:
            return []
        filtered: list[tuple[int, float]] = []
        for _hit_point, tri_id, dist in self.cast_ray(origin, direction):
            if dist > float(max_distance):
                break
            if ignored_tri_ids and tri_id in ignored_tri_ids:
                continue
            filtered.append((int(tri_id), float(dist)))
        return filtered

    def mesh_visibility_stats(
        self,
        camera_pos: np.ndarray,
        target_points: np.ndarray,
        target_tri_ids: set[int],
        ignored_tri_ids: Optional[set[int]] = None,
        hit_epsilon: float = 0.05,
        sample_triangle_ids: np.ndarray | None = None,
        sample_barycentrics: np.ndarray | None = None,
        vertices: np.ndarray | None = None,
        faces: np.ndarray | None = None,
        local_resample_count: int = _LOCAL_BOUNDARY_RESAMPLE_COUNT,
    ) -> tuple[int, int]:
        """Return visible and valid counts for sampled target surface points.

        Back-facing samples — where the target's own front surface sits between
        the camera and the sample — are excluded from the denominator. Mixed
        rays that first pass through the target and then another object before
        reaching the sampled point are refined with local same-triangle
        resampling when sample metadata is available.

        For each ray the method queries:

        1. The first intersection with a *target* triangle (via
           ``first_hits_for_triangles``) and the first intersection with *any*
           triangle (via ``_first_non_ignored_hits``). These handle the common
           visible/external-occlusion cases cheaply.
        2. If the nearest target surface lies in front of the sample, the full
           hit path up to the sample distance is inspected to distinguish
           self-occlusion from mixed target/other-object boundary paths.
        """
        if len(target_points) == 0 or not target_tri_ids:
            return 0, 0

        camera_pos = np.asarray(camera_pos, dtype=np.float64)
        sampled_points = np.asarray(target_points, dtype=np.float64)
        if not self.has_embree and len(sampled_points) > 128:
            if not self._warned_slow_mesh_visibility:
                logger.warning(
                    "Embree unavailable; mesh visibility with %d rays may be slow",
                    len(sampled_points),
                )
                self._warned_slow_mesh_visibility = True

        directions = sampled_points - camera_pos
        expected_dists = np.linalg.norm(directions, axis=1)
        valid_mask = np.isfinite(expected_dists) & (expected_dists > 1e-6)
        if not np.any(valid_mask):
            return 0, 0

        sampled_points = sampled_points[valid_mask]
        directions = directions[valid_mask]
        expected_dists = expected_dists[valid_mask]
        origins = np.broadcast_to(camera_pos, directions.shape).copy()
        triangle_meta = None
        barycentric_meta = None
        if sample_triangle_ids is not None and len(sample_triangle_ids) == len(target_points):
            triangle_meta = np.asarray(sample_triangle_ids, dtype=np.int64)[valid_mask]
        if sample_barycentrics is not None and len(sample_barycentrics) == len(target_points):
            barycentric_meta = np.asarray(sample_barycentrics, dtype=np.float64)[valid_mask]
        vertices_arr = np.asarray(vertices, dtype=np.float64) if vertices is not None else None
        faces_arr = np.asarray(faces, dtype=np.int64) if faces is not None else None

        # Absolute first non-ignored hit per ray (any triangle).
        first_any, _has_any_hit, forced_blocked = self._first_non_ignored_hits(
            origins=origins,
            directions=directions,
            ignored_tri_ids=ignored_tri_ids,
        )

        # First hit on the target's own triangles per ray.
        first_target = self.first_hits_for_triangles(
            origins=origins,
            directions=directions,
            target_tri_ids=target_tri_ids,
            ignored_tri_ids=ignored_tri_ids,
        )

        visible = 0
        valid = 0
        mixed_records: list[tuple[int, float]] = []
        for ray_idx, expected_dist in enumerate(expected_dists):
            if forced_blocked[ray_idx]:
                continue

            t_hit = first_target.get(ray_idx)
            if t_hit is None:
                continue  # ray never reached a target triangle
            _t_point, _t_tri, t_dist = t_hit
            any_hit = first_any.get(ray_idx)

            # Fast path: sampled point itself sits on the front-most target surface.
            if abs(t_dist - float(expected_dist)) <= hit_epsilon:
                valid += 1
                if any_hit is None:
                    visible += 1
                    continue
                any_tri, any_dist = any_hit
                if any_tri in target_tri_ids and abs(any_dist - t_dist) <= hit_epsilon:
                    visible += 1
                continue

            # Full-path inspection is only needed once the target's front surface
            # lies before the sampled point.
            full_hits = self._hits_up_to_distance(
                origin=origins[ray_idx],
                direction=directions[ray_idx],
                max_distance=float(expected_dist) + hit_epsilon,
                ignored_tri_ids=ignored_tri_ids,
            )
            classification = _classify_hit_path(
                full_hits,
                expected_dist=float(expected_dist),
                target_tri_ids=target_tri_ids,
                hit_epsilon=hit_epsilon,
            )
            if classification == "visible":
                valid += 1
                visible += 1
            elif classification == "externally_occluded":
                valid += 1
            elif classification == "mixed_boundary":
                mixed_records.append((ray_idx, float(expected_dist)))

        can_refine_mixed = (
            triangle_meta is not None
            and barycentric_meta is not None
            and vertices_arr is not None
            and faces_arr is not None
            and int(local_resample_count) > 0
        )
        if not can_refine_mixed:
            return visible, valid

        for ray_idx, _expected_dist in mixed_records:
            tri_id = int(triangle_meta[ray_idx])
            if tri_id < 0 or tri_id >= len(faces_arr):
                continue
            tri_vertices = vertices_arr[faces_arr[tri_id]]
            local_points, _local_barys = _local_triangle_resamples(
                triangle_vertices=tri_vertices,
                barycentric=barycentric_meta[ray_idx],
                triangle_id=tri_id,
                n_samples=int(local_resample_count),
            )
            if len(local_points) == 0:
                continue

            local_visible = 0
            local_valid = 0
            for point in local_points:
                direction = np.asarray(point, dtype=np.float64) - camera_pos
                expected_dist = float(np.linalg.norm(direction))
                if not np.isfinite(expected_dist) or expected_dist <= 1e-6:
                    continue
                local_hits = self._hits_up_to_distance(
                    origin=camera_pos,
                    direction=direction,
                    max_distance=expected_dist + hit_epsilon,
                    ignored_tri_ids=ignored_tri_ids,
                )
                classification = _classify_hit_path(
                    local_hits,
                    expected_dist=expected_dist,
                    target_tri_ids=target_tri_ids,
                    hit_epsilon=hit_epsilon,
                )
                if classification == "visible":
                    local_visible += 1
                    local_valid += 1
                elif classification == "externally_occluded":
                    local_valid += 1

            if local_valid >= 2:
                visible += local_visible
                valid += local_valid

        return visible, valid

    def mesh_visibility_ratio(
        self,
        camera_pos: np.ndarray,
        target_points: np.ndarray,
        target_tri_ids: set[int],
        ignored_tri_ids: Optional[set[int]] = None,
        hit_epsilon: float = 0.05,
        sample_triangle_ids: np.ndarray | None = None,
        sample_barycentrics: np.ndarray | None = None,
        vertices: np.ndarray | None = None,
        faces: np.ndarray | None = None,
        local_resample_count: int = _LOCAL_BOUNDARY_RESAMPLE_COUNT,
    ) -> float:
        """Return the visible fraction of valid sampled target surface points."""
        visible, valid = self.mesh_visibility_stats(
            camera_pos=camera_pos,
            target_points=target_points,
            target_tri_ids=target_tri_ids,
            ignored_tri_ids=ignored_tri_ids,
            hit_epsilon=hit_epsilon,
            sample_triangle_ids=sample_triangle_ids,
            sample_barycentrics=sample_barycentrics,
            vertices=vertices,
            faces=faces,
            local_resample_count=local_resample_count,
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
