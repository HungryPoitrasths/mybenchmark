"""Ray casting utilities for occlusion detection using trimesh."""

from __future__ import annotations

from typing import Optional, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import trimesh as _trimesh


class RayCaster:
    """Wraps a trimesh scene for batched ray-intersection queries."""

    def __init__(self, mesh):
        self.mesh = mesh
        # Build a ray-mesh intersector (uses embree if available, else slow fallback)
        try:
            import pyembree  # noqa: F401
            import trimesh
            self.intersector = trimesh.ray.ray_pyembree.RayMeshIntersector(mesh)
        except (ImportError, AttributeError):
            self.intersector = mesh.ray

    @classmethod
    def from_ply(cls, ply_path: str, axis_alignment: Optional[np.ndarray] = None) -> "RayCaster":
        import trimesh
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

        Returns list of (hit_point, triangle_index, distance) sorted by distance.
        """
        direction = direction / np.linalg.norm(direction)
        locations, index_ray, index_tri = self.intersector.intersects_location(
            ray_origins=origin.reshape(1, 3),
            ray_directions=direction.reshape(1, 3),
            multiple_hits=True,
        )
        if len(locations) == 0:
            return []

        distances = np.linalg.norm(locations - origin, axis=1)
        order = np.argsort(distances)
        return [
            (locations[i], int(index_tri[i]), float(distances[i])) for i in order
        ]

    def check_occlusion(
        self,
        camera_pos: np.ndarray,
        target_center: np.ndarray,
        blocker_tri_ids: Optional[set[int]] = None,
    ) -> str:
        """Determine occlusion status of *target* as seen from *camera_pos*.

        If *blocker_tri_ids* is provided, only hits on those triangles count as
        blocking.  Otherwise any hit closer than the target counts.

        Returns one of: "fully_visible", "partially_occluded", "fully_occluded".
        """
        direction = target_center - camera_pos
        dist_to_target = np.linalg.norm(direction)
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
        dist_to_center = np.linalg.norm(target_center - camera_pos)

        visible_count = 0
        rng = np.random.RandomState(42)
        for _ in range(n_samples):
            offset = rng.uniform(-1, 1, size=3) * half_extents * 0.8
            sample_point = target_center + offset
            direction = sample_point - camera_pos
            dist = np.linalg.norm(direction)

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
