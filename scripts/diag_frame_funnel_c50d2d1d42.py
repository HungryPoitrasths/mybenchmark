"""Standalone diagnostic: reproduce select_frames() funnel for c50d2d1d42.

Run from repo root:
    python scripts/diag_frame_funnel_c50d2d1d42.py
"""
import sys
import types
from pathlib import Path
from collections import Counter

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

# ---------------------------------------------------------------------------
# Environment workarounds for THIS local diagnostic run only (do not affect
# the real pipeline / server behavior):
#  1. Non-ASCII path (contains Chinese characters) breaks cv2.imread() on
#     Windows -> monkeypatch to use np.fromfile + cv2.imdecode instead.
#  2. open3d is not installed locally -> shim open3d.io.read_triangle_mesh
#     with a minimal binary-PLY vertex reader (faces are not needed for the
#     bbox-only object geometry used by select_frames()/get_visible_objects).
# ---------------------------------------------------------------------------
_orig_imread = cv2.imread


def _unicode_safe_imread(path, flags=cv2.IMREAD_COLOR):
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
        if data.size == 0:
            return None
        return cv2.imdecode(data, flags)
    except Exception:
        return None


cv2.imread = _unicode_safe_imread


class _FakePlyMesh:
    def __init__(self, vertices, faces):
        self._vertices = vertices
        self._faces = faces

    def __getattr__(self, name):
        if name == "vertices":
            return self._vertices
        if name == "triangles":
            return self._faces
        raise AttributeError(name)


def _read_binary_ply_vertices_only(path):
    """Minimal reader for binary_little_endian PLY: x,y,z (+ optional uchar rgb)."""
    with open(path, "rb") as f:
        header = b""
        while True:
            line = f.readline()
            if not line:
                raise ValueError("Unexpected EOF in PLY header")
            header += line
            if line.strip() == b"end_header":
                break
        header_text = header.decode("ascii", errors="replace")
        n_vertex = None
        vertex_props = []
        in_vertex_elem = False
        for line in header_text.splitlines():
            line = line.strip()
            if line.startswith("element vertex"):
                n_vertex = int(line.split()[-1])
                in_vertex_elem = True
                continue
            if line.startswith("element ") and not line.startswith("element vertex"):
                in_vertex_elem = False
                continue
            if in_vertex_elem and line.startswith("property"):
                parts = line.split()
                vertex_props.append((parts[1], parts[2]))  # (type, name)
        if n_vertex is None:
            raise ValueError(f"No 'element vertex' in {path}")

        type_map = {
            "float": "<f4", "float32": "<f4", "double": "<f8",
            "uchar": "u1", "uint8": "u1", "int": "<i4", "int32": "<i4",
        }
        np_fields = [(name, type_map[t]) for t, name in vertex_props]
        dtype = np.dtype(np_fields)
        data = np.fromfile(f, dtype=dtype, count=n_vertex)
        verts = np.stack([data["x"], data["y"], data["z"]], axis=1).astype(np.float64)
        return verts


_fake_o3d_io = types.SimpleNamespace(
    read_triangle_mesh=lambda path_str: _FakePlyMesh(
        _read_binary_ply_vertices_only(path_str),
        np.zeros((1, 3), dtype=np.int64),  # dummy non-empty faces array
    )
)
_fake_o3d = types.SimpleNamespace(io=_fake_o3d_io)
sys.modules["open3d"] = _fake_o3d

from src.datasets.scannetpp import ScanNetPPDataSource  # noqa: E402
from src.scene_parser import parse_scene  # noqa: E402
from src import frame_selector as fs  # noqa: E402

SCENE_DIR = REPO / "scannetpp" / "c50d2d1d42"
LOCAL_FRAME_DIR = REPO / "scannetpp" / "c50d2d1d42_frame"  # flat, stride-10 partial extraction

FRAME_STRIDE = fs.FRAME_STRIDE
MIN_VISIBLE_OBJECTS = fs.MIN_VISIBLE_OBJECTS

print(f"FRAME_STRIDE={FRAME_STRIDE} MIN_VISIBLE_OBJECTS={MIN_VISIBLE_OBJECTS}")

# ---------------------------------------------------------------------------
# 1. data source + poses
# ---------------------------------------------------------------------------
data_source = ScanNetPPDataSource(SCENE_DIR, sensor="iphone", frame_root=LOCAL_FRAME_DIR)

poses = data_source.load_poses()
print(f"\n=== STAGE 0: load_poses() ===")
print(f"total poses returned: {len(poses)}")
keys = list(poses.keys())
print("first 5 keys:", keys[:5])
print("last 5 keys:", keys[-5:])
nums = [int(k.split("_")[1].split(".")[0]) for k in keys]
is_sorted = all(nums[i] <= nums[i+1] for i in range(len(nums)-1))
print(f"keys numerically sorted ascending? {is_sorted}  min={min(nums)} max={max(nums)}")

intrinsics = data_source.load_intrinsics()
print(f"intrinsics: {intrinsics.width}x{intrinsics.height}")

# ---------------------------------------------------------------------------
# 2. stride sampling (mirrors select_frames() lines ~1170-1190)
# ---------------------------------------------------------------------------
print(f"\n=== STAGE 1: stride sampling (every {FRAME_STRIDE}th pose in dict order) ===")
sampled = []
for i, (image_name, pose) in enumerate(poses.items()):
    if i % FRAME_STRIDE != 0:
        continue
    sampled.append((image_name, pose))
print(f"stride-sampled count: {len(sampled)}")

# ---------------------------------------------------------------------------
# 3. image existence check
# ---------------------------------------------------------------------------
print(f"\n=== STAGE 2: image existence ===")

def resolve_via_datasource(name):
    return data_source.image_path(name)

def resolve_local_flat(name):
    return LOCAL_FRAME_DIR / name

example_name = sampled[0][0]
ds_path = resolve_via_datasource(example_name)
flat_path = resolve_local_flat(example_name)
print(f"data_source.image_path({example_name!r}) -> {ds_path}  exists={ds_path.exists()}")
print(f"flat local resolver     -> {flat_path}  exists={flat_path.exists()}")

exists_flat = 0
missing_flat = 0
missing_names = []
for name, pose in sampled:
    p = resolve_local_flat(name)
    if p.exists():
        exists_flat += 1
    else:
        missing_flat += 1
        missing_names.append(name)

print(f"stride-sampled with LOCAL image present (flat resolver): {exists_flat}")
print(f"stride-sampled with LOCAL image missing: {missing_flat}")
print("first 10 missing:", missing_names[:10])

exists_ds = sum(1 for name, _ in sampled if resolve_via_datasource(name).exists())
print(f"stride-sampled with image present via data_source.image_path(): {exists_ds}")

# ---------------------------------------------------------------------------
# 4. quality filter
# ---------------------------------------------------------------------------
print(f"\n=== STAGE 3: quality filter (_filter_sampled_frames_by_quality) ===")
sampled_frames_for_quality = []
idx = 0
for name, pose in sampled:
    p = resolve_local_flat(name)
    if not p.exists():
        continue
    sampled_frames_for_quality.append({
        "image_name": name,
        "pose": pose,
        "image_path": p,
        "sampled_index": idx,
    })
    idx += 1

quality_kept, quality_stats = fs._filter_sampled_frames_by_quality(sampled_frames_for_quality)
print(f"input to quality filter: {len(sampled_frames_for_quality)}")
print(f"quality_stats: {quality_stats}")
print(f"quality-kept count: {len(quality_kept)}")

# ---------------------------------------------------------------------------
# 5. get_visible_objects on quality-surviving frames
# ---------------------------------------------------------------------------
print(f"\n=== STAGE 4: get_visible_objects (bbox-projection based, MIN_VISIBLE_OBJECTS={MIN_VISIBLE_OBJECTS}) ===")
dist = Counter()
try:
    scene = parse_scene(SCENE_DIR, dataset="scannetpp")
except Exception as exc:
    print(f"parse_scene FAILED: {type(exc).__name__}: {exc}")
    scene = None

if scene is None:
    print("Cannot proceed to visibility stage: scene parsing failed.")
else:
    objects = scene["objects"]
    print(f"scene object count: {len(objects)}")

    n_visible_counts = []
    for entry in quality_kept:
        visible = fs.get_visible_objects(objects, entry["pose"], intrinsics)
        n_visible_counts.append(len(visible))

    for n in n_visible_counts:
        if n == 0:
            dist["0"] += 1
        elif n == 1:
            dist["1"] += 1
        elif n == 2:
            dist["2"] += 1
        else:
            dist[">=3"] += 1

    print(f"n_visible distribution over {len(n_visible_counts)} quality-kept frames: {dict(dist)}")
    print(f"frames passing MIN_VISIBLE_OBJECTS>=3: {dist.get('>=3', 0)}")
    if n_visible_counts:
        print(f"min/mean/max n_visible: {min(n_visible_counts)}/{sum(n_visible_counts)/len(n_visible_counts):.2f}/{max(n_visible_counts)}")

    print("\nSample per-frame n_visible (first 30 quality-kept frames):")
    for entry, n in list(zip(quality_kept, n_visible_counts))[:30]:
        print(f"  {entry['image_name']}: n_visible={n}")

print("\n=== SUMMARY FUNNEL ===")
print(f"total poses (COLMAP registered):      {len(poses)}")
print(f"after stride={FRAME_STRIDE} sampling:         {len(sampled)}")
print(f"after LOCAL image-exists check:        {exists_flat}  (missing={missing_flat})")
print(f"after quality filter:                  {quality_stats.get('final_quality_kept_count')}")
if scene is not None:
    print(f"after MIN_VISIBLE_OBJECTS>=3 filter:    {dist.get('>=3', 0)}")
