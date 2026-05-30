"""Analyze all allocentric/rotate triplets to check if new code can generate agent/distance questions."""
import json, os, glob, numpy as np
from collections import defaultdict

# ===================== Configuration =====================
SCANS_DIR = r'D:\Scannet\data\scans'
OUTPUT_DIR = r'output\pilot'

# ===================== Load scene metadata =====================
meta_map = {}
for f in glob.glob(f'{OUTPUT_DIR}/*/scene_metadata/*.json'):
    sid = os.path.basename(f).replace('.json', '')
    meta_map[sid] = f

# ===================== Helper: ground-projected direction (NEW CODE) =====================
def compute_direction_new(camera_pose, obj1_center, obj2_center):
    """NEW CODE: project camera forward+right to ground (Z=0), normalize, then compute direction."""
    R = camera_pose[:3, :3]
    cam_forward_3d = R[:3, 2]  # 3rd column = forward in world
    cam_right_3d = R[:3, 0]    # 1st column = right in world

    # Project camera axes to ground (Z=0 in ScanNet), normalize
    forward_proj = np.array([cam_forward_3d[0], cam_forward_3d[1], 0.0])
    right_proj = np.array([cam_right_3d[0], cam_right_3d[1], 0.0])
    f_norm = np.linalg.norm(forward_proj)
    r_norm = np.linalg.norm(right_proj)
    if f_norm < 1e-10 or r_norm < 1e-10:
        return None, None  # degenerate (camera looking straight down/up)
    forward_proj /= f_norm
    right_proj /= r_norm

    # Vector from obj1 to obj2 in world, horizontal only (Z is up in ScanNet)
    wx = obj2_center[0] - obj1_center[0]
    wy = obj2_center[1] - obj1_center[1]

    # Components along projected forward and right
    cam_z = wx * forward_proj[0] + wy * forward_proj[1]  # forward component
    cam_x = wx * right_proj[0] + wy * right_proj[1]       # right component

    angle_deg = np.degrees(np.arctan2(cam_x, cam_z))

    # Direction bins (8-way)
    if -22.5 <= angle_deg < 22.5:
        d = 'front'
    elif 22.5 <= angle_deg < 67.5:
        d = 'front-right'
    elif 67.5 <= angle_deg < 112.5:
        d = 'right'
    elif 112.5 <= angle_deg < 157.5:
        d = 'back-right'
    elif angle_deg >= 157.5 or angle_deg < -157.5:
        d = 'back'
    elif -157.5 <= angle_deg < -112.5:
        d = 'back-left'
    elif -112.5 <= angle_deg < -67.5:
        d = 'left'
    elif -67.5 <= angle_deg < -22.5:
        d = 'front-left'
    else:
        d = 'front'

    return d, angle_deg


def compute_distance(obj1_center, obj2_center):
    """Horizontal distance between two object centers (ground-projected)."""
    dx = obj2_center[0] - obj1_center[0]
    dy = obj2_center[1] - obj1_center[1]
    return float(np.sqrt(dx*dx + dy*dy))


def get_attachment_chain(attachment_graph, start_id):
    """Get all objects that move when start_id moves (including transitive)."""
    moved = {start_id}
    if start_id in attachment_graph:
        for child in attachment_graph[start_id]:
            moved.add(child)
            if child in attachment_graph:
                for grandchild in attachment_graph[child]:
                    moved.add(grandchild)
    return moved


def move_object_center(center, delta):
    return [center[0] + delta[0], center[1] + delta[1], center[2] + delta[2]]


# ===================== Collect all triplets =====================
triplets = []
for bench_path in sorted(glob.glob(f'{OUTPUT_DIR}/*/benchmark.json')):
    with open(bench_path, encoding='utf-8') as f:
        data = json.load(f)
    for q in data.get('questions', []):
        if q.get('type') in ('object_move_allocentric', 'object_rotate_object_centric') and q.get('attachment_remapped'):
            triplets.append({
                'scene_id': q['scene_id'],
                'frame_id': q['image_name'].replace('.jpg', ''),
                'frame': q['image_name'],
                'moved_id': q['moved_obj_id'],
                'moved_label': q['moved_obj_label'],
                'query_id': q['query_obj_id'],
                'query_label': q['query_obj_label'],
                'ref_id': q['obj_ref_id'],
                'ref_label': q['obj_ref_label'],
                'delta': q['delta'],
                'type': q['type'],
                'trace_reason': q.get('trace_reason', ''),
                'relation_unchanged': q.get('relation_unchanged'),
            })

print(f'Total triplets to analyze: {len(triplets)}')

# ===================== Analyze each triplet =====================
results = []
errors = []
scene_cache = {}

for i, t in enumerate(triplets):
    sid = t['scene_id']
    fid = t['frame_id']

    # Load scene metadata (cached)
    if sid not in scene_cache:
        meta_path = meta_map.get(sid)
        if meta_path is None:
            errors.append(f'{sid}: no metadata')
            continue
        with open(meta_path, encoding='utf-8') as f:
            meta = json.load(f)
        obj_map = {int(o['id']): o for o in meta['objects']}
        attach_graph = {int(k): [int(vv) for vv in v] for k, v in meta.get('attachment_graph', {}).items()}
        scene_cache[sid] = (obj_map, attach_graph)

    obj_map, attach_graph = scene_cache[sid]

    # Load camera pose
    pose_path = os.path.join(SCANS_DIR, sid, 'pose', f'{fid}.txt')
    if not os.path.exists(pose_path):
        errors.append(f'{sid}/{fid}: pose not found')
        continue

    try:
        c2w = np.loadtxt(pose_path)
        if c2w.shape != (4, 4):
            errors.append(f'{sid}/{fid}: bad pose shape {c2w.shape}')
            continue
    except Exception as e:
        errors.append(f'{sid}/{fid}: {e}')
        continue

    # Get moved IDs (attachment chain)
    moved_ids = get_attachment_chain(attach_graph, t['moved_id'])

    # Check objects exist in metadata
    try:
        query_obj = obj_map[t['query_id']]
        ref_obj = obj_map[t['ref_id']]
    except KeyError as e:
        errors.append(f'{sid}/{fid}: object {e} not found')
        continue

    delta = t['delta']

    # === OLD (before movement) ===
    old_dir, old_angle = compute_direction_new(c2w, query_obj['center'], ref_obj['center'])
    old_dist = compute_distance(query_obj['center'], ref_obj['center'])

    # === NEW (after movement) ===
    new_query_center = move_object_center(query_obj['center'], delta) if t['query_id'] in moved_ids else query_obj['center'][:]
    new_ref_center = move_object_center(ref_obj['center'], delta) if t['ref_id'] in moved_ids else ref_obj['center'][:]
    new_dir, new_angle = compute_direction_new(c2w, new_query_center, new_ref_center)

    if old_dir is None or new_dir is None:
        errors.append(f'{sid}/{fid}: degenerate camera pose')
        continue

    dir_changed = old_dir != new_dir
    new_dist = compute_distance(new_query_center, new_ref_center)
    dist_diff = abs(new_dist - old_dist)

    # Cross bin check
    dist_bins = [0.5, 1.0, 2.0, 3.0, 5.0]
    cross_bin = any(
        (old_dist < th <= new_dist) or (new_dist < th <= old_dist)
        for th in dist_bins
    )

    # NEW CODE thresholds:
    # Agent: direction changed
    # Distance: any meaningful change (no 0.2m threshold)
    agent_possible = dir_changed
    dist_possible = dist_diff > 0.01 or cross_bin  # > 1cm or cross bin

    results.append({
        **t,
        'old_dir': old_dir, 'old_angle': round(old_angle, 1),
        'new_dir': new_dir, 'new_angle': round(new_angle, 1),
        'old_dist': round(old_dist, 3), 'new_dist': round(new_dist, 3),
        'dist_diff': round(dist_diff, 3),
        'dir_changed': dir_changed,
        'cross_bin': cross_bin,
        'agent_possible': agent_possible,
        'dist_possible': dist_possible,
        'moved_ids': sorted(moved_ids),
        'query_moved': t['query_id'] in moved_ids,
        'ref_moved': t['ref_id'] in moved_ids,
    })

    if (i + 1) % 50 == 0:
        print(f'  Processed {i+1}/{len(triplets)}...')

# ===================== Summary =====================
print(f'\n{"="*60}')
print(f'RESULTS SUMMARY')
print(f'{"="*60}')
print(f'Successful: {len(results)}')
print(f'Errors: {len(errors)}')
if errors:
    for e in errors[:20]:
        print(f'  ERROR: {e}')

agent_yes = sum(1 for r in results if r['agent_possible'])
agent_no = sum(1 for r in results if not r['agent_possible'])
dist_yes = sum(1 for r in results if r['dist_possible'])
dist_no = sum(1 for r in results if not r['dist_possible'])

print(f'\n--- AGENT questions ---')
print(f'  CAN generate:  {agent_yes}/{len(results)} ({100*agent_yes/len(results):.1f}%)')
print(f'  CANNOT generate: {agent_no}/{len(results)} ({100*agent_no/len(results):.1f}%)')

print(f'\n--- DISTANCE questions ---')
print(f'  CAN generate:  {dist_yes}/{len(results)} ({100*dist_yes/len(results):.1f}%)')
print(f'  CANNOT generate: {dist_no}/{len(results)} ({100*dist_no/len(results):.1f}%)')

# ===== Find triplets that CANNOT generate EITHER =====
cannot_either = [r for r in results if not r['agent_possible'] and not r['dist_possible']]
print(f'\n{"="*60}')
print(f'CANNOT GENERATE EITHER agent or distance: {len(cannot_either)} triplets')
print(f'{"="*60}')

# Group by reason
for r in cannot_either:
    reasons = []
    if not r['agent_possible']:
        reasons.append(f'dir same ({r["old_dir"]})')
    if not r['dist_possible']:
        reasons.append(f'dist unchanged ({r["old_dist"]}->{r["new_dist"]}, diff={r["dist_diff"]})')
    print(f'  {r["scene_id"]} f{r["frame_id"]}: {r["moved_label"]}#{r["moved_id"]}->{r["query_label"]}#{r["query_id"]}->{r["ref_label"]}#{r["ref_id"]}')
    print(f'    delta={r["delta"]} | trace={r["trace_reason"]}')
    print(f'    reasons: {"; ".join(reasons)}')
    print(f'    query_moved={r["query_moved"]} ref_moved={r["ref_moved"]}')

# ===== CANNOT generate agent (but CAN generate distance) =====
no_agent = [r for r in results if not r['agent_possible'] and r['dist_possible']]
print(f'\n{"="*60}')
print(f'CANNOT generate AGENT (but CAN generate distance): {len(no_agent)} triplets')
print(f'{"="*60}')
for r in no_agent[:20]:
    print(f'  {r["scene_id"]} f{r["frame_id"]}: {r["moved_label"]}#{r["moved_id"]}->{r["query_label"]}#{r["query_id"]}->{r["ref_label"]}#{r["ref_id"]}')
    print(f'    dir: {r["old_dir"]}->{r["new_dir"]} (unchanged) | dist: {r["old_dist"]}->{r["new_dist"]} (diff={r["dist_diff"]})')
    print(f'    delta={r["delta"]} query_moved={r["query_moved"]} ref_moved={r["ref_moved"]}')
if len(no_agent) > 20:
    print(f'  ... and {len(no_agent)-20} more')

# ===== CANNOT generate distance (but CAN generate agent) =====
no_dist = [r for r in results if r['agent_possible'] and not r['dist_possible']]
print(f'\n{"="*60}')
print(f'CANNOT generate DISTANCE (but CAN generate agent): {len(no_dist)} triplets')
print(f'{"="*60}')
for r in no_dist:
    print(f'  {r["scene_id"]} f{r["frame_id"]}: {r["moved_label"]}#{r["moved_id"]}->{r["query_label"]}#{r["query_id"]}->{r["ref_label"]}#{r["ref_id"]}')
    print(f'    dir: {r["old_dir"]}->{r["new_dir"]} | dist: {r["old_dist"]}->{r["new_dist"]} (diff={r["dist_diff"]})')
    print(f'    delta={r["delta"]} query_moved={r["query_moved"]} ref_moved={r["ref_moved"]}')

# Save full results
out_path = os.path.join(os.path.dirname(__file__), 'triplet_analysis.json')
with open(out_path, 'w', encoding='utf-8') as f:
    json.dump({
        'summary': {
            'total': len(results),
            'errors': len(errors),
            'agent_can': agent_yes,
            'agent_cannot': agent_no,
            'dist_can': dist_yes,
            'dist_cannot': dist_no,
            'cannot_either': len(cannot_either),
            'no_agent_but_dist': len(no_agent),
            'no_dist_but_agent': len(no_dist),
        },
        'cannot_either': cannot_either,
        'no_agent': no_agent,
        'no_dist': no_dist,
        'errors': errors,
    }, f, ensure_ascii=False, indent=2)
print(f'\nFull results saved to: {out_path}')
