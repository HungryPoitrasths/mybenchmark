"""Debug test: run generate_l2_object_move with scene0011_00 frame 642 data."""
import sys, json, os, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.qa_generator import generate_l2_object_move
from src.utils.colmap_loader import CameraPose

def test_scene0011_00_agent_direction():
    """Load real scene data and verify agent direction questions are generated."""
    # Load scene metadata
    meta_path = os.path.join(
        os.path.dirname(__file__),
        '..', 'output', 'pilot', '0-9', 'scene_metadata', 'scene0011_00.json'
    )
    with open(meta_path, encoding='utf-8') as f:
        meta = json.load(f)

    all_objects = meta['objects']
    obj_map = {int(o['id']): o for o in all_objects}

    # Camera pose from 642.txt
    c2w = np.array([
        [0.533401, 0.357222, -0.766731, 5.037569],
        [0.845843, -0.231452, 0.480604, 1.042065],
        [-0.005779, -0.904888, -0.425610, 1.421906],
        [0, 0, 0, 1]
    ])
    camera_pose = CameraPose(
        image_name='642.jpg',
        rotation=c2w[:3, :3].copy(),
        translation=c2w[:3, 3].copy(),
    )

    # --- Simulate pipeline filtering for frame 642.jpg ---
    # From frame_debug data:
    referable_ids = {26}          # only sink
    attachment_referable_ids = {25, 26}  # counter + sink (attachment context)
    visible_ids = {1, 8, 9, 25, 26}

    # objects_uniq = objects in referable_set
    objects_uniq = [o for o in all_objects if int(o['id']) in referable_ids]

    # movement_objects = objects in graph_eligible_ids (referable + attachment context)
    graph_eligible_ids = referable_ids | {25}  # 25 = counter, parent in chain
    movement_objects = [o for o in all_objects if int(o['id']) in graph_eligible_ids]

    # collision_objects = visible non-excluded
    collision_objects = [o for o in all_objects if int(o['id']) in visible_ids]

    # attachment_query_objects = objects in attachment_referable_set
    attachment_query_objects = [o for o in all_objects if int(o['id']) in attachment_referable_ids]

    # Attachment graph
    attachment_graph = {25: [26]}
    attached_by = {26: 25}

    print(f"objects_uniq: {[o['id'] for o in objects_uniq]}")
    print(f"movement_objects: {[o['id'] for o in movement_objects]}")
    print(f"collision_objects: {[o['id'] for o in collision_objects]}")
    print(f"attachment_query_objects: {[o['id'] for o in attachment_query_objects]}")
    print(f"attachment_graph: {attachment_graph}")

    # Call generate_l2_object_move WITHOUT occlusion (color_intrinsics=None)
    # This should generate agent and distance questions if direction/distance changes
    questions = generate_l2_object_move(
        objects=objects_uniq,
        attachment_graph=attachment_graph,
        attached_by=attached_by,
        camera_pose=camera_pose,
        templates={},
        room_bounds=None,
        collision_objects=collision_objects,
        movement_objects=movement_objects,
        object_map=obj_map,
        color_intrinsics=None,
        occlusion_backend='depth',
        ray_caster=None,
        instance_mesh_data=None,
        attachment_referable_object_ids=sorted(attachment_referable_ids),
        attachment_query_objects=attachment_query_objects,
    )

    print(f"\nGenerated {len(questions)} questions:")
    from collections import Counter
    types = Counter(q['type'] for q in questions)
    for t, c in sorted(types.items()):
        print(f"  {t}: {c}")
        for q in questions:
            if q['type'] == t:
                moved = q.get('moved_obj_label', '?')
                query = q.get('query_obj_label', q.get('obj_a_label', '?'))
                if t == 'object_move_agent':
                    print(f"    moved={moved}, query={query}, correct={q.get('correct_value')}, old={q.get('old_correct_value')}")
                elif t == 'object_move_distance':
                    print(f"    moved={moved}, query={query}, old_dist={q.get('old_distance_m')}, new={q.get('new_distance_m')}")
                elif t == 'object_move_occlusion':
                    print(f"    moved={moved}, target={q.get('obj_b_label', '?')}")

    # Assertions
    agent_count = types.get('object_move_agent', 0)
    dist_count = types.get('object_move_distance', 0)

    if agent_count == 0:
        print("\n!!! BUG: No agent questions generated despite direction changes!")
        print("Expected: at least 4 agent questions (for cabinets, chairs vs counter)")

    if dist_count == 0:
        print("\n!!! BUG: No distance questions generated despite distance changes!")
        print("Expected: at least some distance questions")

    return questions


if __name__ == '__main__':
    test_scene0011_00_agent_direction()
