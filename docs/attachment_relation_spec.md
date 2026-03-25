# Attachment Relation Specification

## Status

Draft v1.

This document defines a migration path from the current `support_graph` to a more general `attachment_graph` for intervention propagation in CausalSpatial-Bench.


## 1. Motivation

The current `support_graph` is used for two different purposes:

1. geometric support detection
2. intervention propagation for `object_move`, `support_chain`, and related virtual operations

These are not the same concept.

Examples such as `pillow -> bed`, `cushion -> sofa`, `picture -> wall`, and `object -> drawer` are important for movement propagation, but they are not always well modeled by a strict "child bottom touches parent top face" rule.

The main failure mode is that some parent instances are geometrically composite. For example, a `bed` instance may include the mattress, headboard, and side panels. In the current implementation:

- parent contact height is tied to `parent.bbox_max[2]`
- `parent.top_hull_xy` is extracted from a thin slice near `bbox_max[2]`

This can make `pillow-on-bed` fail even when the scene semantics are correct, because the extracted "top face" may correspond to the headboard top rather than the mattress surface.


## 2. Primary Relation

The benchmark should use the following primary relation:

`attachment(parent, child): moving parent should also move child in this benchmark`

Interpretation:

- the relation is directional
- the direction matches intervention propagation
- the relation is about joint motion, not about physical support in the narrow sense

Graph orientation:

```python
attachment_graph[parent_id] = [child_id_1, child_id_2, ...]
attached_by[child_id] = parent_id
```

This direction is intentionally the same as the current `support_graph`, so most downstream movement logic can be reused with minimal change.


## 3. Scope of Propagation

### 3.1 Move propagation

For v1:

- if `attachment(parent, child)` is true, moving `parent` should also move `child`

### 3.2 Remove propagation

For v1:

- removal does **not** propagate through the attachment graph
- `object_remove` should not use attachment edges for cascade deletion

Rationale:

- removal semantics are more ambiguous than movement semantics
- keeping removal non-propagating avoids edge-specific exception rules
- this keeps the graph definition simple and stable

Operationally:

- `move_with_parent = True` for every attachment edge in v1
- `remove_with_parent = False` for every attachment edge in v1

The fields may still be stored explicitly for future flexibility, but they should not vary in v1.


## 4. Edge Schema

Each edge should carry structured metadata rather than only a boolean parent-child link.

Recommended representation:

```python
{
    "parent_id": 12,
    "child_id": 34,
    "type": "resting_on_soft_surface",
    "confidence": 0.86,
    "evidence": {
        "geometry_contact": {
            "z_gap": 0.021,
            "contact_z_parent": 0.73,
            "contact_z_child": 0.71
        },
        "xy_overlap": {
            "child_coverage": 0.68,
            "overlap_area": 0.11
        },
        "containment": None,
        "semantic_prior": {
            "parent_label": "bed",
            "child_label": "pillow"
        }
    },
    "move_with_parent": True,
    "remove_with_parent": False
}
```

Field meanings:

- `type`: subtype of attachment relation
- `confidence`: normalized confidence score in `[0, 1]`
- `evidence`: structured reasons used to create the edge
- `move_with_parent`: whether movement propagates across this edge
- `remove_with_parent`: whether removal propagates across this edge


## 5. Attachment Subtypes

v1 should support the following four subtypes.

### 5.1 `supported_by`

Meaning:

- the child rests on a relatively rigid supporting surface of the parent

Examples:

- `book -> table`
- `cup -> desk`
- `lamp -> cabinet`

Typical evidence:

- small vertical contact gap
- substantial XY overlap
- parent has a plausible contact surface near the child


### 5.2 `resting_on_soft_surface`

Meaning:

- the child rests on a soft or deformable parent surface
- exact rigid top-face support geometry is not required

Examples:

- `pillow -> bed`
- `blanket -> bed`
- `cushion -> sofa`

Typical evidence:

- child bottom is close to a plausible upper contact region of the parent
- strong XY overlap with the parent footprint or contact plateau
- compatible label prior such as `pillow/bed`, `cushion/sofa`

Important note:

`pillow-on-bed` should prefer this subtype rather than being forced through a strict `supported_by` rule tied to `parent.bbox_max[2]`.


### 5.3 `contained_in`

Meaning:

- the child is located inside the parent container volume or usable cavity

Examples:

- `apple -> bowl`
- `clothing -> drawer`
- `toy -> box`

Typical evidence:

- child center or child bbox is substantially inside parent XY extent
- child bottom is at or above the parent inner base
- semantic labels are compatible with a container relation

Note:

This relation is about movement dependence, not necessarily full geometric enclosure.


### 5.4 `affixed_to`

Meaning:

- the child is attached, mounted, hanging from, or structurally connected to the parent

Examples:

- `picture -> wall`
- `monitor -> monitor stand`
- `handle -> cabinet`

Typical evidence:

- geometry indicates sustained contact or near-zero separation
- XY overlap may be weak or irrelevant depending on orientation
- semantic prior is often important


## 6. Evidence Sources

Each edge should be justified from one or more evidence layers.

### 6.1 `geometry_contact`

Purpose:

- determine whether parent and child are close enough in 3D to plausibly move together

Typical signals:

- vertical gap or directional gap to a candidate contact surface
- face-to-face or region-to-region proximity
- whether the child lies above, within, or against the parent

Important change from current support logic:

- do not bind parent contact height to `parent.bbox_max[2]` only
- instead use one or more candidate contact surfaces or plateaus


### 6.2 `xy_overlap`

Purpose:

- measure how much the child footprint aligns with the relevant parent contact region

Typical signals:

- child footprint overlap with parent footprint
- child footprint overlap with candidate parent contact plateau
- overlap normalized by child area

Important note:

- use footprint overlap as evidence, not as the sole definition of the relation
- some edge types such as `affixed_to` may rely less on XY overlap


### 6.3 `containment`

Purpose:

- detect object-in-container relations that are not well represented by top-surface support

Typical signals:

- child bbox mostly within parent XY extent
- child center within parent extent
- child bottom above the parent base and below the parent top opening


### 6.4 `semantic_prior`

Purpose:

- inject weak class-level priors when geometry alone is insufficient or ambiguous

Typical compatible pairs:

- `pillow -> bed`
- `cushion -> sofa`
- `monitor -> monitor stand`
- `clothing -> drawer`

Requirements:

- priors should boost or disambiguate, not create edges from nothing
- geometry should still supply at least one positive signal


## 7. Geometry Fix Before Attachment Migration

This is the first implementation stage and should happen before replacing `support_graph`.

### 7.1 Problem in the current code

Current behavior in `scene_parser.py` and `support_graph.py`:

- parent top contact height is approximated by `bbox_max[2]`
- `top_hull_xy` is extracted from a thin slice near that height
- if that thin slice is sparse, the code falls back to the full XY bbox rectangle

This is fragile for composite instances such as beds, sofas, cabinets, or tables with backs, rails, or raised parts.

### 7.2 Required fix

Replace the single "highest top slice" assumption with a contact-surface candidate model.

Minimum requirement:

- each object should expose one or more plausible upper contact plateaus
- support detection should compare the child against the best matching parent plateau, not only the highest point of the parent

### 7.3 Suggested data representation

Instead of only:

```python
support_geom = {
    "bottom_hull_xy": ...,
    "top_hull_xy": ...
}
```

move toward:

```python
support_geom = {
    "bottom_hull_xy": ...,
    "top_surface_candidates": [
        {
            "z": 0.73,
            "hull_xy": [...],
            "area": 0.91,
            "score": 0.88
        },
        {
            "z": 0.98,
            "hull_xy": [...],
            "area": 0.07,
            "score": 0.22
        }
    ]
}
```

Selection rule:

- for a given child, choose the candidate plateau that maximizes attachment evidence
- do not assume the highest plateau is the correct one

### 7.4 Candidate extraction guidance

Within the current vertex-only parser, candidate plateaus can be approximated by:

- slicing the object into several horizontal bands in Z
- clustering dense upper-surface bands rather than using only `bbox_max[2]`
- computing an XY hull for each band
- scoring candidates by area, density, and stability

This avoids requiring a full face-normal pipeline in the first fix.


## 8. Attachment Decision Rules

The graph builder should evaluate candidate relations in the following order.

### 8.1 Step A: coarse candidate recall

Generate candidate parent-child pairs using cheap tests:

- non-identical object ids
- distance or bbox proximity
- broad XY overlap or containment plausibility
- optional label-based gating to avoid impossible pairs

### 8.2 Step B: subtype-specific scoring

For each candidate pair, compute evidence and assign a subtype.

Suggested priority:

1. `contained_in`
2. `affixed_to`
3. `resting_on_soft_surface`
4. `supported_by`

Reason:

- container and mounted cases are poorly captured by top-face support tests
- soft-surface cases should be identified before strict rigid support rules reject them

### 8.3 Step C: choose the best parent for each child

For v1, each child should have at most one parent in `attached_by`.

Recommended tie-break order:

1. higher attachment confidence
2. stronger subtype-specific evidence
3. smaller contact gap
4. larger normalized overlap or containment score


## 9. Confidence Scoring

Confidence does not need to be a learned model in v1.

A weighted heuristic score is sufficient:

```text
confidence =
    w_geom * geometry_contact_score +
    w_overlap * xy_overlap_score +
    w_contain * containment_score +
    w_prior * semantic_prior_score
```

Guidance:

- `supported_by`: emphasize geometry and overlap
- `resting_on_soft_surface`: emphasize overlap and semantic prior more than exact height
- `contained_in`: emphasize containment
- `affixed_to`: emphasize geometry and semantic prior


## 10. Propagation Rules in Downstream Tasks

### 10.1 `object_move`

Should use `attachment_graph`.

Behavior:

- moving a parent also moves all descendants in the attachment graph

### 10.2 `support_chain`

Should migrate conceptually to `attachment_chain`.

Options:

- keep the old question type name temporarily for compatibility
- internally answer it from `attachment_graph`

### 10.3 `object_remove`

Should not use `attachment_graph` for cascade removal in v1.

Behavior:

- remove only the queried object
- do not recursively remove attached children


## 11. Backward-Compatible Migration Plan

Migration should happen in two stages.

### Stage 1: fix geometry while keeping the current public graph

Goal:

- improve `support_graph` recall, especially for `pillow-on-bed` and similar cases

Actions:

1. change parent contact modeling from a single `bbox_max[2]` top slice to candidate plateaus
2. update support detection to compare the child against the best parent plateau
3. keep the current `support_graph` and `supported_by` API unchanged

Expected outcome:

- better recall without forcing downstream code changes yet

### Stage 2: introduce `attachment_graph`

Goal:

- separate movement dependency from narrow support geometry

Actions:

1. add a new builder, for example `build_attachment_graph(...)`
2. output:
   - `attachment_graph`
   - `attached_by`
   - optional `attachment_edges` metadata list
3. migrate `virtual_ops` and QA generation to use `attachment_graph`
4. keep `support_graph` as a derived compatibility view if needed

Compatibility strategy:

- keep writing `support_graph` for one transition period
- define it as a filtered subset of attachment edges, typically:
  - include `supported_by`
  - optionally include `resting_on_soft_surface` if legacy tasks expect it
- downstream code should gradually switch to `attachment_graph`


## 12. Recommended Compatibility Mapping

During migration:

```python
support_like_types = {
    "supported_by",
    "resting_on_soft_surface",
}
```

Derived compatibility behavior:

- `support_graph` is generated from attachment edges whose `type` is in `support_like_types`
- `attachment_graph` remains the authoritative graph for movement propagation

This lets old evaluation scripts continue to run while new logic is adopted.


## 13. Non-Goals for v1

The following should not be attempted in the first migration:

- multi-parent attachment graphs
- learned attachment classifiers
- removal propagation with edge-specific exceptions
- full physical simulation of falling, sliding, or collision after removal


## 14. Summary of v1 Decisions

1. The benchmark-level relation is `attachment(parent, child)`.
2. Graph direction remains `parent -> child`.
3. The graph is used for movement propagation.
4. Removal does not propagate through the graph in v1.
5. Edge metadata should include `type`, `confidence`, and `evidence`.
6. The initial subtypes are:
   - `supported_by`
   - `resting_on_soft_surface`
   - `contained_in`
   - `affixed_to`
7. The first implementation step is to replace the `bbox_max[2]` top-slice assumption with candidate contact plateaus.

