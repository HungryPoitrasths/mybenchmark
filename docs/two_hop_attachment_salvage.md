# Two-hop Attachment Salvage

This tool creates a human-authored two-hop attachment override cache from
selected ScanNet++ iPhone frames. The review page does not require an existing
VLM/flash referability cache; `run_pipeline` merges the exported JSON with the
normal flash cache at runtime.

## 1. Build the review page

```powershell
python scripts/two_hop_attachment_salvage.py `
  --scene_root C:\path\to\scannetpp `
  --frame_root output\scannetpp_iphone_frames `
  --image "0d2ee665be:001510\002740\001110" `
  --output_html output\two_hop_attachment_salvage.html `
  --output_json two_hop_attachment_salvage.json
```

The scene root must contain the ScanNet++ geometry and iPhone COLMAP files.
Numeric frame IDs are expanded to names such as `frame_001510.jpg`.

Open the generated HTML. The projected boxes show the scene label and object ID
for reference. Manually type an object ID and label for each of four roles:

- `moved`: the root object that is moved;
- `child`: the direct attachment child;
- `grandchild`: the second-hop attachment child;
- `contrast`: the non-chain distractor.

Each ID must belong to an object whose bbox is displayed on that frame. The
typed labels are used as the attachment question wording, so they may be more
specific than the scene's original labels. The three option labels (`child`,
`grandchild`, and `contrast`) must be distinct.

Every card also has two controls:

- **Add** duplicates the same image and projected bboxes immediately after the
  current card, with all four ID/label inputs empty. Use it to annotate another
  two-hop chain in the same frame.
- **Delete** removes that annotation card. If every card for a frame is deleted,
  the frame is omitted from the exported JSON.

Cards for the same frame are numbered automatically. Duplicate four-ID role
sets are rejected, and one object ID must use the same typed label across all
annotations for that frame.

Click **Export JSON**. The downloaded file has referability cache version
`20.0`, includes the selected frames and typed role labels, and supplies a
manual attachment graph.
`--output_json` controls the suggested download filename; the browser chooses
the final download directory.

When a frame has multiple cards, the entry contains all annotations in
`manual_attachment_role_sets`; `manual_attachment_roles` mirrors the first set
for compatibility with older pipeline versions. The current pipeline generates
one forced L3 attachment-chain question for every valid role set, without
discarding human annotations because of automatic question-count caps.

While the Python tool builds the page it prints the total scene/frame count,
scene loading progress, per-frame processing progress, projected bbox counts,
and the final HTML output path.

## 2. Generate questions

```powershell
python scripts/run_pipeline.py `
  --dataset scannetpp `
  --data_root C:\path\to\scannetpp `
  --scannetpp_frame_root output\scannetpp_iphone_frames `
  --referability_cache C:\path\to\flash\scene_status.json `
  --manual_attachment_cache C:\path\to\two_hop_attachment_salvage.json `
  --only_question_types L3_attachment_chain `
  --output_dir output\two_hop_attachment_questions
```

The original flash cache is not modified. For scenes present in the manual
cache, `run_pipeline` replaces the automatic attachment/support graph with the
human chain. Manual scenes and frames are allowed even when absent from the
flash cache, provided the scene exists under `--data_root` and belongs to the
selected split.

For a frame already present in flash, ordinary referability is preserved while
attachment referability and pairs are replaced by the four manual roles. For a
new frame, only the four manual roles are referable; other projected objects
remain visibility metadata and cannot be referenced by generated questions.
Manual scenes and frames are prioritized before `--max_scenes` and
`--max_frames` truncation. Each frame's `manual_attachment_roles` forces the
four object roles, and the typed labels are used in the attachment question,
options, and answer fields.
