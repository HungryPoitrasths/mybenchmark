# Deterministic CoT SFT pipeline

This pipeline converts the benchmark's saved oracle fields into auditable reasoning facts and then renders English SFT responses without invoking a model per question. It covers the 16 reviewed L1/L2/L3 types and deliberately excludes `L3_attachment_move`.

## Guarantees and rejection policy

- The semantic answer and option letter are checked against each other before export.
- Direction reasoning accepts only the eight horizontal directions. Legacy vertical answers are rejected.
- Modern L2 pairwise occlusion is supported; the old unary `object_move_occlusion` format is rejected.
- `not visible` never receives an invented cause. An occluding object is named only when a saved confidence is at least 0.8.
- Internal coordinates, exact closest-point distances, ray counts, and occlusion ratios are not placed in the reasoning text.
- Multi-image order is `image_name`, `auxiliary_image_names`, then `reasoning_frame_2`, with duplicate destination frames removed.
- Validation and test runs generate sidecars and reports but never an MS-SWIFT training file.

Every rejected record is written with a stable UID, error code, and message. A non-empty rejection file is expected when an older mixed benchmark contains vertical directions, `L3_attachment_move`, or unary L2 occlusion.

## Dry run

Run a metadata-only coverage pass before resolving images:

```powershell
python scripts/build_cot_sft.py output_train/benchmark.json `
  --output-prefix output_train/cot/train `
  --split train `
  --allow-missing-images `
  --fail-on-reject
```

Review `train.report.json`, `train.rejected.jsonl`, and a sample of `train.sidecar.jsonl`. Remove `--fail-on-reject` when intentionally filtering known legacy records.
Metadata-only mode deliberately does not write an MS-SWIFT file, so unresolved candidate paths cannot be used for training by mistake.

## Final MS-SWIFT export

Supply the actual image roots and omit `--allow-missing-images`:

```powershell
python scripts/build_cot_sft.py output_train/benchmark.json `
  --output-prefix output_train/cot/train `
  --split train `
  --scannet-image-root D:\datasets\scannet\scans `
  --scannetpp-image-root D:\datasets\scannetpp\data `
  --scannetpp-sensor iphone
```

The generated `train.ms_swift.jsonl` uses `messages` plus `images`. The user message contains one `<image>` token per image, and the assistant message ends with exactly `Answer: C` (or space-separated letters for multiple selection).

## Deterministic templates

Every `signature_id` receives its own 12 reviewed English templates. Their wording is generated
deterministically from the signature's discrete facts (question type, reference frame, movement
state, layout, and result), while the three clauses are filled from saved oracle facts. Building
CoT data therefore needs no template-model call and no image.

To emit response-shaped JSONL for every signature, for audit or a later `merge` operation:

```powershell
python scripts/generate_cot_templates.py offline output_train/benchmark.json `
  --output output_train/cot/template_responses.jsonl
```

Each record has a `signature_id` and its own 12 templates containing `{observation}`,
`{transformation}`, and `{conclusion}`. To persist those per-signature entries in a separate
library, run:

```powershell
python scripts/generate_cot_templates.py merge output_train/cot/template_responses.jsonl `
  --output templates/cot_templates.reviewed.json
```

## Optional external review

The legacy request workflow remains available only when external, signature-specific wording is
needed:

```powershell
python scripts/generate_cot_templates.py prepare output_train/benchmark.json `
  --output output_train/cot/template_requests.jsonl
```

Send only each request's text messages to the template model. Do not add images. Save responses as JSONL records with `signature_id` and a `templates` array of exactly 12 strings. Each string must retain `{observation}`, `{transformation}`, and `{conclusion}`.

Pass a separately reviewed file to the builder with `--template-path`. Template selection is fixed by `SHA256(seed + question_uid + signature_id) mod template_count`; the default seed is 42.

## 10k SFT pilot datasets

The pilot selector operates only on records accepted by the fact, answer, response, and image validators. It deduplicates `question_uid` values before sampling.

Build the fixed 10k training manifest with L1/L2/L3 counts of 4669/661/4670.
The pilot includes every available validated L2 record without duplication and splits the
remaining capacity nearly evenly between L1 and L3:

```powershell
python scripts/build_cot_sft.py output_train/benchmark.json `
  --output-prefix output_train/cot/pilot_train_10k `
  --split train `
  --preset pilot-train-10k `
  --scannet-image-root D:\datasets\scannet\scans `
  --scannetpp-image-root D:\datasets\scannetpp\data
```

Within a level, underfilled types are kept without duplication and their unused quota is redistributed across the other supported types. Within each type, selection compresses signature imbalance and interleaves answer letters and scenes.

Build the fixed 320-question monitoring validation set, with 20 questions for every supported type:

```powershell
python scripts/build_cot_sft.py output_val/benchmark_balanced.json `
  --output-prefix output_val/cot/monitor_val_320 `
  --split val `
  --preset monitor-val-320 `
  --scannet-image-root D:\datasets\scannet\scans `
  --scannetpp-image-root D:\datasets\scannetpp\data
```

The validation command writes `monitor_val_320.ms_swift_eval.jsonl`, not a training export. The preset fails if any of the 16 types has fewer than 20 accepted records.

## Two-A800 pilot training

Install a recent MS-SWIFT release that supports Qwen3-VL before launching. Preview the exact command without starting training:

```powershell
python scripts/run_cot_sft_pilot.py `
  --train-dataset output_train/cot/pilot_train_10k.ms_swift.jsonl `
  --train-sidecar output_train/cot/pilot_train_10k.sidecar.jsonl `
  --monitor-dataset output_val/cot/monitor_val_320.ms_swift_eval.jsonl `
  --monitor-sidecar output_val/cot/monitor_val_320.sidecar.jsonl `
  --output-dir output_train/sft/qwen3_vl_4b_cot_10k `
  --devices 0,1 `
  --dry-run
```

Remove `--dry-run` to train. The launcher uses BF16 LoRA on the language model, leaves the vision encoder frozen, trains the aligner, and uses global batch 32. Its MS-SWIFT callback saves and computes teacher-forced validation loss at the first optimizer step after every 1000-sample milestone. For the fixed 10k, two-epoch run, the schedule is `32, 63, 94, ..., 313, 345, ..., 626`; each checkpoint is at most 31 samples beyond its milestone.

After training, the launcher maps the saved checkpoints to `samples_seen_01000` through `samples_seen_20000`, runs deterministic generation on the same 320 questions for every milestone, and writes:

- `checkpoint_index.json`: global-step to sample-exposure mapping;
- `monitor/*.predictions.jsonl`: raw MS-SWIFT generations;
- `monitor/*.report.json`: strict/macro/level metrics;
- `monitor/*.report.details.json`: per-question parsing details;
- `monitor/learning_curve.json`: the base model and checkpoint learning curve.

To rerun only checkpoint generation/evaluation after an interrupted monitoring pass, use `--skip-train`. Use `--skip-base-eval` when the base report already exists.

The default `max_length` is 8192. Before the full run, inspect image counts and processed token lengths, then lower this value or the processor's visual-token budget if the heaviest samples do not pass a 100-step smoke test.

## Standalone prediction evaluation

MS-SWIFT prediction files can also be evaluated independently:

```powershell
python scripts/evaluate_cot_predictions.py `
  --sidecar output_val/cot/monitor_val_320.sidecar.jsonl `
  --predictions output_train/sft/qwen3_vl_4b_cot_10k/monitor/samples_seen_10000.predictions.jsonl `
  --output output_train/sft/qwen3_vl_4b_cot_10k/monitor/samples_seen_10000.report.json
```

Strict format requires exactly one final `Answer: B` line (or space-separated letters for a multi-select question). Relaxed accuracy is reported separately so format learning is not confused with reasoning accuracy.
