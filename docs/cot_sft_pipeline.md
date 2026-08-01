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

## 8k and 10k SFT pilot datasets

The pilot selector operates only on records accepted by the fact, answer, response, and image validators. It deduplicates `question_uid` values before sampling.

Build the initial 8k training manifest with L1/L2/L3 counts of 3669/661/3670.
It includes every available validated L2 record without duplication:

```powershell
python scripts/build_cot_sft.py output_train/benchmark.json `
  --output-prefix output_train/cot/pilot_train_8k `
  --split train `
  --preset pilot-train-8k `
  --scannet-image-root D:\datasets\scannet\scans `
  --scannetpp-image-root D:\datasets\scannetpp\data
```

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

## Two-GPU pilot training

Install a recent MS-SWIFT release that supports Qwen3-VL before launching. Preview the exact command without starting training:

```powershell
python scripts/run_cot_sft_pilot.py `
  --train-dataset output_train/cot/pilot_train_8k.ms_swift.jsonl `
  --train-sidecar output_train/cot/pilot_train_8k.sidecar.jsonl `
  --monitor-dataset output_val/cot/monitor_val_320.ms_swift_eval.jsonl `
  --monitor-sidecar output_val/cot/monitor_val_320.sidecar.jsonl `
  --output-dir output_train/sft/qwen3_vl_4b_cot_8k `
  --devices 0,1 `
  --max-length 8192 `
  --max-pixels 786432 `
  --attn-impl sdpa `
  --dry-run
```

Remove `--dry-run` to train. The launcher defaults to PyTorch SDPA, so Flash Attention is optional. It disables external experiment reporters (`--report_to none`); the milestone callback writes the required loss, checkpoint, and evaluation artifacts without TensorBoard. It uses the explicit `adamw_torch` AdamW optimizer, BF16 LoRA on the language model, leaves the vision encoder frozen, trains the aligner, and uses global batch 32. Its callback logs training loss every 100 sample exposures, saves a checkpoint every 500 exposures, and computes teacher-forced validation loss every 2000 exposures. Each evaluation prints `eval_loss` to the terminal and writes all evaluation metrics to the corresponding `checkpoint-*/eval_metrics.json`. For the initial 8k, two-epoch run, this produces 160 training-loss records, 32 checkpoints, and 8 validation points. Each reported sample count is at most 31 samples beyond its target milestone.

The launcher sets `CUDA_DEVICE_ORDER=PCI_BUS_ID`, so numeric values passed to `--devices` match the physical indices printed by `nvidia-smi`, including on mixed-GPU servers where CUDA's default fastest-first order differs from PCI order.

After training, the launcher maps the eight evaluation checkpoints to `samples_seen_02000` through `samples_seen_16000`, runs deterministic generation on the same monitoring questions for those checkpoints, and writes:

- `checkpoint_index.json`: global-step to sample-exposure mapping;
- `training_metrics/samples_target_*.json`: training loss every 100 sample exposures;
- `checkpoint-*/eval_metrics.json`: teacher-forced validation loss and evaluation metrics;
- `monitor/*.predictions.jsonl`: raw MS-SWIFT generations;
- `monitor/*.report.json`: strict/macro/level metrics;
- `monitor/*.report.details.json`: per-question parsing details;
- `monitor/learning_curve.json`: the base model and checkpoint learning curve.

To resume interrupted training from the newest complete checkpoint, rerun the same command with:

```powershell
  --resume-from-checkpoint latest `
  --skip-completed-evals
```

The launcher ignores incomplete checkpoint directories when resolving `latest` and refuses to start a fresh training run in an output directory that already contains checkpoints. An explicit checkpoint path may be passed instead of `latest`. Trainer state restores the model adapter, optimizer, scheduler, epoch, global step, and data position. During post-training monitoring, `--skip-completed-evals` reuses existing reports and evaluates existing prediction files before launching any missing inference jobs.

To rerun only checkpoint generation/evaluation after an interrupted monitoring pass, use `--skip-train --skip-completed-evals`. Use `--skip-base-eval` when the base report is intentionally excluded.

The default `max_length` is 8192 and the per-image budget is 786432 pixels (`1024 * 768`). Qwen3-VL uses a patch size of 16 and a spatial merge size of 2, so this is approximately 768 merged visual tokens per image. The pilot contains samples with up to eight images; their visual input is therefore about 6144 tokens at the configured cap. By contrast, 4096 tokens without an image budget can cause MS-SWIFT to delete every over-length retry candidate before training starts. Do not switch to left or right truncation for this failure: truncation can remove either the question or the final `Answer:` supervision. Before the full run, inspect processed token lengths and complete a short smoke test.

## Matched three-target curriculum

This ablation combines the exact old 8k training UIDs with the disjoint stage-two 2k
questions. It then creates three datasets with identical question order, prompts, images,
and repetition schedule:

- `answer_only`: only `Answer: B` (or ordered space-separated multi-select letters);
- `fixed_template_cot`: the existing deterministic fact/template reasoning;
- `teacher_cot`: independent VLM reasoning retained only when its final answer is correct.

The teacher never receives the saved answer or oracle fields. A malformed, repetitive, or
wrong response consumes one semantic attempt; at most two semantic attempts are allowed.
Transport retries do not consume semantic attempts. If both semantic attempts fail, that
question is absent from all three datasets.

First reconstruct the exact 10k candidate pool:

```powershell
python scripts/build_mixed_cot_ablation.py merge `
  --benchmark output_train/benchmark.json `
  --old-sidecar cot/train/pilot_train_8k.sidecar.jsonl `
  --stage2 output_train/stage2_train_2k.json `
  --output cot/train/mixed_train_10k.json
```

Before a full API run, test a small deterministic sample. `--limit-per-type 2` selects at
most two questions of each type; `--limit`, repeated `--question-uid`, or both can narrow it
further. The JSONL cache contains every transport event and semantic attempt, so rerunning
the same command resumes completed questions.

```powershell
$env:OPENAI_API_KEY = "your-key"
python scripts/filter_correct_teacher_cot.py cot/train/mixed_train_10k.json `
  --output cot/train/mixed_train_10k.teacher.small.json `
  --cache-jsonl cot/train/mixed_train_10k.teacher.small.cache.jsonl `
  --base-url https://your-openai-compatible-server/v1 `
  --model your-teacher-vlm `
  --api-provider openai_chat `
  --max-attempts 2 `
  --workers 4 `
  --max-output-tokens 384 `
  --limit-per-type 2 `
  --scannet-image-root D:\datasets\scannet\scans `
  --scannetpp-image-root D:\datasets\scannetpp\data
```

For the full filter, use a separate output/cache and omit all small-sample options:

```powershell
python scripts/filter_correct_teacher_cot.py cot/train/mixed_train_10k.json `
  --output cot/train/mixed_train_10k.teacher.success.json `
  --cache-jsonl cot/train/mixed_train_10k.teacher.cache.jsonl `
  --base-url https://your-openai-compatible-server/v1 `
  --model your-teacher-vlm `
  --api-provider openai_chat `
  --max-attempts 2 `
  --workers 4 `
  --max-output-tokens 384 `
  --scannet-image-root D:\datasets\scannet\scans `
  --scannetpp-image-root D:\datasets\scannetpp\data
```

The VLM calling flags intentionally match `run_sampled_type_vlm_eval.py`. Select
`--api-provider openai_chat` for OpenAI-compatible chat endpoints, including vLLM,
SGLang, DashScope, and most proxy services; select `openai_responses` for the OpenAI
Responses API; or select `anthropic` for Anthropic's native Messages API. The aliases
`--vlm_url`, `--vlm_model`, `--vlm_workers`, `--max_tokens`, and underscore-style
image-root flags are also accepted. Credentials can be passed with `--api-key`, named
with `--api-key-env`, or discovered from `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`,
`ANTHROPIC_AUTH_TOKEN`, or `DASHSCOPE_API_KEY` as appropriate.

Build the matched exports. Every teacher-success question appears at least once. A type
with fewer than 300 teacher-success questions is deterministically repeated to 300 before
the curriculum streams are formed. The builder fails instead of silently omitting unique
questions if the retained pool cannot fit the agreed curriculum capacity.

```powershell
python scripts/build_mixed_cot_ablation.py finalize `
  cot/train/mixed_train_10k.teacher.success.json `
  --output cot/train/mixed_train_curriculum_three_targets.json `
  --output-prefix cot/train/mixed_train_curriculum `
  --type-floor 300 `
  --scannet-image-root D:\datasets\scannet\scans `
  --scannetpp-image-root D:\datasets\scannetpp\data
```

The continuous one-epoch schedule contains 20,480 exposures (640 optimizer steps at global
batch 32). Exposures 1-6,144 are all L1. Stage 2 contains L1/L2/L3 counts of
2,048/6,144/6,144, using the seven-batch pattern `A,B,A,C,A,B,C` repeated 64 times.
The complete L1/L2/L3 counts are therefore 8,192/6,144/6,144. Dataset preprocessing shuffle,
training dataloader shuffle, and length grouping are all disabled, so one optimizer and
scheduler traverse this exact order without a stage restart.

Launch three fresh LoRAs from the base model. This loop deliberately contains neither
`--initial-adapter` nor `--resume-from-checkpoint`; each output directory must be new.

```powershell
$variants = @("answer_only", "fixed_template_cot", "teacher_cot")
foreach ($variant in $variants) {
  python scripts/run_cot_sft_pilot.py `
    --train-dataset "cot/train/mixed_train_curriculum.$variant.ms_swift.jsonl" `
    --train-sidecar "cot/train/mixed_train_curriculum.$variant.sidecar.jsonl" `
    --curriculum-manifest cot/train/mixed_train_curriculum_three_targets.json `
    --monitor-dataset cot/val/monitor_val_320.ms_swift_eval.jsonl `
    --monitor-sidecar cot/val/monitor_val_320.sidecar.jsonl `
    --model Qwen/Qwen3-VL-4B-Instruct `
    --allow-base-model-start `
    --output-dir "cot_result/qwen3_vl_4b_curriculum_$variant" `
    --devices 5,6 `
    --epochs 1 `
    --per-device-batch-size 2 `
    --gradient-accumulation-steps 8 `
    --learning-rate 1e-4 `
    --aligner-learning-rate 1e-5 `
    --lora-rank 32 `
    --lora-alpha 64 `
    --lora-dropout 0.05 `
    --weight-decay 0.01 `
    --warmup-ratio 0.03 `
    --max-grad-norm 1 `
    --max-length 8192 `
    --max-pixels 786432 `
    --log-every-samples 256 `
    --save-every-samples 2048 `
    --eval-every-samples 2048
}
```

Curriculum mode verifies the master schedule hash, all 20,480 sample IDs and row positions,
the exact level/batch composition, base-model initialization, and required hyperparameters
before training. It records the dataset, sidecar, curriculum-file, and schedule SHA256 values
plus both shuffle flags in `pilot_manifest.json`. Checkpoints/evaluations occur at every
2,048 exposures, including the L1 stage boundary at 6,144 and the final exposure at 20,480.

Teacher-forced loss is not directly comparable across the three target styles because they
supervise very different numbers of answer tokens. Use the same generated-answer monitoring
metrics (strict accuracy, per-level accuracy, and per-type accuracy) for the primary ablation;
interpret loss curves only within each target style.

## Standalone prediction evaluation

MS-SWIFT prediction files can also be evaluated independently:

```powershell
python scripts/evaluate_cot_predictions.py `
  --sidecar output_val/cot/monitor_val_320.sidecar.jsonl `
  --predictions output_train/sft/qwen3_vl_4b_cot_8k/monitor/samples_seen_08000.predictions.jsonl `
  --output output_train/sft/qwen3_vl_4b_cot_8k/monitor/samples_seen_08000.report.json
```

Strict format requires exactly one final `Answer: B` line (or space-separated letters for a multi-select question). Relaxed accuracy is reported separately so format learning is not confused with reasoning accuracy.
