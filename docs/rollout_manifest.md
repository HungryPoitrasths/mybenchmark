# Future Rollout Generation and Evaluation

The rollout experiment is split into two isolated stages:

1. Generate an image or video from an answer-free private job.
2. Evaluate the generated media with `scripts/run_sampled_type_vlm_eval.py`
   and a public `predictive-spatial-rollout-v1` manifest.

The original benchmark question is used only by the answering VLM in stage 2.
It is never copied into a generation job.

## Automatic frame selection

`scripts/prepare_future_rollout_jobs.py` automatically creates the private
selection spec before it prepares generation jobs. Users do not provide or
edit a selection spec. A generated entry looks like:

```json
{
  "source_index": 0,
  "motion_frame_path": "frames/F_motion.jpg",
  "camera_rotation_world_to_camera": [
    [1.0, 0.0, 0.0],
    [0.0, 0.94, -0.34],
    [0.0, 0.34, 0.94]
  ],
  "moving_group": [
    {"obj_id": 20, "label": "desk"},
    {"obj_id": 26, "label": "white box on the desk"}
  ],
  "orbit_anchor_label": "chair",
  "picture_eligible": true,
  "video_eligible": true,
  "answer_context_media": [
    {"path": "frames/query.jpg", "role": "query_reference_view"}
  ]
}
```

For every supported L2 question, the selector traverses the scene's registered
camera frames. A frame is eligible only when the moved root, every transitive
attachment, and the query object are fully inside the source image, pass the
projection and edge-margin gates, and are visible. Source visibility uses
registered depth when available and falls back to mesh rays when depth has no
valid samples. The selector applies the benchmark movement or orbit in 3D and
uses counterfactual mesh rays to require the complete moved group to remain
fully in-frame and visible at its future location from the same camera.

The highest-scoring eligible frame is copied to
`media/conditions/<question_uid>/motion.<ext>`. Original and auxiliary
benchmark views are copied to `media/context/<question_uid>/` for answer-time
context. Cosmos, GPT Image, and Qwen each receive the same single motion image;
the extra context views are never generation inputs.

The generated `moving_group` contains the full transitive attachment chain,
with the moved parent first. The selected camera rotation remains in the
private selection input and is used only to convert the benchmark's world-space
motion delta to camera-relative movement. It is absent from all model requests
and public manifests.

The preparation command fixes the paired GPT/Qwen sample and prompt:

```powershell
python scripts/prepare_future_rollout_jobs.py `
  --benchmark_file output/benchmark_subset.json `
  --scannet_root D:/datasets/scannet/scans `
  --scannetpp_root D:/datasets/scannetpp/data `
  --scannetpp_frame_root output/scannetpp_iphone_frames `
  --output_dir rollout/run_20260725 `
  --seed 20260725 `
  --expected_picture_per_type 50
```

Only roots used by the benchmark are required. ScanNet++ defaults to the iPhone
sensor; use `--scannetpp_sensor dslr` for DSLR frames. Frame traversal can be
controlled with `--frame_stride_scannet` and `--frame_stride_scannetpp`.

Selection reports scene and question progress, elapsed time, ETA, and per-scene
cache hit counts on stderr. Image-quality results are cached once per scene and
frame, while source-object visibility is cached once per scene, frame, and
object. Completed question results are appended to a fingerprinted
`private_jobs/selection_checkpoint_*.jsonl` file. Re-running the same command
automatically resumes from that checkpoint and only re-evaluates the question
that was active when the previous process stopped. The fingerprint includes the
benchmark content hash, dataset roots, sensor, frame strides, and selection
algorithm version, so changing any of those settings starts a separate
checkpoint.

Future-frame visibility uses a two-stage selector. All sampled frames first pass
an image-quality and strict bbox-projection prefilter without mesh rays. At most
32 frames per question then receive counterfactual mesh-ray verification using
64 deterministic surface samples and 4 local boundary resamples; bbox probe
rays are disabled because rollout ranking does not consume their metrics. A
passing original question frame is always retained in the shortlist. Tune the
balanced defaults with `--mesh_ray_shortlist_size`,
`--mesh_ray_surface_samples`, and `--mesh_ray_local_resamples`. These reduced
budgets apply only to rollout selection; benchmark generation keeps its full
mesh-ray defaults.

The command writes `private_jobs/selection_spec.json` and the geometry-rich
`private_jobs/selection_audit.json`, then creates private GPT, Qwen, and Cosmos
jobs plus separate public manifests. It selects at most 50 questions for every
supported L2 type by default. If a type has fewer strict single-frame matches,
all available matches are emitted normally and the audit records its
`shortfall`. There is no weaker or manual fallback. The GPT and Qwen job lists
use the same question order, input-image hashes, prompt text, and output
dimensions.

## Generation prompts

GPT Image and Qwen receive exactly one `F_motion` image and the same text. The
text identifies the moving object/attachment labels and states the displacement
in camera coordinates, for example `away from the camera`, `toward the camera`,
`toward image-left`, and `toward image-right`, with distances. It requires the
group to move rigidly, preserves object orientation and the static scene,
removes the old instance, forbids duplicates and annotations, and asks for one
photorealistic result.

Cosmos receives exactly one `F_motion` image and an answer-free action prompt.
The text uses the same camera-relative movement, fixed camera, continuous
motion, rigid attachments, static surroundings, and endpoint settling. Target
duration is `clamp(0.75 + path_length / 0.8, 2, 5)` seconds. The generated
official input JSON contains only `inference_type`, `name`, `prompt`, and
`input_path`.

For orbit questions, the prompts additionally identify the anchor and orbit
operation while preserving the moved group's own orientation. No prompt states
the benchmark query relation or any required occlusion/out-of-frame outcome.

Generation jobs are recursively checked to reject question/options/answer
fields, correct values, future coordinates, future bounding boxes, and future
projections.

## Generate pictures

GPT Image uses `gpt-image-1.5`, `images.edit`, and
`input_fidelity="high"`. Only transport errors, empty responses, or corrupt
files are retried; there is one accepted output per job.

```powershell
$env:OPENAI_API_KEY = "..."
python scripts/run_future_picture_generation.py `
  --backend gpt `
  --jobs rollout/run_20260725/private_jobs/gpt_jobs.json `
  --manifest rollout/run_20260725/manifests/gpt_picture.json `
  --base_url https://api.openai.com/v1 `
  --request_interval 0
```

Qwen runs locally with no best-of-N or retry sampling:

```powershell
python scripts/run_future_picture_generation.py `
  --backend qwen `
  --jobs rollout/run_20260725/private_jobs/qwen_jobs.json `
  --manifest rollout/run_20260725/manifests/qwen_picture.json `
  --qwen_checkpoint /path/to/Qwen-Image-Edit-2511 `
  --device cuda `
  --qwen_cpu_offload `
  --limit 1
```

Remove `--limit 1` after the preflight succeeds. Both backends normalize the
output to the condition image dimensions and update the public manifest
incrementally, so interrupted runs can resume.

## Generate and finalize Cosmos videos

Run each JSON below `private_jobs/cosmos_inputs` with the official
Cosmos-Predict2.5-14B Image2World inference path. These are independent
single-image samples; multiple input files are not multiple conditioning frames
for one 14B sample. Configure the official runner for 480p, 24 FPS, and the
nearest supported duration/frame count to each private job's
`duration_seconds`. Save each result as `<question_uid>.mp4`.

After inference, resolve the videos and extract exactly eight deterministic
frames including both endpoints:

```powershell
python scripts/finalize_cosmos_rollouts.py `
  --jobs rollout/run_20260725/private_jobs/cosmos_jobs.json `
  --manifest rollout/run_20260725/manifests/cosmos_video.json `
  --video_root /path/to/cosmos/outputs
```

The finalizer records the source video hash, FPS, duration, source frame
indices, and all eight frame hashes.

## Public manifest and media order

The public manifest contains question identity, type, scene, eligibility,
ordered media, hashes, and generation provenance. It contains no prompt, pose,
delta, benchmark answer, or GT future geometry.

Picture order is:

1. `motion_reference_view`
2. `predicted_future_view` for full rollout only
3. `destination_to_query_bridge` zero or more
4. `query_reference_view` zero or more

Video order is:

1. `motion_reference_view`
2. eight `predicted_video_frame` items numbered 0 through 7
3. `destination_to_query_bridge` zero or more
4. `query_reference_view` zero or more

`--context_only` removes every `kind: prediction` item and retains identical
real context in identical order. The evaluator inserts fixed role labels before
images and never forwards arbitrary manifest strings to the answering VLM.

Validate completed outputs and strict GPT/Qwen pairing before evaluation:

```powershell
python scripts/validate_rollout_manifest.py --mode picture `
  --manifest rollout/run_20260725/manifests/gpt_picture.json `
  --manifest rollout/run_20260725/manifests/qwen_picture.json `
  --expected_per_type 50 --strict_provenance

python scripts/validate_rollout_manifest.py --mode video `
  --manifest rollout/run_20260725/manifests/cosmos_video.json `
  --expected_per_type 50 --strict_provenance
```

## Answering VLM evaluation

Use distinct output files for each condition. For example:

```powershell
python scripts/run_sampled_type_vlm_eval.py `
  --benchmark_file output/benchmark_subset.json `
  --rollout_manifest rollout/run_20260725/manifests/gpt_picture.json `
  --picture --context_only `
  --model qwen3.5-flash --base_url http://server/v1 `
  --output_json rollout/results/gpt_context.json `
  --output_html rollout/results/gpt_context.html

python scripts/run_sampled_type_vlm_eval.py `
  --benchmark_file output/benchmark_subset.json `
  --rollout_manifest rollout/run_20260725/manifests/gpt_picture.json `
  --picture `
  --model qwen3.5-flash --base_url http://server/v1 `
  --output_json rollout/results/gpt_future.json `
  --output_html rollout/results/gpt_future.html

python scripts/run_sampled_type_vlm_eval.py `
  --benchmark_file output/benchmark_subset.json `
  --rollout_manifest rollout/run_20260725/manifests/cosmos_video.json `
  --video `
  --model qwen3.5-flash --base_url http://server/v1 `
  --output_json rollout/results/cosmos_future.json `
  --output_html rollout/results/cosmos_future.html
```

Run without rollout arguments for the original benchmark baseline.
`--picture` and `--video` are mutually exclusive. Neither can be combined with
`--blind` or `--oracle`; `--direct` remains compatible.

## Leakage boundary

Store future 3D coordinates, projection boxes, simulated visibility, and frame
selection evidence only in a separate audit manifest. Never supply that audit
manifest to a generation client or answering VLM. The public manifest loader
recursively rejects answer and future-geometry fields and verifies optional
SHA-256 values before any answering API call.
