# Future Rollout Generation and Evaluation

The rollout experiment has two isolated stages:

1. Generate an image or video from an answer-free private job.
2. Evaluate the generated media with `scripts/run_sampled_type_vlm_eval.py`
   and a public `predictive-spatial-rollout-v1` manifest.

The original benchmark question is used only by the answering VLM in stage 2.
It is never copied into a generation job.

## Automatic route selection

`scripts/prepare_future_rollout_jobs.py` first creates a private route selection
spec. Each route comes directly from the benchmark's existing fields:

- `image_name`: source view containing the moving group before movement.
- `auxiliary_image_names`: ordered camera path between the endpoints.
- `reasoning_frame_2`: destination view containing the endpoint environment.

Picture generation uses two or three ordered images. When auxiliary frames are
available, the selector chooses the frame whose cumulative camera travel from
the source is closest to 50% of the complete route; an exact tie selects the
earlier frame. Without a bridge it uses source and destination only.

```json
{
  "source_index": 0,
  "generation_images": [
    {"path": "frames/source.jpg", "role": "source_view"},
    {"path": "frames/bridge.jpg", "role": "source_to_destination_bridge"},
    {"path": "frames/destination.jpg", "role": "destination_environment"}
  ],
  "camera_rotation_world_to_camera": [
    [1.0, 0.0, 0.0],
    [0.0, 0.94, -0.34],
    [0.0, 0.34, 0.94]
  ],
  "moving_group": [
    {"obj_id": 20, "label": "desk"},
    {"obj_id": 26, "label": "white box on the desk"}
  ],
  "picture_eligible": true,
  "video_eligible": true
}
```

The source view must pass image-quality, projection, and visibility checks for
every moved object and transitive attachment at its original position. The
destination view must pass image-quality, projection, and counterfactual
visibility checks for the complete moved group at its future position, plus
projection and visibility checks for the static query object. A selected bridge
must exist, have a valid pose, and pass image quality. This avoids requiring one
camera to contain both the original and future object positions.

The generated `moving_group` contains the complete transitive attachment chain,
with the moved parent first. Camera rotation remains private and is used only to
convert world-space movement into camera-relative action text. No masks, boxes,
projection markers, coordinates, or answer fields enter generation requests.

The current pilot selects at most two questions per supported L2 type by
default. Completed results are stored in a fingerprinted
`private_jobs/selection_checkpoint_*.jsonl`; the v2 schema and route algorithm
fingerprint prevent older selector checkpoints from being reused.

```powershell
python scripts/prepare_future_rollout_jobs.py `
  --benchmark_file output/benchmark_subset.json `
  --scannet_root D:/datasets/scannet/scans `
  --scannetpp_root D:/datasets/scannetpp/data `
  --scannetpp_frame_root output/scannetpp_iphone_frames `
  --output_dir rollout/run_20260729 `
  --seed 20260729 `
  --expected_picture_per_type 2
```

## Picture generation

Qwen and GPT receive identical ordered image lists, prompts, seeds, and question
order. The prompt identifies Image 1 as the source, an optional Image 2 as the
route bridge, and the final image as the destination canvas. The output must
preserve the final image's camera, composition, dimensions, and static scene
while placing the rigid moving group at its endpoint exactly once.

This project caps Qwen-Image-Edit-2511 at three input images as an engineering
choice. The Diffusers `QwenImageEditPlusPipeline` accepts an image list, but the
three-image cap is not claimed to be a proven model maximum.

Run Qwen first, initially with one preflight job:

```powershell
python scripts/run_future_picture_generation.py `
  --backend qwen `
  --jobs rollout/run_20260729/private_jobs/qwen_jobs.json `
  --manifest rollout/run_20260729/manifests/qwen_picture.json `
  --qwen_checkpoint /path/to/Qwen-Image-Edit-2511 `
  --device cuda `
  --qwen_cpu_offload `
  --limit 1
```

Then run GPT with the paired inputs:

```powershell
$env:OPENAI_API_KEY = "..."
python scripts/run_future_picture_generation.py `
  --backend gpt `
  --jobs rollout/run_20260729/private_jobs/gpt_jobs.json `
  --manifest rollout/run_20260729/manifests/gpt_picture.json `
  --base_url https://api.openai.com/v1
```

Remove `--limit 1` after preflight. Every input hash is checked at generation
time. Both backends normalize output to the final destination image dimensions
and update manifests incrementally for resumable execution.

## Cosmos single-image baseline

Cosmos-Predict2.5 remains in its standard single-image Image2World mode. It
receives only the source view through one `input_path`; the bridge and
destination views are answer-time context and are not Cosmos conditioning
inputs. Each official input JSON contains only `inference_type`, `name`,
`prompt`, and `input_path`.

Run every JSON under `private_jobs/cosmos_inputs` with the official
Cosmos-Predict2.5-14B Image2World inference path, save each result as
`<question_uid>.mp4`, then extract eight deterministic frames:

```powershell
python scripts/finalize_cosmos_rollouts.py `
  --jobs rollout/run_20260729/private_jobs/cosmos_jobs.json `
  --manifest rollout/run_20260729/manifests/cosmos_video.json `
  --video_root /path/to/cosmos/outputs
```

## Public media order

Picture order is:

1. `source_view`
2. optional `source_to_destination_bridge`
3. `destination_environment`
4. `predicted_future_view` for full rollout only

Cosmos video order remains:

1. `motion_reference_view` (the source image)
2. eight `predicted_video_frame` items numbered 0 through 7
3. optional `destination_to_query_bridge`
4. `query_reference_view` (the destination image)

`--context_only` removes every prediction and retains real context in place.
The evaluator inserts fixed role labels and never forwards arbitrary manifest
strings to the answering VLM.

Validate completed outputs before evaluation:

```powershell
python scripts/validate_rollout_manifest.py --mode picture `
  --manifest rollout/run_20260729/manifests/qwen_picture.json `
  --manifest rollout/run_20260729/manifests/gpt_picture.json `
  --expected_per_type 2 --strict_provenance

python scripts/validate_rollout_manifest.py --mode video `
  --manifest rollout/run_20260729/manifests/cosmos_video.json `
  --expected_per_type 2 --strict_provenance
```

Use `scripts/run_sampled_type_vlm_eval.py --picture` or `--video` with the
corresponding manifest. Add `--context_only` for the real-image context
baseline. Run without rollout arguments for the original benchmark baseline.

## Leakage boundary

Future 3D coordinates, projection boxes, simulated visibility, and selection
evidence stay only in the private audit. Never supply that audit to a generation
client or answering VLM. The public manifest loader rejects answer and
future-geometry fields and verifies optional SHA-256 values before any answering
API call.
