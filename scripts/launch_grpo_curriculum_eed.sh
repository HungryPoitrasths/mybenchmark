#!/usr/bin/env bash
set -euo pipefail

project_root=${PROJECT_ROOT:-/ces124/real/sujinyue/mybenchmark}
asset_root=${ASSET_ROOT:-/ces124/real/sujinyue/grpo}
job_root=${JOB_ROOT:-/ces124/real/sujinyue/codex_jobs/grpo_resume_eed_20260811}
output_root=${OUTPUT_ROOT:-$asset_root/qwen3vl_4b_grpo_curriculum_eed_gpu23}
prepared_root=${PREPARED_ROOT:-$asset_root/prepared}
model_root=${MODEL_ROOT:-/ces124/models/Qwen3-VL-4B-Instruct}
image_root=${SCANNETPP_IMAGE_ROOT:-/ces124/real/sujinyue/datasets/scannetpp/train/iphone_frames}
stage1_checkpoint=${STAGE1_CHECKPOINT:-$output_root/stage1_l1_perception/v0-20260807-170629/checkpoint-3072}
stage2_checkpoint=${STAGE2_CHECKPOINT:-$output_root/stage2_reasoning_replay/v0-20260808-234735/checkpoint-8192}
python_bin=${PYTHON_BIN:-/ces124/real/sujinyue/venvs/grpo/bin/python}
swift_bin=${SWIFT_BIN:-/ces124/real/sujinyue/venvs/grpo/bin/swift}
python_dev_root=${PYTHON_DEV_ROOT:-/ces124/real/sujinyue/python-dev/usr/include}
python_dev_include=${PYTHON_DEV_INCLUDE:-$python_dev_root/python3.11}
triton_driver_source=${TRITON_DRIVER_SOURCE:-/ces124/real/sujinyue/venvs/grpo/lib/python3.11/site-packages/triton/backends/nvidia/driver.c}
devices=${DEVICES:-2,3}
occupier_pid_file=${OCCUPIER_PID_FILE:-$job_root/occupier.pid}

launcher_log=$job_root/logs/launcher.log
status_file=$job_root/status.txt
occupier_log=$job_root/logs/occupier.log
mkdir -p "$job_root/logs" "$output_root"
exec >>"$launcher_log" 2>&1

timestamp() {
  date --iso-8601=seconds
}

set_status() {
  local state=$1
  local detail=${2:-}
  local temporary=$status_file.tmp
  printf 'time=%s\nstate=%s\ndetail=%s\n' "$(timestamp)" "$state" "$detail" >"$temporary"
  mv "$temporary" "$status_file"
}

exec 9>"$job_root/launcher.lock"
if ! flock -n 9; then
  set_status blocked "another target-server GRPO launcher holds the lock"
  exit 3
fi

occupier_pid() {
  [[ -f $occupier_pid_file ]] || return 1
  local pid
  pid=$(<"$occupier_pid_file")
  [[ $pid =~ ^[0-9]+$ ]] || return 1
  kill -0 "$pid" 2>/dev/null || return 1
  printf '%s\n' "$pid"
}

stop_occupier() {
  local pid
  pid=$(occupier_pid) || return 0
  echo "[$(timestamp)] Preflight passed; stopping GPU 2,3 occupier PID $pid."
  kill -TERM "$pid"
  for _ in $(seq 1 60); do
    if ! kill -0 "$pid" 2>/dev/null; then
      rm -f "$occupier_pid_file"
      return 0
    fi
    sleep 1
  done
  echo "[$(timestamp)] Occupier did not exit in 60 seconds; sending SIGKILL."
  kill -KILL "$pid"
  rm -f "$occupier_pid_file"
}

start_occupier() {
  if occupier_pid >/dev/null; then
    echo "[$(timestamp)] GPU occupier is already running."
    return 0
  fi
  echo "[$(timestamp)] Reserving physical GPUs $devices after GRPO exit."
  cd "$project_root" || return 1
  nohup env CUDA_VISIBLE_DEVICES="$devices" "$python_bin" \
    sft_qwen3vl_4B.py \
    --gpus all \
    --memory-gb 45 \
    --utilization-percent 80 >"$occupier_log" 2>&1 </dev/null 9>&- &
  printf '%s\n' "$!" >"$occupier_pid_file"
}

grpo_pid=
occupier_stopped=0
finished=0
exit_signal=

cleanup() {
  local status=$?
  trap - EXIT TERM INT HUP
  if [[ -n $grpo_pid ]] && kill -0 "$grpo_pid" 2>/dev/null; then
    kill -TERM -- "-$grpo_pid" 2>/dev/null || true
  fi
  if [[ $finished -eq 1 && $status -eq 0 ]]; then
    set_status complete "stage 2 resumed at checkpoint-8192 and finished"
  elif [[ -n $exit_signal ]]; then
    set_status interrupted "launcher received $exit_signal"
  else
    set_status failed "launcher exit code $status"
  fi
  if [[ $occupier_stopped -eq 1 ]]; then
    start_occupier || true
  fi
  exit "$status"
}

handle_signal() {
  exit_signal=$1
  exit 143
}

trap cleanup EXIT
trap 'handle_signal SIGTERM' TERM
trap 'handle_signal SIGINT' INT
trap 'handle_signal SIGHUP' HUP

set_status validating "checking model, checkpoints, datasets, images, and launch command"

required_files=(
  "$model_root/config.json"
  "$model_root/model.safetensors.index.json"
  "$model_root/tokenizer.json"
  "$prepared_root/stage1_l1_6144.grpo.jsonl"
  "$prepared_root/stage2_reasoning_18432.grpo.jsonl"
  "$project_root/output_train/grpo_curriculum_stage1_l1_6144.json"
  "$project_root/output_train/grpo_curriculum_stage2_reasoning_18432.json"
  "$stage1_checkpoint/adapter_model.safetensors"
  "$stage2_checkpoint/adapter_model.safetensors"
  "$stage2_checkpoint/optimizer.pt"
  "$stage2_checkpoint/scheduler.pt"
  "$stage2_checkpoint/trainer_state.json"
  "$stage2_checkpoint/rng_state_0.pth"
  "$stage2_checkpoint/rng_state_1.pth"
  "$python_dev_include/Python.h"
  "$triton_driver_source"
)
for required in "${required_files[@]}"; do
  if [[ ! -f $required ]]; then
    set_status failed "required file is missing: $required"
    exit 4
  fi
done

# Triton compiles a small CUDA driver extension during vLLM startup. Python.h
# also includes the architecture-specific header relative to python_dev_root.
export CPATH="$python_dev_include:$python_dev_root${CPATH:+:$CPATH}"
export PSR_CACHE_TRITON_DRIVER_SOURCE=1
printf '#include <Python.h>\n' | gcc -x c -fsyntax-only -

printf '%s  %s\n' \
  edac7703329133edfc53e46ac0081835144c99d7eebf28b71c732694d435224d "$model_root/config.json" \
  58a7841d7bff2548dd91577d216274a83cf1b500bc6a534b809d6c1b1707cf2b "$model_root/model.safetensors.index.json" \
  a5d85b6dcc535e6b93115a9ef287e6132fdbf30270da6218194ba742261173c7 "$model_root/tokenizer.json" \
  8ab00db902724b0fab97cc33cc79e53beb2b08715d3d89b57be641e9dca188ce "$stage2_checkpoint/adapter_model.safetensors" \
  8592db545fd331981dae5391c3914e892206fcc5fbddb1091eb0107a61ab6064 "$stage2_checkpoint/optimizer.pt" \
  | sha256sum --check --strict -

"$python_bin" - "$model_root" "$prepared_root" "$stage2_checkpoint" <<'PY'
import json
import sys
from pathlib import Path

model_root, prepared_root, stage2_checkpoint = map(Path, sys.argv[1:])
index = json.loads((model_root / "model.safetensors.index.json").read_text())
shards = {model_root / name for name in index["weight_map"].values()}
missing_shards = sorted(path for path in shards if not path.is_file() or path.stat().st_size == 0)
if missing_shards:
    raise FileNotFoundError(f"missing model shards: {missing_shards}")

expected_rows = {
    "stage1_l1_6144.grpo.jsonl": 6144,
    "stage2_reasoning_18432.grpo.jsonl": 18432,
}
images = set()
for name, expected in expected_rows.items():
    rows = 0
    with (prepared_root / name).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            rows += 1
            for image in row.get("images", []):
                path = Path(image)
                if not path.is_file():
                    raise FileNotFoundError(f"{name}:{line_number}: {path}")
                images.add(path)
    if rows != expected:
        raise ValueError(f"{name}: expected {expected} rows, found {rows}")
if len(images) != 6129:
    raise ValueError(f"expected 6129 unique images, found {len(images)}")
trainer_state = json.loads((stage2_checkpoint / "trainer_state.json").read_text())
if trainer_state.get("global_step") != 8192:
    raise ValueError(f"expected checkpoint global_step 8192, found {trainer_state.get('global_step')}")
print(
    f"Verified checkpoint-8192, {len(shards)} model shards, "
    f"{sum(expected_rows.values())} rows, and {len(images)} images"
)
PY

"$python_bin" -m py_compile \
  "$project_root/scripts/run_grpo_qwen3vl_4b.py" \
  "$project_root/scripts/run_grpo_curriculum_qwen3vl_4b.py" \
  "$project_root/src/cot/grpo_training.py"

dry_run_log=$job_root/logs/dry_run.log
"$python_bin" "$project_root/scripts/run_grpo_curriculum_qwen3vl_4b.py" \
  --stage1-benchmark "$project_root/output_train/grpo_curriculum_stage1_l1_6144.json" \
  --stage2-benchmark "$project_root/output_train/grpo_curriculum_stage2_reasoning_18432.json" \
  --output-root "$output_root" \
  --prepared-root "$prepared_root" \
  --start-stage 2 \
  --stage2-resume-from-checkpoint "$stage2_checkpoint" \
  --model "$model_root" \
  --swift-bin "$swift_bin" \
  --scannetpp-image-root "$image_root" \
  --devices "$devices" \
  --stage2-epochs 1 \
  --stage2-learning-rate 5e-6 \
  --optim adamw_torch \
  --per-device-batch-size 2 \
  --gradient-accumulation-steps 4 \
  --num-generations 8 \
  --lora-rank 16 \
  --lora-alpha 32 \
  --lora-dropout 0.05 \
  --lora-dtype bfloat16 \
  --vllm-tensor-parallel-size 1 \
  --vllm-gpu-memory-utilization 0.45 \
  --vllm-max-model-len 10240 \
  --vllm-mm-processor-cache-gb 0 \
  --max-length 8192 \
  --max-completion-length 1024 \
  --max-pixels 786432 \
  --warmup-ratio 0.03 \
  --max-grad-norm 0.5 \
  --temperature 1.0 \
  --top-p 1.0 \
  --beta 0.001 \
  --answer-reward-weight 1.0 \
  --format-reward-weight 0.1 \
  --deepspeed none \
  --save-steps 512 \
  --save-total-limit 40 \
  --dataloader-workers 4 \
  --dataset-workers 4 \
  --attn-impl sdpa \
  --seed 42 \
  --reuse-prepared-dataset \
  --dry-run >"$dry_run_log" 2>&1

if grep -q -- '--adapters ' "$dry_run_log"; then
  set_status failed "dry-run incorrectly preloads a policy adapter during Trainer resume"
  exit 5
fi
grep -q -- "--ref_adapters $stage1_checkpoint" "$dry_run_log"
grep -q -- "--resume_from_checkpoint $stage2_checkpoint" "$dry_run_log"
grep -q -- '--reward_funcs psr_answer psr_format' "$dry_run_log"
grep -q -- '--reward_weights 1.0 0.1' "$dry_run_log"
if grep -q -- 'PSR_RESUME_ADAM_STATE_DTYPE' "$dry_run_log"; then
  set_status failed "stage 2 dry-run contains the stage-1 optimizer dtype repair"
  exit 6
fi

stop_occupier
occupier_stopped=1

set_status cuda_smoke "testing BF16 CUDA and two-rank NCCL on physical GPUs $devices"
export CUDA_VISIBLE_DEVICES="$devices"
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
"$python_bin" - <<'PY'
import torch

assert torch.cuda.device_count() == 2, torch.cuda.device_count()
for index in range(2):
    device = torch.device("cuda", index)
    left = torch.randn((1024, 1024), device=device, dtype=torch.bfloat16)
    result = left @ left
    torch.cuda.synchronize(device)
    assert torch.isfinite(result).all()
print(torch.__version__, torch.version.cuda, [torch.cuda.get_device_name(i) for i in range(2)])
PY

if ! timeout -k 10s 90s "$python_bin" -m torch.distributed.run \
  --standalone --nproc_per_node=2 "$project_root/scripts/nccl_smoke.py"; then
  echo "[$(timestamp)] Normal NCCL smoke failed; retrying with P2P and IB disabled."
  export NCCL_P2P_DISABLE=1
  export NCCL_IB_DISABLE=1
  timeout -k 10s 90s "$python_bin" -m torch.distributed.run \
    --standalone --nproc_per_node=2 "$project_root/scripts/nccl_smoke.py"
fi

set_status running "stage 2 restoring checkpoint-8192 on physical GPUs $devices"
cd "$project_root"
command=(
  "$python_bin" scripts/run_grpo_curriculum_qwen3vl_4b.py
  --stage1-benchmark output_train/grpo_curriculum_stage1_l1_6144.json
  --stage2-benchmark output_train/grpo_curriculum_stage2_reasoning_18432.json
  --output-root "$output_root"
  --prepared-root "$prepared_root"
  --start-stage 2
  --stage2-resume-from-checkpoint "$stage2_checkpoint"
  --model "$model_root"
  --swift-bin "$swift_bin"
  --scannetpp-image-root "$image_root"
  --devices "$devices"
  --stage2-epochs 1
  --stage2-learning-rate 5e-6
  --optim adamw_torch
  --per-device-batch-size 2
  --gradient-accumulation-steps 4
  --num-generations 8
  --lora-rank 16
  --lora-alpha 32
  --lora-dropout 0.05
  --lora-dtype bfloat16
  --vllm-tensor-parallel-size 1
  --vllm-gpu-memory-utilization 0.45
  --vllm-max-model-len 10240
  --vllm-mm-processor-cache-gb 0
  --max-length 8192
  --max-completion-length 1024
  --max-pixels 786432
  --warmup-ratio 0.03
  --max-grad-norm 0.5
  --temperature 1.0
  --top-p 1.0
  --beta 0.001
  --answer-reward-weight 1.0
  --format-reward-weight 0.1
  --deepspeed none
  --save-steps 512
  --save-total-limit 40
  --dataloader-workers 4
  --dataset-workers 4
  --attn-impl sdpa
  --seed 42
  --reuse-prepared-dataset
)

setsid "${command[@]}" &
grpo_pid=$!
set +e
wait "$grpo_pid"
status=$?
set -e
grpo_pid=
if [[ $status -ne 0 ]]; then
  exit "$status"
fi
finished=1
