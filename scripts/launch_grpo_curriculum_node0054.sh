#!/usr/bin/env bash
set -euo pipefail

project_root=${PROJECT_ROOT:-/data/home/sujinyue/mybenchmark}
job_root=${JOB_ROOT:-/data/home/sujinyue/codex_jobs/grpo_curriculum_node0054_20260807}
output_root=${OUTPUT_ROOT:-$project_root/grpo/qwen3vl_4b_grpo_curriculum_node0054_gpu45}
prepared_root=${PREPARED_ROOT:-$project_root/grpo/prepared}
resume_checkpoint=${RESUME_CHECKPOINT:-$project_root/grpo/recovery/checkpoint-1024}
image_root=${SCANNETPP_IMAGE_ROOT:-/data/home/sujinyue/datasets/scannetpp/train/iphone_frames}
transfer_marker=${TRANSFER_MARKER:-$project_root/migration/modelscope_transfer.complete}
python_bin=${PYTHON_BIN:-/data/home/sujinyue/venvs/grpo/bin/python}
swift_bin=${SWIFT_BIN:-/data/home/sujinyue/venvs/grpo/bin/swift}
base_model=${BASE_MODEL:-/data/model/Qwen3-VL-4B-Instruct}
devices=${DEVICES:-4,5}
occupier_memory_gb=${OCCUPIER_MEMORY_GB:-45}

launcher_log=$job_root/logs/launcher.log
status_file=$job_root/status.txt
occupier_log=$job_root/logs/occupier_after_grpo.log

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

find_occupiers() {
  pgrep -u "$(id -u)" -f "[s]ft_qwen3vl_4B.py.*--memory-gb $occupier_memory_gb" || true
}

stop_occupier() {
  local pid=$1
  kill -0 "$pid" 2>/dev/null || return
  echo "[$(timestamp)] GRPO is ready for CUDA; stopping SFT occupier PID $pid."
  kill -TERM "$pid"
  for _ in $(seq 1 60); do
    kill -0 "$pid" 2>/dev/null || return 0
    sleep 1
  done
  echo "[$(timestamp)] Occupier did not stop after 60 seconds; sending SIGKILL."
  kill -KILL "$pid"
}

restart_occupier() {
  if pgrep -u "$(id -u)" -f "[s]ft_qwen3vl_4B.py.*--memory-gb $occupier_memory_gb" >/dev/null; then
    echo "[$(timestamp)] SFT occupier is already running."
    return
  fi
  echo "[$(timestamp)] Restarting SFT occupier on CUDA_VISIBLE_DEVICES=$devices."
  cd "$project_root" || return
  nohup env CUDA_VISIBLE_DEVICES="$devices" "$python_bin" \
    sft_qwen3vl_4B.py \
    --model-name-or-path "$base_model" \
    --dataset-path data/train.jsonl \
    --output-dir output/qwen3-vl-4b-sft \
    --num-train-epochs 3 \
    --per-device-train-batch-size 1 \
    --gradient-accumulation-steps 8 \
    --learning-rate 2e-5 \
    --max-seq-length 4096 \
    --lora-r 64 \
    --lora-alpha 128 \
    --bf16 \
    --gradient-checkpointing \
    --gpus all \
    --memory-gb "$occupier_memory_gb" >"$occupier_log" 2>&1 </dev/null &
  echo "[$(timestamp)] SFT occupier PID: $!"
}

curriculum_finished=0
occupier_was_stopped=0
interrupted_signal=
grpo_pid=

on_exit() {
  local status=$?
  trap - EXIT TERM INT HUP
  if [[ -n $grpo_pid ]] && kill -0 "$grpo_pid" 2>/dev/null; then
    kill -TERM -- "-$grpo_pid" 2>/dev/null || true
  fi
  if [[ $curriculum_finished -eq 1 && $status -eq 0 ]]; then
    set_status complete "both GRPO curriculum stages finished"
  elif [[ -n $interrupted_signal ]]; then
    set_status interrupted "launcher received $interrupted_signal"
  else
    set_status failed "launcher exit code $status"
  fi
  if [[ $occupier_was_stopped -eq 1 ]]; then
    restart_occupier
  fi
  exit "$status"
}

on_signal() {
  interrupted_signal=$1
  exit 143
}

trap on_exit EXIT
trap 'on_signal SIGTERM' TERM
trap 'on_signal SIGINT' INT
trap 'on_signal SIGHUP' HUP

if pgrep -u "$(id -u)" -f '[r]un_grpo_curriculum_qwen3vl_4b.py.*qwen3vl_4b_grpo_curriculum_node0054_gpu45' >/dev/null; then
  set_status blocked "another matching GRPO curriculum process is running"
  exit 3
fi

set_status waiting_transfer "waiting for encrypted ModelScope image download"
while [[ ! -f $transfer_marker ]]; do
  sleep 30
done

for required in \
  "$resume_checkpoint/adapter_model.safetensors" \
  "$resume_checkpoint/optimizer.pt" \
  "$resume_checkpoint/scheduler.pt" \
  "$resume_checkpoint/trainer_state.json" \
  "$prepared_root/stage1_l1_6144.grpo.jsonl" \
  "$prepared_root/stage2_reasoning_18432.grpo.jsonl" \
  "$base_model/config.json" \
  "$base_model/model.safetensors.index.json"; do
  if [[ ! -f $required ]]; then
    set_status failed "required file is missing: $required"
    exit 4
  fi
done

set_status validating "verifying prepared rows and every referenced image"
"$python_bin" - "$prepared_root" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
expected = {
    "stage1_l1_6144.grpo.jsonl": 6144,
    "stage2_reasoning_18432.grpo.jsonl": 18432,
}
images = set()
for name, expected_rows in expected.items():
    path = root / name
    rows = 0
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            rows += 1
            for image in row.get("images", []):
                image_path = Path(image)
                if not image_path.is_file():
                    raise FileNotFoundError(f"{path}:{line_number}: {image_path}")
                images.add(image_path)
    if rows != expected_rows:
        raise ValueError(f"{path}: expected {expected_rows} rows, found {rows}")
if len(images) != 6129:
    raise ValueError(f"expected 6129 unique images, found {len(images)}")
print(f"Verified {sum(expected.values())} rows and {len(images)} unique images")
PY

"$python_bin" -m py_compile \
  "$project_root/scripts/run_grpo_qwen3vl_4b.py" \
  "$project_root/scripts/run_grpo_curriculum_qwen3vl_4b.py" \
  "$project_root/src/cot/grpo_training.py"

mapfile -t occupier_pids < <(find_occupiers)
for occupier_pid in "${occupier_pids[@]}"; do
  occupier_was_stopped=1
  stop_occupier "$occupier_pid"
done

set_status nccl_smoke "testing normal NCCL P2P/NVLink on physical GPUs $devices"
export CUDA_VISIBLE_DEVICES="$devices"
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
if ! timeout -k 10s 90s "$python_bin" -m torch.distributed.run \
  --standalone --nproc_per_node=2 "$project_root/scripts/nccl_smoke.py"; then
  echo "[$(timestamp)] Normal NCCL P2P smoke failed; retrying with P2P and IB disabled."
  export NCCL_P2P_DISABLE=1
  export NCCL_IB_DISABLE=1
  timeout -k 10s 90s "$python_bin" -m torch.distributed.run \
    --standalone --nproc_per_node=2 "$project_root/scripts/nccl_smoke.py"
fi

set_status running "stage 1 resumes from checkpoint-1024; stage 2 starts automatically"
cd "$project_root"
command=(
  "$python_bin" scripts/run_grpo_curriculum_qwen3vl_4b.py
  --stage1-benchmark output_train/grpo_curriculum_stage1_l1_6144.json
  --stage2-benchmark output_train/grpo_curriculum_stage2_reasoning_18432.json
  --output-root "$output_root"
  --prepared-root "$prepared_root"
  --stage1-resume-from-checkpoint "$resume_checkpoint"
  --model "$base_model"
  --swift-bin "$swift_bin"
  --scannetpp-image-root "$image_root"
  --scannetpp-sensor iphone
  --devices "$devices"
  --stage1-epochs 1
  --stage2-epochs 1
  --stage1-learning-rate 1e-5
  --stage2-learning-rate 5e-6
  --optim adamw_torch
  --resume-optimizer-state-dtype bfloat16
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

curriculum_finished=1
