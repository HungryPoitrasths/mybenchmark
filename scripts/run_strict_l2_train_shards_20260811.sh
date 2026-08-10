#!/usr/bin/env bash
set -euo pipefail

repo_root="${1:-/home/sujinyue/mybenchmark}"
python_bin="${PYTHON_BIN:-python}"
data_root="/data/zju-151/scannet/data"
frame_root="/home/sujinyue/datasets/scannetpp/train/iphone_frames"
log_root="${repo_root}/output_train/scannetpp_polit/logs"

cd "${repo_root}"
mkdir -p "${log_root}"

for shard in 40-49 50-59 60-69 70-79 80-89 90-99; do
    output_dir="output_train/scannetpp_polit/${shard}l2strict"
    cache="output_train/scannetpp_flash/${shard}/scene_status.json"
    log="${log_root}/${shard}l2strict.log"

    if [[ ! -f "${cache}" ]]; then
        echo "Missing referability cache: ${cache}" >&2
        exit 1
    fi
    if [[ -e "${output_dir}" ]]; then
        echo "Refusing to overwrite existing output: ${output_dir}" >&2
        exit 1
    fi

    echo "[$(date -Iseconds)] starting ${shard}" | tee "${log}"
    "${python_bin}" scripts/run_pipeline.py \
        --dataset scannetpp \
        --scannetpp_sensor iphone \
        --split train \
        --data_root "${data_root}" \
        --scannetpp_frame_root "${frame_root}" \
        --output_dir "${output_dir}" \
        --referability_cache "${cache}" \
        --no_salvage \
        --max_frames 100 \
        --skip_question_vlm_check \
        --repair_referability_cache \
        --only_question_types \
        L2_object_move_agent \
        L2_object_move_distance \
        L2_object_move_occlusion \
        L2_object_rotate_object_centric \
        L2_object_move_allocentric \
        L2_object_remove \
        >> "${log}" 2>&1
    echo "[$(date -Iseconds)] completed ${shard}" | tee -a "${log}"
done
