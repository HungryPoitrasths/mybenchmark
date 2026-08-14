# Classic spatial benchmark evaluation

This evaluation compares the same frozen sample and media manifest across:

- `Qwen/Qwen3-VL-4B-Instruct`
- SFT `checkpoint-768`
- GRPO `checkpoint-7680`
- GRPO `checkpoint-9216`

The full suite has 3,960 questions in 22 subsets. Scores remain separate by
benchmark; the summarizer deliberately does not produce a cross-benchmark
overall score.

| Benchmark | Included subsets | Questions |
|---|---|---:|
| MMSI-Bench | Object Motion (76), Multi-step Reasoning (198) | 274 |
| SPAR-Bench | ViewChg (50), four SpImag variants (50 each) | 250 |
| MindCube | Dynamics / what-if movement | 292 |
| VSI-Bench | Relative Direction, Relative Distance, Route Planning (50 each) | 150 |
| MVBench | Object Shuffle, Moving Direction, Egocentric Navigation (200 each) | 600 |
| CLEVRER | Explanatory, Predictive, Counterfactual (200 each) | 600 |
| BLINK | Labeled validation: Multi-view Reasoning (133), Relative Depth (124), Spatial Relation (143), Visual Correspondence (172) | 572 |
| VSR | Official zero-shot test split (1,222 questions over 731 unique images) | 1,222 |

Each of the four models therefore runs 3,960 examples, for 15,840 total
model-example inferences. VSI-Bench and CLEVRER are deterministically sampled
with seed 42; the other included subsets use their complete pinned evaluation
split. BLINK uses validation because its public test answers are hidden.

## 1. Install

Use the same environment as the Qwen3-VL checkpoints, then add the evaluation
dependencies:

```bash
pip install -r requirements-classic-spatial-eval.txt
```

`CLEVRER` must be downloaded from its official release because its videos are
not fetched automatically. Point `--clevrer-root` at a directory containing
both its validation question annotations and videos. If train, validation, and
test annotations are all present, the preparer selects files whose names contain
`val` or `validation`; ambiguous annotation layouts fail instead of mixing
splits. MVBench videos covered by additional licenses may likewise be supplied
through `--mvbench-root`.

## 2. Prepare and freeze the inputs

All paths are CLI arguments. If a root is omitted, the preparer downloads the
pinned Hugging Face revision. It writes copied images or fixed extracted video
frames under the output directory, so all four models see byte-identical media.

```bash
python scripts/prepare_classic_spatial_benchmarks.py \
  --cache-dir /data/home/sujinyue/.cache/classic_spatial_eval \
  --output-dir /data/home/sujinyue/mybenchmark/classic_eval/prepared \
  --clevrer-root /data/home/sujinyue/datasets/CLEVRER
```

For datasets already downloaded locally, add any of:

```text
--mmsi-root PATH --spar-root PATH --mindcube-root PATH --vsi-root PATH
--mvbench-root PATH --clevrer-root PATH --blink-root PATH --vsr-root PATH
```

Run one item from every subset before materializing the full suite:

```bash
python scripts/prepare_classic_spatial_benchmarks.py \
  --cache-dir /data/home/sujinyue/.cache/classic_spatial_eval \
  --output-dir /data/home/sujinyue/mybenchmark/classic_eval/smoke_prepared \
  --clevrer-root /data/home/sujinyue/datasets/CLEVRER \
  --smoke-per-subset 1
```

Preparation produces `manifest.jsonl`, `benchmark.lock.json`, and
`manifest.summary.json`. Missing or corrupt visual inputs stop preparation;
the tool never substitutes a text-only example.

VSR uses the pinned official zero-shot JSONL directly, avoiding deprecated
Hugging Face dataset scripts. It first looks for each COCO image locally and
downloads the row's pinned source URL only when the local file is absent.

## 3. Run the four models

The runner detects `adapter_config.json` and loads LoRA checkpoints on top of
`--base-model`. A full merged checkpoint can also be passed as a model spec.

```bash
python scripts/run_classic_spatial_eval.py \
  --manifest /data/home/sujinyue/mybenchmark/classic_eval/prepared/manifest.jsonl \
  --output-dir /data/home/sujinyue/mybenchmark/classic_eval/predictions \
  --base-model /data/model/Qwen3-VL-4B-Instruct \
  --model-spec base=base \
  --model-spec sft_768=/data/home/sujinyue/mybenchmark/ckpt/sft/checkpoint-768 \
  --model-spec grpo_7680=/data/home/sujinyue/mybenchmark/ckpt/grpo_stage2/checkpoint-7680 \
  --model-spec grpo_9216=/data/home/sujinyue/mybenchmark/ckpt/grpo_stage2/checkpoint-9216 \
  --devices 0,1 \
  --batch-size 1 \
  --resume
```

Use `--dry-run` to validate the manifest, media hashes, model specifications,
and shard size without loading a model. To distribute the work, launch the
same command with a shared output directory and distinct values of
`--num-shards N --shard-index I`.

For example, two independent 24 GB workers can run shard 0 and shard 1 with the
same arguments except for `--devices` and `--shard-index`:

```text
worker A: --devices 0 --num-shards 2 --shard-index 0
worker B: --devices 0 --num-shards 2 --shard-index 1
```

The prediction files can be copied into one directory before summarization, or
their separate directories can both be passed after `--predictions`.

If a model is already served by vLLM, use `--backend openai` and pass the served
model name as the model-spec path. Run one served model per invocation unless
the endpoint exposes several model names.

## 4. Summarize

```bash
python scripts/summarize_classic_spatial_eval.py \
  --manifest /data/home/sujinyue/mybenchmark/classic_eval/prepared/manifest.jsonl \
  --predictions /data/home/sujinyue/mybenchmark/classic_eval/predictions \
  --output-dir /data/home/sujinyue/mybenchmark/classic_eval/report \
  --base-label base
```

The output contains `metrics.json`, `metrics.csv`, `report.md`, and
`errors.jsonl`. Accuracy confidence intervals and checkpoint deltas use 10,000
paired, scene/video-clustered bootstrap samples with seed 42. CLEVRER reports
both exact per-question accuracy and per-option accuracy. Invalid responses are
counted as wrong and remain in the denominator.
