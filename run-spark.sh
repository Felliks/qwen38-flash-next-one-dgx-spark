#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
readonly SCRIPT_DIR
readonly DATA_DIR="${QWEN_NEXT_DATA_DIR:-${SCRIPT_DIR}/data}"
readonly HF_CACHE="${HF_CACHE:-${DATA_DIR}/huggingface}"
readonly SGLANG_CACHE="${SGLANG_CACHE:-${DATA_DIR}/sglang}"
readonly PLE_CACHE="${PLE_CACHE:-${DATA_DIR}/ple-cache}"
readonly MODEL_REPO="RadixArk/Qwen3.8-Flash-Next-NVFP4"
readonly MODEL_REVISION="7b719225242aacd3dbd3f9407468c2ee9a9d2594"
readonly SNAPSHOT="${HF_CACHE}/hub/models--RadixArk--Qwen3.8-Flash-Next-NVFP4/snapshots/${MODEL_REVISION}"
readonly PLE_RAW="${PLE_CACHE}/ple-fp8.raw"
readonly IMAGE="${QWEN_NEXT_IMAGE:-qwen38-flash-next-gx10:73a255-ple-disk-v3}"
readonly CONTAINER="${QWEN_NEXT_CONTAINER:-qwen38-flash-next-nvfp4-mtp}"
readonly BIND_ADDR="${BIND_ADDR:-127.0.0.1}"
readonly API_PORT="${API_PORT:-8000}"
readonly CPUSET="${CPUSET:-0-19}"

usage() {
  printf '%s\n' \
    'Usage: ./run-spark.sh COMMAND' \
    '' \
    'Commands:' \
    '  prepare  Build the pinned SGLang image, download the model and create the 51.2 GB PLE mmap' \
    '  serve    Start Qwen3.8-Flash-Next NVFP4 + MTP in the background' \
    '  smoke    Send a small OpenAI-compatible chat request' \
    '  status   Show container, memory and GPU status' \
    '  logs     Follow SGLang logs' \
    '  stop     Stop the model cleanly'
}

require_gb10() {
  local arch compute
  arch=$(uname -m)
  compute=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1)
  [[ "$arch" == aarch64 ]] || {
    printf 'Expected aarch64, got %s\n' "$arch" >&2
    return 1
  }
  [[ "$compute" == 12.1 ]] || {
    printf 'Expected NVIDIA GB10 / SM 12.1, got compute capability %s\n' "$compute" >&2
    return 1
  }
}

prepare() {
  require_gb10
  mkdir -p "$HF_CACHE" "$SGLANG_CACHE" "$PLE_CACHE"

  printf 'Building pinned SGLang + GX10 PLE/QSA overlay...\n'
  docker build --network host -t "$IMAGE" "$SCRIPT_DIR"

  printf 'Downloading %s at pinned revision %s...\n' "$MODEL_REPO" "$MODEL_REVISION"
  local hf_env=()
  if [[ -n "${HF_TOKEN:-}" ]]; then
    hf_env=(-e "HF_TOKEN=${HF_TOKEN}")
  fi
  docker run --rm --memory 24g --memory-swap 24g \
    -v "${HF_CACHE}:/root/.cache/huggingface" \
    -e HF_XET_HIGH_PERFORMANCE=1 \
    -e HF_XET_NUM_CONCURRENT_RANGE_GETS=8 \
    "${hf_env[@]}" \
    --entrypoint python3 "$IMAGE" -c \
    "from huggingface_hub import snapshot_download; print(snapshot_download(repo_id='${MODEL_REPO}', revision='${MODEL_REVISION}', max_workers=4))"

  if [[ -f "$PLE_RAW" && -f "${PLE_RAW}.json" ]]; then
    printf 'Verifying existing PLE mmap...\n'
    docker run --rm --memory 8g --memory-swap 8g \
      -v "${HF_CACHE}:/root/.cache/huggingface:ro" \
      -v "${PLE_CACHE}:/ple-cache" \
      --entrypoint python3 "$IMAGE" /opt/gxai/prepare_ple.py \
      --snapshot "/root/.cache/huggingface/hub/models--RadixArk--Qwen3.8-Flash-Next-NVFP4/snapshots/${MODEL_REVISION}" \
      --output /ple-cache/ple-fp8.raw \
      --revision "$MODEL_REVISION" \
      --verify-only \
      --verify-samples 256
  else
    printf 'Creating the verified 51.2 GB FP8 PLE mmap. This takes a while...\n'
    docker run --rm --memory 8g --memory-swap 8g \
      -v "${HF_CACHE}:/root/.cache/huggingface:ro" \
      -v "${PLE_CACHE}:/ple-cache" \
      --entrypoint python3 "$IMAGE" /opt/gxai/prepare_ple.py \
      --snapshot "/root/.cache/huggingface/hub/models--RadixArk--Qwen3.8-Flash-Next-NVFP4/snapshots/${MODEL_REVISION}" \
      --output /ple-cache/ple-fp8.raw \
      --revision "$MODEL_REVISION" \
      --verify-samples 256
  fi

  printf 'Preparation complete. Run: ./run-spark.sh serve\n'
}

require_prepared() {
  docker image inspect "$IMAGE" >/dev/null
  [[ -f "${SNAPSHOT}/config.json" ]] || {
    printf 'Pinned model snapshot not found. Run prepare first.\n' >&2
    return 1
  }
  [[ -f "$PLE_RAW" && -f "${PLE_RAW}.json" ]] || {
    printf 'PLE mmap not found. Run prepare first.\n' >&2
    return 1
  }
}

serve() {
  require_gb10
  require_prepared
  if docker container inspect "$CONTAINER" >/dev/null 2>&1; then
    printf 'Container %s already exists. Stop it first.\n' "$CONTAINER" >&2
    return 1
  fi

  docker run -d --rm --init \
    --name "$CONTAINER" \
    --gpus all \
    --ipc host \
    --shm-size 16g \
    --memory 116g \
    --memory-swap 116g \
    --cpuset-cpus "$CPUSET" \
    -p "${BIND_ADDR}:${API_PORT}:30000" \
    -v "${HF_CACHE}:/root/.cache/huggingface" \
    -v "${SGLANG_CACHE}:/root/.cache/sglang" \
    -v "${PLE_CACHE}:/ple-cache:ro" \
    -e SGLANG_QWEN4_PLE_DISK_CACHE_PATH=/ple-cache/ple-fp8.raw \
    -e SGLANG_QWEN4_PLE_DISK_CACHE_MANIFEST=/ple-cache/ple-fp8.raw.json \
    -e SGLANG_QWEN4_PLE_SOURCE_REVISION="$MODEL_REVISION" \
    -e SGLANG_QWEN4_PLE_CACHE_BYTES=268435456 \
    -e SGLANG_QWEN4_PLE_CACHE_MAX_LOOKUP_ROWS=4096 \
    "$IMAGE" \
    sglang serve \
    --model-path "$MODEL_REPO" \
    --revision "$MODEL_REVISION" \
    --served-model-name qwen38-flash-next-nvfp4-mtp \
    --trust-remote-code \
    --host 0.0.0.0 \
    --port 30000 \
    --quantization modelopt_fp4 \
    --fp4-gemm-backend flashinfer_cutlass \
    --page-size 64 \
    --mamba-radix-cache-strategy extra_buffer \
    --mamba-track-interval 64 \
    --max-mamba-cache-size 20 \
    --mamba-ssm-dtype float32 \
    --chunked-prefill-size 4096 \
    --max-running-requests 4 \
    --max-total-tokens 524288 \
    --context-length 262144 \
    --mem-fraction-static 0.95 \
    --allow-auto-truncate \
    --ple-offload-embedding \
    --reasoning-parser qwen3 \
    --tool-call-parser qwen3_coder \
    --preferred-sampling-params '{"temperature":1.0,"top_p":0.95,"top_k":20,"min_p":0.0,"presence_penalty":0.0,"repetition_penalty":1.0}' \
    --disable-prefill-cuda-graph \
    --cuda-graph-backend-decode disabled \
    --disable-flashinfer-autotune \
    --speculative-algorithm NEXTN \
    --speculative-num-steps 3 \
    --speculative-eagle-topk 1 \
    --speculative-num-draft-tokens 4

  printf 'Starting. First load can take about 10 minutes. Follow it with: ./run-spark.sh logs\n'
  printf 'API when ready: http://%s:%s/v1\n' "$BIND_ADDR" "$API_PORT"
}

smoke() {
  curl --fail --silent --show-error \
    "http://${BIND_ADDR}:${API_PORT}/v1/chat/completions" \
    -H 'Content-Type: application/json' \
    -d '{"model":"qwen38-flash-next-nvfp4-mtp","messages":[{"role":"user","content":"Reply with exactly: GX10 OK"}],"max_tokens":32,"temperature":0}'
  printf '\n'
}

status() {
  docker ps --filter "name=^/${CONTAINER}$" \
    --format 'table {{.Names}}\t{{.Status}}\t{{.Ports}}'
  free -h
  nvidia-smi
}

case "${1:-}" in
  prepare) prepare ;;
  serve) serve ;;
  smoke) smoke ;;
  status) status ;;
  logs) docker logs --follow "$CONTAINER" ;;
  stop) docker stop -t 180 "$CONTAINER" ;;
  *) usage; exit 2 ;;
esac
