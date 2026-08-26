FROM lmsysorg/sglang@sha256:14ed582518584c5c830206b5318a2c2769e68229c3422e48a28b952b3a888bd4

ARG SGLANG_IMAGE_COMMIT=d91c3682b0b429e4c70df63cd57f819588ce29b0
ARG SGLANG_QWEN4_SOURCE_COMMIT=73a255206f916366c8d26d4022f82ddfb0ab558d
LABEL org.opencontainers.image.title="GX10 SGLang Qwen3.8 Flash Next NVFP4"
LABEL org.opencontainers.image.revision="${SGLANG_IMAGE_COMMIT}"
LABEL ai.gx10.qwen4-source-revision="${SGLANG_QWEN4_SOURCE_COMMIT}"
LABEL ai.gx10.ple-cache="bounded-nvme-row-cache-v1"
LABEL ai.gx10.qsa-sm121="pytorch-sdpa-fallback-v1"

COPY patch_qwen4_ple_disk_cache.py /opt/gxai/patch_qwen4_ple_disk_cache.py
COPY patch_sm121_qsa_sdpa.py /opt/gxai/patch_sm121_qsa_sdpa.py
COPY prepare_ple.py /opt/gxai/prepare_ple.py

RUN test "$(git -C /sgl-workspace/sglang rev-parse HEAD)" = "${SGLANG_IMAGE_COMMIT}" \
    && python3 /opt/gxai/patch_qwen4_ple_disk_cache.py \
       /sgl-workspace/sglang/python/sglang/srt/models/qwen4_exp.py \
    && python3 /opt/gxai/patch_sm121_qsa_sdpa.py \
       /sgl-workspace/sglang/python/sglang/srt/layers/attention/qwen_sparse_attn_backend.py \
    && python3 -m py_compile \
       /sgl-workspace/sglang/python/sglang/srt/models/qwen4_exp.py \
       /sgl-workspace/sglang/python/sglang/srt/layers/attention/qwen_sparse_attn_backend.py \
       /opt/gxai/prepare_ple.py
