#!/usr/bin/env python3
"""Add a bounded, disk-backed Qwen4 PLE embedding to a pinned SGLang tree.

The patch is intentionally source-hash guarded.  If the upstream Qwen4
implementation changes, the image build stops instead of silently applying a
possibly incorrect transformation.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path


EXPECTED_SHA256 = "f406977eb2373937393241f453477867f7dc943bd4839216db8fe66fa9f921d8"
MARKER = "class Qwen4ExpDiskCachedEmbedding"


DISK_EMBEDDING = r'''
class Qwen4ExpDiskCachedEmbedding(VocabParallelEmbedding):
    """PLE table backed by an mmap file with a bounded pinned row cache.

    Qwen3.8-Flash-Next has a 51.2 GB FP8 PLE table.  Keeping that table in
    pinned host RAM still consumes the DGX Spark's unified memory.  This class
    keeps the complete table on NVMe, caches decode-sized working sets in
    pinned RAM, and stages only requested FP8 rows to CUDA before converting
    them to BF16.  The checkpoint's scalar ``weight_scale`` is applied by the
    existing PLE path after gather, exactly as for the stock implementation.
    """

    _COPIED_ATTRIBUTES = Qwen4ExpPinnedHostEmbedding._COPIED_ATTRIBUTES

    def __init__(self, embedding: VocabParallelEmbedding) -> None:
        nn.Module.__init__(self)
        if not isinstance(embedding.quant_method, UnquantizedEmbeddingMethod):
            raise NotImplementedError(
                "disk-backed PLE requires an unquantized embedding table"
            )
        if embedding.weight.dtype != torch.float8_e4m3fn:
            raise TypeError(
                "disk-backed PLE requires FP8 E4M3 checkpoint storage, got "
                f"{embedding.weight.dtype}"
            )
        if embedding.tp_size != 1:
            raise NotImplementedError(
                "disk-backed PLE is currently restricted to TP=1"
            )
        if embedding.num_added_embeddings:
            raise NotImplementedError(
                "disk-backed PLE does not support added vocabulary rows"
            )

        for name in self._COPIED_ATTRIBUTES:
            setattr(self, name, getattr(embedding, name))
        self.quant_method = None
        source_weight = embedding.weight
        self._rows = int(source_weight.shape[0])
        self._storage_dtype = source_weight.dtype
        self._raw_path = os.environ["SGLANG_QWEN4_PLE_DISK_CACHE_PATH"]
        self._manifest_path = os.environ.get(
            "SGLANG_QWEN4_PLE_DISK_CACHE_MANIFEST", self._raw_path + ".json"
        )
        self._cache_bytes = int(
            os.environ.get("SGLANG_QWEN4_PLE_CACHE_BYTES", str(1 << 30))
        )
        self._cache_max_lookup_rows = int(
            os.environ.get("SGLANG_QWEN4_PLE_CACHE_MAX_LOOKUP_ROWS", "4096")
        )
        self._log_interval = int(
            os.environ.get("SGLANG_QWEN4_PLE_LOG_INTERVAL", "256")
        )

        # Keep a named parameter so SGLang's generic loader retains its normal
        # parameter topology. PLE shards are intercepted in load_weights and
        # validated, but are not copied into this empty placeholder.
        placeholder = nn.Parameter(
            torch.empty(
                (0, self.embedding_dim),
                dtype=source_weight.dtype,
                device="cpu",
            ),
            requires_grad=False,
        )
        for name, value in vars(source_weight).items():
            setattr(placeholder, name, value)
        placeholder.weight_loader = self.weight_loader
        self.register_parameter("weight", placeholder)
        self.register_buffer("weight_scale", embedding.weight_scale, persistent=True)
        del embedding.weight

        self._mapped = None
        self._fd = None
        self._cache = None
        self._cache_tags = None
        self._cache_cursor = None
        self._cache_sets = 0
        self._cache_ways = 4
        self._calls = 0
        self._hits = 0
        self._lookups = 0
        self._latency_ms = []

    def _ensure_open(self) -> None:
        if self._mapped is not None:
            return
        manifest = json.loads(Path(self._manifest_path).read_text())
        expected = {
            "dtype": "float8_e4m3fn",
            "embedding_dim": self.embedding_dim,
            "rows": self._rows,
        }
        actual = {key: manifest.get(key) for key in expected}
        if actual != expected:
            raise RuntimeError(
                f"PLE cache manifest mismatch: expected {expected}, got {actual}"
            )
        expected_revision = os.environ.get("SGLANG_QWEN4_PLE_SOURCE_REVISION")
        if expected_revision and manifest.get("revision") != expected_revision:
            raise RuntimeError(
                "PLE cache revision mismatch: "
                f"expected {expected_revision}, got {manifest.get('revision')}"
            )
        expected_bytes = self._rows * self.embedding_dim
        actual_bytes = os.path.getsize(self._raw_path)
        if actual_bytes != expected_bytes:
            raise RuntimeError(
                f"PLE cache size mismatch: {actual_bytes} != {expected_bytes}"
            )
        self._mapped = torch.from_file(
            self._raw_path,
            shared=False,
            size=expected_bytes,
            dtype=torch.uint8,
        ).view(self._rows, self.embedding_dim)
        self._fd = os.open(self._raw_path, os.O_RDONLY)
        if hasattr(os, "posix_fadvise"):
            os.posix_fadvise(self._fd, 0, 0, os.POSIX_FADV_RANDOM)

        cache_rows = self._cache_bytes // self.embedding_dim
        cache_rows -= cache_rows % self._cache_ways
        if cache_rows >= self._cache_ways:
            self._cache_sets = cache_rows // self._cache_ways
            self._cache = torch.empty(
                (cache_rows, self.embedding_dim),
                dtype=torch.uint8,
                device="cpu",
                pin_memory=True,
            )
            self._cache_tags = torch.full(
                (self._cache_sets, self._cache_ways),
                -1,
                dtype=torch.int64,
            )
            self._cache_cursor = torch.zeros(
                self._cache_sets,
                dtype=torch.uint8,
            )
        logger.info(
            "Qwen4 PLE disk cache opened: path=%s rows=%d dim=%d cache_bytes=%d",
            self._raw_path,
            self._rows,
            self.embedding_dim,
            0 if self._cache is None else self._cache.numel(),
        )

    def validate_shard(
        self, loaded_weight: torch.Tensor, row_start: int, row_end: int
    ) -> None:
        if loaded_weight.dtype != self._storage_dtype:
            raise TypeError(
                f"PLE shard dtype mismatch: {loaded_weight.dtype} != "
                f"{self._storage_dtype}"
            )
        if loaded_weight.ndim != 2 or loaded_weight.shape[1] != self.embedding_dim:
            raise ValueError(
                "PLE shard shape mismatch: "
                f"{tuple(loaded_weight.shape)}, dim={self.embedding_dim}"
            )
        if row_end - row_start != loaded_weight.shape[0] or row_end > self._rows:
            raise ValueError(
                f"PLE shard range mismatch: [{row_start}, {row_end}) for "
                f"{tuple(loaded_weight.shape)} and {self._rows} rows"
            )

    def allocate_output(
        self, shape: Tuple[int, ...], device: torch.device
    ) -> torch.Tensor:
        return torch.empty(shape, dtype=torch.bfloat16, device=device)

    def _gather_cached(self, ids: torch.Tensor, stage: torch.Tensor) -> None:
        assert self._cache is not None
        assert self._cache_tags is not None
        assert self._cache_cursor is not None
        for output_row, row_id in enumerate(ids.tolist()):
            set_id = row_id % self._cache_sets
            tags = self._cache_tags[set_id]
            hit = (tags == row_id).nonzero()
            if hit.numel():
                way = int(hit[0])
                self._hits += 1
            else:
                way = int(self._cache_cursor[set_id])
                slot = set_id * self._cache_ways + way
                self._cache[slot].copy_(self._mapped[row_id])
                tags[way] = row_id
                self._cache_cursor[set_id] = (way + 1) % self._cache_ways
            slot = set_id * self._cache_ways + way
            stage[output_row].copy_(self._cache[slot])

    def _report_stats(self, elapsed_ms: float, lookups: int) -> None:
        self._calls += 1
        self._lookups += lookups
        self._latency_ms.append(elapsed_ms)
        if len(self._latency_ms) > 2048:
            del self._latency_ms[:1024]
        if self._log_interval <= 0 or self._calls % self._log_interval:
            return
        ordered = sorted(self._latency_ms)
        p95 = ordered[min(len(ordered) - 1, int(len(ordered) * 0.95))]
        hit_rate = self._hits / self._lookups if self._lookups else 0.0
        logger.info(
            "Qwen4 PLE disk cache stats: calls=%d lookups=%d hit_rate=%.4f "
            "last_ms=%.3f p95_ms=%.3f",
            self._calls,
            self._lookups,
            hit_rate,
            elapsed_ms,
            p95,
        )

    def gather(
        self, input_ids: torch.Tensor, out: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        expected_shape = (*input_ids.shape, self.embedding_dim)
        if out is None:
            output = self.allocate_output(expected_shape, input_ids.device)
        else:
            if tuple(out.shape) != expected_shape:
                raise ValueError(
                    f"invalid PLE prefetch output shape: {tuple(out.shape)} != "
                    f"{expected_shape}"
                )
            if out.dtype != torch.bfloat16 or out.device != input_ids.device:
                raise ValueError(
                    "PLE prefetch output must be bfloat16 on the id device"
                )
            output = out

        flat_ids = input_ids.reshape(-1).long()
        if not flat_ids.numel():
            return output
        self._ensure_open()
        started = time.perf_counter()
        host_ids = torch.empty(
            flat_ids.shape,
            dtype=torch.int64,
            device="cpu",
            pin_memory=True,
        )
        host_ids.copy_(flat_ids, non_blocking=True)
        torch.cuda.current_stream().synchronize()
        low, high = torch.aminmax(host_ids)
        if int(low) < 0 or int(high) >= self._rows:
            raise IndexError(
                f"PLE row outside [0, {self._rows}): [{int(low)}, {int(high)}]"
            )

        stage = torch.empty(
            (host_ids.numel(), self.embedding_dim),
            dtype=torch.uint8,
            device="cpu",
            pin_memory=True,
        )
        if self._cache is not None and host_ids.numel() <= self._cache_max_lookup_rows:
            self._gather_cached(host_ids, stage)
        else:
            torch.index_select(self._mapped, 0, host_ids, out=stage)

        gpu_stage = torch.empty(
            stage.shape,
            dtype=self._storage_dtype,
            device=input_ids.device,
        )
        gpu_stage.copy_(stage.view(self._storage_dtype), non_blocking=True)
        output.view(-1, self.embedding_dim).copy_(gpu_stage)
        current_stream = torch.cuda.current_stream()
        gpu_stage.record_stream(current_stream)
        # This ARM64 PyTorch build has no CPU record_stream kernel. More
        # importantly, disk-fed rows cannot be frozen into a decode CUDA graph.
        # The launch config therefore uses eager decode, and this stream-local
        # barrier keeps pinned staging buffers alive until both copies finish.
        current_stream.synchronize()
        self._report_stats(
            (time.perf_counter() - started) * 1000.0,
            host_ids.numel(),
        )
        if (
            host_ids.numel() > self._cache_max_lookup_rows
            and self._fd is not None
            and hasattr(os, "posix_fadvise")
        ):
            os.posix_fadvise(self._fd, 0, 0, os.POSIX_FADV_DONTNEED)
        return output

    def reduce(self, output: torch.Tensor) -> torch.Tensor:
        return output

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.gather(input_ids)


'''


def replace_once(source: str, old: str, new: str, label: str) -> str:
    count = source.count(old)
    if count != 1:
        raise RuntimeError(f"expected one {label}, found {count}")
    return source.replace(old, new, 1)


def patch(path: Path) -> None:
    source = path.read_text()
    if MARKER in source:
        print(f"already patched: {path}")
        return
    digest = hashlib.sha256(source.encode()).hexdigest()
    if digest != EXPECTED_SHA256:
        raise RuntimeError(
            f"refusing to patch {path}: sha256 {digest} != {EXPECTED_SHA256}"
        )

    source = replace_once(
        source,
        "import math\n",
        "import json\nimport math\nimport os\nimport time\nfrom pathlib import Path\n",
        "import block",
    )
    source = replace_once(
        source,
        "class Qwen4ExpPinnedHostEmbedding(VocabParallelEmbedding):\n",
        "class Qwen4ExpPinnedHostEmbedding(VocabParallelEmbedding):\n",
        "pinned embedding class",
    )
    # The disk class reuses the stock class' copied-attribute contract, so it
    # must be inserted after the complete pinned class and before PLELayer.
    source = replace_once(
        source,
        "\n\nclass Qwen4ExpPLELayer(nn.Module):\n",
        "\n\n" + DISK_EMBEDDING + "class Qwen4ExpPLELayer(nn.Module):\n",
        "PLE layer boundary",
    )
    source = replace_once(
        source,
        '''        if config.ple_offload_embedding:\n            self.ple_embedding.ngram_embedding = Qwen4ExpPinnedHostEmbedding(\n                self.ple_embedding.ngram_embedding\n            )\n''',
        '''        if config.ple_offload_embedding:\n            embedding_cls = (\n                Qwen4ExpDiskCachedEmbedding\n                if os.environ.get("SGLANG_QWEN4_PLE_DISK_CACHE_PATH")\n                else Qwen4ExpPinnedHostEmbedding\n            )\n            self.ple_embedding.ngram_embedding = embedding_cls(\n                self.ple_embedding.ngram_embedding\n            )\n''',
        "PLE offload constructor",
    )
    source = replace_once(
        source,
        '''            if ov_start < ov_end:\n                local_start = ov_start - tp_start\n                src_start = ov_start - row_start\n                n_rows = ov_end - ov_start\n                emb.weight.data[local_start : local_start + n_rows].copy_(\n                    loaded_weight[src_start : src_start + n_rows].to(\n                        device=emb.weight.device, dtype=emb.weight.dtype\n                    )\n                )\n''',
        '''            if isinstance(emb, Qwen4ExpDiskCachedEmbedding):\n                emb.validate_shard(loaded_weight, row_start, row_end)\n                return\n            if ov_start < ov_end:\n                local_start = ov_start - tp_start\n                src_start = ov_start - row_start\n                n_rows = ov_end - ov_start\n                emb.weight.data[local_start : local_start + n_rows].copy_(\n                    loaded_weight[src_start : src_start + n_rows].to(\n                        device=emb.weight.device, dtype=emb.weight.dtype\n                    )\n                )\n''',
        "PLE shard loader",
    )
    source = replace_once(
        source,
        "if isinstance(emb, Qwen4ExpPinnedHostEmbedding):",
        "if isinstance(emb, (Qwen4ExpPinnedHostEmbedding, Qwen4ExpDiskCachedEmbedding)):",
        "FP8 offload type guard",
    )
    path.write_text(source)
    print(f"patched: {path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path)
    args = parser.parse_args()
    patch(args.path)


if __name__ == "__main__":
    main()
