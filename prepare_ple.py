#!/usr/bin/env python3
"""Build and verify the raw FP8 PLE table used by the GX10 experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import shutil
from pathlib import Path

import torch
from safetensors import safe_open


TENSOR_RE = re.compile(
    r"^(?P<prefix>.+\.ple\.ple_embedding\.ngram_embedding)\."
    r"shard_(?P<index>\d+)\.weight$"
)
DTYPE = torch.float8_e4m3fn


def load_index(snapshot: Path) -> dict[str, str]:
    path = snapshot / "model.safetensors.index.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    data = json.loads(path.read_text())
    return data["weight_map"]


def discover_shards(weight_map: dict[str, str]) -> list[tuple[int, str, str]]:
    shards: list[tuple[int, str, str]] = []
    prefixes: set[str] = set()
    for name, filename in weight_map.items():
        match = TENSOR_RE.match(name)
        if match:
            prefixes.add(match.group("prefix"))
            shards.append((int(match.group("index")), name, filename))
    if len(prefixes) != 1:
        raise RuntimeError(f"expected one PLE table, found prefixes={sorted(prefixes)}")
    shards.sort()
    indexes = [index for index, _, _ in shards]
    if indexes != list(range(len(shards))):
        raise RuntimeError(f"PLE shard indexes are not contiguous: {indexes}")
    return shards


def tensor_shape(snapshot: Path, name: str, filename: str) -> tuple[int, ...]:
    with safe_open(snapshot / filename, framework="pt", device="cpu") as handle:
        return tuple(handle.get_slice(name).get_shape())


def inspect_layout(
    snapshot: Path, shards: list[tuple[int, str, str]]
) -> tuple[int, int, list[int]]:
    rows_per_shard: list[int] = []
    embedding_dim: int | None = None
    for _, name, filename in shards:
        shape = tensor_shape(snapshot, name, filename)
        if len(shape) != 2:
            raise RuntimeError(f"{name} has invalid shape {shape}")
        if embedding_dim is None:
            embedding_dim = shape[1]
        elif shape[1] != embedding_dim:
            raise RuntimeError(f"{name} has inconsistent shape {shape}")
        rows_per_shard.append(shape[0])
    assert embedding_dim is not None
    return sum(rows_per_shard), embedding_dim, rows_per_shard


def available_bytes(path: Path) -> int:
    return shutil.disk_usage(path.parent).free


def build(
    snapshot: Path,
    output: Path,
    revision: str,
    verify_samples: int,
    force: bool,
) -> dict:
    weight_map = load_index(snapshot)
    shards = discover_shards(weight_map)
    rows, embedding_dim, rows_per_shard = inspect_layout(snapshot, shards)
    expected_bytes = rows * embedding_dim
    manifest_path = Path(str(output) + ".json")

    if output.exists() or manifest_path.exists():
        if not force:
            raise FileExistsError(
                f"{output} or {manifest_path} already exists; use --force"
            )
    output.parent.mkdir(parents=True, exist_ok=True)
    if available_bytes(output) < expected_bytes + (8 << 30):
        raise RuntimeError(
            f"insufficient disk: need {expected_bytes + (8 << 30)} bytes free"
        )

    partial = Path(str(output) + ".partial")
    partial.unlink(missing_ok=True)
    digest = hashlib.sha256()
    offset_rows = 0
    try:
        with partial.open("wb", buffering=0) as destination:
            for (index, name, filename), shard_rows in zip(shards, rows_per_shard):
                with safe_open(
                    snapshot / filename, framework="pt", device="cpu"
                ) as handle:
                    tensor = handle.get_tensor(name)
                if tensor.dtype != DTYPE:
                    raise TypeError(f"{name}: expected {DTYPE}, got {tensor.dtype}")
                if tuple(tensor.shape) != (shard_rows, embedding_dim):
                    raise RuntimeError(f"{name}: shape changed to {tuple(tensor.shape)}")
                raw = tensor.contiguous().view(torch.uint8).numpy()
                digest.update(memoryview(raw))
                raw.tofile(destination)
                offset_rows += shard_rows
                print(
                    f"[{index + 1:03d}/{len(shards):03d}] "
                    f"rows={offset_rows}/{rows}",
                    flush=True,
                )
            destination.flush()
            os.fsync(destination.fileno())
        if partial.stat().st_size != expected_bytes:
            raise RuntimeError(
                f"raw PLE size mismatch: {partial.stat().st_size} != {expected_bytes}"
            )
        os.replace(partial, output)
    finally:
        partial.unlink(missing_ok=True)

    manifest = {
        "schema": 1,
        "source": "RadixArk/Qwen3.8-Flash-Next-NVFP4",
        "revision": revision,
        "dtype": "float8_e4m3fn",
        "rows": rows,
        "embedding_dim": embedding_dim,
        "shards": len(shards),
        "size_bytes": expected_bytes,
        "sha256": digest.hexdigest(),
    }
    temporary_manifest = Path(str(manifest_path) + ".partial")
    temporary_manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    os.replace(temporary_manifest, manifest_path)
    verify(snapshot, output, manifest, shards, rows_per_shard, verify_samples)
    return manifest


def verify(
    snapshot: Path,
    output: Path,
    manifest: dict,
    shards: list[tuple[int, str, str]],
    rows_per_shard: list[int],
    samples: int,
) -> None:
    raw = torch.from_file(
        str(output),
        shared=False,
        size=manifest["size_bytes"],
        dtype=torch.uint8,
    ).view(manifest["rows"], manifest["embedding_dim"])
    rng = random.Random(38)
    offsets: list[int] = []
    cursor = 0
    for count in rows_per_shard:
        offsets.append(cursor)
        cursor += count
    by_shard: dict[int, list[int]] = {}
    for _ in range(samples):
        global_row = rng.randrange(manifest["rows"])
        shard_index = max(
            index for index, offset in enumerate(offsets) if offset <= global_row
        )
        by_shard.setdefault(shard_index, []).append(global_row)
    for shard_index, global_rows in by_shard.items():
        _, name, filename = shards[shard_index]
        with safe_open(snapshot / filename, framework="pt", device="cpu") as handle:
            tensor = handle.get_tensor(name).contiguous().view(torch.uint8)
        base = offsets[shard_index]
        for global_row in global_rows:
            if not torch.equal(raw[global_row], tensor[global_row - base]):
                raise RuntimeError(
                    f"PLE verification failed at row {global_row} ({name})"
                )
    print(f"verified {samples} deterministic PLE rows", flush=True)


def verify_existing(snapshot: Path, output: Path, samples: int) -> dict:
    manifest_path = Path(str(output) + ".json")
    manifest = json.loads(manifest_path.read_text())
    if output.stat().st_size != manifest["size_bytes"]:
        raise RuntimeError("existing PLE cache size does not match its manifest")
    weight_map = load_index(snapshot)
    shards = discover_shards(weight_map)
    _, _, rows_per_shard = inspect_layout(snapshot, shards)
    verify(snapshot, output, manifest, shards, rows_per_shard, samples)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--verify-samples", type=int, default=256)
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.verify_only:
        manifest = verify_existing(args.snapshot, args.output, args.verify_samples)
    else:
        manifest = build(
            args.snapshot,
            args.output,
            args.revision,
            args.verify_samples,
            args.force,
        )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
