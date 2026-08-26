#!/usr/bin/env python3
"""Use PyTorch SDPA for QSA decode on GB10 while FA4 SM121 is broken."""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path


EXPECTED_SHA256 = "c959835d05d0f395ad7eae4330cf264af9f6f7c1bff3d45a39bb953d2536f5f2"
MARKER = "def _forward_sm121_sdpa_sparse"


def replace_once(source: str, old: str, new: str, label: str) -> str:
    count = source.count(old)
    if count != 1:
        raise RuntimeError(f"expected one {label}, found {count}")
    return source.replace(old, new, 1)


def main() -> None:
    path = Path(sys.argv[1])
    source = path.read_text()
    if MARKER in source:
        return
    digest = hashlib.sha256(source.encode()).hexdigest()
    if digest != EXPECTED_SHA256:
        raise RuntimeError(f"unexpected QSA source hash: {digest}")

    source = replace_once(
        source,
        "from sglang.srt.model_executor.forward_batch_info import ForwardMode\n",
        "from sglang.srt.model_executor.forward_batch_info import ForwardMode\n"
        "from sglang.srt.utils.common import is_sm121\n",
        "SM121 helper import",
    )
    helper = r'''
    def _forward_sm121_sdpa_sparse(
        self,
        q: torch.Tensor,
        k_buffer: torch.Tensor,
        v_buffer: torch.Tensor,
        layer,
        metadata: QwenSparseAttnMetadata,
        topk_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Packed sparse decode without flash-attn-4's broken SM121 varlen path."""
        slots = self._logical_to_physical(topk_indices, metadata)
        repeats = q.shape[1] // k_buffer.shape[1]
        outputs = []
        for row in range(q.shape[0]):
            valid_slots = slots[row, slots[row] >= 0].long()
            if valid_slots.numel() == 0:
                outputs.append(torch.zeros_like(q[row]))
                continue
            keys = k_buffer.index_select(0, valid_slots).repeat_interleave(
                repeats, dim=1
            )
            values = v_buffer.index_select(0, valid_slots).repeat_interleave(
                repeats, dim=1
            )
            query = q[row].unsqueeze(0).unsqueeze(2)
            keys = keys.permute(1, 0, 2).unsqueeze(0)
            values = values.permute(1, 0, 2).unsqueeze(0)
            output = F.scaled_dot_product_attention(
                query,
                keys,
                values,
                is_causal=False,
                scale=layer.scaling,
            )
            outputs.append(output[0, :, 0, :])
        return torch.stack(outputs).reshape(q.shape[0], -1)

'''
    source = replace_once(
        source,
        "    def forward_decode(\n"
        "        self,\n"
        "        q: torch.Tensor,\n",
        helper
        + "    def forward_decode(\n"
        + "        self,\n"
        + "        q: torch.Tensor,\n",
        "QSA decode method",
    )
    source = replace_once(
        source,
        "        metadata = self._resolve_metadata(forward_batch)\n"
        "        topk_indices = topk_indices.to(torch.int32).contiguous()\n"
        "        trtllm_decode = _resolve_trtllm_sparse_decode()\n",
        "        metadata = self._resolve_metadata(forward_batch)\n"
        "        topk_indices = topk_indices.to(torch.int32).contiguous()\n"
        "        if is_sm121():\n"
        "            return self._forward_sm121_sdpa_sparse(\n"
        "                q, k_buffer, v_buffer, layer, metadata, topk_indices\n"
        "            )\n"
        "        trtllm_decode = _resolve_trtllm_sparse_decode()\n",
        "SM121 decode dispatch",
    )
    path.write_text(source)


if __name__ == "__main__":
    main()
