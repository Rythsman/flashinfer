"""Benchmark DCP MLA wrapper.run GPU time.

This script benchmarks the GPU time of `flashinfer.mla.BatchMLAPagedAttentionWrapper.run`
when DCP-style sharding is enabled (multiple ranks, each with its own page indices).

It supports both non-CUDA-graph timing and CUDA-graph timing via `bench_gpu_time`.

Notes:
- The benchmark only measures `wrapper.run` time. All `wrapper.plan` calls are done
  once during setup and are excluded from timing.
- For CUDA graphs, outputs are pre-allocated and passed into `run` to avoid any
  allocation during capture/replay.

Example:
  python benchmarks/bench_mla_dcp_gpu_time.py --backend fa3 --use-cuda-graph

"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np
import torch

import flashinfer
from flashinfer.testing.utils import bench_gpu_time


@dataclass(frozen=True)
class BenchmarkConfig:
    batch_size: int
    seq_len: int
    tp_size: int
    dcp_size: int
    head_dim_ckv: int
    head_dim_kpe: int
    page_size: int
    backend: str
    dtype: torch.dtype
    sm_scale: float


def _parse_dtype(name: str) -> torch.dtype:
    name = name.lower().strip()
    if name in ("bf16", "bfloat16"):
        return torch.bfloat16
    if name in ("fp16", "float16", "half"):
        return torch.float16
    raise ValueError(f"Unsupported dtype: {name}")


def _compute_sm_scale() -> float:
    # Keep the same scale formula as the original script.
    x = 0.1 * math.log(40.0) + 1.0
    return x * x * ((128.0 + 64.0) ** -0.5)


def _build_dcp_indices(
    kv_lens: torch.Tensor, dcp_size: int, rank: int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build kv_indptr/kv_indices/local_kv_lens for a given DCP rank.

    All outputs are int32 CPU tensors. The wrapper will move/copy them as needed.

    Args:
        kv_lens: Shape [B], int32 CPU tensor.
        dcp_size: DCP world size.
        rank: Rank in [0, dcp_size).

    Returns:
        kv_indptr_cpu: Shape [B + 1], int32.
        kv_indices_cpu: Shape [sum(local_kv_lens)], int32.
        local_kv_lens_cpu: Shape [B], int32.
    """
    if kv_lens.dtype != torch.int32 or kv_lens.device.type != "cpu":
        raise ValueError("kv_lens must be an int32 CPU tensor")

    # Same formula as the original script.
    local_kv_lens = ((kv_lens - rank - 1) // dcp_size) + 1
    local_kv_lens = torch.clamp(local_kv_lens, min=0).to(torch.int32)

    kv_indptr = torch.empty((kv_lens.numel() + 1,), dtype=torch.int32, device="cpu")
    kv_indptr[0] = 0
    kv_indptr[1:] = torch.cumsum(local_kv_lens, dim=0)

    indices: List[torch.Tensor] = []
    offset = 0
    for original_len_i, local_len_i in zip(kv_lens.tolist(), local_kv_lens.tolist()):
        if local_len_i > 0:
            idx = torch.arange(local_len_i, dtype=torch.int32) * dcp_size + rank + offset
            indices.append(idx)
        offset += int(original_len_i)

    kv_indices = (
        torch.cat(indices, dim=0) if indices else torch.empty((0,), dtype=torch.int32)
    )

    if not kv_indptr.is_contiguous():
        kv_indptr = kv_indptr.contiguous()
    if not kv_indices.is_contiguous():
        kv_indices = kv_indices.contiguous()
    if not local_kv_lens.is_contiguous():
        local_kv_lens = local_kv_lens.contiguous()

    return kv_indptr, kv_indices, local_kv_lens


def _make_inputs(cfg: BenchmarkConfig, device: torch.device) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Create all input tensors.

    Returns:
        q_indptr: [B + 1] int32 cuda
        kv_lens: [B] int32 cpu (used to build per-rank indices)
        q_nope: [B, H, D_ckv] cuda
        q_pe: [B, H, D_kpe] cuda
        ckv_cache: [B * seq_len, 1, D_ckv] cuda
        kpe_cache: [B * seq_len, 1, D_kpe] cuda
        out_template: [B, H, D_ckv] cuda (for allocation)
        lse_template: [B, H] cuda (for allocation)
    """
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)

    num_local_heads = 128 // cfg.tp_size * cfg.dcp_size

    q_indptr = torch.arange(0, cfg.batch_size + 1, device=device, dtype=torch.int32)

    # Keep kv_lens on CPU because wrapper.plan copies host indptr/len anyway.
    kv_lens_cpu = torch.full(
        (cfg.batch_size,), cfg.seq_len, dtype=torch.int32, device="cpu"
    )

    q_nope = torch.softmax(
        torch.randn(
            cfg.batch_size,
            num_local_heads,
            cfg.head_dim_ckv,
            dtype=cfg.dtype,
            device=device,
        ),
        dim=-1,
    )
    q_pe = torch.softmax(
        torch.randn(
            cfg.batch_size,
            num_local_heads,
            cfg.head_dim_kpe,
            dtype=cfg.dtype,
            device=device,
        ),
        dim=-1,
    )

    kv_all = torch.softmax(
        torch.randn(
            cfg.batch_size * cfg.seq_len,
            1,
            cfg.head_dim_ckv + cfg.head_dim_kpe,
            dtype=cfg.dtype,
            device=device,
        ),
        dim=-1,
    )
    ckv_cache = kv_all[..., : cfg.head_dim_ckv].contiguous()
    kpe_cache = kv_all[..., cfg.head_dim_ckv :].contiguous()

    # `wrapper.run` requires out.dtype == q_nope.dtype.
    out_template = torch.empty_like(q_nope)
    lse_template = torch.empty(
        (cfg.batch_size, num_local_heads), device=device, dtype=torch.float32
    )

    return (
        q_indptr,
        kv_lens_cpu,
        q_nope,
        q_pe,
        ckv_cache,
        kpe_cache,
        out_template,
        lse_template,
    )


def _build_dcp_wrappers_and_buffers(
    cfg: BenchmarkConfig,
    device: torch.device,
    q_indptr: torch.Tensor,
    kv_lens_cpu: torch.Tensor,
    q_nope: torch.Tensor,
    q_pe: torch.Tensor,
    ckv_cache: torch.Tensor,
    kpe_cache: torch.Tensor,
    out_template: torch.Tensor,
    lse_template: torch.Tensor,
    return_lse: bool,
) -> Tuple[List[flashinfer.mla.BatchMLAPagedAttentionWrapper], List[torch.Tensor], List[Optional[torch.Tensor]]]:
    """Create one wrapper per rank, plan once, and pre-allocate out/lse buffers."""

    num_local_heads = 128 // cfg.tp_size * cfg.dcp_size

    # Share a single workspace buffer across ranks (runs are sequential).
    workspace = torch.empty(128 * 1024 * 1024, dtype=torch.int8, device=device)

    wrappers: List[flashinfer.mla.BatchMLAPagedAttentionWrapper] = []
    outs: List[torch.Tensor] = []
    lses: List[Optional[torch.Tensor]] = []

    for rank in range(cfg.dcp_size):
        wrapper = flashinfer.mla.BatchMLAPagedAttentionWrapper(workspace, backend=cfg.backend)

        kv_indptr_cpu, kv_indices_cpu, local_kv_lens_cpu = _build_dcp_indices(
            kv_lens_cpu, cfg.dcp_size, rank
        )

        # Move indices to device; this also matches the original script behavior.
        kv_indptr = kv_indptr_cpu.to(device=device, non_blocking=True)
        kv_indices = kv_indices_cpu.to(device=device, non_blocking=True)
        local_kv_lens = local_kv_lens_cpu.to(device=device, non_blocking=True)

        wrapper.plan(
            q_indptr,
            kv_indptr,
            kv_indices,
            local_kv_lens,
            num_local_heads,
            cfg.head_dim_ckv,
            cfg.head_dim_kpe,
            cfg.page_size,
            False,  # causal
            cfg.sm_scale,
            q_nope.dtype,
            ckv_cache.dtype,
        )

        out = torch.empty_like(out_template)
        if return_lse:
            lse = torch.empty_like(lse_template)
        else:
            lse = None

        wrappers.append(wrapper)
        outs.append(out)
        lses.append(lse)

    return wrappers, outs, lses


def _make_dcp_run_fn(
    wrappers: Sequence[flashinfer.mla.BatchMLAPagedAttentionWrapper],
    outs: Sequence[torch.Tensor],
    lses: Sequence[Optional[torch.Tensor]],
    q_nope: torch.Tensor,
    q_pe: torch.Tensor,
    ckv_cache: torch.Tensor,
    kpe_cache: torch.Tensor,
    return_lse: bool,
) -> Callable[[], None]:
    """Create a callable that runs all DCP ranks once (measured unit)."""

    if len(wrappers) != len(outs) or len(wrappers) != len(lses):
        raise ValueError("wrappers/outs/lses length mismatch")

    def _run_once() -> None:
        for i in range(len(wrappers)):
            if return_lse:
                wrappers[i].run(
                    q_nope,
                    q_pe,
                    ckv_cache,
                    kpe_cache,
                    out=outs[i],
                    lse=lses[i],
                    return_lse=True,
                )
            else:
                wrappers[i].run(
                    q_nope,
                    q_pe,
                    ckv_cache,
                    kpe_cache,
                    out=outs[i],
                    return_lse=False,
                )

    return _run_once


def _stats_ms(ms_list: Sequence[float]) -> str:
    arr = np.asarray(ms_list, dtype=np.float64)
    p50 = float(np.percentile(arr, 50))
    p10 = float(np.percentile(arr, 10))
    p90 = float(np.percentile(arr, 90))
    return f"p50={p50:.4f} ms, p10={p10:.4f} ms, p90={p90:.4f} ms, n={arr.size}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark DCP MLA wrapper.run GPU time")
    parser.add_argument("--backend", type=str, default="fa3", choices=["auto", "fa2", "fa3"])
    parser.add_argument("--dtype", type=str, default="bf16")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seq-len", type=int, default=8192)
    parser.add_argument("--tp-size", type=int, default=16)
    parser.add_argument("--dcp-size", type=int, default=8)
    parser.add_argument("--head-dim-ckv", type=int, default=512)
    parser.add_argument("--head-dim-kpe", type=int, default=64)
    parser.add_argument("--page-size", type=int, default=1)
    parser.add_argument("--return-lse", dest="return_lse", action="store_true")
    parser.add_argument(
        "--no-return-lse", dest="return_lse", action="store_false", default=True
    )

    parser.add_argument("--enable-cupti", action="store_true", default=False)
    parser.add_argument("--use-cuda-graph", action="store_true", default=False)
    parser.add_argument("--num-iters-within-graph", type=int, default=10)

    parser.add_argument("--dry-run-ms", type=int, default=100)
    parser.add_argument("--repeat-ms", type=int, default=1000)
    parser.add_argument("--no-l2-flush", action="store_true", default=False)

    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark")

    device = torch.device("cuda:0")
    dtype = _parse_dtype(args.dtype)

    cfg = BenchmarkConfig(
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        tp_size=args.tp_size,
        dcp_size=args.dcp_size,
        head_dim_ckv=args.head_dim_ckv,
        head_dim_kpe=args.head_dim_kpe,
        page_size=args.page_size,
        backend=args.backend,
        dtype=dtype,
        sm_scale=_compute_sm_scale(),
    )

    (
        q_indptr,
        kv_lens_cpu,
        q_nope,
        q_pe,
        ckv_cache,
        kpe_cache,
        out_template,
        lse_template,
    ) = _make_inputs(cfg, device)

    wrappers, outs, lses = _build_dcp_wrappers_and_buffers(
        cfg,
        device,
        q_indptr,
        kv_lens_cpu,
        q_nope,
        q_pe,
        ckv_cache,
        kpe_cache,
        out_template,
        lse_template,
        return_lse=args.return_lse,
    )

    run_once = _make_dcp_run_fn(
        wrappers,
        outs,
        lses,
        q_nope,
        q_pe,
        ckv_cache,
        kpe_cache,
        return_lse=args.return_lse,
    )

    # One extra warmup outside of bench_gpu_time to avoid first-use overhead.
    torch.cuda.synchronize()
    run_once()
    torch.cuda.synchronize()

    times_ms = bench_gpu_time(
        run_once,
        dry_run_time_ms=args.dry_run_ms,
        repeat_time_ms=args.repeat_ms,
        l2_flush=not args.no_l2_flush,
        enable_cupti=args.enable_cupti,
        use_cuda_graph=args.use_cuda_graph,
        num_iters_within_graph=args.num_iters_within_graph,
    )

    num_local_heads = 128 // cfg.tp_size * cfg.dcp_size
    print(
        "Config: "
        f"backend={cfg.backend}, dtype={args.dtype}, batch_size={cfg.batch_size}, seq_len={cfg.seq_len}, "
        f"tp_size={cfg.tp_size}, dcp_size={cfg.dcp_size}, num_local_heads={num_local_heads}, "
        f"head_dim_ckv={cfg.head_dim_ckv}, head_dim_kpe={cfg.head_dim_kpe}, page_size={cfg.page_size}, "
        f"return_lse={args.return_lse}"
    )
    print(
        "Timing: "
        f"use_cuda_graph={args.use_cuda_graph}, enable_cupti={args.enable_cupti}, "
        f"num_iters_within_graph={args.num_iters_within_graph}, l2_flush={not args.no_l2_flush}"
    )
    print(_stats_ms(times_ms))


if __name__ == "__main__":
    main()
