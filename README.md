## MLAPlan 任务切分与 Hopper Kernel（BatchMLAPageAttentionHopper）配合机制解析

本文梳理 `include/flashinfer/attention/scheduler.cuh` 中 `MLAPlan` 的构造逻辑：它如何把一个 batch 的 MLA attention 拆分成 work items（任务），在不同 payload（`batch_size/qo_len/num_heads/kv_len/causal`）下任务数如何变化，以及这些调度结果如何被 `include/flashinfer/attention/mla_hopper.cuh` 的 `BatchMLAPageAttentionHopper` / `BatchMLAPageAttentionHopperKernel` 消费与执行。

> 目标：做到“解释能对上代码块”，并保证信息准确。

---

## 1. Host 侧调用链（Plan -> Run -> Hopper Kernel）

### 1.1 Plan 调用点

`MLAPlan` 在 Host 侧的主要调用入口（SM90/Hopper 版本）是：

- `csrc/batch_mla_sm90_plan.cu::BatchMLAPagedAttentionSM90Plan`
  - 直接调用 `MLAPlan(...)` 并返回 `plan_info.ToVector()`

对应代码：

```cpp
// csrc/batch_mla_sm90_plan.cu
cudaError_t status =
    MLAPlan(float_workspace_buffer.data_ptr(), float_workspace_size_in_bytes,
            int_workspace_buffer.data_ptr(), page_locked_int_workspace_buffer.data_ptr(),
            int_workspace_size_in_bytes, plan_info,
            static_cast<IdType*>(qo_indptr.data_ptr()),
            static_cast<IdType*>(kv_indptr.data_ptr()),
            static_cast<IdType*>(kv_len.data_ptr()),
            batch_size, num_heads, head_dim_o, causal, stream);
```

非 Hopper（FA2）路径也会复用同一个 `MLAPlan`：

- `csrc/batch_mla_plan.cu::BatchMLAPagedAttentionPlan`

### 1.2 Run 装配 Params 并调用 Hopper kernel

SM90/Hopper 路径：

- `csrc/batch_mla_sm90_run.cu::BatchMLAPagedAttentionSM90Run`
  - `MLAPlanInfo plan_info; plan_info.FromVector(plan_info_vec)`
  - 通过 `GetPtrFromBaseOffset` 把 `plan_info.*_offset` 对应的数组指针装配到 `Params`（即 `MLAParams`）里
  - 最终调用 `mla::BatchMLAPageAttentionHopper(..., plan_info.num_blks_x, plan_info.num_blks_y, stream)`

对应代码片段（字段映射非常关键）：

```cpp
// csrc/batch_mla_sm90_run.cu
params.q_indptr = GetPtrFromBaseOffset<IdType>(int_buffer_ptr, plan_info.q_indptr_offset);
params.kv_indptr = GetPtrFromBaseOffset<IdType>(int_buffer_ptr, plan_info.kv_indptr_offset);
params.partial_indptr = GetPtrFromBaseOffset<IdType>(int_buffer_ptr, plan_info.partial_indptr_offset);
params.q_len = GetPtrFromBaseOffset<IdType>(int_buffer_ptr, plan_info.q_len_offset);
params.kv_len = GetPtrFromBaseOffset<IdType>(int_buffer_ptr, plan_info.kv_len_offset);
params.q_start = GetPtrFromBaseOffset<IdType>(int_buffer_ptr, plan_info.q_start_offset);
params.kv_start = GetPtrFromBaseOffset<IdType>(int_buffer_ptr, plan_info.kv_start_offset);
params.kv_end = GetPtrFromBaseOffset<IdType>(int_buffer_ptr, plan_info.kv_end_offset);
params.work_indptr = GetPtrFromBaseOffset<IdType>(int_buffer_ptr, plan_info.work_indptr_offset);

params.merge_packed_offset_start = GetPtrFromBaseOffset<IdType>(int_buffer_ptr, plan_info.merge_packed_offset_start_offset);
params.merge_packed_offset_end   = GetPtrFromBaseOffset<IdType>(int_buffer_ptr, plan_info.merge_packed_offset_end_offset);
params.merge_partial_packed_offset_start = GetPtrFromBaseOffset<IdType>(int_buffer_ptr, plan_info.merge_partial_packed_offset_start_offset);
params.merge_partial_packed_offset_end   = GetPtrFromBaseOffset<IdType>(int_buffer_ptr, plan_info.merge_partial_packed_offset_end_offset);
params.merge_partial_stride = GetPtrFromBaseOffset<IdType>(int_buffer_ptr, plan_info.merge_partial_stride_offset);

params.partial_o  = GetPtrFromBaseOffset<DTypeO>(float_buffer_ptr, plan_info.partial_o_offset);
params.partial_lse= GetPtrFromBaseOffset<float>(float_buffer_ptr, plan_info.partial_lse_offset);

cudaError_t status = mla::BatchMLAPageAttentionHopper<MASK_MODE, HEAD_DIM_CKV, HEAD_DIM_KPE>(
    params, plan_info.num_blks_x, plan_info.num_blks_y, stream);
```

非 Hopper（FA2）路径类似，但调用的是 `mla::BatchMLAPagedAttention(...)`（位于 `include/flashinfer/attention/mla.cuh`）。

---

## 2. Plan 输出的数据结构：`MLAPlanInfo` 与 `MLAParams` 的一一对应关系

### 2.1 `MLAPlanInfo`（Plan 的产物）

文件：`include/flashinfer/attention/scheduler.cuh`，结构体：`MLAPlanInfo`。

它包含两类信息：

- **Kernel grid 形状**：`num_blks_x`、`num_blks_y`
- **调度数组在 workspace buffer 内的 offset**：`q_indptr_offset`、`kv_indptr_offset`、`q_start_offset`、`kv_start_offset` 等
- **split-KV 时的 merge 描述**：`merge_*_offset` + `partial_*_offset`

### 2.2 `MLAParams`（Kernel 的入参）

文件：`include/flashinfer/attention/mla_params.cuh`，结构体：`MLAParams`。

`MLAPlanInfo` 的 offset 在 run 时会被解析成 `MLAParams` 里的指针字段。特别要注意：

- `q_indptr/kv_indptr/q_len/kv_len/q_start/kv_start/kv_end/work_indptr/partial_indptr` 全部来自 int workspace
- `partial_o/partial_lse` 来自 float workspace
- `merge_*` 来自 int workspace

---

## 3. `MLAPlan` 的任务切分逻辑（核心）

### 3.1 “packed Q” 视角：`packed_qo_len = qo_len * num_heads`

`MLAPlan` 会把每个 request 的 query（`qo_len` token）和 head 维度打包成一个一维的 packed 序列：

- `packed_qo_len = qo_len * num_heads`

这是后续 Q tile 切分、以及 kernel 内 `num_heads.divmod(packed_offset, q, r)` 反解回 `(token_idx, head_idx)` 的基础。

### 3.2 决定 cluster 形态（影响 gridDim.x）

`MLAPlan` 会根据 batch 的平均 packed Q 长度选择 `cluster_size`：

- 若 `avg_packed_qo_len > 64`：`cluster_size = 2`
- 否则：`cluster_size = 1`

并写入：

- `plan_info.num_blks_x = cluster_size`
- `plan_info.num_blks_y = num_clusters = num_sm / cluster_size`

同时：

- `cta_tile_q = 64`
- `cluster_tile_q = cluster_size * cta_tile_q`（也就是 64 或 128）

代码位置：`include/flashinfer/attention/scheduler.cuh::MLAPlan` 开头。

### 3.3 决定 KV chunk 上限 `kv_len_limit`（决定是否 split KV）

`MLAPlan` 会先计算所有 request 的所有 Q tile 的 **effective KV length** 总和 `total_kv_lens`：

- non-causal：`effective_kv_len = kv_len`
- causal：`effective_kv_len = packed_causal_kv_end(...)`（越靠后的 Q tile，effective KV 往往越大）

然后用 `ceil_div(total_kv_lens, num_clusters)` 得到每个 cluster 平均负载，再通过一个分段函数 `f()` 离散化：

- `x<=8 -> 32`
- `x<=16 -> 64`
- `x<=32 -> 128`
- `x<=64 -> 192`
- 否则对齐到 `256` 倍数

最终：

- `kv_len_limit = f(max(ceil_div(total_kv_lens, num_clusters), 1))`

**重要结论**：

- GPU 的 `num_sm` 越大、`num_clusters` 越大，平均负载越小，`kv_len_limit` 越可能落在较小档位，从而更容易触发 split-KV。

### 3.4 生成 work items（Q tile × KV chunk），并用 heap 做 load balance

对每个 request：

1. 将 packed Q 按 `cluster_tile_q` 切成 `num_qo_tiles = ceil(packed_qo_len / cluster_tile_q)`
2. 对每个 Q tile：
   - 计算 `remaining_len`（causal / non-causal 不同）
   - 若 `remaining_len > kv_len_limit` 则 split KV：
     - `num_kv_chunks = ceil(remaining_len / kv_len_limit)`
     - 每个 KV chunk 生成一个 work
     - 每个 work 记录：`q_indptr/kv_indptr/q_len/kv_len/q_start/kv_start/kv_end`
     - 并为该 work 分配 `partial_indptr`（指向 `partial_o/partial_lse` 的写入位置）
   - 否则：仅生成 1 个 work，并设置 `partial_indptr = -1` 表示直接写 final
3. 每生成一个 work，用 `MinHeap` 将它分配给当前累计 cost 最小的 cluster（对应 `blockIdx.y`）

代价函数位于同文件：

- `include/flashinfer/attention/scheduler.cuh::cost_function(int qo_len, int kv_len)`

---

## 4. Hopper kernel 如何消费 `MLAPlan` 的调度结果

### 4.1 遍历 work：`work_indptr[blockIdx.y]..work_indptr[blockIdx.y+1]`

文件：`include/flashinfer/attention/mla_hopper.cuh`，kernel：`BatchMLAPageAttentionHopperKernel`。

它用 `blockIdx.y` 作为 cluster id，从 `work_indptr` 拿到该 cluster 的 work 区间：

```cpp
for (IdType work_idx = work_indptr[blockIdx.y]; work_idx < work_indptr[blockIdx.y + 1]; ++work_idx) {
  auto [q_indptr, kv_indptr, partial_indptr, q_len, kv_len, packed_qo_start, kv_start, kv_end] =
      get_block_coord(params, work_idx);
  // ...
}
```

其中 `get_block_coord` 只是把 `params` 里的数组按固定顺序取出来。

### 4.2 为什么 `q_start` 必须是 packed offset

Hopper kernel 在加载 Q、写回 O/LSE 时都会把 packed offset 通过 `num_heads.divmod(...)` 反解为 `(token_idx, head_idx)`。

因此 `MLAPlan` 存储的 `q_start` 是“packed Q 起点”，并在 kernel 内与 `blockIdx.x * CTA_TILE_Q` 合成 CTA 负责的 packed Q 范围：

- `qo_packed_idx_base = packed_qo_start + blockIdx.x * CTA_TILE_Q`

这与 plan 中：

- `cluster_q_start = qo_tile_idx * cluster_tile_q`

保持一致。

### 4.3 split KV 时写 partial；否则写 final

Hopper kernel 的 `write_o` 调用点用 `partial_indptr == -1` 作为是否 split-KV 的判据：

- `partial_indptr == -1`：`partial_o/partial_lse` 传 `nullptr`，直接写 `final_o/final_lse`
- `partial_indptr >= 0`：写 `partial_o/partial_lse`，等待最后 merge

### 4.4 第二阶段 merge：`DevicePersistentMergeStates` 消费 `merge_*`

Hopper kernel 在 `grid.sync()` 之后直接调用 merge：

- `DevicePersistentMergeStates(merge_packed_offset_start, merge_packed_offset_end, merge_partial_packed_offset_start, merge_partial_packed_offset_end, merge_partial_stride, ...)`

该函数实现位于 `include/flashinfer/attention/mla.cuh`。

`merge_*` 描述了“某个 CTA（用 `cta_idx = gridDim.x * blockIdx.y + blockIdx.x` 编码）负责把哪一段 packed Q 的 partial 结果跨 stride 归约回 final”。

---

## 5. payload 对任务划分的影响（可直接按代码推导）

下面给出与实现严格一致的推导关系（变量与 `MLAPlan` 中一致）：

- **packed Q 长度**：`packed_qo_len_i = qo_len_i * num_heads`
- **cluster_tile_q**：`cluster_tile_q = cluster_size * 64`，其中 `cluster_size = 2 if avg_packed_qo_len>64 else 1`
- **Q tile 数**：`num_qo_tiles_i = ceil(packed_qo_len_i / cluster_tile_q)`
- **KV split 判据（每个 Q tile）**：`split_kv = remaining_len > kv_len_limit`
- **KV chunk 数**：`num_kv_chunks = ceil(remaining_len / kv_len_limit)`
- **work item 数（近似）**：`sum_i sum_{q_tile} (split? num_kv_chunks : 1)`

其中 `kv_len_limit` 由 `total_kv_lens/num_clusters` 推导，并被 `f()` 离散化，所以它同时受：

- GPU 的 `num_sm`
- `cluster_size`（由 `avg_packed_qo_len` 决定）
- `batch_size/qo_len/num_heads/kv_len/causal`

共同影响。

---

## 6. 一个与常见 decode 配置一致的示例（帮助直觉理解）

若 decode 场景：

- `qo_len = 1`
- `num_heads = 64`（例如 DCP 后 local heads 为 64）
- `kv_len = 8192`

则：

- `packed_qo_len = 1 * 64 = 64`
- `avg_packed_qo_len = 64`（batch=1 时就是它）
- `cluster_size = 1`，`cluster_tile_q = 64`，所以每个 request 的 Q tile 数：`ceil(64/64)=1`

若 GPU `num_sm` 较大导致 `num_clusters` 较大，则 `ceil(total_kv_lens/num_clusters)` 可能较小，经过 `f()` 离散后 `kv_len_limit` 可能变为 `256`（或更小档位），从而：

- `num_kv_chunks = ceil(8192 / 256) = 32`
- 该 request（唯一 Q tile）会产生约 32 个 work items（每个 work 覆盖一个 KV chunk），这些 work 会被 `MinHeap` 尽量均匀分配到不同的 `blockIdx.y`。

这解释了“batch 很小、KV 很长、GPU 很大”时更容易触发 split-KV 并出现大量 work 的现象。

---

## 7. 你可以从哪些字段直接观察“切分结果”

若要在运行时验证某个 payload 的任务划分，可以关注这些数组（都由 `MLAPlan` 写入 int workspace）：

- `work_indptr`：每个 `blockIdx.y` 对应的 work 区间
- `q_indptr/kv_indptr/q_len/kv_len/q_start/kv_start/kv_end`：每个 work 的坐标
- `partial_indptr`：是否 split-KV，以及 partial 写入位置
- `merge_packed_offset_* / merge_partial_*`：merge 阶段每个 CTA 的归约范围与 stride

---

## 8. 相关文件索引

- `include/flashinfer/attention/scheduler.cuh`
  - `MLAPlanInfo`
  - `MLAPlan(...)`
  - `cost_function(...)`
- `include/flashinfer/attention/mla_params.cuh`
  - `MLAParams`
- `include/flashinfer/attention/mla_hopper.cuh`
  - `BatchMLAPageAttentionHopperKernel`
  - `BatchMLAPageAttentionHopper(...)`
- `include/flashinfer/attention/mla.cuh`
  - `DevicePersistentMergeStates(...)`
- `csrc/batch_mla_sm90_plan.cu` / `csrc/batch_mla_sm90_run.cu`
  - Plan/Run 的 host 侧装配路径（SM90/Hopper）
- `csrc/batch_mla_plan.cu` / `csrc/batch_mla_run.cu`
  - 非 Hopper（FA2）路径的装配方式（同样使用 `MLAPlan`）
