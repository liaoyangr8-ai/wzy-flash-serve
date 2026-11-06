# WZY-Flash-Serve 优化会战 — 项目总结与交接文档

**日期**：2026-11-05 深夜 ~ 11-06 凌晨（三个连续会话）
**机器**：`liaoyang@192.168.2.100`（2× RTX 2080 Ti 22GB, E5-2698B v3）
**工程目录**：`~/wzy-flash-serve`（详细过程全部在 `README.md` 7.12–7.17）
**本文件用途**：一页纸看清"做了什么、为什么、当前状态、下轮从哪进"。

---

## 0. 一句话总结

贪心解码不可复现的根因（晋升队列原子序）已修复并落生产；生产配置提速 +7~8%；
stage A（设备侧 epoch）与 CUDA graph 基础设施就绪并全部字节等价验证；graph 实测
burst-neutral 证伪了"+9%"假设；内核合并完成定标（GPU-bound、0.535ms/层）与首次
尝试（F1 等价达成但性能证伪），方法论已校准。**下一步 = F5 mtp drafter 步内合并。**

---

## 1. 已落地的成果（按时间线）

### 7.12 mkey 线索排除 + stage A 三版尝试
- `mkey + l` 实为 `d_ncold_out`（每层冷组数快路标志），与 seq 语义无关；
  真正的"第五处公式"（peer 服务路径）在 pp2 解码路径是死代码。
- stage A 三版：窗口入口写 mapped（竞态）→ 设备端窗口末 bump（确定但分叉）→
  逐位复刻 `cold_seq` 值（又不稳）。**撞出真正的发现：基线贪心解码自身不可复现**
  （同二进制 5 次 = 2 条流：1.92 与 1.78 tok/win 两档）。

### 7.13 ★ 可复现性根因修复（确定性晋升）——本会战最重要的成果
- **根因**：`split_route_hit_cold_kernel` 用 `atomicAdd` 分配冷组下标 → 各组服务
  顺序非确定 → `service_one_cold_layer` 按此顺序 push `pending_promo` 且带
  `size() < max_q` 截断 → **晋升的专家子集每次运行不同** → VRAM 驻留映射不同 →
  累加顺序不同 → 贪心流漂移（`WZY_ADAPT_DEPTH` 的 acc_ema 只是放大器）。
- **修复**（源码已应用）：push 全部合格候选 + `promote_cold` 内按
  `(freq 降序, gkey 升序)` 排序后再截断（`WZY_PROMO_QUEUE`）。
- **验证**：晋升全关 9/9 一致；修复后 9/9 一致（3 个 prompt/长度组合，含自然 stop）。
- 补丁：`tools/patch_promo_order.py`；备份 `/tmp/wzy_stage_v2.cuh.bak-promo`。

### 7.14 stage A 等价验证通过 + 生产调参定版
- stage A（`tools/patch_stageA_epoch.py`：设备侧 epoch、流内窗口末尾
  `bump_epoch_kernel`、值与旧 `cold_seq` 逐位相同）叠上修复后源码：
  **prose 3/3 + code 2/2 字节等价**，ms/win 无回退。
- 晋升参数扫描（全部可复现）：定版 **`WZY_PROMO_MIN_F=2 / PROMO_PER_WIN=16 /
  PROMO_QUEUE=64`**（已写入 config.json）。
- **生产成绩**：prose 62.5 tok/s（+7%）、code 77.6 tok/s（+8%），历史最好区间。

### 7.15 CUDA graph v2：全通但 burst-neutral
- 806 节点/图，exec ≈ **40 MiB**；warmup 宽形状必须用 `WZY_GRAPH_MAXW`（默认 4）
  挡掉，否则真解码形状 OOM；instantiate 失败 → 非致命回退（撤簿记+递归重跑普通路径）。
- **字节等价** 3/3；但 A/B×3×2：`enq+s0 ≈ 14.5ms` 恒定（enq 3.9→0.8，s0 10.8→13.5）
  —— 发射与冷专家服务本就重叠，**"图=+9%"的推导结构性不成立**。
- 图代码保留源码（env 关闭零开销）；诊断 `[GRAPH] shape/nodes/free` 已内置。

### 7.16 层级定标（`WZY_LAYER_PROF=1 WZY_LAYER_PROF_AT=30`）
- 单层 0.535ms × 48 ≈ 25.7ms ≈ s0+s1 ⇒ **GPU-bound，CPU 冷服务（~2-5ms）已隐藏**。
  moe 0.252 / attention 0.176 / router 0.060 / gr_read 0.047，~25 内核/层 ≈ 20µs/个
  —— 发射粒度 bound，非算力 bound。
- mtp 链（3.0ms）：每步已是单图 `step1_exec`，3 × (图+同步+主机门控)，
  步内 ~40 小内核 —— 与主模型同病。

### 7.17 F1 首战：等价达成、性能证伪（已回滚）
- `router 三内核 → router_gate_top10_fused<<<T,512>>>`（逐位复刻累加顺序）：
  **3/3 IDENTICAL**，但 router 相 0.060→0.132ms（ms/win -12%）。
- **根因**：原 gemm 512 warp 并行（每 warp 一行），合并后每 token 只 16 warp
  （每 warp 串行 32 行）——**并行度损失 >> 发射收益**。
- **方法论修正**：合并只允许发生在**不减少总 warp 数**的前提下。

---

## 2. 当前状态（2026-11-06 凌晨）

- 源码 = 7.14 定版（确定性晋升 + stage A epoch + graph 代码 env 关闭），F1 零残留。
- `bin/wzy_pp2_engine` 与 `/tmp/wzy_pp2_engine.graph5` 同语义；服务运行中、健康
  （`http://192.168.2.100:11437`），config env = 新生产配置。
- 贪心解码**完全可复现**；基线流：`/tmp/eq_gbase.json`（prose，hash 3bb743b1…）、
  `/tmp/eq_codeBase.json`（code）。

## 3. 工具与判据（全部就位）

| 工具 | 用法 |
|---|---|
| `/tmp/eqrun_g2.py` | `--bin X --env K=V --ref base.json --times N --save out.json`：逐字节等价 + DEC-PROF + `[GRAPH]` 透传 + stderr 全量落盘 `/tmp/errdump_<tag>.txt` |
| `WZY_LAYER_PROF=1 WZY_LAYER_PROF_AT=30` | 一次运行得四相耗时（基准：moe 0.252 / attn 0.176 / router 0.060 / gr_read 0.047） |
| `WZY_GRAPH_STAGE0=1`（+可选 `WZY_GRAPH_MAXW`） | 图捕获/回放；`[GRAPH]` 行输出节点数与失败诊断 |
| 等价判据 | 任何改动 `--ref /tmp/eq_gbase.json` 必须 IDENTICAL（除非改动语义上允许变流，如 F5，此时须验证构建内自洽） |

## 4. 回滚点

| 场景 | 恢复物 |
|---|---|
| 回到 7.14 定版语义（=当前） | `git`-less：`/tmp/*.bak-promo`、`/tmp/*.bak-preStageA` + `tools/patch_*.py` 可重放 |
| 无 epoch 的纯晋升修复版 | `/tmp/wzy_pp2_engine.detpromo` |
| 图实验版 | `/tmp/wzy_pp2_engine.graph5` |
| config.json | `/tmp/config.json.bak-detpromo`（旧生产参数） |

## 5. 下轮入口（按序）

1. **F5：mtp drafter 步内合并**（首选）——`wzy_mtp.cuh` `run_rest_row0`，单 token
   前向并行度本就低，合并不亏并行度；步内 ~40 小内核 × 3 步 ≈ 3.0ms，目标砍半。
   注意：drafter 只影响"提议"，不影响验证正确性，但会改窗口组成 → token 流可能变
   （构建内自洽即可）。
2. **sgate 并入 top10**（低风险快赢）：只把 rows=1 的 sgate gemm（1/8 warp 利用率）
   并进 `router_top10_fast_multi`，不动 gemm 主干。预计 +0.3-0.5ms/win。
3. F2/F3：需按"不减少 warp 总数"重新设计（persistent kernel / megakernel 另立项）。
4. 若 1+2 落地：29.0 → ~26.5ms/win ≈ **68-70 tok/s，追平 Strata**。

## 6. 遗留与风险

- **温度>0 服务端路径**有自己的采样 rng，不在引擎级可复现范围（LoopGuard 等不变）。
- `wzy_batch_engine` 共享晋升代码，修复生效但未单独测。
- 7.8 遗留 `exec_body1/exec_full` 死成员（无害）。
- 新生产配置只在 128-token 短跑做了 3× 重复 + 质量抽查（dup20=0、自然 stop、文本连贯）；
  长文本大规模回归未做。
- README 7.12–7.17 为完整过程记录（含所有失败的中间态与证据链），交接以本文 + 7.17.3 为准。


---

## 7. 追加（2026-11-06：F5 准备 + F1b，README 7.18；profile 对照 7.20）

- F5 准备完成：drafter 每步 ~45 节点，可行路线收益比一般，未实施。
- F1b 已应用：sgate 并入 top10 空闲 warp 1，逐位等价（3/3 IDENTICAL），实测中性。
- 合并结论：发射开销被 pipeline 掩盖，逐个省发射收益约 0；合并收益必须来自减少 GPU 串行工作量。F2-F5 收益下修。
- profile 对照（7.20）：学习版 profile（11-02 真实流量学出）+1.2 tok/s（65.7 vs 64.5，vram_hit +0.39pp）；A2 的 67.1 归因修正。当前生产 = strata 官方词表 + 学习版 profile，蟓蟹题实测 65.7 tok/s，比会话开始基线 +5.8%。建议周期性 SAVE_PROFILE。
- 下轮入口：内核内部工程（moe 相中间 buffer 消除 / drafter d_head gemm / QSA smem 扫描）。
