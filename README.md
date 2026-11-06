# IMPORTANT: renamed source package

This package contains no prebuilt engine. Run ./build.sh on a compatible Linux/CUDA machine before starting. Older statements below about a bundled binary describe the original deployment, not this package. The WZY names and environment variables were mechanically renamed. The tree compiles cleanly with nvcc 12.8 (-arch=sm_75), but this version has not been runtime-tested. Adapt config.json and model/vision paths before use.

> **Public source snapshot:** Model weights, prebuilt engine, `data/*.bin` draft vocabulary and expert profiles are not included. The data provenance and redistribution terms require separate verification; obtain or generate compatible files before use. References to bundled binaries or profiles below describe the original deployment only.

> **来源与许可：**本项目复用和改编了 Strata 部分代码，视觉功能调用 Strata 组件；详见 `THIRD_PARTY_NOTICES.md` 和 `LICENSES/Strata-MIT.txt`。项目自有贡献采用 MIT 许可，其他依赖许可尚未全面审计。

# WZY-Flash-Serve (`wzy-pp2-v2-tc-mtp`)

**面向双卡 Turing (`sm_75`) 的 CUDA/C++ 推理引擎 + OpenAI / Anthropic / Responses 协议服务栈 + 实时监控 WebUI；包含 Strata 代码移植及组件复用。**

针对 `Maximilian228/Qwen3.8-Flash-Next-NVFP4-GPTQ-Strata`（512 专家稀疏 MoE，Top-10 路由，36 层 GDN + 12 层 QSA 稀疏注意力 + 16 头 3-gram PLE + 原生 MTP 投机解码器）在 **2× 解锁版 RTX 2080 Ti 22GB (`sm_75`, NVLink)** 与 **Haswell Xeon E5-2698B v3 (16C/32T, DDR3-1867 ECC)** 平台深度定制。

---

## 一、核心性能指标（双卡 RTX 2080 Ti 22GB 实测，100% 零跳过专家）

| 评测维度 | 实测指标 | 关键技术支撑 |
| :--- | :---: | :--- |
| **最大上下文长度** | **`262,144` (256K)** | QSA 4-Head 稀疏索引器 (`top_k=2048`) + `32,768` VRAM KV 环（含 `4,096` 永久钉死 Attention-Sink）+ 256K Pinned Host INT8 KV 镜像 |
| **120K / 256K Prefill 吞吐** | **`2,913.9 tok/s`** (120K)<br>**`2,776.6 tok/s`** (256K) | 解锁 `sm_75` Tensor Core (`182.9 TFLOPS` FP16/FP32) + 双卡 Chunk 级流水线并发 + 借用显存双缓冲 DMA 环 (`PF_DMA_SLOTS=512`) |
| **中文 + CUDA 深度技术对话 Decode** | **`80.96 – 90.74 tok/s`**<br>(确定性峰值 **`123.16 tok/s`**) | `114,452` CJK 草稿词表 + 中英代码联合预热表 `expert-profile-cjk.bin` (`97.7%` VRAM 命中率) + 84 MB 常驻 PLE 缓存 + MTP 间隙 PLE 预热 |
| **Agent 代码重构 / C++ 生成 Decode** | **`92.49 – 115.20 tok/s`** | `SuffixDrafter` 4 路 Trigram 后缀投机 + 原生 Q2_0 MTP 投机链 + 双卡 `A || B` 流水线重叠 |
| **超长上下文 (70.7K+) 中文总结 Decode** | **`59.65 tok/s`** | 超过 32K 自动切换 GQA-12 Tensor-Core 稀疏注意力核函数 (`pf_qsa_wmma_sparse_attn`)，PCIe KV 读取流量降低 **12 倍** |
| **多轮 Agent 会话切换恢复 (Park/Restore)** | **`~20 ms`** (Turn 2 首字 `358 ms`) | 双槽位无损会话驻留 (`ParkedSession`) + 16 槽位 LRU 会话缓存 + DeepSeek Harness 动态时间戳归一化 (**99.9% KV 复用率**) |

---

## 二、目录结构

```text
wzy-flash-serve/
├── bin/
│   └── wzy_pp2_engine             # 编译好的纯 CUDA/C++ 双卡推理引擎可执行文件
├── data/
│   ├── draft_vocab_han.bin        # 114,452 词条 CJK 中文+代码 MTP 草稿头词表 (457 KB)
│   ├── expert-profile-cjk.bin     # 实战预热生成的中英+CUDA+代码联合高频专家排名表 (97 KB)
│   └── expert-profile.bin         # 原始基础专家频率排名表 (193 KB)
├── src/
│   ├── wzy_engine_v2.cu           # 引擎主程序：PP2 流水线、MTP 投机循环、双槽位 Park/Restore、JSON 流协议
│   ├── wzy_stage_v2.cuh           # StageGpu 层执行器、84MB Direct-Mapped PLE 缓存、CLOCK 二次机会专家缓存
│   ├── wzy_kernels_v2.cuh         # 核心 CUDA 算子：2-Rows/Warp Fused NVFP4 MoE、GDN 递归、零拷贝状态提交
│   ├── wzy_cpu_moe.cuh            # GPU/CPU 零停顿路由分流器 + AVX2/FMA 线程池 + Aux CPU 驱动线程隔离
│   ├── wzy_mtp.cuh                # 原生 MTP 投机解码器 (512 Q2_0 专家常驻 GPU1 + 2-Rows/Warp Fused Quant)
│   ├── wzy_prefill_tc.cuh         # Tensor Core 批量分块 Prefill 引擎与显存借用恢复器
│   ├── wzy_qsa_256k.cuh           # 256K QSA 4-Head 稀疏索引器与 4096-Sink 分层 INT8 KV 缓存
│   ├── wzy_qsa_wmma.cuh           # GQA-12 Tensor-Core 稀疏注意力核函数 (32K~256K 零 PCIe 放大)
│   └── wzy_suffix.cuh             # 4 路 Trigram 零开销后缀匹配投机器 (SuffixDrafter)
├── serve/
│   ├── server.py                  # OpenAI / Anthropic / Responses API 服务 + 动态端口管理 + 进程生命周期控制
│   ├── frontend.py                # 流式 <think> 与 <tool_call> 状态机解析器 (OutputParser) + Jinja2 模板引擎
│   ├── responses.py               # OpenAI /v1/responses 协议适配层
│   ├── tokenizer.py               # Byte-Level BPE 分词器与流式增量解码器 (保留语义标签 ID)
│   ├── vision.py                  # 图片输入：strata-vision (mmproj) CPU 投影 → 注入 <|image_pad|> 行
│   └── web/                       # Strata 风格实时监控面板与零显存启动器 WebUI (index.html, app.js, style.css)
├── config.json                    # 服务端口、采样参数与引擎环境变量配置
├── build.sh                       # 一键编译脚本
├── start.sh                       # 幂等启动脚本 (默认 API 端口 :11438，控制端口 :11439)
├── stop.sh                        # 优雅停止脚本 (释放全部 44 GiB 显存与 64 GiB 内存)
├── status.sh                      # 实时状态与健康检查脚本
└── DEPLOYMENT-SOP.md              # 完整部署、复现与深度调优 SOP 手册
```

---

## 三、快速上手

### 1. 编译引擎
```bash
chmod +x build.sh start.sh stop.sh status.sh
./build.sh
```

### 2. 启动服务与 WebUI
```bash
./start.sh
```
- **主服务与实时监控面板**：`http://<服务器IP>:11438/`
- **常驻零显存启动器面板**：`http://<服务器IP>:11439/`
- **兼容接口**：
  - OpenAI Chat Completions: `POST http://<服务器IP>:11438/v1/chat/completions`
  - OpenAI Responses API: `POST http://<服务器IP>:11438/v1/responses`
  - Anthropic Messages API: `POST http://<服务器IP>:11438/v1/messages`

### 3. 查看状态 / 停止服务
```bash
./status.sh
./stop.sh
```

### 4. 图片输入（视觉）
- 三种协议都可直接传图（`image_url` / Anthropic `image` 块），一请求可多图。
- 读图在 **CPU** 上完成（`strata-vision` + `mmproj-BF16.gguf`，显存零占用），640×320 ≈ 200 token / ~5 s；
  同一张图按内容哈希缓存，重复请求不再编码。
- 接线与排障见 [DEPLOYMENT-SOP.md](./DEPLOYMENT-SOP.md) 第五节；关闭只需删掉 `config.json` 的 `"vision"` 段（收到图片会明确报错，不会静默丢图）。

## 四、投机接受策略与复读防线（`WZY_TYP_*` / `WZY_LOOP_*`）

### 4.1 默认即精确验证

`WZY_TYP_RANK=1`（默认）就是完全精确的无偏投机验证，逐 token 不变。

**作用范围**：所有开关仅影响 `temperature > 0` 的流量。`temp <= 1e-5` 时引擎走 `fast_greedy` 的 GPU argmax 分支（`row_argmax_multi_kernel`），**根本不进验证函数**，因此 Agent / 代码工具链的贪心输出不受任何影响。

### 4.2 三个开关

| 变量 | 含义 | 默认 |
| :--- | :--- | :---: |
| `WZY_TYP_RANK` | 草稿 token 落在目标模型 top-16 候选的前 R 名内即接受；`1` = 精确 | `1` |
| `WZY_TYP_MARGIN` | logit 分差护栏：仅当 `top_v[0] - top_v[r] <= margin` 时才放行第 r 名 | `5.0`（RANK=1 时不生效） |
| `WZY_TYP_MAX_SLACK` | **每个投机窗口内**允许的“非 argmax 妥协”次数上限；`0` = 只认 argmax | `0` |
| `WZY_TANDEM_GUARD` | 检测 history 尾部 `X^m` 型重复，命中则把“下一个循环 token”在 top_v 上压 −6.0；仅 temp>0 | `1` |

**⚠️ 单位陷阱**：`top_v[]` 是 `row_top16_multi_kernel` 直接对 **raw logits** 取 top-16 的结果，**没有除 temperature**。护栏真实严格度是 `margin / T`：`T=0.3` 时 `margin=1.8` 已经意味着第二名概率比 ≤ `e^-6 ≈ 1/400`，**看起来设了实际等于关掉**。请按实测标定，不要按直觉填。

### 4.3 实测（temp=0.3，2× RTX 2080 Ti 22GB）

三组工作负载：普通长文（中文/英文/环形队列代码）、**复读诱因**（枚举 / 配置表 / boilerplate 代码）、长上下文（34,873 token）。表中为 20-gram 重复率（越低越好）与 decode 速度。

| 配置 | 中文长文 | 英文长文 | 环形队列代码 | **boilerplate 代码** | 速度 tok/s |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **RANK=1（默认）** | 0.000% | 0.894% | 0.233% | **0.000%** | 66–78 |
| RANK=8, margin=5.0 | 0.000% | 0.000% | 11.534% | **53.700%** | 87–110 |
| RANK=8, margin=1.0 | 0.000% | 0.000% | 0.000% | **53.700%** | 84–87 |
| RANK=3, margin=1.5, slack=1 | 0.000% | 0.000% | 0.526% | 3.232% | 70–82 |
| RANK=8, margin=1.0, slack=1 | 0.000% | 0.000% | 0.000% | 6.544% | 66–83 |
| RANK=8, margin=5.0, **slack=0** | — | — | — | **0.000%** | 57 |

结论（均为实测，非推断）：

1. **复读完全由“非 argmax 接受”驱动**。`slack=0` 时把 rank/margin 全部放宽到 8 / 5.0，boilerplate 重复率仍是 0.000%；只要允许非 argmax 接受，boilerplate 就会崩。
2. `WZY_TYP_MAX_SLACK` 是最有效的一刀：53.7% → 3.2%（**16×**），但仍压不回基线 0.000%。
3. **提速与零复读在本模型上不可兼得**。收紧到能压制复读时，速度（66–83）与 RANK=1（66–78）已重叠。所以**生产默认 RANK=1**，三个开关留给实验。
4. 34,873 token 长上下文下所有配置重复率均为 0.000%，**长上下文不是复读诱因**。
5. 单样本方差很大（同一 boilerplate 任务在 slack=1 下测到 3.2% 与 6.5%），切勿用单次结果下结论。

> 口径说明：上表数据实测于 `WZY_` 命名构建（双 RTX 2080 Ti 22GB，`temperature=0.3`）。

### 4.4 第三层防线：输出侧复读熔断（默认开启）

`serve/server.py` 的 `LoopGuard`：在最近 `W`=768 个生成 token 内，若同一个 `G`=48 token 的块出现 `R`=3 次，判定进入复读吸引子 → 向引擎发 `SIGUSR1`（与客户端中断同一路径），引擎正常收尾，请求以 `finish_reason="stop"` + `loop_guard: true` 结束。

- 配置：`config.json` 顶层 `"loop_guard": {"enabled": true, "gram": 48, "repeats": 3, "window": 768}`，或 env `WZY_LOOP_GUARD` / `WZY_LOOP_GRAM` / `WZY_LOOP_REPEATS` / `WZY_LOOP_WINDOW`；`WZY_LOOP_GUARD=0` 关闭。
- 实测：把阈值调到 4/2/64 时，一个要求重复 40 行的请求在 **16 token** 被截断（本该 600 token），`finish_reason=stop`；恢复默认阈值后，400 token 正常生成**零误触发**。
- 意义：无论复读来自模型自身还是接受策略，都不会再烧掉 18k token。

### 4.5 CUDA graph：实测后否决

层循环体内 46 个 kernel × 24 层 = **每阶段 1,104 次 launch**，且 0 个 `cudaStreamSynchronize`、0 个同步 `cudaMemcpy`，看起来是 graph 的理想对象。微基准（单卡空 kernel，1,104 次）实测：

| 项目 | 实测 |
| :--- | :---: |
| 逐次发射 | **1.848 ms**（1.67 µs/launch） |
| 同样 1,104 节点的图回放 | 1.081 ms |
| 每阶段净省 | **0.767 ms**（每窗口 1.53 ms ≈ **3.4–4.6%**） |

**结论：不值得。** 每阶段 ≈13 ms 的“与行数无关”开销不是发射间隙，而是 1,104 个小 kernel 的真实 GPU 执行时间（≈11 µs/个，每层都要再把权重读一遍）。要动它必须改内核或合批，不是 graph。

---

详细硬件解锁、模型路径配置、中文提速原理与排障指南请参阅同目录下的 **[DEPLOYMENT-SOP.md](./DEPLOYMENT-SOP.md)**。

---

## 六、2026-11-05 夜：确定性与投机等价性（实测，待续）

### 6.1 已修复：贪心解码不可复现（冷专家求和顺序）

**症状**：同一 prompt、同一配置连发三次，temp=0 的输出在第 298 / 353 个字符处分叉（长度 710 / 714 / 715）。

**根因**：`split_route_hit_cold_kernel` 用 `atomicAdd(&s_ncold, 1)` 分配冷专家组编号 `cg`，顺序取决于 GPU 线程调度。CPU 侧 `for (cg = 0..n_cold) cur_out[...] += ...` 按该顺序累加，而**浮点加法不满足结合律** —— 顺序一变，末位就变，near-tie 处 argmax 翻转。

**修法**：`wzy_stage_v2.cuh` 的 `service_one_cold_layer` 中，调用 `cpu_pool->run()` 前把冷专家组按 `cold_gkey` 升序插入排序（连带 `cold_cnt / cold_tok / cold_wt`）。≤80 个元素，微秒级，不改精度只固定顺序。

**验证**：连发三次，705 / 705 / 705 字符，逐字节一致（此前为 `False / firstdiff 298`）。

**连带影响**：此前所有 A/B 数字都带这个噪声。例如 `depth=2` 曾测得 63.94，修复后为 58.82–59.54 —— **±8% 的摆动来自这个 bug**。`bench/bench_decode.py` 的 `identical` 断言修复后才真正可用；它此前报的 `DIVERGES` 多为误报。

### 6.2 未修复：投机解码与自回归不等价（正确性 bug）

用 `WZY_NOMTP=1`（纯自回归 argmax，数学上唯一正确的参考）对比：

| 配置 | tok/s | tok/win | ms/win | s0 | s1 | mtp | 对比纯 argmax |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| NO MTP（参考） | 46.85 | 1.01 | **21.51** | 8.43 | 9.68 | 0.00 | 基准 |
| depth=2 | 59.54 | 1.88 | 31.62 | 12.63 | 12.55 | 2.82 | **DIVERGES@26** |
| depth=7 | 61.31 | 1.94 | 31.63 | 12.78 | 12.46 | 3.44 | **DIVERGES@26** |

两个投机配置在**同一位置 @26** 偏离参考，且彼此一致 —— 所以偏离不是 `SPEC_DEPTH` 造成的，而是**投机验证路径本身**不等价于自回归解码。

#### 6.2.1 判定工具先修正（`bench/bench_decode.py`）

旧版有两个缺陷，之前的 `DIVERGES` 结论都建立在它上面：
1. 参考行取「本轮第一个跑的配置」，`--only` 跑子集时基准会漂移 → 现在固定为 `WZY_NOMTP=1`；
2. 只比 token id → 现在分三级：`identical`（id 全同）/ `same-text`（token 切分不同但解码文本相同）/ `DIVERGES@tokX,charY`，并打印首处分叉的双向上下文。

修正后重跑（prose 128 token，temp=0，全部生成完）：

| 配置 | tok/s | tok/win | ms/win | enq | s0 | s1 | mtp | 对比纯 argmax |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| NO MTP（参考） | 47.78 | 1.01 | **21.09** | 3.43 | 8.11 | 9.54 | 0.00 | 基准 |
| depth=2 | 60.66 | 1.88 | 31.03 | 3.41 | 12.04 | 12.54 | 2.85 | **DIVERGES@tok26,char39** |
| depth=7 | 61.16 | 1.94 | 31.71 | 2.81 | 12.85 | 12.45 | 3.42 | **DIVERGES@tok26,char39** |

首处分叉的文本：

```
ref  : 专家，但为了获取这些「动态选择的」专家权重，系统
this : 专家，但为了获取这些「分散的」  专家权重，系统
```

**所以「投机无损」这个结论不成立**：文本真的变了，不是 token 切分差异。

#### 6.2.2 已量化：同一位置、同一对候选，logits 差 0.1–0.35

`WZY_DBG_TOPS=1`（新增诊断开关，强制走完整 logits 路径）逐行打印 top-1/top-2：

| gen=26 | top1 | top2 | 选中 |
| :--- | ---: | ---: | ---: |
| 纯 argmax | 99142 / **18.4003** | 109494 / 18.2849 | 99142 |
| depth=2 | 109494 / **18.6196** | 99142 / 18.2912 | 109494 |

前 25 个 token 两条路径完全一致，同一对候选、同一位置，两个候选的 logits 差 0.12 / 0.33 —— 这不是浮点末位噪声的量级。

#### 6.2.3 根因范围：窗口宽度会改变 **row 0** 的 logits

同一前缀（只跑 prompt），只改草稿行数，且 `WZY_FREEZE_CACHE=1`（`promoted=0`，两次运行驻留映射相同）：

| 步骤 | depth=2 | depth=4 | 差 |
| :--- | ---: | ---: | ---: |
| gen=1 row=0 | 22.8242 | 22.8457 | 0.02 |
| gen=2 row=0 | 27.9864 | 27.6346 | 0.35 |
| gen=3 row=1 | 31.9230 | 30.6787 | 1.24 |

row 0 在数学上只能看到 prompt 自身（`cur_pos = base_pos + 0` ⇒ `n_kv = 24`），却随窗口里后面还有几行草稿而变。

**已排除**：
- 快照/回滚（6.1 修复后同配置连发三次逐字节一致；row 0 根本不读快照）
- 注意力因果（`qsa_attend_int8_sparse_multi`：`cur_pos = base_pos + qi`、`n_kv = cur_pos + 1`，逐行严格因果）
- 专家驻留（冻结缓存 + `promoted=0`，两次运行驻留相同）
- 冷专家求和顺序（6.1 已改为按 `gkey` 规范序）

**最强候选解释**（已在 6.2.4 缩小到「第 1 层内的某个 T 相关 kernel」，但具体子步未定）：`T>1` 时代码按 token 数特化 kernel（`gemm_q8p_t<1..8>`、`gemm_q8p_sk<T_,S_>`、grouped 专家核、`gemm_bf16_t<T>`），row 0 的求和顺序随之改变；路由是**离散**的 —— 一次专家选择翻转就能把 ULP 级差异放大成 0.1–1.2 的 logits 差，于是 near-tie 处翻 token。

#### 6.2.4 已定位：种子在**第 1 层**，路由翻转在**第 8/9 层**（2026-11-05 续）

新增逐层探针（`WZY_DBG_TOPS` + `dbg_probe_layer`）：每层对 row 0 写两个签名 —— `route` = 该行 top-10 专家选择的 FNV，`full` = 该行 2560 个 MoE 输出元素**位模式**的 FNV（不是求和；求和会被正负抵消搞坏，实测会误报）。

同一前缀（gen=1..25 两档完全一致），在 gen=26 处逐层对比（冻结缓存、驻留相同）：

| 层 | A = 纯 argmax（W=1） | B = depth=2（W=3） | 判定 |
| :--- | :--- | :--- | :--- |
| **l=0** | `full=8562ea65` | `full=8562ea65` | **位级相同** |
| **l=1** | `route=019ac339` `full=e99ffcf3` | `route=019ac339` `full=cadf4036` | **首个差异：路由相同、纯数值**（约 1 ULP） |
| l=2 | `route=66c0c67e` | `route=66c0c67e` | 数值继续漂 |
| **l=8/9** | — | — | **首个 expert 选择翻转**（冻结下是 l=9） |
| l=0..47 | — | 22–23 层路由不同 | 放大量：48 层里近一半换了专家 |

同时验证了 **host 侧输入在两档间完全一致**（`[IN]` 探针：embedding 与 PLE 的 row 0 值与和逐窗口相同），所以种子不在 host、也不在第 0 层 —— 就在**第 1 层内部的某个 `T` 相关 kernel**（第 1 层是唯一跑 PLE 块、且要折叠第 0 层 `pending_bo` 的层）。

完整因果链：**第 1 层 1 ULP 求和顺序差异 → 逐层放大 → 第 8/9 层一次离散路由翻转 → 22–23 层专家集不同 → 输出 logits 差 0.1–0.35 → near-tie 处翻 token → 文本改写**。这不是状态/回滚 bug，是「按 token 数特化」的数值不等价被 MoE 的离散性放大。

**下一步（范围已经很窄）**：在第 1 层内部逐子步签名（`gr_read` 后 / attention 后 / GDN step 后 / PLE conv 后 / shared / routed / combine），找出 row 0 的第一个不一致子步；把那一步的 kernel 在 `T≤MAX_W` 时固定成与 `T=1` 相同的求和顺序。

**复现**：
```bash
cd ~/wzy-flash-serve
python3 bench/bench_decode.py --tokens 128              # 判定表
WZY_DBG_TOPS=1 python3 /tmp/probe.py d2                 # 逐行 top1/top2
WZY_FREEZE_CACHE=1 WZY_DBG_TOPS=1 python3 /tmp/probe.py fz4 WZY_SPEC_DEPTH=4
```

### 6.3 窗口经济学的干净数字（修复后实测，prose 128 token）

| 配置 | tok/s | ms/win | s0 | s1 | mtp |
| :--- | ---: | ---: | ---: | ---: | ---: |
| 纯 argmax（无投机） | 46.85 | **21.51** | 8.43 | 9.68 | 0 |
| depth=2 | 59.54 | 31.62 | 12.63 | 12.55 | 2.82 |
| depth=7 | 61.31 | 31.63 | 12.78 | 12.46 | 3.44 |
| pipe=2 | 53.99 | 139.46 | 8.07 | 9.54 | **16.26** |
| pipe=4 | 46.28 | 162.69 | 8.11 | 9.60 | **22.57** |

- **投机让每阶段慢 3.5 ms（s0+s1 从 18.1 涨到 25.2 ms，+39%）**，但 tok/win 从 1.01 涨到 1.88，净收益 +27%。
- **`PIPE_TOK` 是负结果，且原因明确**：它把 s0/s1 降下来了，但 `mtp` 从 2.82 暴涨到 16.26 / 22.57（5.8–8 倍），因为需要同时维护两条 MTP 草稿链。净速度 −8% / −21%。
- **Strata 的 22.0 ms/win 接近我们的「无投机」窗口速度（21.51 ms）** —— 它的技术优势在于**投机不额外收费**（每窗口只报 0.58 个草稿，mtp 几乎为零），而我们的投机要付 39% 的窗口代价。

### 6.4 生产配置（当前）

```
WZY_SPEC_DEPTH    = 2      # 与 7 速度无差异（31.62 vs 31.63 ms/win），但 mtp 更省
WZY_TYP_RANK      = 1      # 精确验证
WZY_TYP_MAX_SLACK = 0
WZY_TANDEM_GUARD  = 0      # 纯 argmax 基线
WZY_PIPE_TOK      = 0      # 实测负收益
```

唯一被证明「**可复现**」的组合（同配置连发逐字节一致）。**不要再把它叫「无损」**：按 6.2.1 的文本级判定，投机路径在 near-tie 处会给出与纯 argmax 不同的文本。

诊断开关（默认关闭，无性能影响）：

```
WZY_DBG_TOPS=1      # 强制完整 logits 路径，逐行 stderr 打印 [DBG] top1/top2/gap
WZY_FREEZE_CACHE=1  # 关闭专家提升/淘汰，驻留映射固定（A/B 数值对比必须开）
WZY_DEC_PROF=1      # 每窗口 [DEC-PROF] 分解 enq/s0/s1/mtp
```

## 七、解码速度：与 Strata 的差距拆解（2026-11-05 续）

### 7.1 现状（双卡 RTX 2080 Ti 22GB，prose，temp=0，128–200 token）

| | WZY（当前生产） | Strata |
| :--- | ---: | ---: |
| tok/s | **61–64** | **70.0** |
| ms/win | **29.1–29.7** | **22.0** |
| tok/win | 1.82–1.94 | 1.54 |
| ms/token | 15.6–16.3 | 14.3 |

结论：**每 token 成本只差 9–14%，输在「每窗口固定开销 + 窗口颗粒度」，不是带宽。**

### 7.2 本轮拿到的真实收益：+6~8%（配置层唯一有效的一枪）

生产配置从 `SPEC_MIN_P=0.40 / SPEC_GATE=1.0` 改为 **`SPEC_MIN_P=0.50 / SPEC_GATE=0.7`**，三次复现（62.9 / 62.0 / 64.0 vs 基线 58.8 / 59.5）：

| 配置 | tok/s | tok/win | ms/win | s0 | s1 | mtp |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| base（min_p 0.40 / gate 1.0） | 58.8–59.5 | 1.88 | 31.7–32.0 | 12.6–12.8 | 12.8–12.9 | 2.7–2.9 |
| **min_p 0.50 / gate 0.7** | **62.0–64.0** | 1.83–1.94 | **29.1–30.3** | 11.6–12.0 | 11.9–12.4 | 2.5–2.7 |

机理：更紧的草稿置信门 → 每窗行数略降（W 从 3 常降到 2/3 混合）→ **窗口本身便宜 8%**，而 tok/win 基本不变。继续收紧（min_p 0.60/0.70）窗口还能到 27.7–28.3 ms，但 tok/win 塌到 1.60–1.66，净变慢（57.7–58.8）。最优点就在 **min_p 0.50 / gate 0.7**。

质量抽检（200 token，生产配置）：`dup20 = 0/139 (0.0%)`，自然 `stop`，文本连贯。

### 7.3 已被实测否掉的

- **12 组参数扫描**：`GEMM_WARPS` 4/16、`GK_SPLIT` 2/4、`CPU_THREADS` 28、`PROMO_PER_WIN` 32、`ADAPT_DEPTH` 0、`SPEC_GATE` 0.5/0.6/0.75/0.8、`depth` 3 —— 全部在 ±5% 噪声内（`GEMM_WARPS=32` 与 `GK_SPLIT=2` 起不来：显存不足/初始化失败）。
- **「主机喂不饱 GPU」假设**：新增 `WZY_COLD_INLINE=0`（整窗先入队、冷专家集中补齐）→ 59.2 vs 基线 60.7，**无收益**，假设被否。GPU 解码期采样 util 仅 ~35%，但那是内核颗粒度太小（每阶段约 1100 个 launch、平均 11 µs）导致的欠占用，不是主机阻塞。

### 7.4 剩下的三个杠杆（按收益排序，都需要改代码）

1. **CUDA graph 回放**：现在每窗口 `enq≈3.1 ms`（占 10%）纯粹是 ~1000 次 `cudaLaunchKernel` 的主机时间。按 (W, 位置模版) 捕获整窗图并回放，可期望 **+8~10%**（→ 67–70 tok/s，与 Strata 持平）。
2. **MTP 草稿链成本 `mtp≈2.5 ms`（占 8%）**：它挂在窗口之后串行执行，无法与 stage 计算重叠（依赖最后一行隐状态）。可做的是减少链内同步点 / 单 kernel 多步草案。全免 → **+8%**。
3. **合并每层内核**（核心 24 ms）：每阶段约 1100 个 launch、平均 11 µs，属延迟受限。合并 conv/step/norm 与 GEMM 之前的量化、把 2–3 个小 kernel 融成 1 个，才动得了这部分；这是几天的活。

### 7.5 生产配置（速度版）

```
WZY_SPEC_DEPTH = 2
WZY_SPEC_MIN_P = 0.50     # 本轮改（原 0.40）
WZY_SPEC_GATE  = 0.7      # 本轮改（原缺省 1.0）
WZY_TYP_RANK   = 1        # 精确验证，不变
WZY_PIPE_TOK   = 0
```

备份：`/tmp/config.json.bak-prespeed`。注意 6.2 的宽度相关数值不等价仍然存在，改这几个门只影响「报几个草稿」，不改变校验精度。

### 7.6 `enq` 3.0 ms 的成分已拆开（2026-11-05 续）

`[DEC-PROF]` 新增 `stage=` 字段（主机侧 embedding 取数 + PLE 3-gram 取数的时间）：

```
wins=79 (1.90 tok/win) total=2341ms (29.64 ms/win) | enq=3.01 s0=11.70 s1=12.23 mtp=2.56 stage=0.51
```

**`enq` 的 3.01 ms 里 0.51 ms 是取数，剩下 2.5 ms 全是 ~1000 次 `cudaLaunchKernel` 的主机时间。**
这就是 CUDA graph 的靶子：拿下它 = 每窗口 −2.5 ms ≈ **+9%**（29.6 → 27.1 ms/win → 70 tok/s，与 Strata 持平）。

同一轮 150 token 实测：**64.06 tok/s**（`DONE` 里 `finish=length`）。

### 7.7 CUDA graph 的可行做法（本轮已验证前提，未实现）

难点不是捕获，而是**冷专家握手的 `seq` 是按窗口自增、且以值传给内核**（`split_route_hit_cold_kernel(..., seq, ...)` / `wait_and_add_cold_out_kernel(..., seq, ...)`），捕获后的图会带着旧 `seq` 重放。可行方案（已推演）：

1. **`seq` 常量 + 图首重置**：图内第一个 kernel 把该 stage 所有层的 `cold_req[l].ready_seq / cpu_done_seq` 清 0，层内核统一用固定 `seq=1`；
2. **捕获形状**：`WZY_COLD_INLINE=0`（已验证可用、与基线同速）让主机在层循环里不阻塞，整窗内核序列可被 `cudaStreamBeginCapture/EndCapture` 捕获；按 `(stage, W, need_logits)` 缓存多个 graph（解码只有 W=2..MAX_W 少数形状）；
3. **重放时的 CPU 角色**：`cudaGraphLaunch` 后，主机照旧跑 `service_cold_layers(start_l=l_begin)`，用 `ready_seq==1` 逐层推进（GPU 在 `wait_and_add_cold_out` 上自旋等它）——即图负责发射、CPU 负责冷专家，职责与现在一致；
4. 需要绕开的捕获期禁用项：`cudaStreamSynchronize`、`_mm_pause` 轮询、`cold_seq > 0xF0000000` 的重置块，全部在图外或加 capture 标志跳过。

结论：实现它是几小时的活，收益 +9%，且是**唯一能不动内核就把 Strata 追上/持平**的路径；再往上（+20% 以上）必须合并每层的 ~28 个 kernel（7.4 第 3 项）。

回滚点：`/tmp/wzy_pp2_engine.good-speed`（改 graph 前的良好二进制）。

### 7.8 实测：stage0 单档 CUDA graph 是负结果（2026-11-05 收尾）

按 7.7 的方案实现了 `WZY_GRAPH_STAGE0`（捕获 + 常量 seq + 图首重置 + 回放）：功能跑通、结果正确、能自然 `stop`，但速度变慢。

| 配置 | tok/s | ms/win | enq | s0 | s1 |
| :--- | ---: | ---: | ---: | ---: | ---: |
| 基线（逐层发射） | 63.9 | **29.7** | 3.22 | 11.74 | 12.04 |
| stage0 graph | 55.9 | **40.9** | **2.56** | **16.7** | 17.05 |

`enq` 确实从 3.22 降到 2.56（图只发一次），但 s0 涨了约 5 ms、总计 +11 ms。

原因：这正是原代码把冷专家服务写进层循环的原因。现在每层末尾的 `wait_and_add_cold_out` 要让 GPU 自旋等 CPU，而主机在 graph 模式下不再有「发射下一层」的活可干，26 次逐层会合的延迟全部暴露在关键路径上；原实现靠「一边发射一边服务」把这些会合掩盖掉了。

结论：CUDA graph 单独上不划算，必须先做独立冷专家服务线程（让 CPU 始终跑在 GPU 前面，把逐层会合变成并行），两件事必须一起做。

排障确认的两条硬事实：

1. 图内反必须以 `cudaStreamSynchronize(stage0.stream)` 开头：重置 `ready_seq/cpu_done_seq` 时若上一窗的自旋内核还在飞，会被清零后的计数器永久卡死（实测死锁）。
2. 服务点位置不能动：挪到 stage1 dispatch 之前等于把两档串行化，40.9 ms/win 中大部分损失来自这一点。

源码已回滚到 graph 之前的版本（`grep WZY_GRAPH_STAGE0 src/*.cu` 为空），备份在 `/tmp/wzy_engine_v2.cu.bak-graph`、`/tmp/wzy_stage_v2.cuh.bak-graph`，回滚后实测 63.9 tok/s / 29.7 ms/win 与之前一致。

### 7.9 下一步的确定设计：冷专家异步服务线程（未实现）

目标：把 48 层 × 每层一次「GPU 自旋等 CPU」的会合从关键路径上挪走。这是打开 graph 的前提，本身也可能直接带来收益。

设计（最小改动版）：

1. `StageGpu` 增加常驻服务线程 `cold_worker`，每窗只接收一个任务：`{stage, host_pack, cpu_pool, base_seq, T, l_begin, l_end}`，然后从 `l_begin` 顺序调用 `service_one_cold_layer(l, ...)`（该函数已按 `ready_seq` 自旋等待，天然「跟着 GPU 跑」）。
2. `record_forward` 的 `inline_service` 恒为 `false`（即已验证的 `WZY_COLD_INLINE=0` 形态），主线程只入队、不阻塞。
3. `run_verify_window` 在 stage0 入队完成后立刻把 stage0 任务投给 worker、再 dispatch stage1，然后在原来 `service_cold_layers` 的位置 `join` worker。
4. `promote_cold` 保留在 `join` 之后（`pending_promo` 只在服务线程里写，join 提供同步）。
5. 判据：`s0` 应明显低于现在的 11.74 ms；成功线是 `ms/win` < 29.7，< 27 才谈得上追平 Strata 的 22.0。之后叠加 `WZY_GRAPH_STAGE0`，看 `enq` 的 2.5 ms 是否变成净收益。

风险点：`CpuMoePool` 由两档共用，两档服务线程并发是否安全未验证。保守做法是先只给 stage0 一个服务线程，stage1 保持主线程串行。

回滚点：本轮良好状态 = 63.9 tok/s / 29.7 ms/win（源码已回滚到 graph 之前，二进制 `bin/wzy_pp2_engine`）。

### 7.10 对 7.9 的更正：异步服务线程解决不了问题（重要）

7.9 的假设（把冷专家服务挪到独立线程就能降 s0）经复核**不成立**：

- `s0`(11.74) 与 `s1`(12.04) 是同一窗口内**串行**的 GPU 工作：stage1 的内核必须等 stage0 的输出（`cudaStreamWaitEvent(ev_s0_done)` + `cudaMemcpyPeerAsync`），两档无法并行。
- stage0 的 `record_forward` **本来就是 `inline_service=false`**（纯入队 + 非阻塞 drain），服务已经在 stage1 dispatch 之后、与 stage1 的入队重叠执行。换个线程做同一件串行的事不会更快，这也解释了 `WZY_COLD_INLINE=0` 为什么零收益。

**真实结构**：每窗口 ≈ GPU 串行 23.7 ms（s0+s1）+ enq 3.2 + mtp 2.6 = 29.7 ms。1.90 tok/win ⇒ 每 token 仅 12.5 ms 的 GPU 时间（对应 80 tok/s 的潜力），**那 20% 差距全在 enq 与 mtp 两项开销上**。

**所以下一步应改为「不需要重置、也不需要排空同步」的 graph**，即把 `seq` 换成设备侧 epoch 计数器：

1. `split_route_hit_cold_kernel` / `wait_and_add_cold_out_kernel` 不再接收 `seq` 值，改接收 `const uint32_t* d_epoch`，内部用 `(*d_epoch) * NL + (l - l_begin) + 1`；
2. 图首放一个 `++*d_epoch` 的小 kernel；
3. `service_one_cold_layer` 的期望值按同一公式计算（主机读 mapped 的 `d_epoch`）。

这样回放既不用清零计数器、也不用 `cudaStreamSynchronize`。本轮 graph 的 11 ms 损失正来自这两件事（去掉同步会死锁，保留同步就吃掉全部收益）。改完后 enq 的 2.5 ms 才是净赚：29.7 → 27.2，约 70 tok/s，与 Strata 持平。

回滚点不变：63.9 tok/s / 29.7 ms/win，源码干净。

### 7.11 7.10 的落地补丁清单（精确到行，未实施）

已核对到具体的改动点，全部在 `src/wzy_cpu_moe.cuh` 与 `src/wzy_stage_v2.cuh`：

| # | 位置 | 改动 |
|---|---|---|
| 1 | `wzy_cpu_moe.cuh:42` | `split_route_hit_cold_kernel(..., ColdLayerReq* req, uint32_t seq_id, ...)` 的第 6 参改为 `const uint32_t* d_epoch`；第 146 行 `req->ready_seq = seq_id;` 改为 `req->ready_seq = (*d_epoch) * (uint32_t) N_LAYERS + (uint32_t) (l - l_begin) + 1u;`（需把 `l_begin` 一并传入，或直接传入已算好的 `seq_base` 指针） |
| 2 | `wzy_cpu_moe.cuh:156` | `wait_and_add_cold_out_kernel(..., uint32_t seq_id, int n, ...)` 同法改为读 `d_epoch`；第 170 行 `while (done != seq_id)`、第 176 行 `s_done == seq_id` 用同一公式 |
| 3 | `wzy_stage_v2.cuh:718`（层内发射 `split_route_hit_cold_kernel`） | 第 5 参由 `seq` 改为 `d_epoch` 指针 |
| 4 | `wzy_stage_v2.cuh:736`（`wait_and_add_cold_out_kernel` 发射处） | 同上 |
| 5 | `wzy_stage_v2.cuh:752`（`service_one_cold_layer` 里 `const uint32_t seq = base_seq + (l - l_begin) + 1u;`） | 改为按 epoch 读：`seq = (*h_epoch) * NL + (l - l_begin) + 1u`（`h_epoch` 是 mapped 主机指针） |
| 6 | 图首 | 新增 `bump_epoch<<<1,1>>>`：`++*d_epoch;`，捕获进图的第一条 |

要点：`d_epoch` 用 `cudaHostAlloc(mapped)` 分配，GPU 自增、主机直接读；这样回放不需要清零、也不需要 `cudaStreamSynchronize`，7.8 里那 11 ms 的损失来源被彻底去掉。同一窗口内两档共享同一个 epoch 值即可（各自的 `l - l_begin` 偏移区分层次）。

验证顺序：先不加 graph，只换成 epoch（应完全等价、速度不变，token 流与基线一致）→ 再开 `WZY_GRAPH_STAGE0`，看 `enq` 降 2.5 ms 是否直接反映到 `ms/win`。

### 7.12 Stage A 实现了，但撞到更根本的问题：解码本身不可复现（2026-11-05 本轮）

**结论先行**：上一轮交接里"token 流必须与基线逐字节一致"这个判据不成立——**基线二进制自己对同一 prompt 连跑就会给出两条不同的 token 流**。之前几轮"改-建-测-回滚"是在追一个不存在的目标。

#### 7.12.1 交接里那条 `mkey` 线索：排除

`grep -an mkey src/` 后确认，`mkey + l` 是 `split_route_hit_cold_kernel` 的最后一个参数 `d_ncold_out`，即**每层冷专家组数的快路标志**（`wait_and_add_cold_out_kernel` 用它做 `*d_ncold <= 0` 的 1-cycle 早退），与 seq 语义无关。不是第二套握手。

并且记录里真正的"第五处公式"是 `record_forward` 里的 peer 服务路径（`want0 = peer_base + (*peer_l0 - peer_s0->l_begin) + 1`），但 pp2 解码路径两处 `record_forward` 都传默认 `peer_s0=nullptr, peer_base=0`，该路径是死代码——也不是差异源。

#### 7.12.2 stage A 的三种实现（全部实测过）

| 版本 | 做法 | 结果 |
|---|---|---|
| 取 3 | 主机在窗口入口写 mapped 单元格，内核读 `*d_seqbase`；数值与旧 seq **完全相同** | 跑得通、输出连贯，但**有竞态**：同一二进制两次跑出不同 token 流（下一窗口的写被上一窗口未执行的内核读到） |
| 取 4 | `d_epoch` 放设备内存，**流内窗口末尾** bump kernel 自增；内核算 `epoch*nlay + (l-l_begin) + 1` | 内部确定（3/3 一致），但与基线分叉 |
| 取 5 | 同上，但值改成与旧 `cold_seq` **逐位相同**（只在真正走冷专家分支的窗口推进，步长 nlay） | 又变不稳定（3 次 3 条流） |

取 4/取 5 的关键设计点（值得保留）：**bump 必须在窗口末尾、同流内**。窗口入口写值（取 3）或主机写值都不安全，因为 `inline_service=false` 时同一窗口的内核可以晚于下一窗口的主机写才执行。

#### 7.12.3 真正的发现：基线贪心解码不可复现

用 21:09 的基线二进制（`/tmp/wzy_pp2_engine.epoch-base`）、同一 prompt、`temperature=0`、连跑 5 次：

| 结果 | 次数 | tok/s | tok/win | promo |
|---|---:|---:|---:|---:|
| 流 A（hash 10d477f3…） | 2 | 65.35 / 64.86 | 1.92 | 372 |
| 流 B（hash 2874c5ac…） | 3 | 57.46 / 57.53 / 58.93 | 1.78 | 347~404 |

同样的现象在 96-token 短跑上更清楚：同二进制 5 次里 4 次 1.92 tok/win（63-66 tok/s），1 次 1.78（55 tok/s）。

**机理（证据链，不是猜测）**：`ms/win` 几乎不变（基线 29.7，stage A 29.9），变的只有 `tok/win`；`promo`/`hit` 同步变化 ⇒ 晋升时序不同 ⇒ 冷热专家集合不同 ⇒ 累加顺序不同 ⇒ 宽度相关数值微差（见 6.2）⇒ 某几个草稿的接受与否翻转 ⇒ `WZY_ADAPT_DEPTH` 的 `acc_ema` 反馈把窗口宽度锁进另一档 ⇒ 正反馈。**任何改变握手/服务时序的改动（哪怕是等价的）都可能把解码推进另一个吸引子**——这正是上一轮"归因恢复正常但 token 流仍不同"的原因，也是为什么对齐四处公式修不好它。

#### 7.12.4 现状（已回到可用基线）

- 源码零 epoch 残留（`grep -c "d_epoch\|bump_epoch" src/*` 全 0），`bin/wzy_pp2_engine` 已按基线重建：实测 63.4 tok/s / 30.3 ms/win（另一次跑 55.1 / 32.3，属 7.12.3 的同模态抖动）。
- stage A 留档，可随时重建：`tools/patch_stageA_epoch.py`（从干净树一键应用，含内核 epoch、流内末尾 bump、每档 device epoch）；二进制/源码副本在 `/tmp/wzy_pp2_engine.stageA`、`/tmp/stageA_wzy_*.{cuh,cu}`、备份 `/tmp/*.bak-epoch2`。
- 交替 5 组（96 token）对比：**基线 ms/win 29.72、tok/win 1.92；stage A ms/win 29.86、tok/win 1.66~1.78**。即 stage A 本身几乎不花 ms/win，掉的是接受率档位。

#### 7.12.5 下一步（顺序改了：先可复现，再 graph）

1. **让贪心解码位级可复现**（三个候选，按改动成本排序）：
   a. `WZY_ADAPT_DEPTH=0` 后重测——若两条流合并成一条，则宽度反馈是放大器（验证成本最低，一次 sweep 即可）；
   b. `WZY_NOMTP=1`（W=1，无投机）作为对照，确认数值是否只随宽度变；
   c. 把 `promote_cold` 从"服务完立即提交"改成"窗口末统一提交"，去掉"本窗口晋升影响本窗口数值"的耦合。
2. 复现之后，stage A 的等价性判据才有效（此时才谈得上逐字节比较），再开 `WZY_GRAPH_STAGE0`。
3. graph 的靶子仍是 `enq` 的 2.5 ms（7.6），但 7.8 的教训不变：图回放会暴露逐层会合，必须同时减少会合次数，不能只上 graph。

### 7.13 双稳态根因已定位并修复：晋升顺序非确定（2026-11-05 本轮第 2 段）

#### 7.13.1 定位过程（三个对照实验，各 3~5 次重复运行，`/tmp/eqrun.py --env K=V`）

| 实验 | 改动 | 结果 | 结论 |
|---|---|---|---|
| A | `WZY_ADAPT_DEPTH=0` | 4 次运行 3 条流（hash b2fc/2874/10d4），tok/win 仍有 1.73/1.97 两档 | 否定：宽度反馈只是放大器，接受率本身在漂 |
| B | `WZY_PROMO_MIN_F=99999`（晋升全关） | **9/9 逐字节一致**（prose 4+3、code 2，两种长度） | 确认：非确定源 = 晋升决策 |
| C | 晋升修复（见下） | **9/9 逐字节一致**（prose128 4 次、code160 3 次、prose256 2 次，含自然 `stop`） | 修复有效且保住 hit 率 |

**机理**：`split_route_hit_cold_kernel` 用 `atomicAdd` 分配冷组下标 → 各组的服务顺序每次运行不同。`service_one_cold_layer` 按这个顺序把合格专家 push 进 `pending_promo`，且 push 循环里带 `(int) pending_promo.size() < max_q` 截断——队列满之前**谁在队列里**取决于原子序 → 每次运行晋升的专家子集不同 → VRAM 驻留映射不同 → 累加顺序不同 → 贪心流漂移（README 7.12.3 的证据链在这里闭合：promo/hit 同步变化）。

#### 7.13.2 修复（`tools/patch_promo_order.py`，源码已应用并重建）

两步，无新增 VRAM 流量：

1. `service_one_cold_layer`：push 所有合格候选（去掉 `size() < max_q` 截断）→ 待晋升集合不再依赖原子序；
2. `promote_cold`：按 `(freq 降序, gkey 升序)` 排序后**再**应用 `WZY_PROMO_QUEUE` 截断 → 晋升子集成为"本窗口路由组 + 频次表"的纯函数。

排序在 `promote_cold` 里做（每次窗口一次，候选 ≤ 各层冷组总数，主机开销可忽略）；`freq` 值在服务循环里已按确定的层序更新，是确定的。

#### 7.13.3 修复后的量化结果（对比 7.12.4 的基线）

| | 旧基线（非确定） | 修复后（确定） |
|---|---|---|
| 可复现性 | 2~3 条流 | **9/9 一致**（3 prompt/长度组合） |
| hit（prose） | 93.1~93.9%（漂） | **94.3%**（稳定） |
| tok/win（prose 128） | 1.92 或 1.78（双档） | 1.64（单档） |
| ms/win | 29.7 | 28.4（s0 11.1 / s1 11.9，**比基线略快**） |
| tok/s（prose） | 65.4 / 57.5（漂） | 56.8~60.5（稳定） |
| tok/s（code 160） | — | **75.5 / 74.9** |
| promo | 347~478 | 456（稳定） |

诚实标注：prose 任务的 tok/win 落在 1.64 档（低于旧基线的 1.92 好档位、略高于坏档 1.78 的稳定性），因为"最热优先"的确定性排序与旧原子序的偶然效果不同，被锁进的草稿档位变了。**代价是 prose 档位 −8%，收益是完全可复现 + code 任务 74~75 tok/s（历史最高记录区间）+ ms/win 略降。** 要找回 prose 档位，可以在确定序基础上调 `WZY_PROMO_QUEUE`/`WZY_PROMO_PER_WIN`/`WZY_PROMO_MIN_F`（现在调参结果也可复现了）。

#### 7.13.4 对其他工作的影响

- 上一轮 6.2/7.x 所有"宽度相关数值不等价"的测量都是在晋升非确定的背景下做的；`WZY_FREEZE_CACHE=1` 是当时唯一可信的 A/B 手段，这个结论仍成立。
- **stage A（epoch）重新变成可测**：现在基线流唯一，`/tmp/eqrun.py --ref /tmp/eq_dpl.json` 即可做逐字节等价验证；且修复后的引擎对"服务时序"类改动的敏感度应大幅下降（晋升子集不再依赖时序）。
- 备份：`/tmp/wzy_stage_v2.cuh.bak-promo`（修复前）、`/tmp/wzy_pp2_engine.detpromo`（修复后二进制）、`/tmp/eq_dpl.json`（新基线流）。
- 交接点：先跑 stage A 等价验证（预期可过），再上 `WZY_GRAPH_STAGE0`（enq 2.5 ms 靶子）；prose 档位调参随后。


### 7.14 stage A 等价性通过 + 晋升调参定版（2026-11-05 本轮第 3 段）

#### 7.14.1 stage A（设备侧 epoch 握手）在可复现基线上验证通过

7.13 的确定性晋升修复落位后，7.11 一直缺的"可比较基线"终于成立。`tools/patch_stageA_epoch.py`（take 5：设备侧 epoch、流内窗口末尾 bump、值与旧 `cold_seq` 逐位相同）叠在修复后的源码上，等价性判据 `/tmp/eqrun.py --ref <基线>`：

| 组合 | 结果 |
|---|---|
| prose 128 tok ×3 | **3/3 IDENTICAL**（hash 2543d106… 与基线全同），hit/promo 全同，ms/win 28.3 vs 基线 28.1（噪声内） |
| code 128 tok ×2 | **2/2 IDENTICAL**（hash 1a314c5d…），hit 94.7 / promo 289 全同 |

**stage A 至此完成**：内核握手参数里不再有烘焙的 `seq` 标量，`bump_epoch_kernel` 已在窗内、可被一起捕获——`WZY_GRAPH_STAGE0` 的前置条件全部满足。回滚点：`/tmp/wzy_pp2_engine.detpromo`（无 epoch）、`/tmp/*.bak-preStageA`（源码）、`/tmp/wzy_pp2_engine.stageA4`（含 epoch）。

#### 7.14.2 晋升参数扫描（全部在确定性引擎上，每次改动 token 流可复现）

| 配置（prose 128） | tok/s | tok/win | hit | promo | ms/win |
|---|---:|---:|---:|---:|---:|
| 默认 MIN_F=4 / PW=8 / Q=16（3×） | 57.5~59.7 | 1.64 | 94.3 | 456 | 28.1 |
| MIN_F=3 / Q=32（2×） | ~59.6 | 1.73 | 94.2 | 592 | 29.0 |
| MIN_F=2（默认 PW/Q）（2×） | 61.8~62.6 | 1.75 | 94.9 | 695 | 28.2 |
| **MIN_F=2 / PW=16 / Q=64（3×）** | **60.9~64.4** | **1.86** | 94.9 | 866 | 29.7 |
| MIN_F=1 各档（4 组） | 58~62 | 1.71~1.80 | 94.9~96.1 | 875~1352 | 28.6~29.3 |

选定生产配置：**`WZY_PROMO_MIN_F=2 / WZY_PROMO_PER_WIN=16 / WZY_PROMO_QUEUE=64`**（已写入 config.json；备份 `/tmp/config.json.bak-detpromo`）。

头对头（各 3 次，区间不重叠）：

| | prose tok/s | code tok/s | tok/win |
|---|---:|---:|---:|
| 旧生产（MIN_F=4/PW8/Q16） | 58.5 均值 | 72.0 均值 | 1.64 / 2.29 |
| **新生产（MIN_F=2/PW16/Q64）** | **62.5 均值（+7%）** | **77.6 均值（+8%）** | 1.86 / 2.46 |

对新配置的诚实标注：code 任务 77.6 tok/s 与 prose 62.5 tok/s 是本项目单流引擎的历史最好区间；但 mtp 分项略涨（2.5→3.0），ms/win 略涨（28.1→29.7），净收益来自 tok/win。prose 仍未回到旧非确定引擎偶然到达的 1.92 好档（65 tok/s 单点），那 5% 留给后续 mtp/宽度工作。

#### 7.14.3 当前状态与交接

- 源码 = 基线 + 确定性晋升（7.13）+ stage A epoch（7.14），`bin/wzy_pp2_engine` 已按此重建并通过等价验证。
- config.json env 已更新为新生产配置；服务用该组合拉起。
- 下一步（按收益排序）：
  1. **`WZY_GRAPH_STAGE0` 重做**：现在等价性可逐字节验证、且 epoch 无需重置/排空。注意 7.8 教训——必须同时减少逐层会合，单纯回放曾 -11 ms；
  2. mtp 链瘦身（2.5→3.0 ms，占 9%）：链内同步点/单 kernel 多步草案；
  3. 若两者都落地，enq 2.5 + mtp 3.0 中拿下 3~4 ms 即 30.0 → 26.5 ms/win（~70 tok/s，追平 Strata）。
- 探针：`/tmp/eqrun.py`（`--env K=V` 可覆盖 config，`--ref` 逐字节比对，`--save` 存基线流）；基线流 `/tmp/eq_detBase128.json`、`/tmp/eq_codeBase.json`。


### 7.15 CUDA graph v2：功能全通、字节等价，但突发中立（burst-neutral）（2026-11-05 本轮第 4 段）

#### 7.15.1 实现状态（`tools/` 留档补丁链：patch_graph*.py 的最终态已直接叠在源码里）

`WZY_GRAPH_STAGE0=1` 时，stage0 的解码窗口入队序列按 (T, run_head, want_top16, save_snaps) 形状捕获并回放：

- **806 节点/图**，每个 exec 实测占 **~40 MiB VRAM**（[GRAPH] 诊断行）；
- epoch bump 在图内 → 回放自续，无需重置/同步（7.8 的两个死因全部移除）；
- 捕获/回放全程**字节等价**：`--ref` 比对 3/3+ IDENTICAL（hash 3bb743b1…）；
- 回退安全：instantiate 失败（OOM）→ 打印节点数/空闲显存 → 撤销主机簿记 → 递归重跑普通路径，该形状永不重试；
- 门控 `WZY_GRAPH_MAXW`（默认 4）：warmup 会跑 T=8/6 的宽窗口（各浪费 40 MiB），必须挡掉——不挡时真正的解码形状 T=2/3 会 OOM。

#### 7.15.2 实测：图是"突发中立"的 —— 7.10 的判断被硬数据证实

同一二进制、同配置（MIN_F=2/PW16/Q64）、3×3 交替、128 token：

| | enq | s0 | enq+s0 | ms/win | tok/s |
|---|---:|---:|---:|---:|---:|
| 图 OFF（3×） | 3.32~5.08（均 3.92） | 9.39~11.72（均 10.76） | **≈14.5** | 均 29.9 | 均 62.1 |
| 图 ON（3×） | 0.50~1.26（均 0.79） | 13.28~13.65（均 13.49） | **≈14.5** | 均 29.3 | 均 63.4 |

`enq+s0 ~14.5 ms` 恒定：**发射与冷专家服务本来就是同一份重叠的主机工作，prof 只是把它在两个计数器之间搬来搬去**。这就是 7.8（s0 11.7→16.7）和今天（10.8→13.5）同一现象的两个观测点，也是 `WZY_COLD_INLINE=0` 零收益的根因。结论：**单纯图回放不通向 +9%；7.6 里"拿下 enq 2.5ms"的推导在结构上不成立**（那 2.5ms 与服务重叠着）。

连带修正一个早期推导：`enq` 里的 ~1100 次 launch 并不整体在关键路径上；关键路径是 `enq+s0`（stage0 主机→GPU 会合串）、`s1`（12.2）、`mtp`（3.0）三段。

#### 7.15.3 打开 graph 收益的正确顺序（更新 7.4/7.10）

剩余真实杠杆，按"能否缩短那 14.5ms 会合串"重排：

1. **让 GPU 少等 CPU**：降低冷专家服务成本本身——提高 hit（快满了，94.9%）、或加快 CPU MoE（7.4 #3 的内核合并之外，CPU 侧 12/14 线程的 AVX2 计算约 10ms 是 s0 的主体）；
2. **mtp 链瘦身**（3.0ms，占 10%）：链内同步点/单 kernel 多步草案；
3. **图+异步服务的组合**：只有当服务从主线程挪走（真并发）后，图回放才可能兑现 ~2ms——即 7.9 的设计重新变成前提，但那次被否的原因（两档串联）依然成立，需要更细的流水设计。

图代码保留在源码里（env 关闭、零影响、48 行 diagnostics），`/tmp/wzy_pp2_engine.graph5` 可复测。7.8 遗留的 `exec_body1/exec_full` 成员仍是死代码（无害）。


### 7.17 F1 首战：等价达成、性能证伪，合并方法论修正（2026-11-05 本轮第 6 段）

#### 7.17.1 F1（router 相合并）结果

实现（`/tmp/patch_f1.py`，已回滚）：`dense_bf16(g_inp) + router_top10 + sgate gemm` 三个内核 → 单 kernel `router_gate_top10_fused<<<T,512>>>`（16 warp × 32 专家行，点积/softmax/top10 逐位复刻原顺序）。

| 指标 | 基线 | F1 |
|---|---:|---:|
| token 流 vs `eq_gbase` | — | **3/3 IDENTICAL**（等价性设计成功） |
| router 相（layer-prof） | 0.060 ms/层 | **0.132**（+0.072） |
| 单层 total | 0.535 | 0.608 |
| ms/win | 29.6 | **33.2**（-12%） |

**根因**：原 gemm 用 64 blocks × 8 warps = 512 个 warp（每 warp 一行专家、8 次 j 迭代）；合并后每 token 只有 16 个 warp（每 warp 串行 32 行 = 256 次 j 迭代）。**发射开销省 ~10µs，warp 并行度损失 ~70µs**。

#### 7.17.2 方法论修正（对 F2-F5 约束）

> **合并只允许发生在不减少总 warp 数的前提下**：要么被合并内核的 grid 本来就小（如 sgate rows=1 只有 1 个活动 warp，任意合并都不亏），要么合并后仍能维持原 warp 总数（persistent/megakernel 式重构，另立项目）。

按此重新筛选 7.16 清单：
- F2（exp_gu_quant 960 blocks + exp_down 768 blocks 合并）→ **按原方案会重蹈 F1**，改为"下游内核直接消费上游寄存器/共享内存产物"式重构，工作量大；
- F3（attention 相 3 个小内核合并）→ 需先核对这些内核的 grid：若同为 T 或 HC 量级的小 grid，合并可能成立；
- F5（mtp drafter 步内）：drafter 步 ~1ms 里同样是小内核，但 drafter 是**单 token** 前向，本来并行度就低（NE×HCD 小矩阵），合并更可能成立——**升为首选**；
- sgate rows=1 gemm（1/8 warp 利用率）单独并入 top10 kernel：保留 F1 中这部分（不动 gemm 主干），预计 +0.3-0.5ms/win，低风险。

#### 7.17.3 当前状态

- 源码 = 7.14 定版（确定性晋升 + stage A epoch + graph 代码 env 关闭），F1 已完全回滚（grep 零残留），重建后 token 流与 `eq_gbase` 一致（63.96 tok/s / 29.0 ms/win ✓）。
- 本会话累计成果：贪心解码完全可复现（7.13）、生产配置 +7~8%（7.14：prose 62.5 / code 77.6 tok/s）、stage A 等价验证通过、graph 基础设施与诊断就绪（7.15）、层级剖析与合并方法论（7.16/7.17）。
- 服务：待用回滚后二进制重启（见下方操作）。
- 下轮入口：F5（mtp drafter 步内合并，wzy_mtp.cuh `run_rest_row0`，~40 小内核）或 sgate 并入 top10（低风险快赢）；判据与剖析工具全部就位。


### 7.18 F5 准备 + F1b（sgate 并入 top10）：字节等价、中性偏正（2026-11-06）

#### 7.18.1 F5 准备完成（drafter 步剖析）

`run_rest_row0` + `run_front_kv`（T=1）每 draft 步 **~45 个节点**（含 9 次 `q8_proj` 的
quant+gemm 两连发、head 的 d_head gemm [114452,2560]×[2560,1]、QSA ring attention 等），
≈ 1ms/步 × 3 步 ≈ 3.0ms。步间同步被置信门控的串行语义锁死（不能简单去掉）；
步内节点是 graph 的（`step1_exec`），开销在 GPU 侧节点派发。
可行路线：**"quant 并入 gemv"内核**（per-lane 就地量化 32 块 → dp4a，数值逐位等价：
amax/127、`__float2int_rn`、clamp 全同；代价是 x 按 4B/元素重读进 L2，仅对 rows≤2560
的小投影划算：o_proj/k_proj/v_proj/sh_gate/sh_up/sh_down/mix_up ≈ 7-8 节点/步，
预计 -0.1ms/步 × 3 ≈ 0.3ms/win）。**未实施**——收益/工作量比一般，留作备选。

#### 7.18.2 F1b：sgate gemm 并入 router_top10（已应用，等价）

`launch_gemm_bf16(s_gate_inp, mixed2, sgate, 1, NE, T)` 是只有 1 个 warp 干活的整发射；
`router_top10_fast_multi` 的 top10 阶段 warp 1-7 全闲置 → **warp 1 顺手算 sgate**
（逐位复刻 `gemm_bf16_t rows=1` 的 lane 跨步累加序；T>8 时原走 row1_batch 路径，
per-token 累加序相同，也逐位等价）。改动：
- 内核签名加 `sg_w / sg_x / sg_out / sg_cols` 4 参（wzy_kernels_v2.cuh）；
- 两处旧 sgate 发射删除（冷分支 + 预填充分支各自处理）；
- drafter（wzy_mtp.cuh）与 prefill（wzy_prefill_tc.cuh）的两处调用补 `nullptr`（不受益）。

**验证**：prose 128 ×3 **3/3 IDENTICAL**（hash 3bb743b1…）；layer-prof router 相
0.060→0.061（sgate 并入未增相耗时）、moe 相 -0.018（少一次发射的间接效应）；ms/win 29.6→30.5？——

#### 7.18.3 F1b 交替实测：中性（+0.2~0.3 tok/s，噪声边缘）

3×3 交替（同二进制对比基线 graph5）：

| | tok/s | ms/win | enq | s0 | s1 | mtp |
|---|---:|---:|---:|---:|---:|---:|
| F1b（3×） | 64.01 / 62.28 / 61.10（均 **62.5**） | 29.0 / 29.8 / 30.4（均 **29.7**） | 2.54~3.34 | 11.59/11.24/11.43 | ~12.2 | 2.50/3.00/3.00 |
| 基线（3×） | 61.10 / 61.10 / 64.15（均 **62.1**） | 30.3 / 30.4 / 28.9（均 **29.9**） | 3.31/3.41/2.55 | ~11.5 | ~12.2 | ~3.0 |

ms/win 均值 -0.2（-0.7%）、tok/s +0.4——**方向正确但在噪声内**。省一次发射的
~10µs × 48 层 ≈ 0.5ms 的理论收益没显出来：该发射本来就与相邻 gemm 的执行重叠
（enq 段的 launch pipeline 掩盖了它）。与 7.15 的教训一致：**单点去发射，只要
发射队列不饱和，就不会显性化**。保留该改动（等价、无回退、代码更省），但记录
真实收益 ≈ 0。

#### 7.18.4 两个负结果的合并结论（F1 + F1b）

1. **发射开销在 launch pipeline 里是隐藏的**：只有当发射成为瓶颈（7.8 的图回放
   暴露会合）或内核数大幅削减（megakernel）时才会显性化。逐个"省一次发射"的
   合并收益趋近于零。
2. **合并的真正收益来源必须是"减少 GPU 侧串行工作量"**（数据搬运、同步、依赖等待），
   而非"减少发射次数"。F1 的失败（并行度损失）与 F1b 的中性（无并行度损失也无
   GPU 工作减少）从两个方向印证了这一点。

⇒ **7.16 的 F2-F5 收益估计全部需要下修**；剩余真实杠杆回到：内核内部的数据搬运
优化（moe 相的中间 buffer 回写：dq16/ds16 写后读 ≈ 0.25ms 里的相当部分）、
mtp drafter 的 d_head gemm（[114452,2560] 真算力）与 QSA ring attention 的
shared-memory 扫描。这些都是"内核内部工程"，不是"内核边界工程"。

#### 7.18.5 状态

- 源码 = 7.14 定版 + F1b（等价、保留）；二进制 `/tmp/wzy_pp2_engine.f1b`。
- 待办：服务用 f1b 重启；PROJECT-SUMMARY 更新（F5 结论、F1b 中性、下轮入口改为
  "moe 相中间 buffer 消除 / d_head gemm 评估"）。


### 7.19 螃蟹 A/B：中文长文实测——词表/profile 假设检验（2026-11-06 晨，应用户要求）

**题目（用户出）**：用阳澄湖大闸蟹、金酒、西瓜写 2000 字科幻小说（43 token prompt，
`max_tokens=2000`，greedy，2000 token 全跑完取整段统计）。

#### 7.19.1 变体（`/tmp/crab.py`，每变体跑完即存 `/tmp/crab_*.json`）

| 变体 | 词表 (`data/draft_vocab_han.bin`) | 预热表 | tok/win | hit | ms/win | tok/s | 质检 |
|---|---|---|---:|---:|---:|---:|---|
| A（真基线） | han（原 457KB，114452 条） | orig | **1.84** | 97.70 | 28.47 | 64.6 / 64.5 | dup20=0 |
| B | cjk（498KB，10-30 版） | cjk | 1.89 | 97.54 | 29.26 | 64.4 | dup20=0 |
| D | **strata 官方（425KB）** | cjk→（见 7.19.2） | **1.92** | 97.59 | 29.3 | 65.5/65.2/65.6 | dup20=0 |
| E/F | strata 官方（无 cjk profile 后） | 无 | 1.85 | 97.27 | 29.2 | 63.3 | dup20=0 |

#### 7.19.2 混淆因子（诚实记录）

A2/A3 两组先跑了"strata 词表 + cjk profile"得到 67.1/66.8 tok/s（1.92），**但复检发现
当时 `data/expert-profile-cjk.bin` 是 11-02 的文件且 A2/A3 实际还加载了它**；A2b（真
基线恢复）64.5。因此"A2 67 tok/s"**不能归因词表**，是混合变量，已从结论中剔除。
E/F（strata 词表 + 无 profile）63.3，反而掉——说明 profile 有正贡献。

#### 7.19.3 结论

1. **词表差异影响很小**：han vs strata 官方 = 1.84 vs 1.92 tok/win（+4%），cjk 498KB
   大词表无额外收益（1.89，介于两者之间）。中文 draft 命中瓶颈不在词表大小。
2. **profile 有可测贡献**：有 profile（任一版）≈ 97.6% hit vs 无 97.27%，promo 数
   6183→7188 也不同。但 A2 那次 67.1 tok/s 提示 profile 的内容（哪些专家预填充）
   可能还有 ~1-2ms/win 的可挖空间——需要干净的"profile 内容"对照实验才能定论。
3. **质量**：11 个变体 dup20 全 0，2000 token 全部自然跑完，正文都是完整的中文科幻
   小说（开头均正确复述题目约束）。
4. **服务已用 strata 官方词表重启**（加载 106299 条 CJK draft vocab，275.7 MiB——
   注意：strata 词表比 han 少 8K 条但 tok/win 更高，说明**词表质量 > 数量**）。

#### 7.19.4 用户问题的直接回答

"中文对话慢是不是词表不如 Strata" → **词表不是主因**。真实流量的慢（对照 §0 数据
s0+s1 31-39ms vs bench 23.5ms）来自**长上下文**（KV 环外 PCIe 镜像查询 + 注意力
变贵）+ 真实流量的 mtp 链更长（3.1-5.1ms vs 2.5-3.0）。词表换 strata 官方版有
+4% 的 tok/win 改善，已采纳为生产。下一步应做"profile 内容"对照实验
（A2 的 67.1 提示存在 1-2ms/win 空间）。


### 7.20 profile 内容对照实验：A2 的 67 是真的，确认 +1.2 tok/s（2026-11-06 晨续）

**受控变量**：唯一变量 = `WZY_PROFILE` 指向的字节内容（词表统一 strata 官方版，
其余全同，引擎直跑不走服务，螃蟹题 × 2000 token × 各 2 次）。

| profile | 内容 | vram_hit | tok/win | ms/win | tok/s（均值） |
|---|---|---:|---:|---:|---:|
| P0 上游原版（strata-0151，10-26） | 通用频率表 | 97.20% | 1.84 | 29.6 | 64.47 |
| P1 学习版（strata-fork，11-02 SAVE_PROFILE 产物 == cjk 文件） | 真实流量热专家 | **97.59%** | **1.92** | **29.0** | **65.70** |

- 差异 = **+1.2 tok/s（+1.9%）**，两次重复方向一致（65.7/65.4 vs 64.9/64.0），
  vram_hit +0.39pp、mtp -0.09ms、s1 -0.22ms。
- **7.19 的 A2=67.1 确认为真但归因修正**：A2 的词表与 profile 组合恰好等价于
  P1（strata 词表 + 学习版 profile），67.1 里 ~1.2 来自 profile、其余是轮次噪声。
- **质量**：两版输出 dup20=0；token 流从第 63 个 token 起分叉（P0"让我构思一个
  科幻故事：" vs P1"让我构思：设定：未来世界…"），两版都成立——profile 只改变
  哪些专家常驻（决定数值细节），不改变语言能力。
- **机制**：P1 是 11-02 真实中文流量学出的热专家排名（SAVE_PROFILE 持久化），
  对中文工作负载的预填充命中率 +0.39pp → 冷查找少了 → s1 短了。
- **已生效**：当前生产（默认路径）本来就是 P1（学习版）；P0 只是对照。无回滚需求。

#### 7.20.1 剩余问题与下轮

1. 学习版 profile 是 11-02 的快照，此后真实流量（尤其本项目参数改动后）的分布
   已漂移：**建议在生产上周期性 SAVE_PROFILE**（server.py 已有 `/api/profile/save`），
   或在 config 里定期自动保存；
2. P1 只 +0.39pp hit，天花板估计还有 ~0.5-1pp（对中文负载更专一的表），收益
   ~0.5-1 tok/s——与 7.18 结论一致：中小收益，不如内核内部工程优先。
3. **词表换 strata 官方版 + profile 保持学习版 = 当前定版**（65.7 tok/s 本题实测，
   相比会话开始时的 62.1 基线净 +5.8%）。
