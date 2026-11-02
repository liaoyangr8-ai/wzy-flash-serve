# IMPORTANT: renamed source package

This package contains no prebuilt engine. Run ./build.sh on a compatible Linux/CUDA machine before starting. Older statements below about a bundled binary describe the original deployment, not this package. The WZY names and environment variables were mechanically renamed; this version has not yet been compiled or tested. Adapt config.json and model/vision paths before use.

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

详细硬件解锁、模型路径配置、中文提速原理与排障指南请参阅同目录下的 **[DEPLOYMENT-SOP.md](./DEPLOYMENT-SOP.md)**。
