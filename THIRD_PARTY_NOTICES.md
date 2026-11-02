# Third-party notices / 第三方来源声明

## Strata

This project incorporates and adapts portions of Strata, including substantially matching `serve/frontend.py` and `serve/responses.py`, and a port of `SuffixDrafter` in `src/wzy_suffix.cuh`. The vision integration also calls the separately obtained Strata `strata-vision` executable. Other files may contain additional adapted material; this is not an exhaustive provenance audit.

Strata is copyright (c) 2026 Niko1221 and the Strata contributors. The full MIT copyright and permission notice from the locally examined Strata fork is provided unchanged in `LICENSES/Strata-MIT.txt`. Preserve it when redistributing the relevant code or substantial portions. The exact upstream commit of the local fork was not established.

本项目包含和改编了 Strata 的部分代码，包括上述协议模块和后缀投机模块；视觉功能依赖另行获取的 Strata 可执行程序。这份清单并非完整的来源审计。分发相关代码时请保留 `LICENSES/Strata-MIT.txt` 中的原始版权和许可声明。

## Project license and remaining checks

The root `LICENSE` applies to project contributions that liaoyangr8-ai has the authority to license; third-party components retain their own copyright and license terms. Other contributors' rights and additional provenance remain subject to review.

Model weights, tokenizers, model data/profiles, CUDA and other dependencies, `strata-vision`, llama.cpp/mtmd, fonts or other third-party assets may have separate terms. They have not been comprehensively audited here; identify their actual provenance and include any required notices, source, attribution, or license terms before redistribution. Model weights and `strata-vision` are not bundled in this source package. This notice is not legal advice and does not itself establish full compliance.

根目录 `LICENSE` 适用于 liaoyangr8-ai 有权授权的项目贡献，不代替第三方许可。模型权重、词表、数据、CUDA、视觉程序、llama.cpp/mtmd 及其他依赖可能适用各自条款，本次未完成全面审计；再分发前请逐项核对来源与条件。
