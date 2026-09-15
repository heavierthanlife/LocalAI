# QA-Loop Round 027 (2026-09-15)
基线: last_head=c504ca7 | 模式: 清理（移除 Headroom）+ 文案（休眠指标）| 触发: manual

## 背景
用户全景审计确认：34/45 清标指标 `checker:'skip'`（设计如此，需平台/评标数据）；Headroom 对中文零压缩；`loadProviderSelector` 死代码；(A) LLM 悬空——**经复查实为用户已自建 `id='mimo'` 自定义 provider 并选中，无悬空**（前判基于旧快照，已更正）。

## 用户决定
- B 移除 Headroom：**(i) 最小方案**（仅移除依赖 + 清缓存，零代码改动）。
- C 休眠指标：仅**文案 + 文档标注**（不做功能）。
- 释放 C 盘 headroom 模型空间 + `docker image prune` + vhdx 压缩。

## ⑤ IMPLEMENT
- **B（FIX-053）**：`requirements.txt` 删 `headroom-ai==0.27.0` / `magika==0.6.3` / `onnxruntime==1.20.1`；`headroom_utils.py` 保留为 soft no-op。
- **C**：`document_analysis_svc.py` 两处占位文案 →「○ 需交易平台数据（当前不可用）」；`else` 分支追加 `skip_reason`；`docs/ARCHITECTURE.md` 新增「休眠指标分类」（13 可激活 / 21 恒不可用，经脚本 `indicator_defs` 精确枚举）。
- `fix_registry` FIX-053；回归 `test_headroom_removed_and_skip_label`；CHANGELOG。

## 释放 C 盘
- 主机：删 `…\.cache\huggingface\hub\models--chopratejas--kompress-v2-base`(834.1MB) + `models--answerdotai--ModernBERT-base`(2.1MB)。
- 容器：`rm -rf /app/data/hf_cache/hub/models--chopratejas--kompress-v2-base`(~262MB)。
- `docker image prune -f` 删旧镜像层；Docker Desktop vhdx Compact（手动）。
- 保留 sentence-transformers 三模型（bge/MiniLM/distiluse）。

## ⑧ 验证
- 回归/verify_fixes/doc_drift + 容器 `import headroom` 应 ImportError + `/check_auth`=200 + 缓存已删。

## 复核发现（@code-reviewer）
- Medium：`tests/fixtures/clearance_baseline/scores.json` 内 34 处 `result` 仍是旧占位串（快照过期）；`test_clearance_baseline_scores` 仅比对 composite/n_bidders，**不受影响**。无生成脚本 → 记为 backlog：下次基线校准/新增生成器时一并刷新（含 `skip_reason` 后缀）。
- Low：`_sr` 变量命名 → 已改为 `skip_reason_text`。

