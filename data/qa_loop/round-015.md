# QA-Loop Round 015 (2026-09-09)

基线: last_head=22160b3 | 模式: full（VL 识别可靠性：OCR ground-truth + 最强 provider + 交叉验证） | 触发: user（mimo VL 低 IQ，如何确保图片识别被正确解读）

## ① COLLECT 摘要（用户问题）
- mimo v2.5/pro 低 IQ → 图片识别可能"自信地读错"。如何确保被正确解读？

## ② VERIFY 现状
- 全文本扫描 PDF 路径已 **OCR-first**（file_processing:1041-1090，VL 仅 OCR 失败页兜底）——没问题。
- **可靠性空洞**：图片抽检（`describe_images_batch` 批量 VL-only）+ admin vl_test——mimo 单模型无 ground-truth，空/无法识别结果**静默丢弃**。
- `vl_cross_check`/`describe_with_crosscheck` 是死代码，且只对同一模型问两次——对"整体低 IQ"模型无效。

## ③④ CROSS-EXAM / CONFIRM（用户拍板）
1. **最强-configured-first**（auto 解析 dashscope→nvidia→mimo）；显式 pin（如 mimo）尊重但受 OCR+verifier 守卫
2. **接受 EasyOCR CPU 成本**
3. **仅 OCR-empty 图像做交叉验证**
4. **报告 source 标注**（OCR / VL识别 / VL+复核 / 需人工复核 / 无法识别）

## ⑤ IMPLEMENT（self）
- `ocr.py`：`ocr_text_from_bytes(image_bytes)`（bytes→np→EasyOCR）
- `vl_model.py`：`VL_STRENGTH`；auto 顺序 dashscope→nvidia→mimo；`select_vl_pair()`；`verify_image()`（primary + 按强度序候选 verifier 交叉，数字集不一致→consistent False，容忍坏 key 顺延）
- `file_processing._describe_sampled_images`：OCR-first + OCR-empty 子集 verify_image + `[来源]` 标签 desc + 不静默丢（无法识别显式）；`describe_images_in_file` 守卫放宽（OCR 或 VL 任一可用即可跑）
- `admin_regeneration /admin/vl_test`：响应加 `ocr/provider/verifier_desc/consistent/note`
- `review.js handleVLTest`：OCR 对照块 + VL 描述 + 一致/复核徽标 + 推理 + 复核文本并列展示

## ⑥ DOCS
- CHANGELOG [2026-09-09] round-015 条目 + fix_registry FIX-2026-09-09-022。

## ⑦ PUSH
- `22160b3..ebcc902` 已推送（LocalAI master）。last_head 更新。

## ⑧ 验证
- 回归 **128/128**（+4 VL 单测：select_vl_pair 排序 / 交叉数字不一致 / 一致 / 无 verifier）· verify_fixes **105/105**
- 容器实机：OCR 从生成图精确读出 `12345.67`/`88000`（确定性层 PASS）；vl pair=mimo|dashscope；真实 verify_image：mimo 主读正确 → dashscope 401（容器 key 失效）→ **顺延 nvidia 交叉一致**（坏 key 容忍 PASS）
- 注：容器 DASHSCOPE_API_KEY 失效（401）→ 交叉验证自动落到 nvidia；用户如需 dashscope 复核请更新 key

## ⑨ RE-CHECK / 停跑判定
- 无新增 Critical/High → 闸门通过。
- backlog：UI 审计计划（T0 静态路由交叉 / T1 Playwright / T2 深度）在 VL 可靠性之后排队；dashscope key 更新后交叉验证覆盖最强模型。
