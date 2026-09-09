# QA-Loop Round 014 (2026-09-09)

基线: last_head=ece4ceb | 模式: focused（admin VL 测试组件 404 修复） | 触发: user（VL 模型测试报网络错误 + /admin/vl_test 404）

## ① COLLECT 摘要（用户反馈）
- admin 面板测试 VL 模型（xiaomi token plan / mimo v2.5）时：`POST /admin/vl_test 404` + `Unexpected token '<', "<!doctype"... is not valid JSON`。

## ② VERIFY 根因
- **`POST /admin/vl_test` 后端路由从未实现**。review.js:1296 `handleVLTest`（VL 测试组件，上传图片 POST）指向不存在的端点 → Flask 返回 HTML 404 → 前端 `r.json()` 解析失败报 "Unexpected token '<'"。
- 后端已有：`GET /admin/vl_status`（admin_regeneration.py:1450）、`vl_model` 单例（VL_PROVIDER_CONFIG 含 mimo/xiaomi：`token-plan-cn.xiaomimimo.com/v1`）。只缺测试端点。

## ③④ CROSS-EXAM / CONFIRM
- 修复 = 实现该端点 + 让推理模型（mimo）的 reasoning_content 也能展示。前端契约（`d.status==='ok'` → `d.data.{description, reasoning}`）已就绪，无需改前端。

## ⑤ IMPLEMENT（self）
- `app/services/vl_model.py`：新增 `describe_image_v2(image_bytes, prompt)` → `{description, reasoning}`（一次 chat 调用读 `message.content` + `reasoning_content`；失败沿用 ⚠️ 错误串模式；不动现有 `describe_image` 的字符串调用方）
- `app/routes/admin_regeneration.py`：在 `admin_vl_status` 后加 `POST /admin/vl_test`（`@admin_required`；multipart `image`；裸 jsonify 匹配前端契约：成功 `{status:'ok',data:{description,reasoning}}`、无文件 400、⚠️ 串→`{status:'error'}`）
- `tests/test_regression.py`：+3 单测（describe_image_v2 读 content+reasoning / 非推理空 reasoning / 不可用 ⚠️）
- `tests/integration/test_admin.py`：+`TestAdminVLTest` 4 用例（admin_required 403 / 无文件 400 / 成功 200 / 失败 status error）；修正文件上传用 `io.BytesIO`（werkzeug 对 raw bytes tuple 不当文件解析）

## ⑥ DOCS
- CHANGELOG [2026-09-09] round-014 条目 + round-014.md。

## ⑦ PUSH
- `ece4ceb..<待填>` 已推送（LocalAI master）。last_head 更新。

## ⑧ 验证
- 回归 **124/124**（+3 单测）· verify_fixes **102/102** · check_system 133/137
- 容器实机：真实 mimo/mimo-v2.5 VL 调用 PASS（红底图→描述"VL TEST 123" + reasoning_content 正确返回）；路由注册确认；路由契约三态 PASS（成功 200 / 无文件 400 / ⚠️→status error）
- 注：真实调用一次返回 mimo provider 500（该 provider 侧瞬时/该图问题），同图直接调用 PASS；路由层经 monkeypatch 验证契约

## ⑨ RE-CHECK / 停跑判定
- 无新增 Critical/High → 闸门通过。db 标记的 admin 路由测试待 `-m db` 环境（本机/容器 PG）跑通（已在容器内以等价脚本验证）。