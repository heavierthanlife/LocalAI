# 编辑器选型 PoC 报告

**目的**：为 Wiki 升级方案 U9b（编辑器内联合规标注）选择前端编辑器。

测试环境：Chrome 128, Windows 11, 中国招标文件样本（~2,500 字、4 章结构）

---

## 候选编辑器

| 选项 | 方案 | 适合度 | 说明 |
|------|------|--------|------|
| **Tiptap/ProseMirror** | 推荐 ✅ | 高 | 原生 Mark 概念，标注一等公民 |
| **Quill** | 备选 | 中 | API 简单，标注需自定义 Attributor（hack） |
| CodeMirror 6 | ❌ 排除 | 低 | 代码编辑器，不适用文档编辑 |
| contenteditable | ❌ 排除 | — | 跨浏览器兼容地狱，无结构化支持 |

---

## Benchmark 对比

| 指标 | Tiptap | Quill | 差异 |
|------|--------|-------|------|
| 初始化时间 | ~180 ms | ~60 ms | Quill 更快（更轻量） |
| 内存占用 | ~8.5 MB | ~4.2 MB | Tiptap 大一倍 |
| 中文字符渲染 | ✅ 完美 | ✅ 完美 | 无差异 |
| 大文档（50KB+） | ✅ 流畅 | ✅ 流畅 | 无差异 |
| CDN 加载方式 | ESM importmap | `<script src>` | Quill 更简单 |
| 打包体积 | ~200 KB | ~70 KB | Quill 更轻 |

## 标注能力对比（核心维度）

### Tiptap — 原生 Mark API

```javascript
// 定义合规标注 Mark（一次注册，全局可用）
const ComplianceViolation = Mark.create({
  name: 'complianceViolation',
  renderHTML: () => ['span', { class: 'compliance-violation' }, 0],
});

// 选中文本 → 一键标注
editor.chain().focus().setMark('complianceViolation').run();

// 遍历文档 → 收集所有标注
doc.descendants((node, pos) => {
  node.marks.forEach(mark => {
    if (mark.type.name.startsWith('compliance')) { /* 收集 */ }
  });
});
```

**优势**：Mark 是 ProseMirror 一等公民，渲染、序列化、查询都是原生支持。

### Quill — 自定义 Attributor 类

```javascript
// 每个标注类型需单独注册 Attributor（API 较弱）
class ComplianceViolation extends Quill.import('parchment').Attributor.Class {}
Quill.register(new ComplianceViolation(), true);

// 标注
quill.format('compliance-violation', true);

// 遍历 → 需手动解析 Delta（扁平结构）
quill.getContents().ops.filter(op => op.attributes?.['compliance-violation']);
```

**劣势**：Quill 的 Delta 格式是扁平 Ops 数组，标注叠加会生成多个碎片 Op，遍历复杂度 O(n)。

### 结论

| 能力 | Tiptap | Quill |
|------|--------|-------|
| 标注定义 | Mark.create（简洁） | Attributor 子类（冗长） |
| 标注查询 | 树遍历 descendent() | 扁平 Delta 遍历 |
| 标注叠加 | 天然支持多 Mark | 叠加产生碎片 |
| 序列化/反序列化 | 自动 | 需手动处理 |
| 侧边栏联动 | 简单 | 需维护位置映射 |

**Tiptap 在标注场景优势明显**：Mark 是 ProseMirror 文档模型的一等公民，合规标注（OK/WARN/VIOLATION/CRITICAL 四种严重级别 + 内联 + 侧边栏）是 Tiptap 的最佳场景。

---

## 集成复杂度评估（与 app.js）

| 维度 | Tiptap | Quill |
|------|--------|-------|
| 加载方式 | `importmap` + ESM `<script type="module">` | `<script src>` 直引 |
| 兼容现有架构 | ✅ 正常（ESM 在现代浏览器通用） | ✅ 更简单 |
| JS 初始化 | `new Editor({ element, extensions })` | `new Quill('#el', { theme })` |
| 与现有 DOM 事件交互 | 需通过 ProseMirror API | 可以直接操作 DOM |
| 数据导入格式 | HTML / JSON | HTML / Delta |
| 学习曲线 | 中高（需理解 Schema/Mark/Node） | 低（类 textarea API） |

**关键点**：Tiptap 的 ESM importmap 在现代浏览器（Chrome 89+, Edge 89+, Safari 16.4+, Firefox 108+）完全支持。项目目标用户为国内招标代理，Chrome/Edge 占比 > 95%。

---

## 最终推荐：Tiptap/ProseMirror

**推荐理由（4 条）**：

1. **标注能力压倒性优势**：U9b 的核心需求是"内联合规标注"，Tiptap 的 Mark API 为四种严重级别标注提供了原生支持，Quill 需要 4 个 Attributor 子类 + 碎片式查询。

2. **ProseMirror 文档模型支持结构化合规数据**：U12（知识图谱）需要从文档中提取结构化关系，ProseMirror 的 Node+Mark 模型比 Quill 的扁平 Delta 更适合。

3. **可扩展性**：Tiptap 有活跃的扩展生态（collaboration, table, mention, etc.），未来需求不会受限于编辑器能力。

4. **团队学习成本可控**：学习曲线虽高，但主要复杂度在注册阶段（Mark 定义 + 工具栏集成），与 app.js 的交互接口少。

**风险与缓解**：

| 风险 | 缓解 |
|------|------|
| ESM 加载可能失败（CDN 不可用） | 回退到 Quill CDN（Quill 也是有效选择） |
| 团队不熟悉 ProseMirror 概念 | PoC 页面即可作为模板，自定义 Mark 只需复制 |
| 与 app.js 13K 行文件交互 | 封装 `ComplianceEditor` 类，暴露 4 个方法 |

---

## 实施建议

1. **U9b 编辑器代码**放在 `static/js/editor/compliance_editor.js`（独立文件，不在 app.js 中）
2. **初始化接口**：
   ```javascript
   // 暴露给 app.js 的最简接口
   const ce = new ComplianceEditor('#editor-container', { content });
   ce.highlight(range, 'violation');   // 标注
   ce.clearHighlights();               // 清除
   ce.getAnnotations();                // 获取所有标注
   ce.setContent(html);                // 设置内容
   ```
3. **PoC 页面可作为模板**：`docs/editor_poc/tiptap.html` → 直接升级为生产代码基础
