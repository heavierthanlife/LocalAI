# Skill MCP Server - 使用指南

## 概述

Skill MCP Server 提供了一个标准化的接口来访问共享技能库中的236个技能。它通过 MCP (Model Context Protocol) 协议提供技能列表、技能详情和技能调用功能。

## 配置

### 1. MCP 服务器配置

在 `.mcp.json` 文件中添加 skill 服务器：

```json
{
  "mcpServers": {
    "skill": {
      "command": "node",
      "args": ["tools/skill-mcp/index.js"],
      "env": {
        "SHARED_SKILLS_DIR": "D:\\AI_Tools\\shared-agent-infra\\skills",
        "LOCAL_SKILLS_DIR": ".opencode/skills"
      }
    }
  }
}
```

### 2. 技能目录结构

- **共享技能库**: `D:\AI_Tools\shared-agent-infra\skills\` (236个技能)
- **本地技能目录**: `.opencode/skills\` (32个技能)
- **技能优先级**: 本地技能覆盖同名共享技能

## 可用工具

### 1. `list_skills`

列出所有可用的技能。

**参数**: 无

**返回**: 技能列表，包含名称、描述、来源和路径

**示例**:
```json
{
  "name": "list_skills",
  "arguments": {}
}
```

### 2. `get_skill`

获取特定技能的详细信息。

**参数**:
- `name` (必需): 技能名称

**返回**: 技能的完整信息，包括内容

**示例**:
```json
{
  "name": "get_skill",
  "arguments": {
    "name": "api-and-interface-design"
  }
}
```

### 3. `invoke_skill`

调用一个技能。

**参数**:
- `name` (必需): 技能名称
- `args` (可选): 传递给技能的参数

**返回**: 技能调用指令和内容

**示例**:
```json
{
  "name": "invoke_skill",
  "arguments": {
    "name": "api-and-interface-design",
    "args": {
      "task": "Design a REST API for a todo list application",
      "context": "Using Flask and PostgreSQL"
    }
  }
}
```

## 使用示例

### 命令行测试

```bash
# 列出所有技能
echo '{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"list_skills","arguments":{}}}' | node tools/skill-mcp/index.js

# 获取特定技能
echo '{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"get_skill","arguments":{"name":"api-and-interface-design"}}}' | node tools/skill-mcp/index.js

# 调用技能
echo '{"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"invoke_skill","arguments":{"name":"api-and-interface-design","args":{"task":"Design API"}}}}' | node tools/skill-mcp/index.js
```

### 在 lmcode 中使用

当 lmcode 启动时，它会自动连接到配置的 MCP 服务器。然后你可以使用以下命令：

1. **列出技能**: 系统会自动显示可用的技能列表
2. **调用技能**: 使用 `skill:<skill-name>` 语法调用技能

## 技能分类

共享技能库包含以下类别的技能：

### 开发相关
- `api-and-interface-design` - API 和接口设计
- `code-review-and-quality` - 代码审查和质量
- `test-driven-development` - 测试驱动开发
- `debugging-and-error-recovery` - 调试和错误恢复

### 文档处理
- `docx` - Word 文档处理
- `pdf` - PDF 处理
- `powerpoint` - PowerPoint 处理
- `xlsx` - Excel 处理

### AI/ML
- `axolotl` - LLM 微调
- `flash-attention` - 高效注意力机制
- `pytorch-lightning` - PyTorch Lightning
- `unsloth` - LLM 训练优化

### 工具集成
- `github-*` - GitHub 集成
- `obsidian-*` - Obsidian 集成
- `hermes-*` - Hermes Agent 集成

### 中国特定
- `chinese-gov-procurement-system` - 中国政府采购系统
- `chinese-gov-tender-audit` - 中国政府招标审计
- `company-background-check-cn` - 中国公司背景调查

## 故障排除

### 1. 服务器无法启动

检查 Node.js 版本：
```bash
node --version
```

确保依赖已安装：
```bash
cd tools/skill-mcp && npm install
```

### 2. 技能未找到

检查技能目录是否存在：
```bash
ls -la "D:\AI_Tools\shared-agent-infra\skills"
ls -la ".opencode/skills"
```

### 3. 权限问题

确保服务器有权限读取技能目录。

## 开发

### 添加新技能

1. 在技能目录中创建新文件夹
2. 添加 `SKILL.md` 文件
3. 在 frontmatter 中定义 `name` 和 `description`
4. 编写技能内容

### SKILL.md 格式

```markdown
---
name: my-skill
description: Description of what this skill does
---

# My Skill

## Instructions

Detailed instructions for using this skill...
```

## 性能优化

- 技能列表缓存 1 分钟
- 技能文件按需加载
- 支持本地技能覆盖共享技能

## 安全注意事项

- 服务器只读取技能文件，不执行任何写操作
- 技能内容是纯文本，不包含可执行代码
- 参数传递是安全的，不会导致代码注入
