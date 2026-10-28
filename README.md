# 🤖 RAG智能问答Agent

基于RAG（检索增强生成）技术的智能问答Agent系统，支持文档上传、知识库构建、多轮对话和Agent式智能推理。采用模块化工程化设计，代码清晰易懂，适合作为课程项目和求职作品集。

## ✨ 功能特性

- 📄 **多格式文档支持**：支持PDF、TXT格式文档上传，自动解析和处理
- 🧠 **智能知识库构建**：自动文本分割、向量化存储，构建专属知识库
- 🔍 **语义检索**：基于向量相似度的语义检索，精准定位相关内容
- 🤖 **Agent式对话**：采用ReAct模式的智能Agent，自主决策调用工具
- 🔌 **MCP 双向接入**：既能作为 MCP Server 被 Claude Desktop / Cursor 调用，
  也能作为 MCP Client 反向调用社区里的任意 MCP 服务器（如 bili-subtitle-mcp）
- 💬 **多轮对话**：支持上下文记忆，实现流畅的多轮对话体验
- 🎨 **图形界面**：基于Gradio的友好界面，操作简单直观
- ⚙️ **灵活配置**：支持多种大模型API，支持API/本地两种Embedding方式
- 🛡️ **确定性降级**：无 API Key 时走抽取式兜底答案，离线可演示、结果可复现
- 📊 **可解释性**：支持查看Agent思考过程和回答来源
- ✅ **可回归**：55 个单元测试覆盖工具层、MCP 服务层与真实 stdio 端到端链路

## 🛠️ 技术栈

| 技术 | 说明 |
|------|------|
| Python 3.10+ | 开发语言 |
| LangChain | 大模型应用开发框架 |
| Chroma | 轻量级向量数据库 |
| Gradio | 图形界面框架 |
| OpenAI API | 大模型API（兼容所有OpenAI格式的API） |
| sentence-transformers | 开源Embedding模型（可选） |
| **MCP (Model Context Protocol)** | **把知识库标准化暴露给任意 AI 客户端；SDK 1.x/2.x 双版本兼容** |

## 🏗️ 系统架构

### 整体架构图

```
┌─────────────────────────────────────────────────────────┐
│                     用户界面层                           │
│                      (Gradio)                           │
│  文档上传  │  知识库管理  │  智能对话  │  思考过程展示   │
└─────────────────────────────────────────────────────────┘
                            │
┌─────────────────────────────────────────────────────────┐
│                     业务逻辑层                           │
│                                                        │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐   │
│  │ 文档加载模块 │  │ 向量存储模块 │  │ 对话管理模块 │   │
│  └─────────────┘  └─────────────┘  └─────────────┘   │
│                                                        │
│  ┌─────────────────────────────────────────────────┐   │
│  │              RAG Agent核心模块                   │   │
│  │  ┌─────────┐  ┌─────────┐  ┌─────────────────┐ │   │
│  │  │  大模型  │  │  工具集  │  │ ReAct推理引擎  │ │   │
│  │  └─────────┘  └─────────┘  └─────────────────┘ │   │
│  └─────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────┘
                            │
┌─────────────────────────────────────────────────────────┐
│                     数据存储层                           │
│                                                        │
│  ┌─────────────────┐  ┌─────────────────────────────┐  │
│  │  向量数据库     │  │        临时文件存储          │  │
│  │   (Chroma)      │  │        (本地文件)           │  │
│  └─────────────────┘  └─────────────────────────────┘  │
└─────────────────────────────────────────────────────────┘
```

### Agent工作流程

```
用户提问
    ↓
┌─────────────┐
│  思考(Think)│  → 分析问题，判断是否需要调用工具
└─────────────┘
    ↓
需要工具？ ──否──→ 直接生成回答
    │是
    ↓
┌─────────────┐
│  行动(Act)  │  → 选择并调用相应工具
└─────────────┘
    ↓
┌─────────────┐
│ 观察(Observe)│ → 获取工具返回结果
└─────────────┘
    ↓
信息足够？ ──否──→ 回到思考步骤
    │是
    ↓
生成最终回答
```

## 🚀 快速开始

### 1. 环境准备

- Python 3.10 或更高版本
- 一个可用的大模型API Key（支持OpenAI、DeepSeek、智谱AI等所有OpenAI兼容的API）

### 2. 安装依赖

```bash
# 克隆或下载项目到本地
cd rag-agent-project

# 安装依赖包
pip install -r requirements.txt
```

> 💡 如果需要使用本地开源Embedding模型，会自动下载sentence-transformers模型，首次使用需要联网下载。

### 3. 配置API Key

复制 `.env.example` 文件为 `.env`，然后修改其中的配置：

```bash
cp .env.example .env
```

编辑 `.env` 文件，填入你的API信息：

```env
# 以DeepSeek为例（推荐，性价比高）
LLM_API_BASE=https://api.deepseek.com/v1
LLM_API_KEY=你的DeepSeek_API_Key
LLM_MODEL_NAME=deepseek-chat

# Embedding配置（默认使用API方式）
EMBEDDING_TYPE=api
EMBEDDING_MODEL_NAME=text-embedding-ada-002
```

> 💡 支持所有OpenAI兼容的API，包括但不限于：
> - OpenAI官方
> - DeepSeek
> - 智谱AI（GLM）
> - 通义千问
> - 月之暗面（Kimi）
> - 本地部署的Ollama等

### 4. 运行项目

```bash
python app.py
```

启动成功后，在浏览器中访问 `http://localhost:7860` 即可使用。

## 📁 项目结构

```
rag-agent-project/
├── src/                          # 核心源码目录
│   ├── __init__.py              # 包初始化文件
│   ├── config.py                # 配置管理模块
│   ├── document_loader.py       # 文档加载与处理模块
│   ├── vector_store.py          # 向量数据库模块
│   ├── chat_history.py          # 对话历史管理模块
│   ├── rag_chain.py             # RAG问答链模块
│   ├── agent.py                 # Agent核心模块
│   ├── mcp_tools.py             # MCP工具层（纯业务逻辑，零依赖、可单测）
│   ├── mcp_server.py            # MCP服务层（工具注册 / Schema 契约 / 三种传输）
│   └── mcp_client.py            # MCP客户端适配层（反向调用外部 MCP 服务器）
├── examples/                     # 示例文档
│   ├── RAG技术介绍.txt
│   ├── AI_Agent技术介绍.txt
│   └── LangChain框架介绍.txt
├── tests/                        # 单元测试
│   ├── test_mcp_tools.py        # 工具层：输入校验、检索、降级、协议兼容
│   ├── test_mcp_server.py       # 服务层：工具注册、Schema 契约、端到端行为
│   └── test_mcp_client.py       # 客户端：入参解析 + 真实子进程 stdio 端到端
├── data/                        # 数据存储目录
│   ├── vector_db/               # 向量数据库存储
│   └── temp/                    # 临时文件存储
├── app.py                       # Gradio界面主程序
├── run_mcp_server.py            # MCP 服务器启动脚本
├── requirements.txt             # 完整依赖（含 RAG 主程序）
├── requirements-mcp.txt         # 只跑 MCP 服务层的最小依赖
├── .env.example                 # 环境变量示例
├── .gitignore                   # Git忽略文件
└── README.md                    # 项目说明文档
```

## 📖 使用说明

### 1. 构建知识库

1. 在左侧"知识库管理"区域，点击上传文件
2. 选择要上传的PDF或TXT文档（支持多选）
3. 点击"上传并构建知识库"按钮
4. 等待处理完成，右侧会显示知识库状态

### 2. 智能对话

1. 在右侧对话区域的输入框中输入问题
2. 按回车或点击"发送"按钮
3. Agent会自主思考并回答问题
4. 可以开启"显示Agent思考过程"查看详细的推理步骤

### 3. 其他功能

- **刷新状态**：查看最新的知识库信息
- **清空知识库**：删除所有知识库内容
- **清空对话历史**：重置对话上下文

## 🔌 MCP 服务层（Model Context Protocol）

> 这一段是本项目相对"课程作业"最有区别度的部分：**把私有的 RAG 能力变成一个标准协议节点**。
> 改造前它是一个只能在自己的 Gradio 页面里用的问答脚本；
> 改造后，任何支持 MCP 的 AI 客户端（Claude Desktop、Cursor、Claude Code、Continue）
> 都能直接调用它，它也能反过来调用别人的 MCP 服务器。

### 作为 Server：被 AI 客户端调用

暴露三个工具：

| 工具 | 入参 | 说明 |
|------|------|------|
| `search_knowledge_base` | `query`(必填), `top_k`(1–20，默认3) | 语义检索，返回原文片段 + 出处 + 相似度分数 |
| `ask_knowledge_base` | `question`(必填), `top_k` | 带引用的问答；有 LLM 走生成式答案，无 LLM 走确定性原文摘录 |
| `knowledge_base_stats` | 无 | 知识库规模、来源列表、检索方式 |

三步跑起来（**不需要 Chroma、不需要 Embedding、不需要 API Key**）：

```bash
pip install -r requirements-mcp.txt
python run_mcp_server.py --demo        # 演示模式：内置示例文档
```

然后在 MCP 客户端里注册（以 Claude Desktop 的 `claude_desktop_config.json` 为例）：

```json
{
  "mcpServers": {
    "rag-kb": {
      "command": "python",
      "args": ["E:/求职/github_upload/rag-qa-agent/run_mcp_server.py", "--demo"],
      "env": { "RAG_MCP_DEMO": "1" }
    }
  }
}
```

接真实知识库时去掉 `--demo` 即可（会用 Chroma 里已导入的文档）；再加 `--with-llm`
就会接入 LLM 生成自然语言答案，否则始终用抽取式兜底答案。

也支持非 stdio 传输，方便自建客户端或远端调用：

```bash
python run_mcp_server.py --demo --transport streamable-http --port 8765
```

### 作为 Client：反向调用别人的 MCP 服务器

`src/mcp_client.py` 把外部 MCP 服务器的工具即时接入本地 ReAct Agent 的工具池，
**接入一个新的 MCP 服务器不需要改任何代码**——运行时拉取对端的 `tools/list`，
自动转成 LangChain `Tool`：

```python
from src.mcp_client import MCPServerConfig, MCPToolBridge

bridge = MCPToolBridge(config=MCPServerConfig(
    command="bili-subtitle-mcp",          # 社区里的任意 MCP server
    args=[],
))
remote_tools = bridge.connect()           # initialize + tools/list
tools = bridge.to_langchain_tools()       # 直接塞进 AgentExecutor
```

### 设计要点

| 问题 | 做法 |
|------|------|
| 工具层如何做到零依赖可测 | **依赖倒置**：检索能力抽象成 `Retriever` 协议注入，测试用假实现，不碰向量库 |
| 没有 LLM 时工具还能用吗 | **确定性降级**：抽取式兜底答案，结果可复现，且 LLM 报错时自动回退 |
| MCP stdio 会话怎么管 | **单一常驻 task 持有会话**：anyio task group 的 enter/exit 必须在同一 task，否则关闭时抛 `Attempted to exit cancel scope in a different task` |
| 模型乱填参数怎么办 | 双重防线：手写 JSON Schema 声明契约 + 运行期入参净化（`top_k` 夹到 1–20，空 query 拒绝） |
| SDK 升级会炸吗 | `src/mcp_server.py` 做了 1.x/2.x 兼容层（`FastMCP` → `MCPServer`），`inputSchema`/`input_schema` 字段名差异也一并处理 |
| 失败会不会拖垮 Agent | 工具层异常统一转成给模型看的中文错误文本，不抛到外层 |

### 测试

```bash
python -m unittest discover -s tests        # 55 个用例
```

其中 `tests/test_mcp_client.py` 里的 `TestStdioEndToEnd` 会**真的拉起一个子进程**
走完 `initialize → tools/list → tools/call`，而不是只 mock 一下。

## 🌟 技术亮点

### 1. RAG检索增强生成
- 完整实现了RAG技术的全流程：文档加载 → 文本分割 → 向量化 → 语义检索 → 生成回答
- 支持自定义文本分割参数，优化检索效果
- 支持多种Embedding方式，灵活适配不同场景

### 2. 向量数据库与语义检索
- 使用Chroma轻量级向量数据库，部署简单
- 基于余弦相似度的语义检索，比关键词检索更智能
- 支持持久化存储，知识库可以重复使用

### 3. Prompt工程与上下文管理
- 精心设计的RAG提示词，确保回答的准确性
- Agent提示词遵循ReAct模式，引导模型正确推理
- 对话历史自动管理，避免上下文过长

### 4. Agent式对话设计
- 采用ReAct（推理+行动）Agent架构
- Agent自主决策是否需要调用工具
- 支持多步推理，能够处理复杂问题
- 可查看完整的思考过程，具有良好的可解释性

### 5. 完整的系统架构设计
- 模块化设计，各模块职责清晰
- 配置与代码分离，易于维护和扩展
- 完善的错误处理，系统稳定性好
- 代码注释详细，可读性强

### 6. 工程化思维
- 清晰的项目结构，符合Python项目规范
- 依赖管理规范，使用requirements.txt
- 环境变量配置，敏感信息不硬编码
- 完整的文档说明，易于上手

## 💼 求职相关

这个项目可以很好地体现在AI应用开发方面的能力，适合放在求职作品集里：

### 体现的技术能力
- ✅ 大语言模型应用开发（LLM Application）
- ✅ RAG检索增强生成技术
- ✅ 向量数据库与语义检索
- ✅ AI Agent开发（ReAct模式）
- ✅ Prompt工程
- ✅ LangChain框架使用
- ✅ Python工程化开发
- ✅ 模块化设计与架构设计
- ✅ 图形界面开发

### 可以扩展的方向
- 加入更多工具（搜索引擎、计算器、代码解释器等）
- 支持更多文档格式（Word、PPT、Markdown等）
- 加入重排序（Rerank）提升检索效果
- 支持多用户和权限管理
- 加入对话记忆的持久化
- 支持多模态（图片、音频）

## ❓ 常见问题

### Q: 支持哪些大模型？
A: 支持所有OpenAI兼容的API，包括OpenAI、DeepSeek、智谱AI、通义千问、Kimi等。只需要在.env文件中配置对应的API地址和密钥即可。

### Q: 可以不用API，完全离线使用吗？
A: Embedding部分可以使用本地开源模型（设置EMBEDDING_TYPE=local），但大模型部分目前需要API。如果有本地部署的大模型（如Ollama），只要提供OpenAI兼容的接口，也可以使用。

### Q: 支持哪些文档格式？
A: 目前支持PDF和TXT格式。可以根据需要扩展支持更多格式，如Word、Markdown等。

### Q: 知识库保存在哪里？
A: 知识库保存在 `data/vector_db` 目录下，是本地持久化存储的，重启应用后知识库仍然存在。

## 📝 许可证

MIT License

---

**项目作者**：程序设计实践课程项目
**版本**：v1.0.0
