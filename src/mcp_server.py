# -*- coding: utf-8 -*-
"""MCP 服务层：把 ``KnowledgeBaseService`` 以标准 MCP 工具的形式暴露出去。

对外可通过 stdio（被 Claude Desktop / Cursor / Claude Code 拉起）
或 streamable-http / sse（给自建客户端或远端调用）两种方式接入。

关于 SDK 版本：MCP Python SDK 在 2.0 把 ``FastMCP`` 更名为 ``MCPServer``，
这里用一个极薄的兼容层同时兼容两个大版本，避免升级 SDK 就炸。
"""

from __future__ import annotations

import asyncio
import os
import re
from typing import Any, Dict, List, Optional

from .mcp_tools import (
    DEFAULT_TOP_K,
    MAX_TOP_K,
    MIN_TOP_K,
    KnowledgeBaseService,
    ToolInputError,
)

try:  # mcp >= 2.0
    from mcp.server.mcpserver import MCPServer as _MCPApp
except ModuleNotFoundError:  # mcp < 2.0
    from mcp.server.fastmcp import FastMCP as _MCPApp  # type: ignore[assignment]

SERVER_NAME = "rag-knowledge-base"
SERVER_INSTRUCTIONS = (
    "一个本地知识库的检索与问答服务。回答私有文档相关问题时，"
    "优先用 ask_knowledge_base 拿到带引用的答案；需要看原文片段时用 "
    "search_knowledge_base；不确定知识库里有什么时先用 knowledge_base_stats。"
)


# 手写 JSON Schema：既是给模型的入参契约，也被单元测试用来校验实现没有漂移。
TOOL_SPECS: List[Dict[str, Any]] = [
    {
        "name": "search_knowledge_base",
        "title": "知识库语义检索",
        "description": (
            "在本地知识库中做语义检索，返回相似度最高的原文片段及其出处。"
            "当你需要引用原文、核对细节、或者不确定该不该回答时使用该工具。"
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "minLength": 1,
                    "maxLength": 500,
                    "description": "检索关键词或自然语言问题",
                },
                "top_k": {
                    "type": "integer",
                    "minimum": MIN_TOP_K,
                    "maximum": MAX_TOP_K,
                    "default": DEFAULT_TOP_K,
                    "description": "返回的片段数量",
                },
            },
            "required": ["query"],
            "additionalProperties": False,
        },
    },
    {
        "name": "ask_knowledge_base",
        "title": "知识库问答",
        "description": (
            "基于知识库回答一个问题，答案附带引用来源。"
            "配置了 LLM 时返回生成的自然语言答案；没有 LLM 时返回确定性的原文摘录，"
            "不会编造知识库之外的内容。"
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "question": {
                    "type": "string",
                    "minLength": 1,
                    "maxLength": 500,
                    "description": "要回答的问题",
                },
                "top_k": {
                    "type": "integer",
                    "minimum": MIN_TOP_K,
                    "maximum": MAX_TOP_K,
                    "default": DEFAULT_TOP_K,
                    "description": "参考的片段数量",
                },
            },
            "required": ["question"],
            "additionalProperties": False,
        },
    },
    {
        "name": "knowledge_base_stats",
        "title": "知识库统计",
        "description": (
            "查看知识库规模：文档片段总数、来源文件列表、检索方式与默认返回条数。"
            "无入参。在回答“知识库里有什么/有多少文档”这类问题时使用。"
        ),
        "inputSchema": {
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False,
        },
    },
]


def _fail(exc: Exception) -> str:
    """把异常渲染成对模型友好的中文错误文本，而不是让调用直接崩掉。"""
    if isinstance(exc, ToolInputError):
        return f"参数错误：{exc}"
    return f"工具执行失败：{type(exc).__name__}: {exc}"


def create_app(service: Optional[KnowledgeBaseService] = None) -> Any:
    """构建一个已注册全部工具的 MCP 应用实例。

    Args:
        service: 知识库服务对象；不传则惰性构造默认的（会加载真实向量库）。
            传值是依赖注入，方便测试与复用已有 Chroma 实例。
    """
    kb = service if service is not None else build_service()
    app = _MCPApp(SERVER_NAME, instructions=SERVER_INSTRUCTIONS)

    @app.tool(
        name="search_knowledge_base",
        description=TOOL_SPECS[0]["description"],
    )
    async def search_knowledge_base(query: str, top_k: int = DEFAULT_TOP_K) -> str:
        try:
            return KnowledgeBaseService.render_search(kb.search(query, top_k))
        except Exception as exc:  # noqa: BLE001 - 边界保护，任何异常都要回给模型
            return _fail(exc)

    @app.tool(
        name="ask_knowledge_base",
        description=TOOL_SPECS[1]["description"],
    )
    async def ask_knowledge_base(question: str, top_k: int = DEFAULT_TOP_K) -> str:
        try:
            return KnowledgeBaseService.render_ask(kb.ask(question, top_k))
        except Exception as exc:  # noqa: BLE001
            return _fail(exc)

    @app.tool(
        name="knowledge_base_stats",
        description=TOOL_SPECS[2]["description"],
    )
    async def knowledge_base_stats() -> str:
        try:
            return KnowledgeBaseService.render_stats(kb.stats())
        except Exception as exc:  # noqa: BLE001
            return _fail(exc)

    return app


def build_service(top_k: Optional[int] = None, with_llm: bool | None = None) -> KnowledgeBaseService:
    """按当前配置构造知识库服务。

    两种模式，自动选择：

    * **演示模式**（环境变量 ``RAG_MCP_DEMO=1``）：内置一份固定的示例文档，
      不需要 Chroma、不需要 Embedding、不需要 API Key，装完 MCP SDK 就能跑通整条链路；
    * **正常模式**：加载真实的 Chroma 向量库。

    Args:
        top_k: 覆盖默认返回条数。
        with_llm: 是否接入 LLM 生成答案。None 时跟随环境变量 ``RAG_MCP_LLM``。
    """
    if with_llm is None:
        with_llm = os.getenv("RAG_MCP_LLM", "").lower() in ("1", "true", "yes")

    if os.getenv("RAG_MCP_DEMO") == "1":
        return build_demo_service(top_k=top_k, with_llm=with_llm)

    # 惰性导入：只有真正要跑服务时才触碰 Chroma / Embedding，
    # 这样 `import src.mcp_server` 在没有装向量库依赖的环境里也不会失败。
    from .config import config
    from .vector_store import VectorStore

    answerer = _build_llm_answerer() if with_llm else None

    return KnowledgeBaseService(
        retriever=VectorStore(),
        answerer=answerer,
        default_top_k=top_k or config.RETRIEVE_TOP_K,
    )


# ---------------------------------------------------------------------------
# 演示模式：零依赖内置语料，用于在没有 API Key / 向量库时完整跑通 MCP 链路
# ---------------------------------------------------------------------------
DEMO_DOCS: List[Dict[str, Any]] = [
    {
        "content": (
            "本项目使用 Chroma 作为向量数据库，文档切片经过 Embedding 后持久化存储在 "
            "data/vector_db 目录下，重启服务后无需重新建库。"
        ),
        "meta": {"source": "架构说明.md"},
    },
    {
        "content": (
            "Embedding 支持两种模式：一是调用 OpenAI 兼容的在线接口，"
            "二是使用本地 sentence-transformers 开源模型。切换只改环境变量 EMBEDDING_TYPE。"
        ),
        "meta": {"source": "架构说明.md", "page": 3},
    },
    {
        "content": (
            "ReAct Agent 每轮最多迭代 5 次，超过上限或输出解析失败时会自动恢复并提示，"
            "避免长对话过程中出现跑飞或死循环。"
        ),
        "meta": {"source": "agent设计.md"},
    },
    {
        "content": (
            "文档加载支持 PDF 与 TXT：PDF 走 pypdf 按页解析，"
            "之后按 CHUNK_SIZE=500、CHUNK_OVERLAP=50 做重叠切分以保留上下文。"
        ),
        "meta": {"source": "文档处理.md", "page": 1},
    },
    {
        "content": (
            "MCP 服务层把知识库暴露为 search_knowledge_base、ask_knowledge_base、"
            "knowledge_base_stats 三个工具，可被 Claude Desktop、Cursor 等任意 MCP 客户端调用。"
        ),
        "meta": {"source": "MCP接入.md"},
    },
]


class InMemoryRetriever:
    """演示用的检索器：按关键词命中数打分，纯内存、零依赖、结果确定。"""

    def __init__(self, docs: List[Dict[str, Any]] = None):
        self.docs = docs or DEMO_DOCS

    @staticmethod
    def _tokens(text: str) -> List[str]:
        toks = re.findall(r"[A-Za-z0-9]+|[\u4e00-\u9fff]+", text.lower())
        out: List[str] = []
        for t in toks:
            if len(t) < 2 and not t.isdigit():
                continue
            out.append(t)
            if re.fullmatch(r"[\u4e00-\u9fff]{2,}", t):
                out.extend(t[i : i + 2] for i in range(len(t) - 1))
        return list(dict.fromkeys(out))

    def _rank(self, query: str):
        q_tokens = self._tokens(query)
        scored = []
        for doc in self.docs:
            text = doc["content"].lower()
            hits = sum(1 for t in q_tokens if t in text)
            if hits:
                scored.append((doc, hits / max(len(q_tokens), 1)))
        scored.sort(key=lambda x: -x[1])
        return scored

    def similarity_search(self, query: str, k: Optional[int] = None):
        return [_Hit(d, s) for d, s in self._rank(query)][: k or 3]

    def similarity_search_with_score(self, query: str, k: Optional[int] = None):
        return [(_Hit(d, s), s) for d, s in self._rank(query)][: k or 3]

    def get_document_count(self) -> int:
        return len(self.docs)

    def get_all_sources(self) -> List[str]:
        return sorted({d["meta"].get("source", "未知") for d in self.docs})


class _Hit:
    """极简 Document 替身，字段与 LangChain Document 保持一致。"""

    def __init__(self, doc: Dict[str, Any], score: float):
        self.page_content = doc["content"]
        self.metadata = dict(doc["meta"])
        self.score = score


def build_demo_service(top_k: Optional[int] = None, with_llm: bool = False) -> KnowledgeBaseService:
    """构造演示模式的知识库服务（零依赖、可离线跑）。"""
    return KnowledgeBaseService(
        retriever=InMemoryRetriever(),
        answerer=_build_llm_answerer() if with_llm else None,
        default_top_k=top_k or DEFAULT_TOP_K,
    )


def _build_llm_answerer():
    """构造基于 OpenAI 兼容接口的答案生成器；失败时返回 None 由上层降级。"""
    try:
        from .config import config
        from langchain_openai import ChatOpenAI
    except Exception:  # pragma: no cover - 依赖缺失时静默降级
        return None
    if not config.LLM_API_KEY:
        return None

    llm = ChatOpenAI(
        model=config.LLM_MODEL_NAME,
        api_key=config.LLM_API_KEY,
        base_url=config.LLM_API_BASE,
        temperature=0.2,
    )
    system = (
        "你是严谨的知识库问答助手。只能依据给定的上下文回答，"
        "不得引入上下文之外的信息。若上下文不足以回答，直接说明无法回答。"
        "回答用中文，简洁分点，并在句末用 [序号] 标注依据来源。"
    )

    def answerer(question: str, chunks) -> str:
        context = "\n\n".join(f"{c.citation}\n{c.content}" for c in chunks)
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": f"上下文：\n{context}\n\n问题：{question}"},
        ]
        resp = llm.invoke(messages)
        return getattr(resp, "content", str(resp))

    return answerer


def run(transport: str = "stdio", host: str = "127.0.0.1", port: int = 8765, **kwargs) -> None:
    """启动 MCP 服务。

    Args:
        transport: ``stdio``（默认，供桌面客户端拉起）
            / ``streamable-http`` / ``sse``（供自建客户端或远端调用）。
    """
    app = create_app()
    try:
        app.run(transport=transport, host=host, port=port, **kwargs)
    except TypeError:  # pragma: no cover - 旧版 SDK 不支持 keyword 形式
        if transport == "stdio":
            app.run()
        else:  # pragma: no cover
            asyncio.run(app.run_streamable_http_async(host=host, port=port))


async def list_declared_tools() -> List[Dict[str, Any]]:
    """返回本地声明的工具规格（无需启动服务，便于测试与文档生成）。"""
    return [dict(spec) for spec in TOOL_SPECS]


if __name__ == "__main__":  # pragma: no cover
    run()
