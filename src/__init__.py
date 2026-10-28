# -*- coding: utf-8 -*-
"""RAG 智能问答 Agent - 核心模块。

这里使用 **惰性导出**（PEP 562 ``__getattr__``）：``import src.mcp_tools`` 这类
单点导入不会连带触发 LangChain / Chroma / dotenv 等重型依赖的初始化。
好处是工具层与 MCP 服务层可以在极简环境（例如 CI）里被单独导入和测试。
"""

from typing import Any

_LAZY_EXPORTS = {
    "config": ".config",
    "Config": ".config",
    "DocumentLoader": ".document_loader",
    "VectorStore": ".vector_store",
    "ChatHistoryManager": ".chat_history",
    "RAGChain": ".rag_chain",
    "RAGAgent": ".agent",
    # 轻量模块，无重型依赖
    "KnowledgeBaseService": ".mcp_tools",
    "RetrievedChunk": ".mcp_tools",
    "ToolInputError": ".mcp_tools",
}

__all__ = sorted(_LAZY_EXPORTS.keys())


def __getattr__(name: str) -> Any:
    if name not in _LAZY_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    module = importlib.import_module(_LAZY_EXPORTS[name], __name__)
    return getattr(module, name)


def __dir__():
    return sorted(__all__)
