# -*- coding: utf-8 -*-
"""MCP 服务层测试：覆盖工具注册、JSON Schema 契约与端到端调用结果。

这一组用例需要安装 MCP SDK（不会真正拉起 stdio 进程，只构建 application 对象）。
"""

from __future__ import annotations

import asyncio
import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def _schema_of(tool) -> dict:
    """兼容 mcp 1.x 的 ``inputSchema`` 与 2.x 的 ``input_schema``。"""
    return dict(getattr(tool, "input_schema", None) or getattr(tool, "inputSchema", None) or {})


class _Doc:
    def __init__(self, content, source="手册.pdf", page=None):
        self.page_content = content
        self.metadata = {"source": source}
        if page is not None:
            self.metadata["page"] = page


class _FakeRetriever:
    DOCS = [
        _Doc("使用 Chroma 持久化向量，目录为 data/vector_db。", "架构说明.md"),
        _Doc("Embedding 支持 OpenAI 接口与本地 sentence-transformers 两种模式。", "架构说明.md", page=3),
    ]

    def similarity_search_with_score(self, query, k=None):
        hits = [(d, 0.95 - i * 0.05) for i, d in enumerate(self.DOCS) if query[:2] in d.page_content]
        return hits[: k or 3]

    def get_document_count(self) -> int:
        return len(self.DOCS)

    def get_all_sources(self) -> list:
        return sorted({d.metadata["source"] for d in self.DOCS})


def _build_service():
    from src.mcp_tools import KnowledgeBaseService

    return KnowledgeBaseService(retriever=_FakeRetriever(), default_top_k=3)


def _get_registered(app) -> dict:
    """把 app 已注册的工具按名字索引，用于断言。"""
    tools = asyncio.run(app.list_tools())
    return {t.name: _schema_of(t) for t in tools}


@unittest.skipIf(
    os.getenv("SKIP_MCP_TESTS") == "1",
    "设置 SKIP_MCP_TESTS=1 时跳过依赖 MCP SDK 的用例",
)
class TestToolRegistration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from src.mcp_server import create_app

        cls.app = create_app(_build_service())
        cls.schemas = _get_registered(cls.app)

    def test_all_declared_tools_are_registered(self):
        from src.mcp_server import TOOL_SPECS

        expected = {spec["name"] for spec in TOOL_SPECS}
        self.assertTrue(expected.issubset(set(self.schemas)), f"实际注册：{list(self.schemas)}")

    def test_search_tool_schema(self):
        schema = self.schemas["search_knowledge_base"]
        props = schema["properties"]
        self.assertIn("query", props)
        self.assertIn("top_k", props)
        self.assertEqual(schema["required"], ["query"])
        self.assertEqual(props["top_k"]["type"], "integer")

    def test_ask_tool_schema(self):
        schema = self.schemas["ask_knowledge_base"]
        self.assertEqual(schema["required"], ["question"])
        self.assertIn("top_k", schema["properties"])

    def test_stats_tool_has_no_required_args(self):
        schema = self.schemas["knowledge_base_stats"]
        self.assertEqual(schema.get("required", []), [])

    def test_declared_spec_matches_runtime_schema(self):
        """手写的 TOOL_SPECS 必须与实际注册的一致，防止文档漂移。"""
        from src.mcp_server import TOOL_SPECS

        for spec in TOOL_SPECS:
            runtime = self.schemas[spec["name"]]
            expected_props = set(spec["inputSchema"]["properties"])
            self.assertEqual(expected_props, set(runtime["properties"]), spec["name"])
            self.assertEqual(spec["inputSchema"]["required"], runtime.get("required", []), spec["name"])


@unittest.skipIf(os.getenv("SKIP_MCP_TESTS") == "1", "需要 MCP SDK")
class TestToolBehavior(unittest.TestCase):
    """直接调用注册后的工具函数，验证端到端行为（含降级与错误处理）。"""

    @classmethod
    def setUpClass(cls):
        from src.mcp_server import create_app

        cls.app = create_app(_build_service())
        cls.handlers = _tool_handlers(cls.app)

    def test_search_returns_source_and_score(self):
        out = _run(self.handlers["search_knowledge_base"](query="使用 Chroma", top_k=2))
        self.assertIn("来源：架构说明.md", out)
        self.assertIn("相似度", out)

    def test_search_no_hit_is_friendly(self):
        out = _run(self.handlers["search_knowledge_base"](query="完全不存在的东西"))
        self.assertIn("没有检索到", out)

    def test_out_of_range_top_k_is_clamped_not_crashed(self):
        """超出上界的 top_k 应被夹到最大值而不是报错或把库全量捞出。"""
        out = _run(self.handlers["search_knowledge_base"](query="使用 Chroma", top_k=9999))
        self.assertIn("来源：架构说明.md", out)

    def test_wrong_type_top_k_is_reported_not_crashed(self):
        out = _run(self.handlers["search_knowledge_base"](query="使用 Chroma", top_k="abc"))
        self.assertIn("参数错误", out)

    def test_empty_query_is_reported_not_crashed(self):
        out = _run(self.handlers["search_knowledge_base"](query="   "))
        self.assertIn("参数错误", out)

    def test_stats_tool_output(self):
        out = _run(self.handlers["knowledge_base_stats"]())
        self.assertIn("文档片段总数：2", out)
        self.assertIn("架构说明.md", out)

    def test_ask_falls_back_to_extractive_without_llm(self):
        out = _run(self.handlers["ask_knowledge_base"](question="Embedding"))
        self.assertIn("抽取式兜底答案", out)
        self.assertIn("引用来源", out)


def _tool_handlers(app) -> dict:
    """从 app 实例里取出已注册的工具函数。

    不同 SDK 版本的内部结构不同，这里做多路径兼容；取不到时直接抛错提醒，
    避免测试用例默默失效。
    """
    candidates = {}
    manager = getattr(app, "_tool_manager", None)
    tools_obj = getattr(manager, "tools", None) or getattr(manager, "_tools", None)
    if isinstance(tools_obj, dict):
        for name, tool in tools_obj.items():
            fn = getattr(tool, "fn", None) or getattr(tool, "func", None)
            if fn is not None:
                candidates[name] = fn
    if not candidates:
        for attr in ("_tools", "tools"):
            obj = getattr(app, attr, None)
            if isinstance(obj, dict):
                for name, tool in obj.items():
                    fn = getattr(tool, "fn", None) or getattr(tool, "func", None) or tool
                    candidates[name] = fn
    if not candidates:
        raise AssertionError("无法从 MCP app 中取出已注册工具，请检查 SDK 版本兼容性")
    return candidates


def _run(coro):
    return asyncio.run(coro)


if __name__ == "__main__":
    unittest.main(verbosity=2)
