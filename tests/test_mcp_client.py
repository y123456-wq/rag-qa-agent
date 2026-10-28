# -*- coding: utf-8 -*-
"""MCP 客户端适配层测试 + 真实 stdio 端到端联通测试。

* 纯函数部分（入参解析、prompt 增强、结果渲染）不依赖网络与子进程，永远会跑；
* 端到端部分会真的拉起一个 ``python -m src.mcp_server`` 子进程，
  用真实的 MCP 客户端走完 initialize → tools/list → tools/call，
  这是验证"协议真的通了"而不是"代码看起来对"的唯一方式。
"""

from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.mcp_client import (  # noqa: E402
    MCPServerConfig,
    MCPToolBridge,
    RemoteToolSpec,
    _enhanced_description,
    _parse_payload,
    _render_call_result,
)

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


class TestPayloadParsing(unittest.TestCase):
    def test_single_arg_passthrough(self):
        self.assertEqual(_parse_payload("hello", ["q"]), {"q": "hello"})

    def test_single_arg_with_equal_sign_still_passthrough(self):
        # 单参数工具即使值里含等号，也应整体作为值传入（避免被误切成 kv）
        self.assertEqual(_parse_payload("a=1", ["q"]), {"q": "a=1"})

    def test_multi_arg_kv_parsing(self):
        out = _parse_payload("count=5; page=2", ["count", "page"])
        self.assertEqual(out, {"count": "5", "page": "2"})

    def test_multi_arg_ignores_unknown_keys(self):
        out = _parse_payload("count=5; evil=xx", ["count"])
        self.assertEqual(out, {"count": "5"})

    def test_json_payload(self):
        self.assertEqual(_parse_payload('{"count": 5}', ["count"]), {"count": 5})

    def test_no_arg_tool_returns_empty(self):
        self.assertEqual(_parse_payload("whatever", []), {})

    def test_non_kv_multi_arg_falls_back_to_first(self):
        out = _parse_payload("随便一句话", ["count", "page"])
        self.assertEqual(out, {"count": "随便一句话"})


class TestSpecAndPrompt(unittest.TestCase):
    def test_no_arg_description(self):
        spec = RemoteToolSpec(name="ping", description="健康检查")
        rendered = _enhanced_description(spec)
        self.assertIn("不需要参数", rendered)

    def test_description_includes_types_and_required(self):
        spec = RemoteToolSpec(
            name="get_comments",
            description="获取评论",
            input_schema={
                "properties": {"video": {"type": "string"}, "count": {"type": "integer", "minimum": 1, "maximum": 49}},
                "required": ["video"],
            },
        )
        rendered = _enhanced_description(spec)
        self.assertIn("video（必填）", rendered)
        self.assertIn("count: integer", rendered)
        self.assertIn("范围 1~49", rendered)

    def test_render_call_result_text_blocks(self):
        class Block:
            text = "第一行"

        class Result:
            content = [Block()]
            isError = False

        self.assertEqual(_render_call_result(Result()), "第一行")

    def test_render_call_result_error(self):
        class Result:
            content = []
            isError = True

        self.assertIn("工具返回错误", _render_call_result(Result()))

    def test_render_call_result_empty(self):
        class Result:
            content = []
            isError = False

        self.assertIn("没有返回内容", _render_call_result(Result()))

    def test_spec_from_sdk_tool_handles_both_field_names(self):
        class V1Tool:  # mcp<2 用 inputSchema
            name = "t"
            description = "d"
            inputSchema = {"properties": {"a": {"type": "string"}}}

        class V2Tool:  # mcp>=2 用 input_schema
            name = "t"
            description = "d"
            input_schema = {"properties": {"a": {"type": "string"}}}

        for tool in (V1Tool(), V2Tool()):
            self.assertEqual(RemoteToolSpec.from_sdk_tool(tool).arg_names, ["a"])


@unittest.skipIf(
    os.getenv("SKIP_MCP_E2E") == "1",
    "设置 SKIP_MCP_E2E=1 时跳过真实子进程端到端用例",
)
class TestStdioEndToEnd(unittest.TestCase):
    """真的拉起子进程，走完 MCP 协议全流程。"""

    @classmethod
    def setUpClass(cls):
        env = dict(os.environ)
        env["PYTHONPATH"] = REPO_ROOT
        env["PYTHONIOENCODING"] = "utf-8"
        env["RAG_MCP_DEMO"] = "1"
        cls.bridge = MCPToolBridge(
            config=MCPServerConfig(
                command=sys.executable,
                args=["-m", "src.mcp_server"],
                env=env,
                cwd=REPO_ROOT,
                label="rag-demo-server",
            )
        )
        cls.remote_tools = cls.bridge.connect()

    @classmethod
    def tearDownClass(cls):
        cls.bridge.close()

    def test_remote_tool_discovery(self):
        names = [t.name for t in self.remote_tools]
        for expected in ("search_knowledge_base", "ask_knowledge_base", "knowledge_base_stats"):
            self.assertIn(expected, names)

    def test_remote_schema_is_available(self):
        specs = {t.name: t for t in self.remote_tools}
        self.assertIn("query", specs["search_knowledge_base"].arg_names)

    def test_call_search_over_the_wire(self):
        out = self.bridge.call_tool("search_knowledge_base", {"query": "Embedding 模式", "top_k": 2})
        self.assertIn("架构说明.md", out)
        self.assertIn("相似度", out)

    def test_call_stats_over_the_wire(self):
        out = self.bridge.call_tool("knowledge_base_stats", {})
        self.assertIn("文档片段总数", out)

    def test_call_ask_over_the_wire(self):
        out = self.bridge.call_tool("ask_knowledge_base", {"question": "向量存在哪里"})
        self.assertIn("抽取式兜底答案", out)
        self.assertIn("引用来源", out)

    def test_bad_arguments_return_error_text_not_exception(self):
        out = self.bridge.call_tool("search_knowledge_base", {"query": ""})
        self.assertIsInstance(out, str)
        self.assertTrue(out)

    def test_translate_to_langchain_tools(self):
        try:
            import langchain  # noqa: F401
        except Exception as exc:  # pragma: no cover
            self.skipTest(f"未安装 langchain：{exc}")
        tools = self.bridge.to_langchain_tools()
        names = {t.name for t in tools}
        self.assertIn("knowledge_base_stats", names)
        stats = next(t for t in tools if t.name == "knowledge_base_stats")
        self.assertIn("文档片段总数", stats.func(""))


if __name__ == "__main__":
    unittest.main(verbosity=2)
