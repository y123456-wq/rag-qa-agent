# -*- coding: utf-8 -*-
"""mcp_tools 层单元测试。

这一层刻意不依赖 MCP SDK、Chroma、Embedding 和网络，
全部用例用 FakeRetriever 跑，因此可以进 CI、可以秒级回归。
"""

from __future__ import annotations

import os
import re
import sys
import unittest
from typing import List, Sequence

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.mcp_tools import (  # noqa: E402
    MAX_TOP_K,
    KnowledgeBaseService,
    RetrievedChunk,
    ToolInputError,
)


class FakeDoc:
    """模拟 LangChain Document：只有 page_content 与 metadata 两个属性。"""

    def __init__(self, content: str, source: str = "手册.pdf", page=None):
        self.page_content = content
        self.metadata = {"source": source}
        if page is not None:
            self.metadata["page"] = page


class FakeRetriever:
    """可控的假检索器：命中关键词，未命中返回空列表。

    同时实现/不实现 similarity_search_with_score 两种形态，用于覆盖两条分支。
    """

    def __init__(self, docs: Sequence[FakeDoc] = (), with_score: bool = True, raise_on: str = ""):
        self.docs: List[FakeDoc] = list(docs)
        self.with_score = with_score
        self.raise_on = raise_on
        self.last_k = None

    def _guard(self, query: str):
        if self.raise_on and self.raise_on in query:
            raise RuntimeError("向量库连接中断")

    def similarity_search(self, query: str, k=None):
        self._guard(query)
        self.last_k = k
        hits = [d for d in self.docs if _overlap(query, d.page_content)]
        return hits[: k or 3]

    def similarity_search_with_score(self, query: str, k=None):
        if not self.with_score:
            raise NotImplementedError
        self._guard(query)
        self.last_k = k
        hits = [(d, 0.9 - i * 0.1) for i, d in enumerate(self.docs) if _overlap(query, d.page_content)]
        return hits[: k or 3]

    def get_document_count(self) -> int:
        return len(self.docs)

    def get_all_sources(self) -> List[str]:
        return sorted({d.metadata["source"] for d in self.docs})


_TOKEN_RE = re.compile(r"[A-Za-z0-9]+|[\u4e00-\u9fff]+")


def _tokens(text: str) -> List[str]:
    """与 KnowledgeBaseService 同款的极简分词，用于让假检索器有可预期的召回行为。"""
    out = []
    for tok in _TOKEN_RE.findall(text.lower()):
        if len(tok) == 1 and not tok.isdigit():
            continue
        out.append(tok)
        if re.fullmatch(r"[\u4e00-\u9fff]{2,}", tok):
            out.extend(tok[i : i + 2] for i in range(len(tok) - 1))
    return list(dict.fromkeys(out))


def _overlap(query: str, text: str) -> bool:
    """只要有任一分词命中即视为相关（模拟向量库的模糊召回）。"""
    return any(t in text.lower() for t in _tokens(query))


DOCS = [
    FakeDoc(
        "本项目使用 Chroma 作为向量数据库，向量持久化在 data/vector_db 目录下。",
        "架构说明.md",
    ),
    FakeDoc(
        "Embedding 支持两种模式：调用 OpenAI 兼容接口，或使用本地 sentence-transformers 模型。",
        "架构说明.md",
        page=3,
    ),
    FakeDoc(
        "ReAct Agent 每轮最多迭代 5 次，解析失败时会自动恢复，避免长对话跑飞。",
        "agent设计.md",
    ),
]


class TestInputValidation(unittest.TestCase):
    def setUp(self):
        self.kb = KnowledgeBaseService(retriever=FakeRetriever(DOCS))

    def test_empty_query_rejected(self):
        for bad in ["", "   ", "\n\t"]:
            with self.assertRaises(ToolInputError):
                self.kb.clean_query(bad)

    def test_query_is_normalized(self):
        self.assertEqual(self.kb.clean_query(" 向量   数据库 \n 怎么用 "), "向量 数据库 怎么用")

    def test_query_is_truncated(self):
        long_q = "啊" * 1000
        self.assertEqual(len(self.kb.clean_query(long_q)), 500)

    def test_top_k_clamped_to_upper_bound(self):
        self.assertEqual(self.kb.clean_top_k(999), MAX_TOP_K)

    def test_top_k_clamped_to_lower_bound(self):
        self.assertEqual(self.kb.clean_top_k(0), 1)
        self.assertEqual(self.kb.clean_top_k(-5), 1)

    def test_top_k_none_uses_default(self):
        kb = KnowledgeBaseService(retriever=FakeRetriever(DOCS), default_top_k=7)
        self.assertEqual(kb.clean_top_k(None), 7)

    def test_top_k_rejects_non_integer(self):
        for bad in ["3", 3.5, True, []]:
            with self.assertRaises(ToolInputError):
                self.kb.clean_top_k(bad)


class TestSearch(unittest.TestCase):
    def setUp(self):
        self.kb = KnowledgeBaseService(retriever=FakeRetriever(DOCS))

    def test_returns_structured_chunks(self):
        res = self.kb.search("向量数据库")
        self.assertGreater(res["count"], 0)
        self.assertEqual(res["count"], len(res["chunks"]))
        first = res["chunks"][0]
        self.assertIn("content", first)
        self.assertIn("source", first)
        self.assertIsInstance(first["score"], float)

    def test_source_and_page_extracted(self):
        res = self.kb.search("Embedding")
        hit = [c for c in res["chunks"] if c["page"] == 3]
        self.assertTrue(hit, "应能从 metadata 中提取出 page 字段")

    def test_top_k_limits_results(self):
        res = self.kb.search("向量", top_k=1)
        self.assertEqual(res["count"], 1)

    def test_no_hit_returns_friendly_message(self):
        res = self.kb.search("量子计算zzz")
        self.assertEqual(res["count"], 0)
        self.assertIn("没有检索到", res["message"])
        self.assertIn("没有检索到", KnowledgeBaseService.render_search(res))

    def test_falls_back_when_scoring_unsupported(self):
        kb = KnowledgeBaseService(retriever=FakeRetriever(DOCS, with_score=False))
        res = kb.search("Embedding 模式")
        self.assertGreater(res["count"], 0)
        self.assertIsNone(res["chunks"][0]["score"])

    def test_retriever_exception_becomes_runtime_error(self):
        kb = KnowledgeBaseService(retriever=FakeRetriever(DOCS, raise_on="崩溃"))
        with self.assertRaises(RuntimeError):
            kb.search("崩溃")

    def test_render_includes_citation_and_content(self):
        text = KnowledgeBaseService.render_search(self.kb.search("ReAct"))
        self.assertIn("[1]", text)
        self.assertIn("ReAct", text)


class TestStats(unittest.TestCase):
    def test_counts_and_sources(self):
        kb = KnowledgeBaseService(retriever=FakeRetriever(DOCS))
        res = kb.stats()
        self.assertEqual(res["chunk_count"], 3)
        self.assertEqual(res["source_count"], 2)
        self.assertIn("agent设计.md", res["sources"])
        rendered = KnowledgeBaseService.render_stats(res)
        self.assertIn("文档片段总数：3", rendered)

    def test_empty_knowledge_base(self):
        kb = KnowledgeBaseService(retriever=FakeRetriever([]))
        res = kb.stats()
        self.assertEqual(res["chunk_count"], 0)
        self.assertIn("尚未导入文档", KnowledgeBaseService.render_stats(res))


class TestAsk(unittest.TestCase):
    def test_extractive_answer_without_llm(self):
        kb = KnowledgeBaseService(retriever=FakeRetriever(DOCS))
        res = kb.ask("向量存在哪个目录")
        self.assertEqual(res["mode"], "extractive")
        self.assertIn("vector_db", res["answer"])
        self.assertTrue(res["citations"])
        self.assertIn("引用来源", KnowledgeBaseService.render_ask(res))

    def test_extractive_answer_is_deterministic(self):
        kb = KnowledgeBaseService(retriever=FakeRetriever(DOCS))
        q = "Embedding 有哪些模式"
        self.assertEqual(kb.ask(q)["answer"], kb.ask(q)["answer"])

    def test_uses_llm_answerer_when_provided(self):
        def fake_llm(question, chunks):
            assert isinstance(chunks[0], RetrievedChunk)
            return f"【LLM】{question} -> 共{len(chunks)}条依据"

        kb = KnowledgeBaseService(retriever=FakeRetriever(DOCS), answerer=fake_llm)
        res = kb.ask("Embedding 模式")
        self.assertEqual(res["mode"], "llm")
        self.assertIn("【LLM】", res["answer"])

    def test_degrades_when_llm_raises(self):
        def broken_llm(question, chunks):
            raise TimeoutError("LLM 超时")

        kb = KnowledgeBaseService(retriever=FakeRetriever(DOCS), answerer=broken_llm)
        res = kb.ask("Embedding 模式")
        self.assertEqual(res["mode"], "extractive-fallback")
        self.assertIn("【抽取式兜底答案", res["answer"])

    def test_no_hit_question(self):
        kb = KnowledgeBaseService(retriever=FakeRetriever([]))
        res = kb.ask("是什么")
        self.assertEqual(res["mode"], "empty")
        self.assertIn("无法回答", res["answer"])
        self.assertEqual(res["citations"], [])


class TestProtocolCompatibility(unittest.TestCase):
    def test_accepts_dict_documents(self):
        class DictRetriever:
            def similarity_search_with_score(self, q, k=None):
                return [({"page_content": "字典形态文档内容", "metadata": {"source": "a.md"}}, 0.5)]

            def get_document_count(self):
                return 1

            def get_all_sources(self):
                return ["a.md"]

        res = KnowledgeBaseService(retriever=DictRetriever()).search("字典")
        self.assertEqual(res["chunks"][0]["source"], "a.md")
        self.assertIn("字典形态", res["chunks"][0]["content"])

    def test_is_runtime_checkable_protocol(self):
        from src.mcp_tools import Retriever

        self.assertTrue(isinstance(FakeRetriever(DOCS), Retriever))


if __name__ == "__main__":
    unittest.main(verbosity=2)
