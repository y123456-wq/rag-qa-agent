# -*- coding: utf-8 -*-
"""知识库 MCP 工具层：纯业务逻辑，不依赖 MCP SDK，也不依赖具体向量库实现。

三个设计要点：

1. **依赖倒置**：检索能力通过 ``Retriever`` 协议注入。单元测试用 ``FakeRetriever``
   替换即可，不需要 Chroma / Embedding / 网络 / API Key 就能跑，
   所以这一层可以被 CI 完整覆盖。
2. **确定性降级（graceful degradation）**：没有注入 LLM 回答器时，使用抽取式
   兜底策略（extractive fallback）生成"基于原文片段"的答案，
   保证离线可演示、结果可复现；有 LLM 时才走真正的生成式回答。
3. **统一输入校验**：所有工具入口先做参数净化，非法参数抛 ``ToolInputError``，
   由 MCP 层转成结构化错误信息返回给调用方，而不是让服务端进程崩掉。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Protocol, Sequence, runtime_checkable

__all__ = [
    "ToolInputError",
    "Retriever",
    "RetrievedChunk",
    "KnowledgeBaseService",
]

# ---------------------------------------------------------------------------
# 常量：工具入参边界。集中在这里定义，方便 MCP Schema 与运行期校验保持一致。
# ---------------------------------------------------------------------------
DEFAULT_TOP_K = 3
MIN_TOP_K = 1
MAX_TOP_K = 20
MAX_QUERY_CHARS = 500
MAX_CHUNK_CHARS = 800


class ToolInputError(ValueError):
    """工具入参不合法。MCP 层会把它渲染成对模型友好的错误文本。"""


@runtime_checkable
class Retriever(Protocol):
    """检索能力的抽象协议。

    只要对象实现了这些方法，就可以被 ``KnowledgeBaseService`` 消费：
    真实环境传 ``VectorStore``，测试环境传轻量假实现。
    """

    def similarity_search(self, query: str, k: Optional[int] = None) -> Sequence[Any]:
        """返回相似度最高的 k 个文档片段。"""
        ...

    def similarity_search_with_score(
        self, query: str, k: Optional[int] = None
    ) -> Sequence[tuple]:
        """返回 ``(文档, 分数)`` 序列；不支持时可抛 NotImplementedError。"""
        ...

    def get_document_count(self) -> int:
        """知识库中的文档片段总数。"""
        ...

    def get_all_sources(self) -> Sequence[str]:
        """知识库中出现过的来源文件名列表。"""
        ...


@dataclass(frozen=True)
class RetrievedChunk:
    """标准化后的检索结果，屏蔽不同向量库返回对象的差异。"""

    index: int
    content: str
    source: str
    page: Optional[int] = None
    score: Optional[float] = None

    @property
    def citation(self) -> str:
        """人类/模型可读的引用标记，例如 ``[1] report.pdf p.12``。"""
        base = f"[{self.index}] {self.source}"
        if self.page is not None:
            base += f" p.{self.page}"
        return base

    def to_dict(self) -> Dict[str, Any]:
        return {
            "index": self.index,
            "content": self.content,
            "source": self.source,
            "page": self.page,
            "score": self.score,
        }


def _doc_content(doc: Any) -> str:
    """兼容 LangChain ``Document``、字典和裸文本三种返回形态。"""
    if isinstance(doc, str):
        return doc
    if isinstance(doc, dict):
        return str(doc.get("page_content") or doc.get("content") or "")
    return str(getattr(doc, "page_content", doc))


def _doc_metadata(doc: Any) -> Dict[str, Any]:
    if isinstance(doc, dict):
        meta = doc.get("metadata")
        return dict(meta) if isinstance(meta, dict) else {}
    meta = getattr(doc, "metadata", None)
    return dict(meta) if isinstance(meta, dict) else {}


@dataclass
class KnowledgeBaseService:
    """把向量知识库包装成一组可被 MCP 暴露的确定性工具。

    Args:
        retriever: 实现了 ``Retriever`` 协议的检索对象（真实环境为 VectorStore）。
        answerer: 可选的可调用对象 ``(question, chunks) -> str``，用于接入 LLM 生成答案。
            不传时自动退化为抽取式兜底答案。
        default_top_k: 未显式指定 top_k 时使用的返回条数。
        max_chunk_chars: 单条片段回显给模型的最大字符数，控制上下文开销。
    """

    retriever: Retriever
    answerer: Optional[Callable[[str, List[RetrievedChunk]], str]] = None
    default_top_k: int = DEFAULT_TOP_K
    max_chunk_chars: int = MAX_CHUNK_CHARS
    name: str = "rag-knowledge-base"

    # ---- 输入净化 -------------------------------------------------------
    @staticmethod
    def clean_query(query: str) -> str:
        """校验并清洗查询串。"""
        if not isinstance(query, str) or not query.strip():
            raise ToolInputError("query 不能为空，请输入要检索的问题或关键词。")
        cleaned = re.sub(r"\s+", " ", query).strip()
        if len(cleaned) > MAX_QUERY_CHARS:
            cleaned = cleaned[:MAX_QUERY_CHARS]
        return cleaned

    def clean_top_k(self, top_k: Any) -> int:
        """把 top_k 夹到合法区间，非整数输入直接拒绝。"""
        if top_k is None:
            return self.default_top_k
        if isinstance(top_k, bool) or not isinstance(top_k, int):
            raise ToolInputError(f"top_k 必须是整数（{MIN_TOP_K}-{MAX_TOP_K}），收到：{top_k!r}")
        return max(MIN_TOP_K, min(int(top_k), MAX_TOP_K))

    # ---- 核心能力 -------------------------------------------------------
    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[RetrievedChunk]:
        """执行检索并标准化结果。底层异常会被转成 ``ToolInputError`` 之外的 RuntimeError。"""
        query = self.clean_query(query)
        k = self.clean_top_k(top_k)

        try:
            scored = self.retriever.similarity_search_with_score(query, k)
        except NotImplementedError:
            scored = None
        except Exception as exc:  # pragma: no cover - 依赖真实向量库时才可能触发
            raise RuntimeError(f"检索失败：{exc}") from exc

        chunks: List[RetrievedChunk] = []
        if scored is not None:
            for i, pair in enumerate(scored, 1):
                doc, score = pair if isinstance(pair, tuple) else (pair, None)
                meta = _doc_metadata(doc)
                page = meta.get("page")
                chunks.append(
                    RetrievedChunk(
                        index=i,
                        content=_doc_content(doc).strip(),
                        source=str(meta.get("source") or "未知来源"),
                        page=int(page) if isinstance(page, int) or str(page).isdigit() else None,
                        score=round(float(score), 4) if score is not None else None,
                    )
                )
        else:
            try:
                docs = self.retriever.similarity_search(query, k)
            except Exception as exc:  # pragma: no cover
                raise RuntimeError(f"检索失败：{exc}") from exc
            for i, doc in enumerate(docs, 1):
                meta = _doc_metadata(doc)
                page = meta.get("page")
                chunks.append(
                    RetrievedChunk(
                        index=i,
                        content=_doc_content(doc).strip(),
                        source=str(meta.get("source") or "未知来源"),
                        page=int(page) if isinstance(page, int) or str(page).isdigit() else None,
                    )
                )
        return chunks

    def search(self, query: str, top_k: Optional[int] = None) -> Dict[str, Any]:
        """工具 1：纯检索。返回结构化结果 dict（便于上层再序列化）。"""
        chunks = self.retrieve(query, top_k)
        if not chunks:
            return {
                "query": query,
                "count": 0,
                "message": "知识库中没有检索到相关内容，请确认已上传文档或换个问法。",
                "chunks": [],
            }
        return {"query": query, "count": len(chunks), "chunks": [c.to_dict() for c in chunks]}

    def stats(self) -> Dict[str, Any]:
        """工具 2：知识库统计信息。"""
        try:
            total = int(self.retriever.get_document_count())
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(f"获取文档数量失败：{exc}") from exc
        try:
            sources = list(self.retriever.get_all_sources())
        except Exception:  # pragma: no cover - 来源列表失败不影响主结果
            sources = []
        return {
            "chunk_count": total,
            "source_count": len(sources),
            "sources": sources,
            "default_top_k": self.default_top_k,
            "retrieval_mode": "semantic (embedding similarity)",
        }

    def ask(self, question: str, top_k: Optional[int] = None) -> Dict[str, Any]:
        """工具 3：带引用的问答。优先 LLM 回答器，缺失时走抽取式兜底。"""
        question = self.clean_query(question)
        chunks = self.retrieve(question, top_k)
        if not chunks:
            return {
                "question": question,
                "answer": "根据现有知识库无法回答该问题：没有检索到相关文档片段。",
                "citations": [],
                "mode": "empty",
            }

        if self.answerer is not None:
            mode = "llm"
            try:
                answer = str(self.answerer(question, chunks)).strip()
            except Exception:
                # LLM 不可用（无 Key / 限流 / 超时）时应降级而不是让整个工具失败
                mode = "extractive-fallback"
                answer = self._extractive_answer(question, chunks)
        else:
            mode = "extractive"
            answer = self._extractive_answer(question, chunks)

        return {
            "question": question,
            "answer": answer,
            "citations": [c.citation for c in chunks],
            "mode": mode,
        }

    # ---- 降级策略 -------------------------------------------------------
    def _keywords(self, text: str) -> List[str]:
        """极简中文/英文关键词切分：按非字母数字与常见标点切开，去停用词。"""
        tokens = re.findall(r"[A-Za-z0-9]+|[\u4e00-\u9fff]+", text.lower())
        stop = {"的", "了", "是", "在", "和", "与", "及", "吗", "呢", "请", "我", "你", "它", "这", "那", "什么", "如何", "怎么", "为什么"}
        out = []
        for t in tokens:
            if len(t) < 2 and not t.isdigit():
                continue
            if t in stop:
                continue
            out.append(t)
            # 中文长词再按 2-gram 切一次，提升短query的召回
            if re.fullmatch(r"[\u4e00-\u9fff]{3,}", t):
                out.extend(t[i : i + 2] for i in range(len(t) - 1))
        return list(dict.fromkeys(out))

    def _extractive_answer(self, question: str, chunks: Sequence[RetrievedChunk]) -> str:
        """抽取式兜底：按关键词命中打分挑出最相关句子，拼成"有出处"的答案。

        这是确定性的（不调用任何模型），因此同样的输入永远得到同样的输出，
        便于回归测试，也保证没有 API Key 时服务依然可用。
        """
        keywords = self._keywords(question)
        sentences: List[tuple] = []
        for chunk in chunks:
            body = chunk.content[: self.max_chunk_chars * 2]
            for sent in re.split(r"(?<=[。！？；.!?;])\s*|\n", body):
                sent = sent.strip()
                if len(sent) < 6:
                    continue
                hit = sum(1 for kw in keywords if kw in sent.lower()) if keywords else 0
                if hit:
                    sentences.append((hit, -len(sent), sent, chunk.citation))

        if not sentences:
            top = chunks[0]
            return (
                "【抽取式兜底答案】未命中与问题直接相关的句子，以下是相似度最高的原文片段：\n"
                f"{top.citation}：{top.content[: self.max_chunk_chars]}"
            )

        sentences.sort(key=lambda x: (-x[0], x[1]))
        picked = sentences[:3]
        lines = [f"{s[3]}：{s[2]}" for s in picked]
        return (
            "【抽取式兜底答案 · 未在知识库外补充信息】\n" + "\n".join(lines) +
            "\n\n提示：以上为原文摘录，未接入 LLM 生成；配置 answerer 后可获得润色后的自然语言答案。"
        )

    # ---- 渲染 -----------------------------------------------------------
    @staticmethod
    def render_search(result: Dict[str, Any], max_chars: int = MAX_CHUNK_CHARS) -> str:
        if result["count"] == 0:
            return result["message"]
        lines = [f"共检索到 {result['count']} 条相关片段（查询：{result['query']}）：", ""]
        for c in result["chunks"]:
            score = f"，相似度 {c['score']}" if c.get("score") is not None else ""
            content = c["content"][:max_chars]
            if len(c["content"]) > max_chars:
                content += "…"
            lines.append(f"[{c['index']}] 来源：{c['source']}{score}")
            lines.append(content)
            lines.append("")
        return "\n".join(lines).rstrip()

    @staticmethod
    def render_stats(result: Dict[str, Any]) -> str:
        lines = [
            "知识库统计：",
            f"- 文档片段总数：{result['chunk_count']}",
            f"- 来源文件数：{result['source_count']}",
            f"- 检索方式：{result['retrieval_mode']}",
            f"- 默认返回条数：{result['default_top_k']}",
        ]
        if result["sources"]:
            lines.append("- 来源列表：")
            lines.extend(f"  · {s}" for s in result["sources"][:50])
            if len(result["sources"]) > 50:
                lines.append(f"  …（另有 {len(result['sources']) - 50} 个来源未列出）")
        else:
            lines.append("- 来源列表：（空，尚未导入文档）")
        return "\n".join(lines)

    @staticmethod
    def render_ask(result: Dict[str, Any]) -> str:
        lines = [result["answer"]]
        if result["citations"]:
            lines.append("")
            lines.append("引用来源：" + " ".join(result["citations"]))
        return "\n".join(lines)
