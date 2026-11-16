# -*- coding: utf-8 -*-
"""RAG Agent 离线回归评测基线（基于自研 TraceKit）。

不依赖 LLM / Chroma / 网络 / MCP SDK：用固定语料 + 关键词检索器驱动
``KnowledgeBaseService`` 的抽取式（extractive）问答，作为可复现的端到端回归基线。
每次运行把 Trace 落盘（runs/<id>/trace.jsonl + summary.json），评测报告存
reports/latest.json；失败时退出码非 0，可直接作为 CI 门禁。

运行::

    python evals/rag_eval_baseline.py           # 跑评测 + 落盘 trace
    python evals/rag_eval_baseline.py --diff    # 与上一次运行做回放 diff

依赖 pocketflow-tracekit（见 requirements-evals.txt）。本地开发若尚未发布/克隆，
可用 ``PYTHONPATH=<tracekit 仓库路径>`` 指向源码。
"""

from __future__ import annotations

import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..")))

from src.mcp_tools import KnowledgeBaseService, ToolInputError  # noqa: E402

try:
    from tracekit import EvalSuite, TraceRecorder, diff_runs, list_runs, load_run
    from tracekit.evalkit import contains, not_contains
except ImportError:  # pragma: no cover
    sys.stderr.write(
        "缺少 pocketflow-tracekit：pip install -r requirements-evals.txt，"
        "或设置 PYTHONPATH 指向 tracekit 仓库。\n"
    )
    raise

RUN_DIR = os.path.join(HERE, "runs")
REPORT_PATH = os.path.join(HERE, "reports", "latest.json")

# 固定语料：模拟已导入知识库的文档片段（来源/页码齐全）。
CORPUS = [
    ("本项目使用 Chroma 作为向量数据库，向量持久化在 data/vector_db 目录下。", "架构说明.md", None),
    ("Embedding 支持两种模式：调用 OpenAI 兼容接口，或使用本地 sentence-transformers 模型。", "架构说明.md", 3),
    ("ReAct Agent 每轮最多迭代 5 次，解析失败时会自动恢复，避免长对话跑飞。", "agent设计.md", None),
]


class KeywordRetriever:
    """离线确定性检索器：关键词重叠即命中，模拟向量库的模糊召回。"""

    def __init__(self, corpus):
        self.docs = [{"page_content": c, "metadata": {"source": s, **({"page": p} if p is not None else {})}} for c, s, p in corpus]

    @staticmethod
    def _tokens(text):
        out = []
        for tok in re.findall(r"[A-Za-z0-9]+|[一-鿿]+", text.lower()):
            if len(tok) == 1 and not tok.isdigit():
                continue
            out.append(tok)
            if re.fullmatch(r"[一-鿿]{2,}", tok):
                out.extend(tok[i : i + 2] for i in range(len(tok) - 1))
        return list(dict.fromkeys(out))

    def _hits(self, query):
        toks = self._tokens(query)
        return [d for d in self.docs if any(t in d["page_content"].lower() for t in toks)]

    def similarity_search_with_score(self, query, k=None):
        return [(d, 0.9 - i * 0.1) for i, d in enumerate(self._hits(query))][: k or 3]

    def similarity_search(self, query, k=None):
        return self._hits(query)[: k or 3]

    def get_document_count(self):
        return len(self.docs)

    def get_all_sources(self):
        return sorted({d["metadata"]["source"] for d in self.docs})


_SERVICE = KnowledgeBaseService(retriever=KeywordRetriever(CORPUS))
_REC = TraceRecorder(run_dir=RUN_DIR, name="rag-eval-baseline")


def runner(question):
    """被测对象：一次端到端问答（抽取式、确定性）。空输入转成可读错误而非崩溃。"""
    with _REC.span("RAG.ask", inputs=question) as sp:
        try:
            answer = _SERVICE.ask(question)["answer"]
        except ToolInputError as exc:
            answer = str(exc)
        sp.set_outputs(answer)
        return answer


def build_suite():
    suite = EvalSuite(name="rag-qa-agent-baseline", runner=runner)
    suite.case("命中要点答到 Chroma", "向量数据库用什么", contains("Chroma"))
    suite.case("检索不到如实拒答不编造", "公司去年的净利润是多少", contains("无法回答"))
    suite.case("空输入不崩溃", "", contains("请输入"))
    suite.case("不泄漏内部堆栈", "向量数据库用什么", not_contains("Traceback"))
    suite.case("无 LLM 走降级标记", "ReAct 每轮最多迭代几次", contains("抽取式兜底答案"))
    return suite


def main(argv):
    suite = build_suite()
    report = suite.run()
    summary = _REC.finish()
    suite.save(report, REPORT_PATH)

    print(report.render())
    print(f"\nTrace 已落盘：{os.path.join(RUN_DIR, summary.run_id)}（span {summary.span_count} 个）")
    print(f"报告：{REPORT_PATH}")

    if "--diff" in argv:
        runs = list_runs(RUN_DIR)
        if len(runs) >= 2:
            base, cur = runs[-2]["run_id"], runs[-1]["run_id"]
            d = diff_runs(load_run(base, RUN_DIR), load_run(cur, RUN_DIR))
            print("\n回放 diff（上一次 vs 本次）：")
            print(d.render())
        else:
            print("\n（仅一次运行，暂无可 diff 的基准；再跑一次即可对比）")

    return 0 if report.ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))