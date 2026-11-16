# RAG Agent 离线回归评测基线

用自研 [TraceKit](https://github.com/y123456-wq/pocketflow-tracekit) 给 RAG 问答建立
**端到端回归基线**：不依赖 LLM / Chroma / 网络，固定语料 + 抽取式问答，结果确定可复现。

## 用例（5 条）

| 用例 | 约束 |
|---|---|
| 命中要点答到 Chroma | 关键事实不被"幻觉掉" |
| 检索不到如实拒答不编造 | 无依据时明确说"无法回答" |
| 空输入不崩溃 | 非法入参转成可读错误 |
| 不泄漏内部堆栈 | 输出不含 Traceback |
| 无 LLM 走降级标记 | 离线降级路径被显式标记 |

## 运行

```bash
pip install -r requirements-evals.txt   # 或 PYTHONPATH 指向 tracekit 源码
python evals/rag_eval_baseline.py        # 评测 + Trace 落盘，失败退出码非 0
python evals/rag_eval_baseline.py --diff # 与上一次运行做回放 diff
```

产物：`evals/runs/<run_id>/{trace.jsonl,summary.json}`、`evals/reports/latest.json`。
改 prompt / 换检索策略后重跑，`--diff` 会按节点对齐指出哪些输出变了。

## CI

`.github/workflows/evals.yml` 在 push / PR 时跑该基线，任一用例失败即门禁失败。
（git 依赖需 tracekit 仓库已公开；本地/未发布时用 PYTHONPATH。）