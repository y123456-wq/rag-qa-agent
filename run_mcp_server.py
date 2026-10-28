# -*- coding: utf-8 -*-
"""一键启动 MCP 服务器。

用法：

    pip install -r requirements-mcp.txt

    # 演示模式（零依赖，内置示例文档，无需 API Key / Chroma）
    set RAG_MCP_DEMO=1          # Windows PowerShell: $env:RAG_MCP_DEMO="1"
    python run_mcp_server.py

    # 正常模式（使用真实的 Chroma 知识库，可额外开 RAG_MCP_LLM=1 接 LLM 生成答案）
    python run_mcp_server.py

    # 切换传输方式（默认 stdio，供桌面客户端拉起）
    python run_mcp_server.py --transport streamable-http --port 8765
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.mcp_server import run  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="启动 RAG 知识库 MCP 服务器")
    parser.add_argument(
        "--transport",
        default=os.getenv("RAG_MCP_TRANSPORT", "stdio"),
        choices=["stdio", "sse", "streamable-http"],
        help="传输方式，默认 stdio",
    )
    parser.add_argument("--host", default=os.getenv("RAG_MCP_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.getenv("RAG_MCP_PORT", "8765")))
    parser.add_argument(
        "--demo",
        action="store_true",
        help="强制使用内置演示语料（等价于设置 RAG_MCP_DEMO=1）",
    )
    parser.add_argument(
        "--with-llm",
        action="store_true",
        help="接入 LLM 生成答案；不指定时使用确定性的抽取式兜底答案",
    )
    args = parser.parse_args()

    if args.demo:
        os.environ["RAG_MCP_DEMO"] = "1"
    if args.with_llm:
        os.environ["RAG_MCP_LLM"] = "1"

    run(transport=args.transport, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
