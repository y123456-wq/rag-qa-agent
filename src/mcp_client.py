# -*- coding: utf-8 -*-
"""MCP 客户端适配层：让本地 ReAct Agent 反过来调用外部 MCP 服务器。

到这里，本仓库同时具备两种角色：

* **作为 Server**：被 Claude Desktop / Cursor 等客户端调用（``src/mcp_server.py``）；
* **作为 Client**：调用别人的 MCP 服务器，例如社区里的 ``bili-subtitle-mcp``，
  把外部能力即时接入自己的 ReAct Agent 工具池。

关键技术点：

1. **长生命周期会话**：MCP 的 stdio 传输是一条有状态的双向管道，
   不能在每次工具调用时"起进程 → 用完就关"，否则子进程反复冷启动且上下文丢失。
   这里用一个后台 ``asyncio`` 事件循环线程持有会话，业务线程同步调用。
2. **工具发现 → 工具注册**：运行时拉取远端 ``tools/list``，
   自动把 JSON Schema 描述的远端工具转成 LangChain ``Tool``，
   即"接一个新 MCP 服务器不需要改代码"。
3. **失败隔离**：远端工具报错会被转成给模型看的字符串，不会让 Agent 整体崩溃。
"""

from __future__ import annotations

import asyncio
import threading
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence

__all__ = ["MCPServerConfig", "RemoteToolSpec", "MCPToolBridge"]


@dataclass(frozen=True)
class MCPServerConfig:
    """如何拉起一个 MCP 服务器进程（stdio 传输）。"""

    command: str
    args: Sequence[str] = ()
    env: Optional[Dict[str, str]] = None
    cwd: Optional[str] = None
    label: str = ""

    def resolved_label(self) -> str:
        return self.label or self.command


@dataclass(frozen=True)
class RemoteToolSpec:
    """远端工具的描述快照（tools/list 的结果标准化后的样子）。"""

    name: str
    description: str
    input_schema: Dict[str, Any] = field(default_factory=dict)
    server: str = ""

    @staticmethod
    def schema_of(tool: Any) -> Dict[str, Any]:
        """兼容 SDK 两个大版本的字段名差异（``input_schema`` / ``inputSchema``）。"""
        schema = getattr(tool, "input_schema", None) or getattr(tool, "inputSchema", None)
        return dict(schema or {})

    @classmethod
    def from_sdk_tool(cls, tool: Any, server: str = "") -> "RemoteToolSpec":
        return cls(
            name=tool.name,
            description=(getattr(tool, "description", "") or "").strip(),
            input_schema=cls.schema_of(tool),
            server=server,
        )

    @property
    def arg_names(self) -> List[str]:
        props = self.input_schema.get("properties") or {}
        return list(props.keys())

    @property
    def required_args(self) -> List[str]:
        return list(self.input_schema.get("required") or [])

    def render_for_prompt(self) -> str:
        """给 ReAct prompt 用的一行式工具说明。"""
        args = ", ".join(self.arg_names) if self.arg_names else "（无参数）"
        return f"{self.name}({args}) —— {self.description}"


class _BackgroundLoop:
    """在独立线程里持有一个常驻 asyncio 事件循环。

    MCP 会话由 asyncio 驱动，而 LangChain 的工具函数是同步调用，
    用这个桥接受即可让两边都保持各自习惯的写法。
    """

    def __init__(self) -> None:
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._serve, daemon=True, name="mcp-loop")
        self._thread.start()

    def _serve(self) -> None:
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    def run(self, coro):
        """提交协程到常驻循环并同步等待结果。"""
        return asyncio.run_coroutine_threadsafe(coro, self._loop).result()

    def submit(self, coro):
        """提交协程但不等它结束（用于长生命周期任务），返回 Future。"""
        return asyncio.run_coroutine_threadsafe(coro, self._loop)

    def shutdown(self) -> None:
        self._loop.call_soon_threadsafe(self._loop.stop)


class MCPToolBridge:
    """把一个外部 MCP 服务器的工具，桥接成本地 Agent 可以直接调用的工具。

    Args:
        config: 远端服务器启动方式。
        session_factory: 可注入的会话工厂，形如 ``f(stdio_client, server_params)``，
            返回一个异步上下文管理器。默认是真正的 stdio 客户端；
            测试时可以注入假会话，无需真的起子进程。
    """

    def __init__(
        self,
        config: Optional[MCPServerConfig] = None,
        session_factory: Optional[Callable[[Any, Any], Any]] = None,
        timeout_seconds: float = 60.0,
    ) -> None:
        self.config = config
        self.session_factory = session_factory or _default_session_factory
        self.timeout_seconds = timeout_seconds
        self._loop = _BackgroundLoop()
        self._tools: Dict[str, RemoteToolSpec] = {}
        self._session: Any = None
        self._ready = threading.Event()
        self._stop = threading.Event()

    # ---- 生命周期 -------------------------------------------------------
    def connect(self) -> List[RemoteToolSpec]:
        """建立会话并完成工具发现。返回远端工具清单。

        会话由后台循环里的**单个常驻任务**持有，直到 ``close()`` 才退出。
        这样设计是因为 MCP 的 stdio 传输基于 anyio task group，
        进入和退出必须在同一个 task 里完成，分两个 task 会抛
        "Attempted to exit cancel scope in a different task"。
        """
        import asyncio as _asyncio

        self._stop_event = _asyncio.Event()
        self._owner = self._loop.submit(self._session_owner())
        if not self._ready.wait(timeout=self.timeout_seconds):
            raise TimeoutError(
                f"连接 MCP 服务器超时（{self.timeout_seconds}s）：{self.config and self.config.resolved_label()}"
            )
        return self._discovered

    def close(self) -> None:
        """通知常驻任务收尾，最后关闭后台循环。"""
        try:
            if getattr(self, "_owner", None) is not None:
                self._loop._loop.call_soon_threadsafe(self._stop_event.set)
                try:
                    self._owner.result(timeout=10)
                except Exception:  # noqa: BLE001 - 关闭阶段的异常不应影响调用方
                    pass
        finally:
            self._loop.shutdown()

    async def _session_owner(self) -> None:
        """持有 stdio 连接的整个生命周期（enter ... 阻塞等待 stop ... exit）。"""
        from mcp import ClientSession
        from mcp.client.stdio import StdioServerParameters, stdio_client

        if self.config is None:
            raise ValueError("未提供 MCPServerConfig，无法启动远端服务器。")

        params = StdioServerParameters(
            command=self.config.command,
            args=list(self.config.args),
            env=self.config.env,
            cwd=self.config.cwd,
        )
        async with self.session_factory(stdio_client, params) as streams:
            async with ClientSession(streams[0], streams[1]) as session:
                self._session = session
                await session.initialize()
                result = await session.list_tools()
                self._discovered = [
                    RemoteToolSpec.from_sdk_tool(t, server=self.config.resolved_label())
                    for t in result.tools
                ]
                self._tools = {s.name: s for s in self._discovered}
                self._ready.set()
                await self._stop_event.wait()

    def call_tool(self, name: str, arguments: Optional[Dict[str, Any]] = None) -> str:
        """同步调用远端工具；返回拼好的纯文本结果。"""
        try:
            return self._loop.run(self._call_tool_async(name, arguments or {}))
        except Exception as exc:  # noqa: BLE001 - 远端失败必须降级成对模型可见的文本
            return f"调用远端工具 {name} 失败：{type(exc).__name__}: {exc}"

    async def _call_tool_async(self, name: str, arguments: Dict[str, Any]) -> str:
        session = getattr(self, "_session", None)
        if session is None:
            raise RuntimeError("会话尚未建立，请先调用 connect()。")
        result = await session.call_tool(name, arguments)
        return _render_call_result(result)

    # ---- 转成 LangChain 工具 -------------------------------------------
    def to_langchain_tools(self) -> List[Any]:
        """把远端工具转成 LangChain ``Tool`` 列表，可直接塞进 AgentExecutor。

        远端工具大多只需要一个入参，因此这里约定：
        单参数工具直接透传字符串；多参数/无参数工具则按 ``key=value`` 形式解析，
        这样改造的是 Agent 的工具池，而不是 prompt 模板。
        """
        from langchain.agents import Tool

        tools = []
        for spec in self._tools.values():
            tools.append(
                Tool(
                    name=spec.name,
                    func=self._make_func(spec),
                    description=_enhanced_description(spec),
                )
            )
        return tools

    def _make_func(self, spec: RemoteToolSpec) -> Callable[[str], str]:
        names = spec.arg_names

        def _call(payload: str = "") -> str:
            return self.call_tool(spec.name, _parse_payload(payload, names))

        _call.__name__ = spec.name
        _call.__doc__ = spec.description
        return _call


def _default_session_factory(stdio_client: Callable, params: Any) -> Any:
    return stdio_client(params)


def _enhanced_description(spec: RemoteToolSpec) -> str:
    """把结构化入参信息补进描述里，显著降低模型乱填参数的概率。"""
    desc = spec.description or spec.name
    if not spec.arg_names:
        return f"{desc}\n调用方式：直接输入空字符串即可（该工具不需要参数）。"
    args_desc = "，".join(
        f"{n}{'（必填）' if n in spec.required_args else '（可选）'}" for n in spec.arg_names
    )
    props = spec.input_schema.get("properties") or {}
    hints = []
    for n in spec.arg_names:
        p = props.get(n) or {}
        t = p.get("type", "string")
        extra = f"，取值 {p['enum']}" if p.get("enum") else ""
        bounds = ""
        if isinstance(p.get("minimum"), (int, float)) or isinstance(p.get("maximum"), (int, float)):
            bounds = f"，范围 {p.get('minimum', '?')}~{p.get('maximum', '?')}"
        hints.append(f"{n}: {t}{extra}{bounds}")
    single = len(spec.arg_names) == 1
    call_hint = (
        f"直接输入 {spec.arg_names[0]} 的值即可。"
        if single
        else f"请按 \"{spec.arg_names[0]}=值\" 的形式用分号分隔多个参数输入。"
    )
    return f"{desc}\n参数说明：{args_desc}\n字段类型：{'; '.join(hints)}\n调用方式：{call_hint}"


def _parse_payload(payload: str, names: Sequence[str]) -> Dict[str, Any]:
    """把模型的自然语言输入解析成结构化入参。

    兼容三种形态：
    ``值``（单参数简写）、``k=v; k2=v2``、以及 JSON。
    解析不出来时退化为按位置填第一个参数，保证工具永远能被调到。
    """
    if not names:
        return {}
    payload = (payload or "").strip()
    if payload.startswith("{") and payload.endswith("}"):
        try:
            import json

            data = json.loads(payload)
            if isinstance(data, dict):
                return data
        except Exception:  # noqa: BLE001 - 非法 JSON 时按文本处理
            pass
    if len(names) == 1 and "=" not in payload:
        return {names[0]: payload}
    parsed: Dict[str, Any] = {names[0]: payload}
    parts = [p.strip() for p in payload.replace("，", ";").split(";") if p.strip()]
    kv = {}
    for part in parts:
        if "=" in part:
            k, v = part.split("=", 1)
            k, v = k.strip(), v.strip().strip('"').strip("'")
            if k in names:
                kv[k] = v
    if kv:
        parsed = kv
    return parsed


def _render_call_result(result: Any) -> str:
    """把 MCP 的 CallToolResult 渲染成纯文本给模型看。"""
    pieces: List[str] = []
    raw = getattr(result, "content", None)
    for block in raw or []:
        text = getattr(block, "text", None)
        if text is not None:
            pieces.append(text)
        elif isinstance(block, dict) and "text" in block:
            pieces.append(block["text"])
        else:
            pieces.append(str(block))
    if getattr(result, "isError", False):
        return f"工具返回错误：{chr(10).join(pieces) or '未知错误'}"
    structured = getattr(result, "structuredContent", None)
    if not pieces and structured:
        pieces.append(str(structured))
    return "\n".join(pieces) or "（工具没有返回内容）"
