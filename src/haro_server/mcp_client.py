"""GitHub MCP client -- connects to GitHub's official MCP server
(github/github-mcp-server, run as `github-mcp-server stdio` with a
GITHUB_PERSONAL_ACCESS_TOKEN env var) over the stdio transport, and
exposes its tools in OpenAI tool-call format so they can be handed
straight to litellm's `tools=` parameter alongside this project's other
function-calling tools.

Verified against the actually-installed package versions before writing
this file (same practice as chatterbox_tts.py/pockettts_tts.py document
for their own libraries) -- `pip show mcp litellm` at the time this was
written: mcp==2.2.0, litellm==1.99.0. Real findings from
`help(mcp.ClientSession)`, `help(mcp.client.stdio)`,
`help(litellm.experimental_mcp_client.load_mcp_tools)` and
`help(litellm.experimental_mcp_client.call_openai_tool)`:

- `mcp.ClientSession` is NOT itself an "open a stdio connection" entry
  point -- it's constructed over an already-open (read_stream,
  write_stream) pair. The real stdio transport is
  `mcp.client.stdio.stdio_client` (also re-exported at the top level as
  `mcp.stdio_client`), an `@asynccontextmanager` function that takes an
  `mcp.StdioServerParameters(command, args, env)` and yields that stream
  pair. `stdio_client`'s own source confirms it merges the given `env`
  on top of `get_default_environment()` (so PATH etc. still reach the
  subprocess) rather than replacing the environment outright -- passing
  just GITHUB_PERSONAL_ACCESS_TOKEN, as the brief describes, is enough.
- Neither `stdio_client` nor `ClientSession` has a standalone
  `close()`/`open()` method (confirmed via `dir(mcp.ClientSession)`,
  which lists no `close`) -- both are async context managers only.
  Because this client needs the session to stay open across separate
  `connect()`/`call_tool()`/`close()` calls rather than for the
  lifetime of one `async with` block, `_open_stdio_session` enters both
  context managers manually through a `contextlib.AsyncExitStack` and
  stashes that stack as a plain attribute on the returned session
  (`session._haro_exit_stack`) so `close()` can unwind it later. This
  attribute is our own bookkeeping, not part of mcp's public API --
  `ClientSession` has no `__slots__`, so this is safe, but it's a detail
  of this file, not something documented by mcp itself.
- `ClientSession`'s own docstring says to "enter as an async context
  manager, then call initialize()" -- `_open_stdio_session` does exactly
  that; skipping `initialize()` leaves the session unable to answer
  other requests.
- Both `stdio_client` and `ClientSession.__aenter__` bind an anyio cancel
  scope to whichever asyncio task enters them (confirmed via
  `inspect.getsource`: `stdio_client`'s body runs inside
  `async with anyio.create_task_group() as tg:`, and
  `ClientSession.__aenter__` does
  `self._task_group = anyio.create_task_group(); await
  self._task_group.__aenter__()`). Consequently **`connect()` and
  `close()` must be awaited from the same asyncio task** -- exiting from
  a different task raises `RuntimeError: Attempted to exit cancel scope
  in a different task than it was entered in`. This is a real
  constraint on how a caller (e.g. Task 10's server-startup wiring) must
  use this class, not just a style note -- see the docstring on
  `McpToolClient.close()` below.
- `litellm.experimental_mcp_client.load_mcp_tools(session, format)` and
  `.call_openai_tool(session, openai_tool)` match the brief's guessed
  names and signatures exactly. `load_mcp_tools(session, "openai")`
  returns `list[ChatCompletionFunctionToolParam]` (plain dicts at
  runtime). `call_openai_tool` returns an `mcp.types.CallToolResult` --
  a pydantic model, *not* a string -- so `McpToolClient.call_tool`
  converts it via `.model_dump_json()`, which always yields a string
  regardless of the tool's content shape (text/image/embedded resource),
  ready to go straight into a `role: "tool"` message's `content` field.
"""

from contextlib import AsyncExitStack
from typing import Any

import litellm.experimental_mcp_client as emc
from mcp import ClientSession, StdioServerParameters, stdio_client


async def _open_stdio_session(command: str, args: list[str], env: dict[str, str]) -> ClientSession:
    """Spawn `command args...` (with `env` merged on top of the default
    environment -- see this module's docstring) as an MCP server
    subprocess, open a stdio-transport session to it, initialize it, and
    return the live session.

    Kept as its own module-level function (rather than inlined into
    McpToolClient.connect()) specifically so tests can monkeypatch it and
    avoid spawning a real subprocess -- see tests/test_mcp_client.py.
    """
    params = StdioServerParameters(command=command, args=args, env=env)
    # `async with AsyncExitStack() as exit_stack:` rather than a bare
    # `AsyncExitStack()` -- if `enter_async_context` (either call) or
    # `session.initialize()` below raises, this block's `__aexit__` runs
    # and unwinds whatever was already entered (killing the subprocess,
    # closing its pipes), instead of leaking them: AsyncExitStack does
    # NOT auto-unwind on its own just because it becomes unreachable: it
    # unwinds only when something calls `.aclose()` on it (or, here, its
    # own `async with` exits) -- reproduced experimentally: a bare
    # `AsyncExitStack()` with no `async with`/`aclose()` around a raising
    # `enter_async_context` call left the spawned subprocess running.
    async with AsyncExitStack() as exit_stack:
        read_stream, write_stream = await exit_stack.enter_async_context(stdio_client(params))
        session = await exit_stack.enter_async_context(ClientSession(read_stream, write_stream))
        await session.initialize()
        # Success path: `pop_all()` moves ownership of the already-entered
        # context managers to a fresh stack *without* closing them, so
        # they stay open past this function's return (needed since this
        # session must outlive `_open_stdio_session` itself, across
        # separate connect()/call_tool()/close() calls) -- while the
        # `async with` above still auto-unwinds everything on any
        # exception raised before reaching this line.
        session._haro_exit_stack = exit_stack.pop_all()
        return session


async def _load_mcp_tools(session: ClientSession, format: str) -> list[dict]:
    """Thin wrapper around litellm.experimental_mcp_client.load_mcp_tools
    -- its own module-level function purely so tests can monkeypatch it
    without a real MCP session (see tests/test_mcp_client.py)."""
    return await emc.load_mcp_tools(session, format)


class McpToolClient:
    """One connection to GitHub's official MCP server, exposing its tools
    in OpenAI tool-call format (`self.tools`) and a way to invoke them
    (`call_tool`) for use in this project's LLM tool-calling loop.
    """

    def __init__(self, github_token: str) -> None:
        self._github_token = github_token
        self._session: ClientSession | None = None
        self.tools: list[dict] = []

    async def connect(self) -> None:
        self._session = await _open_stdio_session(
            "github-mcp-server",
            ["stdio"],
            {"GITHUB_PERSONAL_ACCESS_TOKEN": self._github_token},
        )
        self.tools = await _load_mcp_tools(self._session, "openai")

    async def call_tool(self, tool_call: Any) -> str:
        result = await emc.call_openai_tool(self._session, tool_call)
        # result is an mcp.types.CallToolResult (pydantic model), not a
        # string -- see module docstring. model_dump_json() always
        # produces a string, regardless of the tool's content shape, fit
        # to go straight into a role: "tool" message's content field.
        return result.model_dump_json()

    async def close(self) -> None:
        """Close the stdio session/subprocess opened by connect().

        MUST be awaited from the same asyncio task that awaited connect().
        Both `stdio_client` and `ClientSession` bind an anyio cancel scope
        to whichever task enters them (see module docstring); unwinding
        that scope from a different task raises `RuntimeError: Attempted
        to exit cancel scope in a different task than it was entered in`.
        This is a real constraint on any caller of this class (e.g. Task
        10's server-startup wiring must call connect() and close() from
        the same task/coroutine, not e.g. close() from a signal handler
        or a separately-scheduled task), not just an implementation
        detail of this method.
        """
        if self._session is None:
            return
        exit_stack = getattr(self._session, "_haro_exit_stack", None)
        self._session = None
        if exit_stack is not None:
            await exit_stack.aclose()
