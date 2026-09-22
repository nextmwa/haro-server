from unittest.mock import AsyncMock, MagicMock

from haro_server.mcp_client import McpToolClient


async def test_connect_populates_tools_from_the_mcp_session(monkeypatch):
    fake_tools = [{"type": "function", "function": {"name": "list_pull_requests", "parameters": {}}}]

    async def fake_load_mcp_tools(session, format):
        assert format == "openai"
        return fake_tools

    monkeypatch.setattr("haro_server.mcp_client._load_mcp_tools", fake_load_mcp_tools)
    monkeypatch.setattr("haro_server.mcp_client._open_stdio_session", AsyncMock(return_value=MagicMock()))

    client = McpToolClient(github_token="fake-token")
    await client.connect()

    assert client.tools == fake_tools
