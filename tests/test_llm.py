from unittest.mock import AsyncMock, patch

from haro_server.llm import LiteLlmClient


class FakeDelta:
    def __init__(self, content: str | None) -> None:
        self.content = content


class FakeChoice:
    def __init__(self, content: str | None) -> None:
        self.delta = FakeDelta(content)


class FakeChunk:
    def __init__(self, content: str | None) -> None:
        self.choices = [FakeChoice(content)]


async def _fake_stream(chunks):
    for content in chunks:
        yield FakeChunk(content)


async def test_stream_reply_yields_non_empty_deltas_in_order():
    client = LiteLlmClient(model="claude-sonnet-5")

    with patch("haro_server.llm.litellm.acompletion", new=AsyncMock()) as mock_acompletion:
        mock_acompletion.return_value = _fake_stream(["Ciao", None, "!", ""])

        collected = [chunk async for chunk in client.stream_reply("ciao")]

    assert collected == ["Ciao", "!"]


async def test_stream_reply_passes_model_and_transcript():
    client = LiteLlmClient(model="gpt-5.1")

    with patch("haro_server.llm.litellm.acompletion", new=AsyncMock()) as mock_acompletion:
        mock_acompletion.return_value = _fake_stream(["ok"])

        [_ async for _ in client.stream_reply("che ore sono")]

    call_kwargs = mock_acompletion.call_args.kwargs
    assert call_kwargs["model"] == "gpt-5.1"
    assert call_kwargs["stream"] is True
    messages = call_kwargs["messages"]
    assert messages[-1] == {"role": "user", "content": "che ore sono"}
    assert messages[0]["role"] == "system"
