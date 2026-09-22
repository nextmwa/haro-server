from unittest.mock import AsyncMock, patch

from haro_server import db
from haro_server.llm import DEFAULT_SYSTEM_PROMPT, LiteLlmClient


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


class FakeMessage:
    def __init__(self, content: str) -> None:
        self.content = content
        self.tool_calls = None


class FakeCompletionChoice:
    def __init__(self, content: str) -> None:
        self.message = FakeMessage(content)


class FakeCompletion:
    def __init__(self, content: str) -> None:
        self.choices = [FakeCompletionChoice(content)]


class FakeToolCall:
    def __init__(self, id: str, name: str, arguments: str = "{}") -> None:
        self.id = id
        self.function = FakeToolCallFunction(name, arguments)


class FakeToolCallFunction:
    def __init__(self, name: str, arguments: str) -> None:
        self.name = name
        self.arguments = arguments


class FakeToolCallMessage:
    """Stands in for litellm's assistant message when the model decided
    to call a tool instead of replying directly -- .content is None (no
    text yet) and .tool_calls carries what to call. model_dump() only
    needs to round-trip through stream_reply()'s own messages.append()
    call, not match litellm's real serialization exactly.
    """

    def __init__(self, tool_calls: list[FakeToolCall]) -> None:
        self.content = None
        self.tool_calls = tool_calls

    def model_dump(self):
        return {"role": "assistant", "tool_calls": self.tool_calls}


def _client(tmp_path, model="claude-sonnet-5") -> LiteLlmClient:
    db_path = str(tmp_path / "haro.db")
    db.init_db(db_path)
    return LiteLlmClient(model=model, db_path=db_path)


async def test_stream_reply_yields_non_empty_deltas_in_order(tmp_path):
    client = _client(tmp_path)

    with patch("haro_server.llm.litellm.acompletion", new=AsyncMock()) as mock_acompletion:
        mock_acompletion.return_value = _fake_stream(["Ciao", None, "!", ""])

        collected = [chunk async for chunk in client.stream_reply("ciao")]

    assert collected == ["Ciao", "!"]


async def test_stream_reply_passes_model_and_transcript(tmp_path):
    client = _client(tmp_path, model="gpt-5.1")

    with patch("haro_server.llm.litellm.acompletion", new=AsyncMock()) as mock_acompletion:
        mock_acompletion.return_value = _fake_stream(["ok"])

        [_ async for _ in client.stream_reply("che ore sono")]

    call_kwargs = mock_acompletion.call_args.kwargs
    assert call_kwargs["model"] == "gpt-5.1"
    assert call_kwargs["stream"] is True
    messages = call_kwargs["messages"]
    assert messages[-1] == {"role": "user", "content": "che ore sono"}
    assert messages[0]["role"] == "system"


async def test_stream_reply_uses_default_system_prompt_with_no_override_or_memories(tmp_path):
    client = _client(tmp_path)

    with patch("haro_server.llm.litellm.acompletion", new=AsyncMock()) as mock_acompletion:
        mock_acompletion.return_value = _fake_stream(["ok"])
        [_ async for _ in client.stream_reply("ciao")]

    system_content = mock_acompletion.call_args.kwargs["messages"][0]["content"]
    assert system_content == DEFAULT_SYSTEM_PROMPT


async def test_stream_reply_uses_admin_overridden_prompt(tmp_path):
    db_path = str(tmp_path / "haro.db")
    db.init_db(db_path)
    db.set_config_value(db_path, "system_prompt", "Sei un pirata.")
    client = LiteLlmClient(model="claude-sonnet-5", db_path=db_path)

    with patch("haro_server.llm.litellm.acompletion", new=AsyncMock()) as mock_acompletion:
        mock_acompletion.return_value = _fake_stream(["ok"])
        [_ async for _ in client.stream_reply("ciao")]

    system_content = mock_acompletion.call_args.kwargs["messages"][0]["content"]
    assert system_content == "Sei un pirata."


async def test_stream_reply_injects_stored_memories_into_system_prompt(tmp_path):
    db_path = str(tmp_path / "haro.db")
    db.init_db(db_path)
    db.add_memory(db_path, "il compleanno di Leonardo e' il 5 maggio", None)
    client = LiteLlmClient(model="claude-sonnet-5", db_path=db_path)

    with patch("haro_server.llm.litellm.acompletion", new=AsyncMock()) as mock_acompletion:
        mock_acompletion.return_value = _fake_stream(["ok"])
        [_ async for _ in client.stream_reply("ciao")]

    system_content = mock_acompletion.call_args.kwargs["messages"][0]["content"]
    assert "il compleanno di Leonardo e' il 5 maggio" in system_content
    assert system_content.startswith(DEFAULT_SYSTEM_PROMPT)


async def test_extract_facts_parses_json_facts_from_the_reply(tmp_path):
    client = _client(tmp_path)

    with patch("haro_server.llm.litellm.acompletion", new=AsyncMock()) as mock_acompletion:
        mock_acompletion.return_value = FakeCompletion('{"facts": ["ama la pizza margherita"]}')

        facts = await client.extract_facts("mi piace la pizza margherita", "Bello saperlo!")

    assert facts == ["ama la pizza margherita"]
    assert mock_acompletion.call_args.kwargs["stream"] is False


async def test_extract_facts_returns_empty_list_when_nothing_notable(tmp_path):
    client = _client(tmp_path)

    with patch("haro_server.llm.litellm.acompletion", new=AsyncMock()) as mock_acompletion:
        mock_acompletion.return_value = FakeCompletion('{"facts": []}')

        facts = await client.extract_facts("che ore sono", "Sono le tre.")

    assert facts == []


async def test_extract_facts_strips_markdown_code_fence(tmp_path):
    client = _client(tmp_path)

    with patch("haro_server.llm.litellm.acompletion", new=AsyncMock()) as mock_acompletion:
        mock_acompletion.return_value = FakeCompletion('```json\n{"facts": ["test"]}\n```')

        facts = await client.extract_facts("transcript", "reply")

    assert facts == ["test"]


async def test_extract_facts_returns_empty_list_on_malformed_json(tmp_path):
    client = _client(tmp_path)

    with patch("haro_server.llm.litellm.acompletion", new=AsyncMock()) as mock_acompletion:
        mock_acompletion.return_value = FakeCompletion("non sono JSON valido")

        facts = await client.extract_facts("transcript", "reply")

    assert facts == []


async def test_extract_facts_returns_empty_list_when_acompletion_raises(tmp_path):
    client = _client(tmp_path)

    with patch("haro_server.llm.litellm.acompletion", new=AsyncMock()) as mock_acompletion:
        mock_acompletion.side_effect = RuntimeError("provider down")

        facts = await client.extract_facts("transcript", "reply")

    assert facts == []


class FakeTrack:
    def __init__(self, title: str, artist: str, album: str) -> None:
        self.title = title
        self.artist = artist
        self.album = album


async def test_extract_music_query_returns_stripped_model_output(tmp_path):
    client = _client(tmp_path)

    with patch("haro_server.llm.litellm.acompletion", new=AsyncMock()) as mock_acompletion:
        mock_acompletion.return_value = FakeCompletion("  Vasco Rossi  ")

        query = await client.extract_music_query("metti della musica di Vasco Rossi")

    assert query == "Vasco Rossi"
    assert mock_acompletion.call_args.kwargs["stream"] is False


async def test_pick_best_track_returns_none_for_empty_candidates(tmp_path):
    client = _client(tmp_path)

    result = await client.pick_best_track("metti qualcosa", [])

    assert result is None


async def test_pick_best_track_returns_the_indexed_candidate(tmp_path):
    client = _client(tmp_path)
    candidates = [
        FakeTrack("Vita Spericolata", "Vasco Rossi", "Bollicine"),
        FakeTrack("Albachiara", "Vasco Rossi", "Non siamo mica..."),
    ]

    with patch("haro_server.llm.litellm.acompletion", new=AsyncMock()) as mock_acompletion:
        mock_acompletion.return_value = FakeCompletion("1")

        picked = await client.pick_best_track("metti Albachiara", candidates)

    assert picked is candidates[1]


async def test_pick_best_track_falls_back_to_first_candidate_on_malformed_response(tmp_path):
    client = _client(tmp_path)
    candidates = [FakeTrack("Song A", "Artist A", "Album A")]

    with patch("haro_server.llm.litellm.acompletion", new=AsyncMock()) as mock_acompletion:
        mock_acompletion.return_value = FakeCompletion("non e' un numero")

        picked = await client.pick_best_track("qualcosa", candidates)

    assert picked is candidates[0]


async def test_pick_best_track_returns_none_when_model_returns_negative_index(tmp_path):
    client = _client(tmp_path)
    candidates = [FakeTrack("Song A", "Artist A", "Album A")]

    with patch("haro_server.llm.litellm.acompletion", new=AsyncMock()) as mock_acompletion:
        mock_acompletion.return_value = FakeCompletion("-1")

        picked = await client.pick_best_track("qualcosa di completamente diverso", candidates)

    assert picked is None


async def test_stream_reply_without_tools_is_unchanged(tmp_path):
    # Existing behavior: no tools configured -> the original single
    # streamed call, no tool-calling machinery involved at all.
    client = _client(tmp_path)

    with patch("haro_server.llm.litellm.acompletion", new=AsyncMock()) as mock_acompletion:
        mock_acompletion.return_value = _fake_stream(["[emotion:neutral] ciao"])
        chunks = [c async for c in client.stream_reply("ciao")]

    assert "".join(chunks) == "[emotion:neutral] ciao"
    assert mock_acompletion.call_count == 1
    assert "tools" not in mock_acompletion.call_args.kwargs


async def test_stream_reply_calls_a_tool_when_the_model_requests_one(tmp_path):
    fake_tool_call = FakeToolCall(id="call_1", name="list_pull_requests")
    decision_response = FakeCompletion("")
    decision_response.choices[0].message = FakeToolCallMessage([fake_tool_call])

    async def fake_call_tool(tool_call):
        assert tool_call is fake_tool_call
        return "2 open pull requests"

    client = _client(tmp_path)
    client.set_tools(
        tools=[{"type": "function", "function": {"name": "list_pull_requests"}}],
        call_tool=fake_call_tool,
    )

    with patch("haro_server.llm.litellm.acompletion", new=AsyncMock()) as mock_acompletion:
        mock_acompletion.side_effect = [
            decision_response,
            _fake_stream(["[emotion:neutral] hai 2 PR aperte"]),
        ]
        chunks = [c async for c in client.stream_reply("ho PR aperte?")]

    assert "".join(chunks) == "[emotion:neutral] hai 2 PR aperte"
    assert mock_acompletion.call_count == 2  # one non-streamed decision call, one streamed follow-up
    first_call, second_call = mock_acompletion.call_args_list
    assert first_call.kwargs["tools"] == client._tools
    assert first_call.kwargs["stream"] is False
    assert second_call.kwargs["stream"] is True
    assert "tool_call_id" in str(second_call.kwargs["messages"])  # the tool result was folded in
