import json
import logging
from typing import AsyncIterator, Protocol

import litellm

from . import db


class MusicCandidateLike(Protocol):
    """Duck-typed to avoid llm.py importing navidrome.py directly -- the
    real argument at the call site (session.py) is a list of
    navidrome.Track, matched structurally here the same way session.py's
    own SttEngineLike/LlmClientLike/TtsEngineLike Protocols avoid a hard
    dependency on the concrete engine classes.
    """

    title: str
    artist: str
    album: str

logger = logging.getLogger(__name__)

DEFAULT_SYSTEM_PROMPT = """Sei Haro, un assistente vocale amichevole su un piccolo robot da scrivania. Rispondi in modo naturale e conciso, adatto a una conversazione vocale (frasi brevi, colloquiali, senza formattazione markdown).

Inizia SEMPRE la tua risposta con un tag di emozione tra parentesi quadre, scegliendo esattamente uno tra: happy, sad, confused, neutral. Formato esatto, senza eccezioni: [emotion:happy] seguito dal testo della risposta.

Esempio: [emotion:happy] Certo, posso aiutarti con quello!"""

# Editable via the /admin web UI (admin.py) -- stored in the same db.py
# config table set_base_system_prompt()/get_base_system_prompt() use, so an
# edit takes effect on the very next turn with no server restart.
_SYSTEM_PROMPT_CONFIG_KEY = "system_prompt"

FACT_EXTRACTION_PROMPT = """Analizza il seguente scambio tra un utente e un assistente vocale e individua eventuali fatti degni di nota da ricordare a lungo termine sull'utente (preferenze, eventi, informazioni personali, richieste ricorrenti). Se non c'e' nulla di rilevante, restituisci una lista vuota.

Rispondi SOLO con un oggetto JSON nel formato esatto: {"facts": ["fatto 1", "fatto 2"]} oppure {"facts": []}. Nessun altro testo, nessun blocco di codice markdown."""

MUSIC_QUERY_EXTRACTION_PROMPT = """Estrai SOLO il termine di ricerca musicale (nome di un artista, brano o album) dalla richiesta vocale dell'utente. Rispondi con il solo termine di ricerca, senza altro testo, senza virgolette, senza spiegazioni."""

PICK_TRACK_PROMPT = """L'utente ha chiesto di ascoltare musica. Di seguito la sua richiesta originale e un elenco numerato di brani trovati in libreria. Scegli l'indice del brano che meglio corrisponde alla richiesta.

Rispondi SOLO con il numero dell'indice (es. "0"), senza altro testo. Se nessun risultato corrisponde ragionevolmente alla richiesta, rispondi con "-1"."""


def get_base_system_prompt(db_path: str) -> str:
    return db.get_config_value(db_path, _SYSTEM_PROMPT_CONFIG_KEY) or DEFAULT_SYSTEM_PROMPT


def set_base_system_prompt(db_path: str, prompt: str) -> None:
    db.set_config_value(db_path, _SYSTEM_PROMPT_CONFIG_KEY, prompt)


def _build_effective_system_prompt(db_path: str) -> str:
    base = get_base_system_prompt(db_path)
    memories = db.get_memories(db_path)
    if not memories:
        return base
    facts = "\n".join(f"- {m.fact}" for m in memories)
    return f"{base}\n\nCose che sai gia' sull'utente, da usare quando pertinenti (non elencarle a meno che non ti venga chiesto):\n{facts}"


def _strip_code_fence(text: str) -> str:
    # LLMs asked for "JSON only" still sometimes wrap it in a ```json
    # fence -- strip one if present before parsing, rather than failing
    # extraction on a formatting quirk.
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else ""
        if text.endswith("```"):
            text = text[:-3]
    return text.strip()


class LiteLlmClient:
    def __init__(
        self,
        model: str,
        db_path: str,
        tools: list[dict] | None = None,
        call_tool=None,
    ) -> None:
        self._model = model
        self._db_path = db_path
        self._tools = tools
        self._call_tool = call_tool

    def set_tools(self, tools: list[dict], call_tool) -> None:
        """Attaches MCP tools after construction -- server.py's startup
        event (Task 10) connects to the MCP server asynchronously, which
        can only happen once uvicorn's event loop is running, by which
        point this object already exists. The constructor's tools/
        call_tool parameters above stay for tests and any future case
        that has tools available up front.
        """
        self._tools = tools
        self._call_tool = call_tool

    async def stream_reply(self, transcript: str) -> AsyncIterator[str]:
        messages = [
            {"role": "system", "content": _build_effective_system_prompt(self._db_path)},
            {"role": "user", "content": transcript},
        ]

        if not self._tools:
            response = await litellm.acompletion(model=self._model, messages=messages, stream=True)
            async for chunk in response:
                delta = chunk.choices[0].delta.content
                if delta:
                    yield delta
            return

        # Tools configured: decide tool use via a non-streamed call first
        # -- see this task's Context note for why streaming isn't used
        # for this decision step.
        decision = await litellm.acompletion(
            model=self._model, messages=messages, tools=self._tools, tool_choice="auto", stream=False
        )
        message = decision.choices[0].message
        if not message.tool_calls:
            if message.content:
                yield message.content
            return

        messages.append(message.model_dump())
        for tool_call in message.tool_calls:
            result = await self._call_tool(tool_call)
            messages.append({"role": "tool", "tool_call_id": tool_call.id, "content": result})

        # Follow-up call, no tools offered this time -- a plain streamed
        # reply, identical in shape to the no-tools-configured path above.
        response = await litellm.acompletion(model=self._model, messages=messages, stream=True)
        async for chunk in response:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta

    async def extract_facts(self, transcript: str, reply: str) -> list[str]:
        """Best-effort background extraction of long-term-worthy facts from
        one turn. Never raises: this runs detached from the voice turn
        (session.py fires it after response_end, not awaited inline, so
        extraction latency/failures never delay or break the spoken reply)
        -- a malformed/non-JSON model response just yields no facts.
        """
        messages = [
            {"role": "system", "content": FACT_EXTRACTION_PROMPT},
            {"role": "user", "content": f"Utente: {transcript}\nAssistente: {reply}"},
        ]
        try:
            response = await litellm.acompletion(model=self._model, messages=messages, stream=False)
            content = response.choices[0].message.content or "{}"
            data = json.loads(_strip_code_fence(content))
            facts = data.get("facts", [])
            return [f for f in facts if isinstance(f, str) and f.strip()]
        except Exception:
            logger.exception("fact extraction failed, skipping")
            return []

    async def extract_music_query(self, transcript: str) -> str:
        messages = [
            {"role": "system", "content": MUSIC_QUERY_EXTRACTION_PROMPT},
            {"role": "user", "content": transcript},
        ]
        response = await litellm.acompletion(model=self._model, messages=messages, stream=False)
        return (response.choices[0].message.content or "").strip()

    async def pick_best_track(
        self, transcript: str, candidates: list[MusicCandidateLike]
    ) -> MusicCandidateLike | None:
        """Never raises: falls back to the first (Navidrome's own top-
        relevance) candidate if the model's answer isn't a parseable index,
        rather than failing music playback outright over a formatting
        quirk in the disambiguation call.
        """
        if not candidates:
            return None
        listing = "\n".join(
            f"{i}: {c.title} - {c.artist} ({c.album})" for i, c in enumerate(candidates)
        )
        messages = [
            {"role": "system", "content": PICK_TRACK_PROMPT},
            {"role": "user", "content": f"Richiesta: {transcript}\n\nRisultati:\n{listing}"},
        ]
        try:
            response = await litellm.acompletion(model=self._model, messages=messages, stream=False)
            content = _strip_code_fence(response.choices[0].message.content or "")
            index = int(content)
            if index < 0:
                return None
            if index < len(candidates):
                return candidates[index]
        except Exception:
            logger.exception("pick_best_track failed, falling back to the first result")
        return candidates[0]
