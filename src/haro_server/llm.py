import datetime
import json
import logging
from typing import AsyncIterator, Protocol

import litellm
from litellm.types.utils import ChatCompletionMessageToolCall, Function

from . import db
from .assistant_tools import describe_now


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

La tua risposta viene letta ad alta voce da una sintesi vocale, non mostrata su uno schermo: scrivi solo frasi parlate, ognuna chiusa da punto, punto esclamativo o punto interrogativo. Non usare mai elenchi puntati o numerati, trattini a inizio riga, titoli, grassetto, virgolette, emoji o simboli. Se devi proporre più opzioni, dille in una o due frasi normali, per esempio: "Potresti guardare un film, uscire a fare una passeggiata oppure provare una ricetta nuova."

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


_ROUTINES_CONFIG_KEY = "routines"


def get_routines(db_path: str) -> str:
    return db.get_config_value(db_path, _ROUTINES_CONFIG_KEY) or ""


def set_routines(db_path: str, routines: str) -> None:
    db.set_config_value(db_path, _ROUTINES_CONFIG_KEY, routines)


def _build_effective_system_prompt(db_path: str, now: datetime.datetime | None = None) -> str:
    prompt = get_base_system_prompt(db_path)
    memories = db.get_memories(db_path)
    if memories:
        facts = "\n".join(f"- {m.fact}" for m in memories)
        prompt += f"\n\nCose che sai gia' sull'utente, da usare quando pertinenti (non elencarle a meno che non ti venga chiesto):\n{facts}"
    # Without the current date/time the model can't resolve "oggi",
    # "domani" or "svegliami alle 7" for its tools (assistant_tools.py).
    now = now or datetime.datetime.now(datetime.UTC)
    prompt += (
        f"\n\nAdesso e' {describe_now(now)} (ora italiana). Le citta' dell'utente sono Lucca e Pisa. "
        "Per meteo, impegni, sveglie e timer usa sempre gli strumenti a disposizione, non inventare."
    )
    routines = get_routines(db_path).strip()
    if routines:
        prompt += (
            "\n\nRoutine definite dall'utente: quando dice una di queste frasi (anche con parole simili), "
            f"esegui le istruzioni usando gli strumenti e rispondi in modo naturale:\n{routines}"
        )
    return prompt


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

    async def stream_reply(
        self, transcript: str, history: list[tuple[str, str]] | None = None
    ) -> AsyncIterator[str]:
        """`history`: the conversation so far, oldest first, as (user said,
        Haro replied) pairs -- so a follow-up like "sì, la seconda" has
        something to refer to. See session.py's Conversation."""
        messages = [{"role": "system", "content": _build_effective_system_prompt(self._db_path)}]
        for user_text, reply_text in history or []:
            messages.append({"role": "user", "content": user_text})
            messages.append({"role": "assistant", "content": reply_text})
        messages.append({"role": "user", "content": transcript})

        if not self._tools:
            response = await litellm.acompletion(model=self._model, messages=messages, stream=True)
            async for chunk in response:
                delta = chunk.choices[0].delta.content
                if delta:
                    yield delta
            return

        # Tools configured: ONE streamed call that can either answer or ask
        # for tools. Text deltas are yielded as they arrive (a plain reply
        # has the same time-to-first-audio as with no tools); tool-call
        # deltas (id/name once, JSON arguments in pieces) are collected.
        # Replaced a non-streamed "decision" call made before every turn,
        # which added a whole round trip to every reply once weather/
        # calendar/alarm tools became always-on (2026-09-24).
        response = await litellm.acompletion(
            model=self._model, messages=messages, tools=self._tools, tool_choice="auto", stream=True
        )
        pending: dict[int, dict[str, str]] = {}
        async for chunk in response:
            delta = chunk.choices[0].delta
            if delta.content:
                yield delta.content
            for fragment in getattr(delta, "tool_calls", None) or []:
                slot = pending.setdefault(fragment.index, {"id": "", "name": "", "arguments": ""})
                if fragment.id:
                    slot["id"] = fragment.id
                function = getattr(fragment, "function", None)
                if function is not None:
                    slot["name"] += function.name or ""
                    slot["arguments"] += function.arguments or ""
        if not pending:
            return

        tool_calls = [pending[i] for i in sorted(pending)]
        messages.append({
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {"id": c["id"], "type": "function", "function": {"name": c["name"], "arguments": c["arguments"] or "{}"}}
                for c in tool_calls
            ],
        })
        for c in tool_calls:
            # litellm's own type: MCP's call_openai_tool() reads it by
            # item (tool_call["function"]["name"]), local tools by attribute.
            tool_call = ChatCompletionMessageToolCall(
                id=c["id"], type="function", function=Function(name=c["name"], arguments=c["arguments"] or "{}")
            )
            try:
                result = await self._call_tool(tool_call)
            except Exception as exc:
                # A failing tool (weather API down, MCP server died, ...)
                # must not end the turn with no spoken reply: feed the
                # model an error string so it can say something sensible.
                logger.exception("tool call failed: %s", c["name"])
                result = f"Error calling tool: {exc}"
            messages.append({"role": "tool", "tool_call_id": c["id"], "content": result})

        # Follow-up call, no tools offered this time -- a plain streamed
        # reply built on the tool results.
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
