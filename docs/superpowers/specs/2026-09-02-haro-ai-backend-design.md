# Haro AI Backend — Design

Date: 2026-09-02
Status: Approved for implementation planning

## Purpose

This is the AI server for **Haro**, a personal desk robot (separate project,
already built, at `../haro`). The robot streams the user's speech to this
server over a WebSocket, and this server turns that into a natural spoken
conversation: it transcribes the speech, recalls relevant context from past
conversations, asks a large language model (Claude, OpenAI, or Gemini —
whichever is configured) for a reply, and streams back synthesized speech
plus an emotion tag that drives the robot's face.

This spec covers the **server side only**. The robot side (wake-word
detection, audio capture/playback, the OLED face, WiFi provisioning) is a
separate, already-implemented project at `../haro`, whose WebSocket protocol
this server must implement exactly — that protocol is a fixed external
contract, not something this project gets to redesign.

## The protocol contract (already fixed by the robot side)

Reference: `../haro/src/haro/protocol.py` and
`../haro/docs/superpowers/specs/2026-09-02-haro-robot-design.md`.

One persistent WebSocket connection per robot session.

- **Client (robot) → Server, JSON text frames:**
  - `{"type": "hello", "session_id": "<string>"}` — sent once, right after connecting.
  - Binary frames — raw 16-bit signed PCM, mono, **16kHz**, sent continuously while the robot is in its `listening` state.
  - `{"type": "end_of_speech"}` — sent once the robot's own VAD detects the user stopped talking.
- **Server → Client, JSON text frames:**
  - `{"type": "emotion", "value": "happy" | "sad" | "confused" | "neutral"}` — sent once per turn, before or alongside the first audio chunk.
  - Binary frames — raw 16-bit signed PCM, mono, **24kHz** (pinned by this spec; matches the robot's `AudioOutput` default `speaker_sample_rate`), the synthesized reply, streamed as it's generated.
  - `{"type": "response_end"}` — sent once the reply is fully sent.
  - `{"type": "error", "message": "<string>"}` — sent on a server-side failure for this turn; the robot shows an error face and the connection is expected to be reconnected by the robot's own retry logic (already implemented robot-side — this server does not need to implement reconnect logic itself, only respond correctly to a fresh `hello` if the robot reconnects).

This server owns nothing about wake-word detection, VAD, or audio playback —
it only ever sees a `listening` window bounded by `hello`-then-frames and
`end_of_speech`, once per turn.

## Architecture

A single Python 3 asyncio service (FastAPI + Uvicorn, with `uvloop` as the
event loop) exposing one WebSocket endpoint that speaks the protocol above.
Each connection runs a per-session turn loop that pipes audio through four
independent, swappable subsystems:

```
robot audio frames ──► STT (streaming) ──► memory lookup ──► LLM (streaming) ──► TTS (streaming) ──► robot audio frames
                                                 ▲                                        │
                                                 └──────────── memory write ───────────────┘
```

## Components

1. **WebSocket handler** — implements the protocol above; one handler
   instance per connection; owns the per-turn state machine
   (`idle → listening → thinking → speaking → idle`, mirroring the robot's
   own state machine so the two sides reason about turns the same way).
2. **STT — streaming transcription.** Wraps NVIDIA Parakeet TDT v3 (0.6B),
   fed audio incrementally as frames arrive (not batched at
   `end_of_speech`), producing a running partial transcript; the transcript
   is finalized when `end_of_speech` arrives. Chosen for: native streaming
   architecture (RNN-Transducer, not a batched model repurposed for
   streaming), Italian + 24 other European languages, CC-BY licensed, faster
   than Whisper Large v3 even on CPU.
3. **Memory — Graphiti + Neo4j.** Before calling the LLM, queries Graphiti
   for context relevant to the new transcript (hybrid vector + full-text +
   graph traversal over a temporal knowledge graph keyed by `session_id`).
   After the turn completes, writes the new exchange (user transcript +
   assistant reply) back into the graph. Apache 2.0, self-hosted, persists
   across robot restarts/disconnects (a `session_id` is a durable identity
   in the graph, not an in-memory-only key).
4. **LLM — LiteLLM (as a Python library, not the hosted gateway).**
   Routes a single OpenAI-style chat completion call to Claude, OpenAI, or
   Gemini depending on which API key(s) are configured and which model
   string is set as the default. The system prompt instructs the model to
   prefix its reply with a one-line emotion tag in a fixed format:
   `[emotion:<happy|sad|confused|neutral>] <reply text>`. The server strips
   that prefix as soon as it appears in the streamed output and forwards
   the `emotion` message immediately, then keeps streaming the remaining
   text into TTS as it arrives — so TTS starts before the LLM has finished
   generating the full reply.
5. **TTS — streaming synthesis.** Wraps Kokoro-82M, fed the LLM's streamed
   text (after the emotion prefix is stripped) and producing 24kHz PCM audio
   chunks as they're synthesized, forwarded to the robot as binary WebSocket
   frames immediately (not buffered until the full reply is synthesized).
   Apache 2.0, sub-1s latency on GPU / 2-3s on CPU.

## Configuration

Environment variables (loaded via `.env` in development, real environment
variables in the Docker deployment — never committed):

- `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `GEMINI_API_KEY` — set whichever
  ones you have; only the one matching `DEFAULT_MODEL` needs to be present.
- `DEFAULT_MODEL` — the LiteLLM model string to use for conversations (e.g.
  `claude-sonnet-5`, `gpt-5.1`, `gemini-2.5-pro`). Selects both the model
  and, implicitly via LiteLLM's routing, which API key is used.
- `NEO4J_URI`, `NEO4J_USER`, `NEO4J_PASSWORD` — connection to the Graphiti
  backing store (defaults match the bundled `docker-compose.yml` service).
- `STT_DEVICE`, `TTS_DEVICE` — `cuda` or `cpu` (default `cpu`; set to `cuda`
  when running on GPU-equipped hardware).
- `HOST`, `PORT` — WebSocket server bind address (default `0.0.0.0:8765`,
  matching the robot's `Config.server_url` default of `ws://localhost:8765`
  during local development).

## Docker

`docker-compose.yml` with two services:

- `app` — the FastAPI server, built from a `Dockerfile` in this repo.
- `neo4j` — official Neo4j image, Graphiti's backing store, with a named
  volume for persistence across container restarts.

A GPU-enabled override (`docker-compose.gpu.yml`, merged in via `-f`) adds
NVIDIA runtime access to the `app` service for STT/TTS acceleration; the
base compose file runs everything on CPU with no special host requirements.

## Error handling

- **LLM provider unreachable / API key invalid/missing:** send
  `{"type": "error", "message": "..."}` to the robot for that turn; the
  robot's own error handling (already implemented) shows an error face and
  reconnects. This server does not retry the LLM call itself beyond
  whatever built-in retry LiteLLM already does for transient failures.
- **STT or TTS model fails to load or errors mid-turn:** same — send
  `error`, let the turn fail cleanly rather than hang.
- **GPU unavailable when `STT_DEVICE`/`TTS_DEVICE=cuda`:** fail fast at
  startup with a clear log message, rather than silently falling back (a
  silent CPU fallback could turn a expected-fast deployment into an
  unexpectedly slow one without anyone noticing) — falling back to CPU is a
  deliberate config change (`STT_DEVICE=cpu`), not automatic behavior.
- **Neo4j unreachable at startup or during a turn:** log the error and
  continue the turn **without memory** (no context retrieved, no write
  performed) rather than failing the whole turn — a personal assistant that
  temporarily forgets is better than one that stops responding entirely.

## Testing

Same philosophy as the robot project: every external boundary (STT engine,
TTS engine, LiteLLM call, Graphiti client) is wrapped behind a small
interface that can be swapped for a fake in tests, so the WebSocket protocol
handling and turn state machine are fully unit-testable without loading any
real model or hitting any real API. Real-model integration (actual STT/TTS
quality, actual LLM responses, actual Neo4j queries) is verified manually,
since it requires downloaded model weights and real API keys that aren't
available in an automated test run.

## Out of scope

- The robot side of the protocol (already built, at `../haro`).
- Any user-facing text/chat interface to this server — it only ever speaks
  the robot's WebSocket protocol.
- Authentication/authorization on the WebSocket endpoint (this is a personal
  project; the server is assumed to run on a private network the robot
  connects to — no internet-exposed deployment is in scope here).
- Multi-tenant support (multiple distinct end users with separate accounts)
  — `session_id` provides per-conversation memory scoping, not user
  authentication.
