# src/haro_server/admin.py
"""Web UI for editing the system prompt and browsing transcripts/long-term
memory (db.py). Protected by HTTP Basic Auth (a single shared password,
config.py's admin_password) -- see create_admin_router()'s docstring for
why server.py refuses to mount this at all when no password is configured.
"""
import secrets
from html import escape

from fastapi import APIRouter, Depends, Form, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse, Response
from fastapi.security import HTTPBasic, HTTPBasicCredentials

from . import camera_preview, db, llm, turn_audio
from .assistant_tools import TIMEZONE


def create_admin_router(db_path: str, admin_password: str) -> APIRouter:
    """`admin_password` must be a real, non-empty string -- server.py only
    calls this when config.admin_password is set, so there is no "admin UI
    with no password" code path here to get wrong.
    """
    router = APIRouter(prefix="/admin")
    security = HTTPBasic()

    def require_auth(credentials: HTTPBasicCredentials = Depends(security)) -> None:
        # secrets.compare_digest for a timing-safe comparison -- a plain
        # `==` leaks the password's length/prefix via response-time
        # differences to anyone who can send requests to this port.
        # Username is accepted as-is: this is single-shared-password auth,
        # not a multi-user account system.
        if not secrets.compare_digest(credentials.password, admin_password):
            raise HTTPException(status_code=401, detail="Unauthorized", headers={"WWW-Authenticate": "Basic"})

    @router.get("", response_class=HTMLResponse)
    async def admin_page(_: None = Depends(require_auth)) -> str:
        prompt = llm.get_base_system_prompt(db_path)
        transcripts = db.get_transcripts(db_path, limit=50)
        memories = db.get_memories(db_path)
        with_audio = {t.id for t in transcripts if turn_audio.path_for(db_path, t.id) is not None}
        return _render_page(
            prompt, transcripts, memories, with_audio, llm.get_routines(db_path), db.get_pending_alarms(db_path)
        )

    @router.post("/prompt")
    async def save_prompt(prompt: str = Form(...), _: None = Depends(require_auth)) -> RedirectResponse:
        llm.set_base_system_prompt(db_path, prompt)
        return RedirectResponse(url="/admin", status_code=303)

    @router.post("/routines")
    async def save_routines(routines: str = Form(""), _: None = Depends(require_auth)) -> RedirectResponse:
        llm.set_routines(db_path, routines)
        return RedirectResponse(url="/admin", status_code=303)

    @router.post("/alarms/{alarm_id}/cancel")
    async def cancel_alarm(alarm_id: int, _: None = Depends(require_auth)) -> RedirectResponse:
        db.set_alarm_status(db_path, alarm_id, "cancelled")
        return RedirectResponse(url="/admin", status_code=303)

    @router.post("/memories/{memory_id}/delete")
    async def delete_memory_route(memory_id: int, _: None = Depends(require_auth)) -> RedirectResponse:
        db.delete_memory(db_path, memory_id)
        return RedirectResponse(url="/admin", status_code=303)

    @router.get("/camera", response_class=HTMLResponse)
    async def camera_page(_: None = Depends(require_auth)) -> str:
        return _CAMERA_PAGE

    @router.get("/audio/{transcript_id}.wav")
    async def turn_audio_wav(transcript_id: int, _: None = Depends(require_auth)) -> FileResponse:
        path = turn_audio.path_for(db_path, transcript_id)
        if path is None:
            raise HTTPException(status_code=404, detail="no audio for this turn")
        return FileResponse(path, media_type="audio/wav")

    @router.get("/camera.jpg")
    async def camera_jpg(_: None = Depends(require_auth)) -> Response:
        jpeg = camera_preview.get_latest_frame()
        if jpeg is None:
            # 204, not 404: "the robot hasn't sent a frame yet" isn't a
            # missing resource, it's an expected transient state right
            # after startup or with the camera not wired up.
            return Response(status_code=204)
        # X-Face-Box travels with the JPEG it was detected in, so the
        # preview page's overlay can never pair a box with a different
        # frame (both are set together by camera_preview.set_latest_frame(),
        # on the same event loop as this handler). "x1,y1,x2,y2" in the
        # frame's own pixels, or "none".
        box = camera_preview.get_latest_face_box()
        return Response(
            content=jpeg,
            media_type="image/jpeg",
            headers={
                "Cache-Control": "no-store",
                "X-Face-Box": ",".join(str(v) for v in box) if box is not None else "none",
            },
        )

    return router


# Live preview: the robot sends ~5 frames/s, so instead of a whole-page
# meta-refresh (flickers, and at 2s showed one frame in ten) a small script
# pulls /admin/camera.jpg back-to-back and draws X-Face-Box over it. The
# box is positioned in percentages of the image's natural size, so it stays
# aligned at any display size. Same-origin fetch reuses the page's Basic
# Auth credentials.
_CAMERA_PAGE = """<!doctype html>
<html lang="it">
<head>
<meta charset="utf-8">
<title>Haro - Camera</title>
<style>
  body { font-family: -apple-system, system-ui, sans-serif; max-width: 900px; margin: 2rem auto; padding: 0 1rem; color: #222; }
  h1 { font-size: 1.4rem; }
  a { color: #06c; }
  #frame { position: relative; display: inline-block; max-width: 100%; }
  #img { display: block; max-width: 100%; border: 1px solid #ddd; }
  #box { position: absolute; border: 2px solid #2ecc40; box-sizing: border-box; pointer-events: none; }
</style>
</head>
<body>
<h1>Haro - Anteprima camera</h1>
<p><a href="/admin">&larr; torna al pannello</a></p>
<p>Anteprima dal vivo. Il riquadro verde mostra il volto rilevato nel fotogramma.</p>
<div id="frame" hidden><img id="img" alt="Ultimo fotogramma dalla camera"><div id="box" hidden></div></div>
<p id="status"><em>In attesa del primo fotogramma...</em></p>
<script>
const img = document.getElementById("img"), box = document.getElementById("box");
const frame = document.getElementById("frame"), status = document.getElementById("status");
let url = null, frames = 0, since = performance.now();
async function tick() {
  try {
    const r = await fetch("/admin/camera.jpg", { cache: "no-store" });
    if (r.status === 204) {
      status.innerHTML = "<em>Nessun fotogramma ricevuto ancora -- controlla che la camera sia collegata e che il robot sia connesso.</em>";
    } else if (r.ok) {
      const faceBox = r.headers.get("X-Face-Box");
      const next = URL.createObjectURL(await r.blob());
      await new Promise((ok, err) => { img.onload = ok; img.onerror = err; img.src = next; });
      if (url) URL.revokeObjectURL(url);
      url = next;
      frame.hidden = false;
      if (faceBox && faceBox !== "none") {
        const [x1, y1, x2, y2] = faceBox.split(",").map(Number);
        const w = img.naturalWidth, h = img.naturalHeight;
        Object.assign(box.style, { left: 100 * x1 / w + "%", top: 100 * y1 / h + "%",
                                   width: 100 * (x2 - x1) / w + "%", height: 100 * (y2 - y1) / h + "%" });
        box.hidden = false;
      } else {
        box.hidden = true;
      }
      frames++;
      const secs = (performance.now() - since) / 1000;
      if (secs >= 2) {
        status.textContent = (frames / secs).toFixed(1) + " fotogrammi/s" + (box.hidden ? " -- nessun volto rilevato" : "");
        frames = 0; since = performance.now();
      }
    } else {
      status.textContent = "Errore " + r.status;
    }
  } catch (e) {
    status.textContent = "Connessione al server persa, riprovo...";
  }
  setTimeout(tick, 100);
}
tick();
</script>
</body>
</html>"""


def _render_page(
    prompt: str,
    transcripts: list[db.Transcript],
    memories: list[db.Memory],
    with_audio: set[int],
    routines: str = "",
    alarms: list[db.Alarm] | None = None,
) -> str:
    def audio_cell(t: db.Transcript) -> str:
        # preload="none": nothing is downloaded until play is pressed, so
        # 50 rows of players don't pull 50 WAVs on every page load.
        if t.id not in with_audio:
            return "&mdash;"
        return f'<audio controls preload="none" src="/admin/audio/{t.id}.wav"></audio>'

    transcript_rows = "\n".join(
        f"<tr><td>{escape(t.created_at)}</td><td>{escape(t.session_id)}</td>"
        f"<td>{escape(t.transcript) or '<em>(vuota)</em>'}</td><td>{escape(t.reply or '')}</td>"
        f"<td>{escape(t.emotion or '')}</td><td>{audio_cell(t)}</td></tr>"
        for t in transcripts
    ) or "<tr><td colspan=\"6\"><em>Nessuna trascrizione ancora.</em></td></tr>"

    alarm_rows = "\n".join(
        f"<tr><td>{'Sveglia' if a.kind == 'alarm' else 'Timer'} n. {a.id}</td>"
        f"<td>{a.fire_at.astimezone(TIMEZONE):%d/%m %H:%M}</td><td>{escape(a.label)}</td>"
        f'<td><form method="post" action="/admin/alarms/{a.id}/cancel">'
        f'<button type="submit">Cancella</button></form></td></tr>'
        for a in alarms or []
    ) or "<tr><td colspan=\"4\"><em>Nessuna sveglia o timer attivo.</em></td></tr>"

    memory_rows = "\n".join(
        f"<tr><td>{escape(m.created_at)}</td><td>{escape(m.fact)}</td>"
        f'<td><form method="post" action="/admin/memories/{m.id}/delete" '
        f'onsubmit="return confirm(\'Eliminare questo ricordo?\')">'
        f'<button type="submit">Elimina</button></form></td></tr>'
        for m in memories
    ) or "<tr><td colspan=\"3\"><em>Nessun ricordo salvato ancora.</em></td></tr>"

    return f"""<!doctype html>
<html lang="it">
<head>
<meta charset="utf-8">
<title>Haro - Admin</title>
<style>
  body {{ font-family: -apple-system, system-ui, sans-serif; max-width: 900px; margin: 2rem auto; padding: 0 1rem; color: #222; }}
  h1 {{ font-size: 1.4rem; }}
  h2 {{ font-size: 1.1rem; margin-top: 2.5rem; border-bottom: 1px solid #ddd; padding-bottom: 0.3rem; }}
  textarea {{ width: 100%; height: 10rem; font-family: inherit; font-size: 0.95rem; box-sizing: border-box; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 0.85rem; }}
  th, td {{ text-align: left; padding: 0.4rem 0.5rem; border-bottom: 1px solid #eee; vertical-align: top; }}
  th {{ color: #666; font-weight: 600; }}
  button {{ cursor: pointer; }}
  audio {{ width: 220px; height: 32px; }}
</style>
</head>
<body>
<h1>Haro - Pannello di amministrazione</h1>
<p><a href="/admin/camera">Anteprima camera &rarr;</a></p>

<h2>Prompt di sistema</h2>
<form method="post" action="/admin/prompt">
  <textarea name="prompt">{escape(prompt)}</textarea>
  <p><button type="submit">Salva</button></p>
</form>

<h2>Routine</h2>
<p>Una routine per riga o paragrafo, in linguaggio naturale. Quando dici la frase (anche con parole simili), Haro esegue le istruzioni usando meteo, agenda e sveglie. Esempio: <em>Quando dico "buongiorno": salutami, dimmi il meteo di oggi a Lucca e Pisa e gli impegni di oggi da tutti i calendari, e ricordami il primo impegno.</em></p>
<form method="post" action="/admin/routines">
  <textarea name="routines">{escape(routines)}</textarea>
  <p><button type="submit">Salva routine</button></p>
</form>

<h2>Sveglie e timer attivi ({len(alarms or [])})</h2>
<table>
  <tr><th>Tipo</th><th>Quando</th><th>Etichetta</th><th></th></tr>
  {alarm_rows}
</table>

<h2>Memoria a lungo termine ({len(memories)})</h2>
<table>
  <tr><th>Quando</th><th>Fatto</th><th></th></tr>
  {memory_rows}
</table>

<h2>Trascrizioni recenti (ultime {len(transcripts)})</h2>
<table>
  <tr><th>Quando</th><th>Sessione</th><th>Trascrizione</th><th>Risposta</th><th>Emozione</th><th>Audio ricevuto</th></tr>
  {transcript_rows}
</table>

</body>
</html>"""
