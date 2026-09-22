# src/haro_server/admin.py
"""Web UI for editing the system prompt and browsing transcripts/long-term
memory (db.py). Protected by HTTP Basic Auth (a single shared password,
config.py's admin_password) -- see create_admin_router()'s docstring for
why server.py refuses to mount this at all when no password is configured.
"""
import secrets
from html import escape

from fastapi import APIRouter, Depends, Form, HTTPException
from fastapi.responses import HTMLResponse, RedirectResponse, Response
from fastapi.security import HTTPBasic, HTTPBasicCredentials

from . import camera_preview, db, llm


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
        return _render_page(prompt, transcripts, memories)

    @router.post("/prompt")
    async def save_prompt(prompt: str = Form(...), _: None = Depends(require_auth)) -> RedirectResponse:
        llm.set_base_system_prompt(db_path, prompt)
        return RedirectResponse(url="/admin", status_code=303)

    @router.post("/memories/{memory_id}/delete")
    async def delete_memory_route(memory_id: int, _: None = Depends(require_auth)) -> RedirectResponse:
        db.delete_memory(db_path, memory_id)
        return RedirectResponse(url="/admin", status_code=303)

    @router.get("/camera", response_class=HTMLResponse)
    async def camera_page(_: None = Depends(require_auth)) -> str:
        return _render_camera_page(
            has_frame=camera_preview.get_latest_frame() is not None,
            face_box=camera_preview.get_latest_face_box(),
        )

    @router.get("/camera.jpg")
    async def camera_jpg(_: None = Depends(require_auth)) -> Response:
        jpeg = camera_preview.get_latest_frame()
        if jpeg is None:
            # 204, not 404: "the robot hasn't sent a frame yet" isn't a
            # missing resource, it's an expected transient state right
            # after startup or with the camera not wired up.
            return Response(status_code=204)
        # Cache-Control: no-store -- this is refreshed by the page's own
        # cache-busting query param (see _render_camera_page()), not by the
        # browser deciding a JPEG at this URL is safe to reuse.
        return Response(content=jpeg, media_type="image/jpeg", headers={"Cache-Control": "no-store"})

    return router


# QVGA -- matches camera_face_track.cpp's config.frame_size
# (FRAMESIZE_QVGA), the coordinate system face_box's pixel values are in.
_FRAME_WIDTH = 320
_FRAME_HEIGHT = 240


def _render_camera_page(has_frame: bool, face_box: tuple[int, int, int, int] | None) -> str:
    # Whole-page meta-refresh (no JavaScript) re-fetches the <img> tag fresh
    # on every reload too -- simpler than a JS-driven cache-busting query
    # param, and consistent with the rest of this file's plain-HTML style.
    if not has_frame:
        body = "<p><em>Nessun fotogramma ricevuto ancora -- controlla che la camera sia collegata e che il robot sia connesso.</em></p>"
    else:
        overlay = ""
        if face_box is not None:
            x1, y1, x2, y2 = face_box
            # Percentages relative to the wrapping <div> below, which -- via
            # display:inline-block hugging the <img> -- always matches the
            # image's own rendered box exactly, at any display size (no JS
            # needed to recompute anything on resize/refresh).
            left_pct = 100 * x1 / _FRAME_WIDTH
            top_pct = 100 * y1 / _FRAME_HEIGHT
            width_pct = 100 * (x2 - x1) / _FRAME_WIDTH
            height_pct = 100 * (y2 - y1) / _FRAME_HEIGHT
            overlay = (
                f'<div style="position:absolute;left:{left_pct:.2f}%;top:{top_pct:.2f}%;'
                f'width:{width_pct:.2f}%;height:{height_pct:.2f}%;'
                f'border:2px solid #2ecc40;box-sizing:border-box;pointer-events:none;"></div>'
            )
        body = (
            '<div style="position:relative;display:inline-block;max-width:100%;">'
            '<img src="/admin/camera.jpg" alt="Ultimo fotogramma dalla camera" '
            'style="display:block;max-width:100%;border:1px solid #ddd;">'
            f"{overlay}"
            "</div>"
        )
        if face_box is None:
            body += "<p><em>Nessun volto rilevato in questo fotogramma.</em></p>"

    return f"""<!doctype html>
<html lang="it">
<head>
<meta charset="utf-8">
<meta http-equiv="refresh" content="2">
<title>Haro - Camera</title>
<style>
  body {{ font-family: -apple-system, system-ui, sans-serif; max-width: 900px; margin: 2rem auto; padding: 0 1rem; color: #222; }}
  h1 {{ font-size: 1.4rem; }}
  a {{ color: #06c; }}
</style>
</head>
<body>
<h1>Haro - Anteprima camera</h1>
<p><a href="/admin">&larr; torna al pannello</a></p>
<p>Aggiornata ogni ~2 secondi (il robot invia un fotogramma ogni ~2s durante il face tracking). Il riquadro verde mostra il volto rilevato, se presente in questo fotogramma.</p>
{body}
</body>
</html>"""


def _render_page(prompt: str, transcripts: list[db.Transcript], memories: list[db.Memory]) -> str:
    transcript_rows = "\n".join(
        f"<tr><td>{escape(t.created_at)}</td><td>{escape(t.session_id)}</td>"
        f"<td>{escape(t.transcript)}</td><td>{escape(t.reply or '')}</td>"
        f"<td>{escape(t.emotion or '')}</td></tr>"
        for t in transcripts
    ) or "<tr><td colspan=\"5\"><em>Nessuna trascrizione ancora.</em></td></tr>"

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

<h2>Memoria a lungo termine ({len(memories)})</h2>
<table>
  <tr><th>Quando</th><th>Fatto</th><th></th></tr>
  {memory_rows}
</table>

<h2>Trascrizioni recenti (ultime {len(transcripts)})</h2>
<table>
  <tr><th>Quando</th><th>Sessione</th><th>Trascrizione</th><th>Risposta</th><th>Emozione</th></tr>
  {transcript_rows}
</table>

</body>
</html>"""
