from fastapi import FastAPI
from fastapi.testclient import TestClient

from haro_server import camera_preview
from haro_server.admin import create_admin_router


def _client(tmp_path) -> TestClient:
    app = FastAPI()
    app.include_router(create_admin_router(str(tmp_path / "haro.db"), "secret"))
    return TestClient(app)


def test_camera_jpg_carries_the_face_box_detected_in_that_frame(tmp_path):
    camera_preview.set_latest_frame(b"\xff\xd8 frame", (10, 20, 110, 140))

    r = _client(tmp_path).get("/admin/camera.jpg", auth=("any", "secret"))

    assert r.status_code == 200
    assert r.content == b"\xff\xd8 frame"
    assert r.headers["X-Face-Box"] == "10,20,110,140"


def test_camera_jpg_reports_no_face_as_none(tmp_path):
    camera_preview.set_latest_frame(b"\xff\xd8 frame", None)

    r = _client(tmp_path).get("/admin/camera.jpg", auth=("any", "secret"))

    assert r.headers["X-Face-Box"] == "none"


def test_camera_jpg_is_204_before_any_frame(tmp_path):
    camera_preview._latest_jpeg = None
    camera_preview._latest_face_box = None

    r = _client(tmp_path).get("/admin/camera.jpg", auth=("any", "secret"))

    assert r.status_code == 204


def test_camera_endpoints_require_the_password(tmp_path):
    client = _client(tmp_path)
    assert client.get("/admin/camera.jpg").status_code == 401
    assert client.get("/admin/camera", auth=("any", "wrong")).status_code == 401


def test_routines_and_alarms_are_editable_from_the_admin_page(tmp_path):
    import datetime

    from haro_server import db as haro_db
    from haro_server import llm

    client = _client(tmp_path)
    db_path = str(tmp_path / "haro.db")
    haro_db.init_db(db_path)
    alarm_id = haro_db.add_alarm(
        db_path, datetime.datetime(2030, 1, 1, 6, 30, tzinfo=datetime.UTC), "palestra", "alarm"
    )

    r = client.post("/admin/routines", data={"routines": 'Quando dico "buongiorno": meteo e agenda.'},
                    auth=("any", "secret"), follow_redirects=False)
    assert r.status_code == 303
    assert llm.get_routines(db_path) == 'Quando dico "buongiorno": meteo e agenda.'

    page = client.get("/admin", auth=("any", "secret")).text
    assert "Quando dico &quot;buongiorno&quot;" in page
    assert "palestra" in page and "07:30" in page  # 06:30 UTC shown in Rome time

    client.post(f"/admin/alarms/{alarm_id}/cancel", auth=("any", "secret"), follow_redirects=False)
    assert haro_db.get_pending_alarms(db_path) == []
