import datetime
import json

from litellm.types.utils import ChatCompletionMessageToolCall, Function

from haro_server import db
from haro_server.assistant_tools import TIMEZONE, AssistantTools, CombinedTools, describe_now, resolve_day

# Thursday 24 September 2026, 23:10 in Rome.
NOW = datetime.datetime(2026, 9, 24, 23, 10, tzinfo=TIMEZONE)


def _call(name: str, **args):
    return ChatCompletionMessageToolCall(
        id="c1", type="function", function=Function(name=name, arguments=json.dumps(args))
    )


def _weather_payload(days):
    return {
        "current": {"temperature_2m": 18.8, "weather_code": 3},
        "daily": {
            "time": days,
            "weather_code": [61] * len(days),
            "temperature_2m_max": [25.7] * len(days),
            "temperature_2m_min": [14.2] * len(days),
            "precipitation_probability_max": [70] * len(days),
            "wind_speed_10m_max": [12.4] * len(days),
        },
    }


class FakeEvents:
    def __init__(self, items):
        self._items = items
        self.requests = []

    def list(self, **kwargs):
        self.requests.append(kwargs)
        return self

    def execute(self):
        return {"items": self._items}


class FakeCalendar:
    def __init__(self, items):
        self.events_api = FakeEvents(items)

    def events(self):
        return self.events_api


def _tools(tmp_path, calendars=(), weather_days=("2026-09-24", "2026-09-25")):
    db_path = str(tmp_path / "haro.db")
    db.init_db(db_path)
    fetched = []

    async def fetch_json(url, params):
        fetched.append(params)
        return _weather_payload(list(weather_days))

    tools = AssistantTools(db_path, list(calendars), fetch_json=fetch_json, now=lambda: NOW)
    return tools, db_path, fetched


def test_resolve_day():
    today = datetime.date(2026, 9, 24)  # a Thursday
    assert resolve_day("oggi", today) == today
    assert resolve_day(None, today) == today
    assert resolve_day("domani", today) == datetime.date(2026, 9, 25)
    assert resolve_day("sabato", today) == datetime.date(2026, 9, 26)
    assert resolve_day("giovedì", today) == today
    assert resolve_day("2026-10-01", today) == datetime.date(2026, 10, 1)


def test_describe_now_is_local_time_in_italian():
    assert describe_now(NOW) == "giovedì 24 settembre 2026, ore 23:10"


async def test_weather_defaults_to_both_cities(tmp_path):
    tools, _, fetched = _tools(tmp_path)

    text = await tools.call(_call("meteo"))

    assert "Lucca oggi: pioggia leggera, minima 14 gradi, massima 26 gradi" in text
    assert "Pisa oggi" in text
    assert "Adesso 19 gradi" in text
    assert len(fetched) == 2


async def test_weather_for_one_city_tomorrow(tmp_path):
    tools, _, fetched = _tools(tmp_path)

    text = await tools.call(_call("meteo", citta="Pisa", giorno="domani"))

    assert text.startswith("Pisa domani:")
    assert "Adesso" not in text
    assert fetched[0]["latitude"] == 43.7228


async def test_agenda_merges_all_accounts_in_time_order(tmp_path):
    lavoro = FakeCalendar([{"summary": "Riunione", "start": {"dateTime": "2026-09-24T10:00:00+02:00"}}])
    casa = FakeCalendar([
        {"summary": "Dentista", "start": {"dateTime": "2026-09-24T08:30:00+02:00"}},
        {"summary": "Compleanno di Anna", "start": {"date": "2026-09-24"}},
    ])
    tools, _, _ = _tools(tmp_path, calendars=[("lavoro", lavoro, ["primary"]), ("casa", casa, ["primary"])])

    text = await tools.call(_call("agenda"))

    assert text == ("Impegni di oggi: alle 08:30 Dentista (casa); alle 10:00 Riunione (lavoro); "
                    "Compleanno di Anna (casa, tutto il giorno).")
    request = lavoro.events_api.requests[0]
    assert request["timeMin"].startswith("2026-09-24T00:00:00+02:00")


async def test_agenda_reports_a_failing_account_but_keeps_the_others(tmp_path):
    class Broken:
        def events(self):
            raise RuntimeError("token scaduto")

    ok = FakeCalendar([])
    tools, _, _ = _tools(tmp_path, calendars=[("casa", ok, ["primary"]), ("lavoro", Broken(), ["primary"])])

    text = await tools.call(_call("agenda", giorno="domani"))

    assert text == "Impegni di domani: nessun impegno. Non sono riuscito a leggere il calendario lavoro."


async def test_alarm_without_a_date_goes_to_tomorrow_when_the_time_has_passed(tmp_path):
    tools, db_path, _ = _tools(tmp_path)

    text = await tools.call(_call("imposta_sveglia", ora="07:00"))

    [alarm] = db.get_pending_alarms(db_path)
    assert alarm.fire_at.astimezone(TIMEZONE) == datetime.datetime(2026, 9, 25, 7, 0, tzinfo=TIMEZONE)
    assert alarm.kind == "alarm"
    assert text == f"Sveglia numero {alarm.id} impostata per domani alle 07:00."


async def test_timer_list_and_cancel(tmp_path):
    tools, db_path, _ = _tools(tmp_path)

    await tools.call(_call("imposta_timer", minuti=10, etichetta="pasta"))
    listed = await tools.call(_call("elenca_sveglie"))
    [timer] = db.get_pending_alarms(db_path)
    cancelled = await tools.call(_call("cancella_sveglia", numero=timer.id))

    assert "timer numero" in listed and "23:20" in listed and "(pasta)" in listed
    assert cancelled == f"Sveglia numero {timer.id} cancellata."
    assert db.get_pending_alarms(db_path) == []


async def test_a_bad_call_comes_back_as_text_not_an_exception(tmp_path):
    tools, _, _ = _tools(tmp_path)

    text = await tools.call(_call("imposta_sveglia", ora="non un orario"))

    assert text.startswith("Errore nello strumento imposta_sveglia")


async def test_combined_tools_route_to_local_or_mcp(tmp_path):
    tools, _, _ = _tools(tmp_path)
    mcp_calls = []

    async def mcp_call(tool_call):
        mcp_calls.append(tool_call.function.name)
        return "ok"

    combined = CombinedTools(tools, [{"type": "function", "function": {"name": "list_pull_requests"}}], mcp_call)

    assert len(combined.specs) == len(tools.specs) + 1
    assert await combined.call(_call("list_pull_requests")) == "ok"
    assert (await combined.call(_call("elenca_sveglie"))).startswith("Nessuna sveglia")
    assert mcp_calls == ["list_pull_requests"]
