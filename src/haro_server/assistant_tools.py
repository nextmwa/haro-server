"""Everyday assistant functions the LLM can call (OpenAI-style tool calling,
see llm.py's stream_reply()): weather, today's agenda across every Google
Calendar account, alarms and timers. The user just talks ("che tempo fa
domani a Pisa?", "svegliami alle 7") and the model picks the tool.

Every tool returns a short plain-Italian string the model turns into its
spoken reply -- never raises into the model loop (errors come back as text
too, so Haro can say what went wrong).
"""
import asyncio
import datetime
import json
import logging
from typing import Any, Awaitable, Callable
from zoneinfo import ZoneInfo

import httpx

from . import db

logger = logging.getLogger(__name__)

TIMEZONE = ZoneInfo("Europe/Rome")

# The user's two reference cities (weather defaults to both).
CITIES = {
    "lucca": ("Lucca", 43.8429, 10.5027),
    "pisa": ("Pisa", 43.7228, 10.4017),
}

# WMO weather interpretation codes, as returned by Open-Meteo.
_WEATHER_CODES = {
    0: "sereno", 1: "prevalentemente sereno", 2: "parzialmente nuvoloso", 3: "coperto",
    45: "nebbia", 48: "nebbia con brina",
    51: "pioviggine leggera", 53: "pioviggine", 55: "pioviggine intensa",
    56: "pioviggine gelata", 57: "pioviggine gelata intensa",
    61: "pioggia leggera", 63: "pioggia", 65: "pioggia forte",
    66: "pioggia gelata", 67: "pioggia gelata forte",
    71: "neve leggera", 73: "neve", 75: "neve forte", 77: "granelli di neve",
    80: "rovesci leggeri", 81: "rovesci", 82: "rovesci violenti",
    85: "rovesci di neve", 86: "forti rovesci di neve",
    95: "temporale", 96: "temporale con grandine", 99: "temporale con grandine forte",
}

_WEEKDAYS = ["lunedì", "martedì", "mercoledì", "giovedì", "venerdì", "sabato", "domenica"]
_MONTHS = ["gennaio", "febbraio", "marzo", "aprile", "maggio", "giugno", "luglio",
           "agosto", "settembre", "ottobre", "novembre", "dicembre"]

_DAY_PARAM = {
    "type": "string",
    "description": "Il giorno: 'oggi', 'domani', 'dopodomani', un giorno della settimana (es. 'sabato') "
                   "oppure una data YYYY-MM-DD. Predefinito: oggi.",
}

TOOL_SPECS = [
    {"type": "function", "function": {
        "name": "meteo",
        "description": "Previsioni del tempo per Lucca e/o Pisa (le città dell'utente).",
        "parameters": {"type": "object", "properties": {
            "citta": {"type": "string", "enum": ["Lucca", "Pisa", "entrambe"],
                      "description": "Predefinito: entrambe."},
            "giorno": _DAY_PARAM,
        }},
    }},
    {"type": "function", "function": {
        "name": "agenda",
        "description": "Impegni di un giorno da tutti i calendari Google dell'utente.",
        "parameters": {"type": "object", "properties": {"giorno": _DAY_PARAM}},
    }},
    {"type": "function", "function": {
        "name": "imposta_sveglia",
        "description": "Imposta una sveglia a un orario. Se la data manca: oggi se l'orario deve ancora "
                       "arrivare, altrimenti domani.",
        "parameters": {"type": "object", "required": ["ora"], "properties": {
            "ora": {"type": "string", "description": "Orario HH:MM, 24 ore."},
            "giorno": _DAY_PARAM,
            "etichetta": {"type": "string", "description": "Cosa ricordare, se detto (es. 'palestra')."},
        }},
    }},
    {"type": "function", "function": {
        "name": "imposta_timer",
        "description": "Avvia un timer (anche per 'rimanda la sveglia di N minuti').",
        "parameters": {"type": "object", "required": ["minuti"], "properties": {
            "minuti": {"type": "number"},
            "etichetta": {"type": "string"},
        }},
    }},
    {"type": "function", "function": {
        "name": "elenca_sveglie",
        "description": "Elenca sveglie e timer attivi.",
        "parameters": {"type": "object", "properties": {}},
    }},
    {"type": "function", "function": {
        "name": "cancella_sveglia",
        "description": "Cancella una sveglia o un timer per numero (vedi elenca_sveglie), oppure tutti.",
        "parameters": {"type": "object", "properties": {
            "numero": {"type": "integer"},
            "tutte": {"type": "boolean"},
        }},
    }},
]

TOOL_NAMES = {spec["function"]["name"] for spec in TOOL_SPECS}


def resolve_day(value: str | None, today: datetime.date) -> datetime.date:
    text = (value or "oggi").strip().lower()
    if text in ("", "oggi"):
        return today
    if text == "domani":
        return today + datetime.timedelta(days=1)
    if text == "dopodomani":
        return today + datetime.timedelta(days=2)
    for i, name in enumerate(_WEEKDAYS):
        if text in (name, name.rstrip("ì") + "i"):  # "lunedì" / "lunedi"
            return today + datetime.timedelta(days=(i - today.weekday()) % 7)
    return datetime.date.fromisoformat(text)


def day_label(day: datetime.date, today: datetime.date) -> str:
    if day == today:
        return "oggi"
    if day == today + datetime.timedelta(days=1):
        return "domani"
    return f"{_WEEKDAYS[day.weekday()]} {day.day} {_MONTHS[day.month - 1]}"


def describe_now(now: datetime.datetime) -> str:
    local = now.astimezone(TIMEZONE)
    return (f"{_WEEKDAYS[local.weekday()]} {local.day} {_MONTHS[local.month - 1]} {local.year}, "
            f"ore {local:%H:%M}")


# (label, Google Calendar API service, calendar ids) per account -- see
# google_calendar_accounts.py's build_calendar_services().
CalendarAccount = tuple[str, Any, list[str]]


class AssistantTools:
    def __init__(
        self,
        db_path: str,
        calendar_accounts: list[CalendarAccount],
        fetch_json: Callable[[str, dict], Awaitable[dict]] | None = None,
        now: Callable[[], datetime.datetime] | None = None,
    ) -> None:
        self._db_path = db_path
        self._calendar_accounts = calendar_accounts
        self._fetch_json = fetch_json or _http_get_json
        self._now = now or (lambda: datetime.datetime.now(datetime.UTC))

    @property
    def specs(self) -> list[dict]:
        return TOOL_SPECS

    async def call(self, tool_call) -> str:
        name = tool_call.function.name
        try:
            args = json.loads(tool_call.function.arguments or "{}")
            handler = getattr(self, f"_tool_{name}")
            return await handler(**args)
        except Exception as exc:
            logger.exception("assistant tool %s failed", name)
            return f"Errore nello strumento {name}: {exc}"

    def _today(self) -> datetime.date:
        return self._now().astimezone(TIMEZONE).date()

    # --- weather -----------------------------------------------------------

    async def _tool_meteo(self, citta: str = "entrambe", giorno: str = "oggi") -> str:
        today = self._today()
        day = resolve_day(giorno, today)
        keys = list(CITIES) if citta.lower() in ("entrambe", "") else [citta.lower()]
        parts = []
        for key in keys:
            name, lat, lon = CITIES[key]
            data = await self._fetch_json("https://api.open-meteo.com/v1/forecast", {
                "latitude": lat, "longitude": lon, "timezone": "Europe/Rome", "forecast_days": 16,
                "current": "temperature_2m,weather_code",
                "daily": "weather_code,temperature_2m_max,temperature_2m_min,"
                         "precipitation_probability_max,wind_speed_10m_max",
            })
            daily = data["daily"]
            if day.isoformat() not in daily["time"]:
                parts.append(f"{name}: previsioni non disponibili per quel giorno.")
                continue
            i = daily["time"].index(day.isoformat())
            text = (f"{name} {day_label(day, today)}: "
                    f"{_WEATHER_CODES.get(daily['weather_code'][i], 'tempo variabile')}, "
                    f"minima {round(daily['temperature_2m_min'][i])} gradi, "
                    f"massima {round(daily['temperature_2m_max'][i])} gradi, "
                    f"probabilità di pioggia {daily['precipitation_probability_max'][i]}%, "
                    f"vento fino a {round(daily['wind_speed_10m_max'][i])} km/h.")
            if day == today and "current" in data:
                text += f" Adesso {round(data['current']['temperature_2m'])} gradi."
            parts.append(text)
        return " ".join(parts)

    # --- calendar ----------------------------------------------------------

    async def _tool_agenda(self, giorno: str = "oggi") -> str:
        today = self._today()
        day = resolve_day(giorno, today)
        if not self._calendar_accounts:
            return "Nessun calendario configurato."
        start = datetime.datetime.combine(day, datetime.time.min, TIMEZONE)
        end = start + datetime.timedelta(days=1)
        # Accounts in parallel, each account's calendars one after another:
        # a Google API client must not be used from two threads at once.
        results = await asyncio.gather(*(
            asyncio.to_thread(_list_account_events, service, ids, start, end)
            for _, service, ids in self._calendar_accounts
        ), return_exceptions=True)
        labels = [label for label, _, _ in self._calendar_accounts]

        timed, all_day, failed = [], [], []
        for label, result in zip(labels, results):
            if isinstance(result, Exception):
                logger.warning("agenda: calendar %s failed: %s", label, result)
                failed.append(label)
                continue
            for item in result:
                summary = item.get("summary", "un impegno")
                if "dateTime" in item.get("start", {}):
                    begin = datetime.datetime.fromisoformat(item["start"]["dateTime"]).astimezone(TIMEZONE)
                    timed.append((begin, f"alle {begin:%H:%M} {summary} ({label})"))
                else:
                    all_day.append(f"{summary} ({label}, tutto il giorno)")
        lines = [text for _, text in sorted(timed)] + all_day
        head = f"Impegni di {day_label(day, today)}: "
        body = "; ".join(lines) + "." if lines else "nessun impegno."
        if failed:
            body += f" Non sono riuscito a leggere il calendario {', '.join(sorted(set(failed)))}."
        return head + body

    # --- alarms and timers -------------------------------------------------

    async def _tool_imposta_sveglia(self, ora: str, giorno: str | None = None, etichetta: str = "") -> str:
        now = self._now().astimezone(TIMEZONE)
        hour, minute = (int(x) for x in ora.split(":"))
        if giorno:
            day = resolve_day(giorno, now.date())
        else:
            day = now.date()
            if datetime.time(hour, minute) <= now.time():
                day += datetime.timedelta(days=1)
        fire_at = datetime.datetime.combine(day, datetime.time(hour, minute), TIMEZONE)
        if fire_at <= now:
            return "Quell'orario è già passato, dimmi un altro orario."
        alarm_id = db.add_alarm(self._db_path, fire_at, etichetta, "alarm")
        return f"Sveglia numero {alarm_id} impostata per {day_label(day, now.date())} alle {fire_at:%H:%M}."

    async def _tool_imposta_timer(self, minuti: float, etichetta: str = "") -> str:
        fire_at = self._now() + datetime.timedelta(minutes=float(minuti))
        alarm_id = db.add_alarm(self._db_path, fire_at, etichetta, "timer")
        local = fire_at.astimezone(TIMEZONE)
        return f"Timer numero {alarm_id} di {minuti:g} minuti, suonerà alle {local:%H:%M}."

    async def _tool_elenca_sveglie(self) -> str:
        today = self._today()
        alarms = db.get_pending_alarms(self._db_path)
        if not alarms:
            return "Nessuna sveglia o timer attivo."
        items = []
        for a in alarms:
            local = a.fire_at.astimezone(TIMEZONE)
            kind = "sveglia" if a.kind == "alarm" else "timer"
            label = f" ({a.label})" if a.label else ""
            items.append(f"{kind} numero {a.id} {day_label(local.date(), today)} alle {local:%H:%M}{label}")
        return "; ".join(items) + "."

    async def _tool_cancella_sveglia(self, numero: int | None = None, tutte: bool = False) -> str:
        if tutte:
            alarms = db.get_pending_alarms(self._db_path)
            for a in alarms:
                db.set_alarm_status(self._db_path, a.id, "cancelled")
            return f"Cancellate {len(alarms)} tra sveglie e timer."
        if numero is None:
            return "Dimmi quale sveglia cancellare."
        pending = {a.id for a in db.get_pending_alarms(self._db_path)}
        if numero not in pending:
            return f"Non c'è una sveglia attiva numero {numero}."
        db.set_alarm_status(self._db_path, numero, "cancelled")
        return f"Sveglia numero {numero} cancellata."


def _list_account_events(service, calendar_ids: list[str], start: datetime.datetime, end: datetime.datetime) -> list[dict]:
    return [item for calendar_id in calendar_ids for item in _list_events(service, calendar_id, start, end)]


def _list_events(service, calendar_id: str, start: datetime.datetime, end: datetime.datetime) -> list[dict]:
    response = service.events().list(
        calendarId=calendar_id, timeMin=start.isoformat(), timeMax=end.isoformat(),
        singleEvents=True, orderBy="startTime",
    ).execute()
    return response.get("items", [])


async def _http_get_json(url: str, params: dict) -> dict:
    async with httpx.AsyncClient(timeout=10) as client:
        response = await client.get(url, params=params)
        response.raise_for_status()
        return response.json()


class CombinedTools:
    """Local assistant tools plus, when connected, the GitHub MCP server's
    tools -- one list for the model, one dispatcher for its calls."""

    def __init__(self, local: AssistantTools, mcp_tools: list[dict] | None = None, mcp_call=None) -> None:
        self._local = local
        self._mcp_tools = mcp_tools or []
        self._mcp_call = mcp_call

    @property
    def specs(self) -> list[dict]:
        return self._local.specs + self._mcp_tools

    async def call(self, tool_call) -> str:
        if tool_call.function.name in TOOL_NAMES or self._mcp_call is None:
            return await self._local.call(tool_call)
        return await self._mcp_call(tool_call)
