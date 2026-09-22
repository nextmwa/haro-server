import random
import re
from dataclasses import dataclass
from typing import Any, Callable

# Deterministic, keyword-matched utility commands (dice, coin flip, ...) that
# bypass the LLM entirely: the LLM has no real source of randomness and can
# misinterpret/embellish a request like "lancia un dado" in ways that don't
# fit a clean "animate -> reveal result" UX on the robot's display. Matched
# against the raw STT transcript in session.py, before any LLM call --
# see Session.handle_end_of_speech().
#
# Extensible by design: adding a new action is just a new entry in ACTIONS
# below, plus the matching animation on the firmware side (see
# face_display_show_action() in the esp32-firmware repo) -- no protocol or
# session.py changes needed.


@dataclass(frozen=True)
class Action:
    name: str
    pattern: re.Pattern[str]
    resolve: Callable[[], Any]


def _roll_dice() -> int:
    return random.randint(1, 6)


def _flip_coin() -> str:
    return random.choice(["testa", "croce"])


ACTIONS: list[Action] = [
    Action(name="dice_roll", pattern=re.compile(r"\bdad[oi]\b", re.IGNORECASE), resolve=_roll_dice),
    Action(
        name="coin_flip",
        pattern=re.compile(r"\bmoneta\b|\btesta o croce\b", re.IGNORECASE),
        resolve=_flip_coin,
    ),
]


def match_action(transcript: str) -> Action | None:
    for action in ACTIONS:
        if action.pattern.search(transcript):
            return action
    return None
