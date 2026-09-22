import re

# Cheap, local pre-filter run on the raw STT transcript, BEFORE any LLM
# call -- mirrors actions.py's match_action() pattern (deterministic intent
# match ahead of the LLM), except a music request still needs the LLM
# afterward to extract the actual search query and pick among Navidrome's
# results (session.py's _play_music()), since song/artist names are far too
# open-ended for a fixed keyword match the way "lancia un dado" is.
_MUSIC_TRIGGER_RE = re.compile(
    r"\b(metti|riproduci|suona|fai sentire|fammi sentire|ascolta\w*)\b", re.IGNORECASE
)


def is_music_request(transcript: str) -> bool:
    return bool(_MUSIC_TRIGGER_RE.search(transcript))
