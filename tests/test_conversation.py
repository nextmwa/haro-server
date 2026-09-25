from haro_server.session import Conversation


class Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


def test_turns_are_remembered_in_order():
    conv = Conversation(clock=Clock())
    conv.add("ciao", "Ciao!")
    conv.add("come stai?", "Bene!")

    assert conv.history() == [("ciao", "Ciao!"), ("come stai?", "Bene!")]


def test_the_conversation_is_forgotten_after_a_long_silence():
    clock = Clock()
    conv = Conversation(clock=clock)
    conv.add("ciao", "Ciao!")

    clock.now += Conversation.IDLE_RESET_SECONDS + 1

    assert conv.history() == []


def test_a_follow_up_within_the_window_keeps_the_context():
    clock = Clock()
    conv = Conversation(clock=clock)
    conv.add("ciao", "Ciao!")
    clock.now += 30

    assert conv.history() == [("ciao", "Ciao!")]


def test_only_the_most_recent_turns_are_kept():
    conv = Conversation(clock=Clock())
    for i in range(Conversation.MAX_TURNS + 3):
        conv.add(f"domanda {i}", f"risposta {i}")

    history = conv.history()
    assert len(history) == Conversation.MAX_TURNS
    assert history[-1] == (f"domanda {Conversation.MAX_TURNS + 2}", f"risposta {Conversation.MAX_TURNS + 2}")
