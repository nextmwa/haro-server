from haro_server import actions


def test_match_action_dice_roll_singular():
    action = actions.match_action("lancia un dado")
    assert action is not None
    assert action.name == "dice_roll"


def test_match_action_dice_roll_plural():
    action = actions.match_action("tira i dadi")
    assert action is not None
    assert action.name == "dice_roll"


def test_match_action_coin_flip():
    action = actions.match_action("lancia una moneta")
    assert action is not None
    assert action.name == "coin_flip"


def test_match_action_coin_flip_testa_o_croce_phrase():
    action = actions.match_action("facciamo a testa o croce")
    assert action is not None
    assert action.name == "coin_flip"


def test_match_action_returns_none_for_unrelated_speech():
    assert actions.match_action("che tempo fa oggi") is None


def test_dice_roll_resolves_to_a_value_between_1_and_6():
    action = next(a for a in actions.ACTIONS if a.name == "dice_roll")
    for _ in range(50):
        assert action.resolve() in range(1, 7)


def test_coin_flip_resolves_to_testa_or_croce():
    action = next(a for a in actions.ACTIONS if a.name == "coin_flip")
    for _ in range(50):
        assert action.resolve() in ("testa", "croce")
