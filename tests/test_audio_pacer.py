from haro_server.server import MAX_AUDIO_LEAD_SECONDS, AudioPacer

ONE_SECOND = 16000 * 2


class FakeTime:
    def __init__(self) -> None:
        self.now = 100.0
        self.slept: list[float] = []

    def clock(self) -> float:
        return self.now

    async def sleep(self, seconds: float) -> None:
        self.slept.append(seconds)
        self.now += seconds


async def test_sends_freely_until_the_lead_limit_then_paces_to_real_time():
    t = FakeTime()
    pacer = AudioPacer(clock=t.clock, sleep=t.sleep)

    # 10 x 1s of audio, generated instantly (far faster than real time).
    sent = 0.0
    for _ in range(10):
        await pacer.wait_before_sending(ONE_SECOND)
        # Lead over playback when this chunk goes out: never above the limit.
        assert sent - (t.now - 100.0) <= MAX_AUDIO_LEAD_SECONDS + 1e-9
        sent += 1.0

    # Chunks 0 and 1 go out at once (lead 0s, 1s); from chunk 2 on each one
    # waits until the lead is back down to the limit -- real-time pacing.
    assert t.slept[0] == 2.0 - MAX_AUDIO_LEAD_SECONDS
    assert t.slept[1:] == [1.0] * 7
    assert t.now - 100.0 == 9.0 - MAX_AUDIO_LEAD_SECONDS


async def test_a_stream_slower_than_real_time_is_never_delayed():
    t = FakeTime()
    pacer = AudioPacer(clock=t.clock, sleep=t.sleep)

    for _ in range(5):
        await pacer.wait_before_sending(ONE_SECOND // 2)
        t.now += 0.6  # generation slower than playback

    assert t.slept == []


async def test_a_new_stream_after_the_previous_one_played_out_starts_fresh():
    t = FakeTime()
    pacer = AudioPacer(clock=t.clock, sleep=t.sleep)
    for _ in range(3):
        await pacer.wait_before_sending(ONE_SECOND)
    t.now += 60  # long pause: previous reply finished long ago
    t.slept.clear()

    await pacer.wait_before_sending(ONE_SECOND)

    assert t.slept == []
