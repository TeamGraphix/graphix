from threading import Thread

from graphix.rng import ensure_rng


def test_new_thread() -> None:
    t = Thread(target=ensure_rng)
    t.start()
