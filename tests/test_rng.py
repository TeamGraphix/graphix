from concurrent.futures import ThreadPoolExecutor
from threading import Thread

from graphix.rng import ensure_rng


def test_new_thread() -> None:
    t = Thread(target=ensure_rng)
    t.start()


def test_threadpool() -> None:
    with ThreadPoolExecutor() as executor:
        tasks = executor.map(lambda _: ensure_rng(), range(100))
        for _ in tasks:
            pass
