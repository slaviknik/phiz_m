"""Воспроизводимость событий: одинаковый seed — одинаковые направления."""

import numpy as np

from source import Source


def test_same_seed_gives_same_event(source):
    # два независимых источника с одинаковым seed дают идентичное событие
    first = Source(seed=7).generate_directions(0.01)
    second = Source(seed=7).generate_directions(0.01)
    assert np.array_equal(first, second)

    # другое seed (в фикстуре source — 42) даёт другое событие
    assert not np.array_equal(first, source.generate_directions(0.01))
