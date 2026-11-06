"""Световыход по умолчанию: Y = 10000 фотонов/МэВ."""

from source import Source


def test_default_light_yield(source):
    assert source.Y == 10_000
    assert Source(Y=12_345).Y == 12_345
