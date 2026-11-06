"""Проверка корректности входных параметров класса Source."""

import pytest

from source import Source


def test_invalid_parameters_raise_value_error(source):
    for bad_energy in (-1.0, -0.001, float('nan'), float('inf')):
        with pytest.raises(ValueError):
            source.generate_directions(bad_energy)
        with pytest.raises(ValueError):
            source.photon_number(bad_energy)

    for bad_yield in (0, -1.0, float('nan'), float('inf')):
        with pytest.raises(ValueError):
            Source(Y=bad_yield)
