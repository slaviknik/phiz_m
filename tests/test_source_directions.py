"""Начальные направления фотонов: форма массива и единичная длина векторов."""

import numpy as np


def test_generate_directions_shape_and_norm(source):
    directions = source.generate_directions(1.0)
    assert directions.shape == (10_000, 3)
    assert directions.dtype == np.float64
    assert np.allclose(np.linalg.norm(directions, axis=1), 1.0)

    # при нулевой энергии фотонов нет
    assert source.generate_directions(0.0).shape == (0, 3)

    # число направлений равно числу фотонов
    for energy in (0.5, 1.0, 2.0):
        assert source.generate_directions(energy).shape[0] == source.photon_number(energy)
