"""Проверка перекрытий плоских фотокатодов на внутренней поверхности сферы."""

import numpy as np


def test_overlaps_are_detected_where_expected(detector_dense, detector_packed):
    # разреженная конфигурация: фотокатоды не перекрываются
    overlaps, min_gap = detector_dense.check_overlaps()
    assert overlaps.size == 0
    assert min_gap > 0.0

    # плотная конфигурация: перекрытия есть, минимальный зазор отрицателен
    dense = detector_packed
    overlaps, min_gap = dense.check_overlaps()
    assert len(overlaps) > 0
    assert min_gap < 0.0

    # каждая найденная пара действительно пересекается
    u = dense.pmt_centers / np.linalg.norm(dense.pmt_centers, axis=1, keepdims=True)
    for i, j in overlaps:
        distance = dense.radius * np.arccos(np.clip(u[i] @ u[j], -1.0, 1.0))
        assert distance < 2.0 * dense.pmt_radius
