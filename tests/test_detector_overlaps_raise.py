"""Режим raise_on_overlap=True выбрасывает ValueError с диагностикой."""

import numpy as np
import pytest


def test_overlaps_raise_value_error(detector_dense, detector_packed):
    with pytest.raises(ValueError, match='перекрыти'):
        detector_packed.check_overlaps(raise_on_overlap=True)

    # для разреженной конфигурации исключения не возникает
    overlaps, min_gap = detector_dense.check_overlaps(raise_on_overlap=True)
    assert overlaps.size == 0
    assert np.isfinite(min_gap)
