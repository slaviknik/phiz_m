"""Общие фикстуры автотестов (pytest).

pytest подхватывает ``conftest.py`` автоматически для всех файлов каталога
``tests/``, поэтому подготовка данных (детекторы, источник света, случайные
точки поверхности) вынесена сюда и не дублируется в отдельных автотестах.

Доступные фикстуры:

* ``detector`` — детектор в конфигурации задания 1 (R = 2 м, N = 40);
* ``detector_dense`` — детектор с 950 ФЭУ (фотокатоды почти касаются);
* ``detector_packed`` — детектор с 2000 ФЭУ (фотокатоды перекрываются);
* ``source`` — источник света Y = 10 000 фотонов/МэВ с фиксированным seed;
* ``event_directions`` — направления фотонов события с энергией 1 МэВ;
* ``rng`` — генератор случайных чисел с фиксированным seed;
* ``make_sphere_points`` — фабрика случайных точек на поверхности детектора;
* ``chi2_crit_999_dof9`` — критическое значение χ² (9 ст. св., уровень 0.001).
"""

import numpy as np
import pytest

from detector import Detector
from source import Source

# Критическое значение χ² при 9 степенях свободы и уровне значимости 0.001
CHI2_CRIT_999_DOF9 = 27.877


@pytest.fixture(scope='session')
def detector():
    """Детектор в конфигурации задания 1: R = 2 м, N = 40, r = 0.1 м, a = 0.5."""
    return Detector(radius=2.0, pmt_number=40, pmt_radius=0.1, a=0.5)


@pytest.fixture(scope='session')
def detector_dense():
    """Детектор с 950 ФЭУ: фотокатоды почти касаются, но не перекрываются."""
    return Detector(radius=2.0, pmt_number=950, pmt_radius=0.1, a=0.5)


@pytest.fixture(scope='session')
def detector_packed():
    """Детектор с 2000 ФЭУ: фотокатоды заметно перекрываются."""
    return Detector(radius=2.0, pmt_number=2000, pmt_radius=0.1, a=0.5)


@pytest.fixture
def source():
    """Источник света: Y = 10 000 фотонов/МэВ, фиксированный seed (42).

    Фикстура функциональная: генератор случайных чисел не должен делить
    состояние между автотестами.
    """
    return Source(Y=10_000, seed=42)


@pytest.fixture
def event_directions(source):
    """Начальные направления фотонов события с энергией 1 МэВ (10 000 фотонов)."""
    return source.generate_directions(1.0)


@pytest.fixture
def rng():
    """Генератор случайных чисел с фиксированным seed — для воспроизводимости."""
    return np.random.default_rng(0)


@pytest.fixture
def make_sphere_points(rng, detector):
    """Фабрика случайных точек на поверхности детектора.

    Возвращает функцию ``make_sphere_points(count, seed=None)``: ``count``
    равномерно распределённых по сфере точек радиуса ``detector.radius``.
    """
    def _make(count, seed=None):
        generator = np.random.default_rng(seed) if seed is not None else rng
        v = generator.normal(size=(count, 3))
        v = v / np.linalg.norm(v, axis=1)[:, None]
        return detector.radius * v
    return _make


@pytest.fixture
def chi2_crit_999_dof9():
    """Критическое значение χ² при 9 степенях свободы (уровень значимости 0.001)."""
    return CHI2_CRIT_999_DOF9
