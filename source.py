"""Источник света в жидкосцинтилляционном детекторе (Задание 2).

Свет излучается изотропно, а выход фотонов линеен по энергии::

    N = Y * E,

где ``Y = 10000`` фотонов/МэВ — световыход, ``E`` — энергия события в МэВ.
Начальное направление каждого фотона — случайный единичный вектор,
равномерно распределённый по сфере: ``cos(theta) ~ U[-1, 1]``,
``phi ~ U[0, 2*pi)``. Важно, что равномерно разыгрывается именно косинус
полярного угла (равномерность по площади сферы), а не сам угол ``theta``.
"""

from __future__ import annotations

import numpy as np

__all__ = ["Source"]


class Source:
    """Изотропный источник фотонов для события с энергией ``E``.

    Параметры:
        Y    : световыход, фотонов/МэВ (по умолчанию 10 000)
        seed : начальное состояние генератора случайных чисел;
               ``None`` — инициализация из энтропии системы

    Атрибуты:
        Y   : световыход, фотонов/МэВ
        rng : генератор случайных чисел ``numpy.random.Generator``,
              общий для всех методов розыгрыша

    Пример:
        >>> src = Source(seed=0)
        >>> v = src.generate_directions(1.0)          # 10 000 фотонов
        >>> v.shape
        (10000, 3)
        >>> np.allclose(np.linalg.norm(v, axis=1), 1.0)
        True
    """

    def __init__(self, Y: float = 10_000.0, seed: int | None = None) -> None:
        if not np.isfinite(Y) or Y <= 0:
            raise ValueError("Световыход Y должен быть конечным положительным числом")
        self.Y = float(Y)
        self.rng = np.random.default_rng(seed)

    def photon_number(self, E: float) -> int:
        """Число фотонов в событии: ``N = Y·E`` (линейный выход по энергии).

        Параметры:
            E : энергия события, МэВ (``E >= 0``)

        Возвращает:
            int : целое число излучённых фотонов. Поскольку число фотонов
                  целочисленно, значение ``Y·E`` округляется до ближайшего
                  целого (при целом ``Y·E`` совпадает с ним точно).

        Пример:
            >>> Source().photon_number(1.0)
            10000
            >>> Source().photon_number(0.5)
            5000
        """
        if not np.isfinite(E) or E < 0:
            raise ValueError("Энергия E должна быть конечным неотрицательным числом, МэВ")
        return int(round(self.Y * float(E)))

    # алиас, удобный в ноутбуке
    n_photons = photon_number

    def _sample_isotropic(self, E: float) -> tuple[np.ndarray, np.ndarray]:
        """Разыграть ``(cos_theta, phi)`` для ``N = Y·E`` фотонов.

        cos(theta) равномерен на [-1, 1], phi — на [0, 2*pi): это даёт
        постоянную плотность направлений на единичной сфере (изотропность).
        Возвращает массивы формы ``(N,)``; при ``E = 0`` — пустые массивы.
        """
        n = self.photon_number(E)
        cos_theta = self.rng.uniform(-1.0, 1.0, size=n)
        phi = self.rng.uniform(0.0, 2.0 * np.pi, size=n)
        return cos_theta, phi

    def generate_angles(self, E: float) -> tuple[np.ndarray, np.ndarray]:
        """Разыграть начальные направления фотонов в сферических углах.

        Параметры:
            E : энергия события, МэВ

        Возвращает:
            theta : ndarray (N,) — полярный угол от оси +z, [0, pi], рад
            phi   : ndarray (N,) — азимутальный угол от оси +x, [0, 2*pi), рад

        Пример:
            >>> theta, phi = Source(seed=0).generate_angles(0.001)
            >>> theta.shape, phi.shape
            ((10,), (10,))
        """
        cos_theta, phi = self._sample_isotropic(E)
        return np.arccos(cos_theta), phi

    def generate_directions(self, E: float) -> np.ndarray:
        """Разыграть начальные направления фотонов для события с энергией ``E``.

        Число направлений равно числу фотонов ``N = Y·E``. Каждое направление —
        единичный вектор в декартовых координатах; направления равномерно
        распределены по сфере (свет излучается изотропно).

        Параметры:
            E : энергия события, МэВ

        Возвращает:
            ndarray (N, 3) : единичные векторы ``(x, y, z)`` начальных
                             направлений фотонов; при ``E = 0`` — пустой
                             массив формы ``(0, 3)``

        Пример:
            >>> src = Source(seed=0)
            >>> v = src.generate_directions(0.001)     # 10 фотонов
            >>> v.shape
            (10, 3)
            >>> np.allclose(np.linalg.norm(v, axis=1), 1.0)
            True
        """
        cos_theta, phi = self._sample_isotropic(E)
        # sin(theta) = sqrt(1 - cos^2(theta)), sin >= 0 при theta в [0, pi]
        sin_theta = np.sqrt(np.maximum(0.0, 1.0 - cos_theta * cos_theta))
        # x = sin(theta)cos(phi), y = sin(theta)sin(phi), z = cos(theta)
        return np.column_stack((
            sin_theta * np.cos(phi),
            sin_theta * np.sin(phi),
            cos_theta,
        ))

    # алиасы, удобные в ноутбуке и при дальнейшем моделировании
    emit = generate_directions
    generate = generate_directions
    directions = generate_directions

    def __repr__(self) -> str:
        return f"Source(Y={self.Y:g} фотонов/МэВ)"
