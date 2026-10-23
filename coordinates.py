import numpy as np

# конвертация систем координат

def cartesian_to_spherical(points, degrees = False):
    """
    Преобразует декартовы координаты (x, y, z) в сферические (r, theta, phi).
    Параметры:
        points : array_like, shape (..., 3)
        degrees : bool — если True, углы возвращаются в градусах
    Возвращает:
        r     : ndarray, shape (...)   — расстояние от центра
        theta : ndarray, shape (...)   — полярный угол от оси +z, [0, π]
        phi   : ndarray, shape (...)   — азимутальный угол от +x, [0, 2π)
    """
    pts = np.asarray(points, dtype=float)
    x, y, z = pts[..., 0], pts[..., 1], pts[..., 2]
    r = np.sqrt(x*x + y*y + z*z)
    # защита от деления на ноль в центре
    with np.errstate(invalid = 'ignore', divide = 'ignore'):
        cos_theta = np.where(r > 0, z / r, 1.0)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    theta = np.arccos(cos_theta)                  # [0, π]
    phi = np.mod(np.arctan2(y, x), 2.0 * np.pi)   # [0, 2π)
    if degrees:
        theta = np.degrees(theta)
        phi   = np.degrees(phi)
    return r, theta, phi

def spherical_to_cartesian(r, theta, phi, degrees=False):
    """Обратное преобразование (r, theta, phi) → (x, y, z)."""
    if degrees:
        theta = np.radians(theta)
        phi   = np.radians(phi)
    sin_t = np.sin(theta)
    x = r * sin_t * np.cos(phi)
    y = r * sin_t * np.sin(phi)
    z = r * np.cos(theta)
    return np.stack([x, y, z], axis=-1)
