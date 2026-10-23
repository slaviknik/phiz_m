__all__ = ["Detector"]


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

class Detector:
    def __init__(self, radius: float = 2.0, pmt_number: int = 200, pmt_radius: float = 0.1, a: float = 0.5):
        self.radius = radius          # радиус сферического детектора, м
        self.pmt_number = pmt_number  # количество фотоумножителей на внутренней поверхности сферического детектора, штук
        self.pmt_radius = pmt_radius
        self.a = a                    # Bероятность выбивания хотя бы одного фотоэлектрона
        # фиксируем координату центра сферы
        self.center = np.array([0.0,0.0,0.0])
        self.pmt_centers, self.pmt_normals, self.pmt_phi_sph, self.pmt_theta_sph = self.place_pmts_on_sphere(pmt_number, radius)

    # функция равномерного распределения плоских круглых фотокатодов фотоумножителей радиуса r 
    # по внутренней поверхности сферы с радиусом R в Евклидовой системе отсчета (декартовы координаты)
    def place_pmts_on_sphere(self, N: int, R: float):                  # N - количество фотоумножителей, R - радиус сферического детектора
        """
        Размещает фотоумножители указанного диаметра и в указанном количестве
        под поверхностью сферического детектора
        
        Параметры:
        N               : количество фотоумножителей, штук
        R               : радиус одного фотоумножителя, м

        Возвращает:

        centers  : декартовы координаты (x, y, z) центров размещенных фотоумножителей
        normals  : минус единичные вектора центров (нормали) размещенных фотоумножителей
        theta    : полярные углы центров размщенных фотоумножитей относительно центра сферического детектора, радиан
        phi_sph  : азимуты от нулевого меридиана сферического детектора, радиан
        

        Пример:
            >>> det = Detector(radius=2.0, pmt_number=2000, pmt_radius=0.1)
            >>> overlaps, min_gap = det.check_overlaps()
            >>> len(overlaps)
            1284
            >>> min_gap < 0
            True
        """
        # вводим индексы 0..N-1 сдвигаем на 0.5 → симметрия относительно полюсов    
        i = np.arange(N, dtype=float) + 0.5
        # z-координата: линейно от ≈+1 до ≈−1 (равномерно по площади единичной сферы, теорема Архимеда)
        z = 1.0 - 2.0 * i / N
        # радиус проекции точки на плоскость xy: ρ = √(1−z²); max(0,·) защищает от отрицательного под корнем
        rho = np.sqrt(np.maximum(0.0, 1.0 - z * z))
        # «золотой угол» ≈ 2.39996 рад — обеспечивает отсутствие спиральных артефактов
        golden_angle = np.pi * (3.0 - np.sqrt(5.0))
        # азимутальный угол: каждый следующий фотоумножитель повёрнут на золотой угол
        phi = i * golden_angle
        # декартова x на единичной сфере: x = ρ·cos(φ)
        x = rho * np.cos(phi)
        # декартова y на единичной сфере: x = ρ·cos(φ)
        y = rho * np.sin(phi)
        # собираем (N,3)-массив и масштабируем на радиус R → координаты центров ФЭУ
        centers = R * np.column_stack([x, y, z])
        # нормаль внутрь сферы: минус единичный вектор центра
        normals = -centers / np.linalg.norm(centers, axis=1, keepdims=True)
        # полярный угол θ = arccos(z); clip защищает от |z|>1 из-за погрешностей округления
        theta = np.arccos(np.clip(z, -1.0, 1.0))
        # азимут φ в диапазоне [0, 2π): arctan2 даёт (−π,π] (по мсодулю)
        phi_sph = np.mod(np.arctan2(y, x), 2.0 * np.pi)
        #возвращается массив фотоумножителей с координатами, углами поворота, 
        #нормалями, азимутами относительно внутренней поверхности сферы
        return centers, normals, phi_sph, theta

    def check_overlaps(self, tol: float = 1e-9, raise_on_overlap: bool = False):
        """
        Проверяет, что плоские круглые фотокатоды радиуса self.pmt_radius
        не перекрываются на внутренней поверхности сферы радиуса self.radius.
        
        Два круга перекрываются, если геодезическое расстояние между их центрами
        по поверхности сферы меньше 2·r:
        d_geo(i, j) = R · arccos(u_i · u_j) < 2r,
        где u_i = C_i / |C_i| — единичные направления центров.
        
        Параметры:
        
        tol             : допуск (м) для численных сравнений;
                          перекрытием считается d_geo < 2r − tol
        raise_on_overlap: если True и найдены перекрытия — бросает ValueError
                          с диагностическим сообщением и рекомендацией по N

        Возвращает:

        overlaps : ndarray (M, 2) — пары индексов (i, j), i < j,
                   чьи фотокатоды пересекаются
        min_gap  : float — минимальный зазор между краями соседних кругов
                   по всем парам (отрицательный, если есть перекрытие)

        Пример:
            >>> det = Detector(radius=2.0, pmt_number=2000, pmt_radius=0.1)
            >>> overlaps, min_gap = det.check_overlaps()
            >>> len(overlaps)
            1284
            >>> min_gap < 0
            True
        """
        # 1) единичные направления центров ФЭУ: u_i = C_i / |C_i|
        u = self.pmt_centers / np.linalg.norm(self.pmt_centers, axis=1, keepdims=True)
        # 2) матрица косинусов углов между всеми парами: cos γ_ij = u_i · u_j
        #    clip защищает arccos от |cos| > 1 из-за погрешностей округления
        cos_gamma = np.clip(u @ u.T, -1.0, 1.0)
        # 3) геодезические расстояния по поверхности сферы: d_ij = R · γ_ij
        geo_dist = self.radius * np.arccos(cos_gamma)
        # 4) порог перекрытия: круги пересекаются, если d_ij < 2r − tol
        threshold = 2.0 * self.pmt_radius
        mask = geo_dist < threshold - tol
        # 5) убираем диагональ (сам с собой) и дубли (i,j)/(j,i)
        np.fill_diagonal(mask, False)
        mask = np.triu(mask, k=1)
        # 6) список перекрывающихся пар
        overlaps = np.argwhere(mask)                     # (M, 2), i < j
        # 7) минимальный зазор по всем парам (верхний треугольник, без диагонали)
        upper = geo_dist[np.triu_indices_from(geo_dist, k=1)]
        min_gap = float(upper.min() - threshold) if upper.size else float('inf')
        # 8) реакция на перекрытия
        if overlaps.size and raise_on_overlap:
            i, j = overlaps[0]
            raise ValueError(
                f"Обнаружено {len(overlaps)} перекрытий фотокатодов "
                f"(r = {self.pmt_radius} м, R = {self.radius} м).\n"
                f"Первая пара: ФЭУ #{i} и #{j}, "
                f"геодезическое расстояние = {geo_dist[i, j]:.4f} м, "
                f"требуется ≥ {threshold:.4f} м.\n"
                f"Минимальный зазор по всем парам: {min_gap:+.4f} м.\n"
                f"Рекомендация: уменьшить N до ≈ {self.pmt_number*0.95} "
                f"или уменьшить r."
           )
        return overlaps, min_gap

    # визуализация распределения центров фотокатодов по поверхности детектора
    def plot_pmt_centres(self):
        fig = plt.figure(figsize=(7, 7))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(self.pmt_centers[:, 0], self.pmt_centers[:, 1], self.pmt_centers[:, 2],
            s=20, c=self.pmt_theta_sph, cmap='viridis')
        ax.set_box_aspect([1, 1, 1])
        ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
        ax.set_title(f'Равномерное распределение {det.pmt_number} ФЭУ по сфере R={det.radius} м')
        plt.tight_layout()
        plt.show()
