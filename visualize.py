import numpy as np
import matplotlib.pyplot as plt
import k3d
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# Визуализация: matplotlib

def plot_detector(det, show_sphere=True, show_normals=False,
                  highlight_overlaps=True, figsize=(10, 10), dpi=100):
    """
    Рисует сферический детектор и фотокатоды ФЭУ как закрашенные круги,
    ориентированные по нормали к сфере.

    Параметры:
        det                : объект Detector с атрибутами
                             radius, pmt_number, pmt_radius, pmt_centers
        show_sphere        : рисовать каркасную сферу
        show_normals       : рисовать стрелки нормалей (к центру)
        highlight_overlaps : подкрашивать красным круги, участвующие в перекрытиях
        figsize, dpi       : параметры фигуры matplotlib

    Возвращает:
        fig, ax : объекты matplotlib
    """
    # --- 1) достаём параметры из объекта ---
    R = det.radius
    N = det.pmt_number
    r = det.pmt_radius
    centers = det.pmt_centers

    # --- 2) ищем перекрытия (для подсветки и статуса в заголовке) ---
    if highlight_overlaps:
        overlaps, min_gap = det.check_overlaps()
        overlap_idx = set(overlaps.flatten().tolist()) if overlaps.size else set()
    else:
        overlaps, min_gap = np.empty((0, 2), dtype=int), float('nan')
        overlap_idx = set()

    # --- 3) фигура и 3D-оси ---
    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = fig.add_subplot(111, projection='3d')

    # --- 4) каркасная сфера ---
    if show_sphere:
        uu = np.linspace(0, 2 * np.pi, 48)
        vv = np.linspace(0, np.pi, 24)
        xs = R * np.outer(np.cos(uu), np.sin(vv))
        ys = R * np.outer(np.sin(uu), np.sin(vv))
        zs = R * np.outer(np.ones_like(uu), np.cos(vv))
        ax.plot_wireframe(xs, ys, zs, color='lightgray', linewidth=0.4, alpha=0.5)

    # --- 5) единичные направления центров ---
    u = centers / np.linalg.norm(centers, axis=1, keepdims=True)

    # --- 6) подготовка окружностей ---
    n_pts = 32
    t = np.linspace(0.0, 2.0 * np.pi, n_pts)
    base_colors = plt.cm.viridis(np.linspace(0, 1, N))

    # --- 7) рисуем каждый фотокатод ---
    for k in range(N):
        C = centers[k]                          # центр круга
        n = u[k]                                # нормаль сферы в этой точке

        # локальный базис в касательной плоскости
        ref = np.array([0.0, 0.0, 1.0]) if abs(n[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
        e1 = np.cross(n, ref);  e1 /= np.linalg.norm(e1)
        e2 = np.cross(n, e1)

        # точки окружности радиуса r
        circle = C[None, :] + r * (
            np.cos(t)[:, None] * e1 + np.sin(t)[:, None] * e2
        )

        # цвет: красный, если участвует в перекрытии
        color = 'crimson' if k in overlap_idx else base_colors[k]

        # закрашенный многоугольник
        poly = Poly3DCollection(
            [list(zip(circle[:, 0], circle[:, 1], circle[:, 2]))],
            facecolor=color, edgecolor='k', linewidth=0.2, alpha=0.85
        )
        ax.add_collection3d(poly)

        # стрелка нормали (внутрь сферы)
        if show_normals:
            ax.quiver(C[0], C[1], C[2], -n[0], -n[1], -n[2],
                      length=0.3 * R, color='black',
                      linewidth=0.6, arrow_length_ratio=0.15)

    # --- 8) оси координат ---
    L = 1.3 * R
    ax.plot([0, L], [0, 0], [0, 0], 'r-', lw=1.2, alpha=0.7)
    ax.plot([0, 0], [0, L], [0, 0], 'g-', lw=1.2, alpha=0.7)
    ax.plot([0, 0], [0, 0], [0, L], 'b-', lw=1.2, alpha=0.7)
    ax.text(L, 0, 0, 'X'); ax.text(0, L, 0, 'Y'); ax.text(0, 0, L, 'Z')

    # --- 9) масштаб, подписи, заголовок ---
    ax.set_box_aspect([1, 1, 1])
    lim = 1.1 * R
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_zlim(-lim, lim)
    ax.set_xlabel('X, м'); ax.set_ylabel('Y, м'); ax.set_zlabel('Z, м')

    status = "OK" if not overlaps.size else f"ПЕРЕКРЫТИЙ: {len(overlaps)}"
    gap_str = f"{min_gap:+.3f} м" if np.isfinite(min_gap) else "—"
    ax.set_title(
        f"Детектор R = {R} м, N = {N}, r = {r} м\n"
        f"Статус: {status}, минимальный зазор = {gap_str}"
    )
    ax.view_init(elev=25, azim=35)
    plt.tight_layout()
    plt.show()
    return fig, ax
    


# Визуализация: k3d
def plot_detector_k3d(det, show_normals=False, sphere_opacity=0.15,
                           n_circle=12, n_theta=16, n_phi=32):
    """
    Быстрая визуализация сферического детектора с ФЭУ в K3D.

    Вся геометрия ФЭУ собирается в ОДИН mesh с вершинным атрибутом цвета,
    что даёт один draw call вместо N.

    Параметры:
        det            : объект Detector (radius, pmt_number, pmt_radius, pmt_centers)
        show_normals   : рисовать нормали одним объектом k3d.line
        sphere_opacity : прозрачность каркасной сферы (0 — скрыть)
        n_circle       : число вершин на окружности диска (12–16 достаточно)
        n_theta, n_phi : разрешение сетки сферы

    Возвращает:
        plot : k3d.Plot
    """
    R = det.radius
    N = det.pmt_number
    r = det.pmt_radius
    centers = det.pmt_centers.astype(np.float32)
    u = (centers / np.linalg.norm(centers, axis=1, keepdims=True)).astype(np.float32)

    plot = k3d.plot()

    # ============================================================
    # 1. СФЕРА — один mesh
    # ============================================================
    if sphere_opacity > 0:
        theta = np.linspace(0, np.pi, n_theta, dtype=np.float32)
        phi = np.linspace(0, 2 * np.pi, n_phi, dtype=np.float32)
        TH, PH = np.meshgrid(theta, phi, indexing='ij')

        sv = np.stack([
            R * np.sin(TH) * np.cos(PH),
            R * np.sin(TH) * np.sin(PH),
            R * np.cos(TH),
        ], axis=-1).reshape(-1, 3).astype(np.float32)

        # индексы треугольников
        i_idx = np.arange(n_theta - 1)[:, None]
        j_idx = np.arange(n_phi - 1)[None, :]
        a = (i_idx * n_phi + j_idx).ravel()
        b = a + 1
        c = a + n_phi
        d = c + 1
        si = np.concatenate([
            np.stack([a, b, d], axis=1),
            np.stack([a, d, c], axis=1),
        ]).astype(np.uint32)

        plot += k3d.mesh(sv, si, color=0x888888, wireframe=True,
                         opacity=sphere_opacity, side='double')

    # ============================================================
    # 2. ВСЕ ФЭУ — ОДИН mesh с вершинным цветом
    # ============================================================
    # Локальный базис для каждого ФЭУ (векторизованно)
    ref = np.where(np.abs(u[:, 2:3]) < 0.9,
                   np.array([0.0, 0.0, 1.0], dtype=np.float32),
                   np.array([1.0, 0.0, 0.0], dtype=np.float32))
    e1 = np.cross(u, ref)
    e1 /= np.linalg.norm(e1, axis=1, keepdims=True)
    e2 = np.cross(u, e1)

    # Точки окружности: (N, n_circle, 3)
    t = np.linspace(0, 2 * np.pi, n_circle, endpoint=False, dtype=np.float32)
    cos_t = np.cos(t)[None, :, None]        # (1, n_circle, 1)
    sin_t = np.sin(t)[None, :, None]
    # circle[k, j] = C[k] + r*(cos_t[j]*e1[k] + sin_t[j]*e2[k])
    circle = (centers[:, None, :]
              + r * (cos_t * e1[:, None, :] + sin_t * e2[:, None, :]))  # (N, n_circle, 3)

    # Вершины: центр + n_circle точек → (N*(1+n_circle), 3)
    disk_verts = np.concatenate([
        centers[:, None, :],                 # (N, 1, 3) — центры
        circle,                              # (N, n_circle, 3)
    ], axis=1).reshape(-1, 3).astype(np.float32)

    # Индексы: для каждого диска — веер из n_circle треугольников
    # база для диска k: k*(n_circle+1); центр = база, точки = база+1..база+n_circle
    k = np.arange(N, dtype=np.uint32)[:, None]
    base = (k * (n_circle + 1)).astype(np.uint32)          # (N, 1)
    j = np.arange(n_circle, dtype=np.uint32)[None, :]      # (1, n_circle)
    v1 = base + 1 + j
    v2 = base + 1 + (j + 1) % n_circle
    tri = np.stack([np.broadcast_to(base, v1.shape), v1, v2], axis=-1)  # (N, n_circle, 3)
    disk_indices = tri.reshape(-1, 3).astype(np.uint32)

    # Цвета вершин: каждому диску — свой оттенок (палитра viridis через k3d-атрибут)
    # k3d ждёт цвета в формате uint32 0xRRGGBB
    hue = np.linspace(0.0, 1.0, N, dtype=np.float32)
    # простая HSV→RGB палитра без matplotlib для скорости
    h = hue * 6.0
    i_h = np.floor(h).astype(int) % 6
    f = h - np.floor(h)
    p, q, v = 1 - f, 1 - f, np.ones_like(f)       # упрощённо, для N больших достаточно
    # используем фиксированную приятную палитру: от синего к красному
    rr = np.clip(1.5 - np.abs(hue * 3.0 - 2.0), 0, 1)
    gg = np.clip(1.5 - np.abs(hue * 3.0 - 1.0), 0, 1)
    bb = np.clip(1.5 - np.abs(hue * 3.0 - 0.0), 0, 1)
    rgb = np.stack([rr, gg, bb], axis=1)                     # (N, 3) в [0,1]
    colors_per_pmt = (rgb * 255).astype(np.uint32)           # (N, 3)
    color_u32 = ((colors_per_pmt[:, 0] << 16)
                 | (colors_per_pmt[:, 1] << 8)
                 | colors_per_pmt[:, 2]).astype(np.uint32)   # (N,)

    # повторить цвет на (1 + n_circle) вершин каждого диска
    vert_colors = np.repeat(color_u32, n_circle + 1)         # (N*(n_circle+1),)

    plot += k3d.mesh(
        disk_verts, disk_indices,
        colors=vert_colors,       # вершинный атрибут цвета
        opacity=0.85,
        side='double',
        wireframe=False,
    )

    # ============================================================
    # 3. НОРМАЛИ — один объект k3d.line
    # ============================================================
    if show_normals:
        p_inner = centers - 0.3 * R * u                      # (N, 3)
        # пары вершин: [C_0, p_0, C_1, p_1, ...]
        line_verts = np.empty((2 * N, 3), dtype=np.float32)
        line_verts[0::2] = centers
        line_verts[1::2] = p_inner
        # индексы: для каждой пары — отрезок (2k, 2k+1)
        seg = np.arange(N, dtype=np.uint32)[:, None]
        line_indices = np.stack([2 * seg, 2 * seg + 1], axis=-1).reshape(-1, 2).astype(np.uint32)

        plot += k3d.line(line_verts, line_indices, color=0xFF0000, width=0.5)

    # ============================================================
    # 4. Камера
    # ============================================================
    plot.camera = [R * 2.5, R * 2.5, R * 2.5, 0, 0, 0, 0, 0, 1]
    plot.display()
    return plot    

