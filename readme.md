## Структура проекта
Проект состоит из следующих модулей:
detector - класс Detector и его методы
source - класс Source и его методы (задание 2)
coordinates - функции преобразования систем отсчета
visualize - визуализация в matplotlib и k3d

## Ноутбуки
Detector.ipynb - содержит результаты выполненного задания, 
импортирует необходимые библиотеки, в том числе detector, coordinates, visualize,
разработанные в рамках задания
Source.ipynb - содержит результаты выполненного задания 2: класс Source
(Y = 10000 фотонов/МэВ) и метод розыгрыша начальных направлений фотонов
для события с энергией E, число фотонов N = Y·E (линейный выход по энергии)

## Окружение
Рабочих интерпретаторов два, в обоих есть numpy, matplotlib, k3d, ipywidgets, pytest.

venv_dev - проектное окружение; версии зафиксированы в requirements.txt (pip freeze).
Запуск: source venv_dev/bin/activate, затем venv_dev/bin/jupyter lab.
Ноутбуки используют ядро "Python 3 (phiz_m)" -> venv_dev/bin/python -m ipykernel_launcher
(kernelspec зарегистрирован командой venv_dev/bin/python -m ipykernel install --user
--name phiz_m --display-name "Python 3 (phiz_m)").

системный python3 - пользовательское окружение (~/.local), запуск: jupyter lab,
ядро "python3".

Конфликт версий matplotlib устранён: системный пакет Debian python3-matplotlib 3.5.1
содержит файл /usr/lib/python3/dist-packages/matplotlib-3.5.1-nspkg.pth, который при
старте интерпретатора подменяет пакет mpl_toolkits устаревшей копией. Вместе с
matplotlib 3.10+ из pip это давало ImportError (matplotlib.docstring) и неработающую
проекцию "3d" в ноутбуке. Исправлено на уровне пользовательского окружения:
~/.local/lib/python3.10/site-packages/mpl_toolkits/__init__.py - каталог объявлен
обычным пакетом, поэтому pip-версия побеждает по порядку sys.path;
~/.local/lib/python3.10/site-packages/usercustomize.py - отменяет подмену из .pth.
После правки уже запущенные ядра JupyterLab нужно перезапустить (Kernel -> Restart),
иначе они продолжают работать со старым окружением в памяти.

## Тесты
Автотесты написаны на pytest, каждый автотест оформлен отдельным файлом
в каталоге tests/:
test_detector_pmts_on_sphere.py - центры ФЭУ лежат на сфере
test_detector_uniformity.py - равномерность распределения ФЭУ по поверхности
test_detector_neighbour_distance.py - шаг между соседними ФЭУ
test_detector_overlaps.py - перекрытия плоских фотокатодов
test_detector_overlaps_raise.py - режим raise_on_overlap
test_detector_point_in_pmt.py - принадлежность точки фотоумножителю
test_coordinates_conversion.py - согласованность преобразований координат
test_coordinates_roundtrip.py - обратное преобразование координат
test_source_default_y.py - световыход Y = 10000 фотонов/МэВ
test_source_photon_number.py - линейный выход фотонов N = Y·E
test_source_directions.py - форма массива и единичная длина направлений
test_source_isotropy.py - изотропность излучения
test_source_angles.py - сферические углы события
test_source_reproducibility.py - воспроизводимость по seed
test_source_validation.py - проверка корректности входных параметров

Общие фикстуры вынесены в tests/conftest.py (pytest подхватывает файл
автоматически для всего каталога tests/):
detector - детектор задания 1 (R = 2 м, N = 40, r = 0.1 м, a = 0.5)
detector_dense - детектор с 950 ФЭУ (фотокатоды почти касаются)
detector_packed - детектор с 2000 ФЭУ (фотокатоды перекрываются)
source - источник света Y = 10000 фотонов/МэВ с фиксированным seed
event_directions - направления фотонов события с энергией 1 МэВ
rng - генератор случайных чисел с фиксированным seed
make_sphere_points - фабрика случайных точек на поверхности детектора
chi2_crit_999_dof9 - критическое значение χ² (9 степеней свободы, 0.001)

Запуск из корня проекта (окружение venv_dev):
pytest tests
pytest tests -v
pytest tests --doctest-modules detector.py source.py coordinates.py
