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

