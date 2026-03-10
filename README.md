## Measuring Box Size Using a Monocular Camera (Webcam)
The goal of the project was to determine how accurately can the size of a box be predicted using only a standard camera that provides a 2D image, and a combination of two models: YOLO for object detection and DepthAnythingV2 Metric for depth estimation.
## How to Run
All prototype logic is located in main.py.

For accurate size estimation, you need to specify the camera’s focal length (FOCAL_LENGTH) and sensor size (SENSOR_SIZE).
## Methodology
1. Capture a frame from the camera
2. Detect objects using YOLO
3. If boxes are detected, estimate scene depth using DepthAnything
4. Compute the real dimensions of the bounding box based on the median distance to the points inside the bounding box, the camera focal length and the camera sensor size
## Size Calculation
![photo_2026-02-10_02-30-29](https://github.com/user-attachments/assets/3b5b7ce6-204b-4b48-afce-83259d365916)

y — object size

d — distance to the object

h — size of the object’s projection on the sensor (calculated using the sensor dimensions and the bounding box dimensions in the sensor image)

a — camera focal length

$y = (h * d) / a$
## Results
![2026-02-1002-48-49-ezgif com-video-to-gif-converter](https://github.com/user-attachments/assets/1146b803-efd1-4951-944b-421c1f05a9d1)

It was possible to achieve a size estimation error not exceeding ±2 cm.
## What`s next
- The box detection isn`t too accurate. It would be beneficial to retrain YOLO specifically for box detection, or train a custom single-class detector.
- Try 3D detectors that provide 3D bounding boxes, which would allow estimation of the actual box dimensions rather than just the 2D bounding box size.
- Combine detection and depth estimation functionality into a single model with a custom architecture to improve overall accuracy.

--------------------------------------------------------------------

## Измерение размера коробок с помощью монокамеры (веб-камеры)
Целью проекта было желание узнать, насколько точно можно предсказывать размер коробки, используя только обычную камеру, дающую 2D картинку, и комбинацию из двух моделей: YOLO для детекции объектов и DepthAnythingV2 Metric для предсказывания расстояния.
### Как запускать
Вся логика прототипа находится в main.py.

Для точного определения размера нужно указать фокальное расстояние (FOCAL_LENGTH) и размер сенсора (SENSOR_SIZE) камеры.
### Методология
1. Берём кадр с камеры
2. Ищем объекты с помощью YOLO
3. Если найдены коробки, то оцениваем глубину сцены с помощью DepthAnything
4. Рассчитываем реальные размеры bounding box на основе медианного растояния до точек внутри bounding box, фокального расстояния камеры и размеров сенсора камеры
### Расчёт размера
![photo_2026-02-10_02-30-29](https://github.com/user-attachments/assets/3b5b7ce6-204b-4b48-afce-83259d365916)

y - Размер объекта

d - Расстояние до объекта

h - Размер проекции объекта на сенсоре (рассчитывается с помощью размеров сенсора и размеров bounding box на картинке с сенсора)

a - Фокальное расстояние камеры

$y = (h * d) / a$
### Результаты
![2026-02-1002-48-49-ezgif com-video-to-gif-converter](https://github.com/user-attachments/assets/1146b803-efd1-4951-944b-421c1f05a9d1)

Удалось достигнуть погрешности в оценке размеров, не превышающей +/- 2см.
### Что можно улучшить
- Есть проблемы с детекцией коробок, стоит перебучить YOLO исключительно на коробки, либо обучить собственный детектор на один класс
- Попробовать 3D детекторы, дающие 3D bounding box, чтобы можно было оценивать конкретно размеры коробки, а не размеры bounding box
- Совместить функционал детекции и оценки глубины сцены в одной модели с кастомной архитектурой для повышения точности
