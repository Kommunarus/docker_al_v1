# Фреймверк активного обучения в cv

Спасибо, что пользуетесь данным проектом для разметки фотографий. Сейчас реализованы методы классификации.

## Установка

1. Установите [git](https://github.com/git-guides/install-git)
2. Скопируйте проект `git clone https://github.com/Kommunarus/docker_al_v1`
3. Установите [docer-compose](https://docs.docker.com/compose/install/other/)
4. Разверните контейнер `docker-compose up -d`, изменив предварительно путь, где будет датасеты:

   ![image.png](help/image_dc.png)
5. Для тестов установите [Postman](https://www.postman.com/downloads/)

## Требования

Наличие ГПУ на хосте. Алгоритмы обучают нейросети.

## Быстрый старт

После разворачивания контейнера, появится вебсервис по адресу 127.0.0.1:5000

Функция active_learning.

Основная функция, запускающая поиск семплов для разметки.


|   | параметр | описание                                                                                                                                                                                |
| - | ---------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1 | backbone         | эмбединги какой сети будут использоваться для классификации картинки. Возможные значения: b0, resnet50, mobilenet |
| 2 | add              | количество сэмплов, которые нужно найти для разметки                                                                                               |
| 3 | method           | метод активного обучения для поиска сэмплов. Возможные значения: margin, least, ratio, entropy, vae, mixture                             |
| 4 | path_to_labels   | путь до файла/папки с разметками                                                                                                                                     |
| 5 | path_to_img      | путь до папки с картинками                                                                                                                                                |

Пример:

![image.png](help/active_learning.png)

результат:

![image.png](help/res1.png)

path_to_labels может быть как папкой, так и файлом. Файлы должны быть следующего формата: имя_файла\tкласс

125937.jpg	0
178223.jpg	0
064001.jpg	0
136649.jpg	0


***method*** делятся на методы неопределенности и методы разнообразия.

Методы неопределенности:

least, margin, ratio, entropy

![image.png](help/n.png)

Методы разнообразия:

vae — метод на основе вариационного кодировщика. Размеченный датасет обучает сеть восстанавливать картинки без ошибок. Неразмеченные картинки ранжируются по ошибке восстановления. Если картинка восстанавливается с большой ошибкой, значит похожей не было в обучаемом датасете, она сильно не похожа на известный размеченный датасет. Такие картинки кандидаты для разметки.

![image.png](help/вае.png)

mixture — размеченный датасет обучает сиамскую сеть для кодирования эмбедингов картинок (эмбединги размерностью 2) так, что бы разные классы не соприкасались. После, методом кластеризации Gaussian Mixture эмбединги размеченных картинок делятся на кластера и предсказываем класс неразмеченных. Из каждого кластера выбираются сэмплы.

![image.png](help/mix.png)

Методы разнообразия лучше использовать, когда классов не очень много.



Функция f1.

Используется для валидации. Сеть определяется backbone-ом.


|   | параметр     | описание                                                                                                                                                            |
| - | -------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1 | backbone             | тоже , что и в active_learning                                                                                                                                     |
| 2 | path_to_labels_train | путь до файла/папки с разметками для обучения простого классификатора                                           |
| 3 | path_to_img_train    | путь до папки с картинками для обучения простого классификатора                                                      |
| 4 | path_to_labels_val   | путь до файла/папки с разметками, по которой будем определять точность работы классификатора |
| 5 | path_to_img_val      | путь до папки с картинками, по которой будем определять точность работы классификатора            |

Пример:

![image.png](help/p2.png)

Результат:

![image.png](help/resf1.png)

Структура папки ds_for_docker:

![image.png](help/val.png)


## Будущее

Реализовать методы Object detection.
