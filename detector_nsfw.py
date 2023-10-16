# Встроенный модуль, необходим для получения списка файлов
import os
# Встроенный модуль, необходим для копирования файлов
from shutil import copyfile
# Собственно классификатор который мы установили
from nudenet import NudeClassifier

# Название папки с фото
FOLDER_NAME = 'dump_1'
# Порог для классификатора (от 0 до 1)
THRESHOLD = 0.3

# Создаём объект классификатора
classifier = NudeClassifier()
# Получаем Список файлов из указанной выше папки
arr = [f'./{FOLDER_NAME}/{f}' for f in os.listdir(f'./{FOLDER_NAME}/')]
# Счётчики нюдсов/простых фото
nudes = 0
plain = 0

# Перебираем файлы, попутно получая результат классификатора
for file, result in classifier.classify(arr).items():
    # Если уверенность нейросети в том что фото - нюдс > порогового значения
    if result['unsafe'] > THRESHOLD:
        # Увеличиваем счётчик нюдсов
        nudes += 1
        # Копируем файл в папку nudes
        copyfile(file, f'./nudes/{file.split("/")[-1]}')
    # Если значение нейросети < порогового значения
    else:
        # Увеличиваем счётчик простых фото
        plain += 1
        # Копируем файл в папку plain
        copyfile(file, f'./plain/{file.split("/")[-1]}')
    print(f"Nudes: {nudes}")
    print(f"Plain: {plain}")

# Выводим результаты
print(f'[Порог {THRESHOLD}] Сортировка завершена. Найдено: {nudes} нюдсов | {plain} обычных')
# Не даём скрипту мгновенно закрыться
input()