import sqlite3
import random

# Подключение к базе данных
conn = sqlite3.connect('real.db')
cursor = conn.cursor()

# Получаем все ID записей, которые нужно обновить
cursor.execute('SELECT id FROM properties')
rows = cursor.fetchall()

# Обновляем каждую запись случайным значением дистанции
for row in rows:
    # Генерация случайного значения от 50 до 1500
    random_distance = random.randint(50, 1500)

    # Обновляем столбец distance
    cursor.execute('UPDATE properties SET distance = ? WHERE id = ?', (random_distance, row[0]))

# Сохраняем изменения и закрываем соединение
conn.commit()
conn.close()

print("Обновление дистанции завершено.")
