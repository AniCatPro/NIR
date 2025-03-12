import sqlite3
import csv

def export_sqlite_to_csv(db_path, table_name, csv_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute(f"SELECT * FROM {table_name}")
    rows = cursor.fetchall()

    column_names = [description[0] for description in cursor.description]

    with open(csv_path, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(column_names)
        writer.writerows(rows)

    conn.close()
    print(f"Данные из таблицы '{table_name}' экспортированы в '{csv_path}'")


export_sqlite_to_csv('real.db', 'properties', 'real.csv')
