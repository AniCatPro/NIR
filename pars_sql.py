#Afanasev P.Y.
#26.02.2025

#Код парсит данные с Avito в SQLite

import sqlite3
import random
import time
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import StaleElementReferenceException


def safe_float(value, default=0.0):
    try:
        return float(value.replace('\xa0', '').replace(',', '.')) if value else default
    except ValueError:
        return default


def safe_int(value, default=0):
    try:
        return int(value) if value else default
    except ValueError:
        return default


def create_database(db_name="real_estate.db"):
    connection = sqlite3.connect(db_name)
    cursor = connection.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS properties (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            price REAL,
            total_area REAL,
            living_area REAL,
            rooms INTEGER,
            floor INTEGER,
            total_floors INTEGER,
            year INTEGER,
            distance REAL,
            balcony TEXT,
            condition TEXT,
            type TEXT,
            district TEXT,
            url TEXT UNIQUE
        )
    ''')
    connection.commit()
    connection.close()


def parse_ad(driver, url):
    driver.get(url)
    time.sleep(3)  # Даем время для загрузки страницы
    try:
        price = driver.find_element(By.CSS_SELECTOR, 'span[itemprop="price"]').get_attribute("content")
    except:
        price = "0"
    try:
        total_area = \
        driver.find_element(By.XPATH, '//li[span[contains(text(), "Общая площадь")]]').text.split(": ")[1].split(" ")[0]
    except:
        total_area = "0"
    try:
        living_area = \
        driver.find_element(By.XPATH, '//li[span[contains(text(), "Жилая площадь")]]').text.split(": ")[1].split(" ")[0]
    except:
        living_area = "0"
    try:
        rooms = driver.find_element(By.XPATH, '//li[span[contains(text(), "Количество комнат")]]').text.split(": ")[1]
    except:
        rooms = "0"
    try:
        floor_info = driver.find_element(By.XPATH, '//li[span[contains(text(), "Этаж")]]').text.split(": ")[1]
        floor, total_floors = map(int, floor_info.split(" из "))
    except:
        floor, total_floors = 0, 0
    try:
        year = driver.find_element(By.XPATH, '//li[span[contains(text(), "Год постройки")]]').text.split(": ")[1]
    except:
        year = "1970"
    try:
        balcony = driver.find_element(By.XPATH, '//li[span[contains(text(), "Балкон")]]').text.split(": ")[1]
    except:
        balcony = "Нет"
    try:
        condition = driver.find_element(By.XPATH, '//li[span[contains(text(), "Ремонт")]]').text.split(": ")[1]
    except:
        condition = "Неизвестно"
    try:
        title = driver.find_element(By.CSS_SELECTOR, 'h1[data-marker="item-view/title-info"]').text
        if ',' in title:
            type_ = title.split(",")[0].split(".")[-1].strip()
        else:
            type_ = "Неизвестно"
    except:
        type_ = "Неизвестно"
    try:
        district = driver.find_element(By.CSS_SELECTOR,
                                       'span.style-item-address-georeferences-item-TZsrp').text.replace("р-н ", "")
    except:
        district = "Неизвестно"

    print(
        f"DEBUG: {price}, {total_area}, {living_area}, {rooms}, {floor}, {total_floors}, {year}, {balcony}, {condition}, {type_}, {district}, {url}")
    return (safe_float(price), safe_float(total_area), safe_float(living_area), safe_int(rooms), floor, total_floors,
            safe_int(year), 0, balcony, condition, type_, district, url)


def save_to_db(data, db_name="real_estate.db"):
    connection = sqlite3.connect(db_name)
    cursor = connection.cursor()
    try:
        cursor.execute('''
            INSERT INTO properties (price, total_area, living_area, rooms, floor, total_floors, year, distance, balcony, condition, type, district, url)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', data)
        connection.commit()
    except sqlite3.IntegrityError:
        print("Объявление уже в базе:", data[-1])
    connection.close()


def main():
    base_url = "https://www.avito.ru/barnaul/kvartiry/prodam-ASgBAgICAUSSA8YQ?p="
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()))

    # Определяем общее количество страниц
    driver.get(base_url + "1")
    time.sleep(2)

    try:
        total_pages = int(driver.find_element(By.XPATH, '//li[a[@data-marker="pagination-button/page(100)"]]').text)
    except:
        total_pages = 10
    print(f"Всего страниц: {total_pages}")

    pages_parsed = 0  # Используем для подсчета страниц
    max_pages = 15  # Установите количество страниц, которое хотите обработать

    while pages_parsed < max_pages:
        random_page = random.randint(1, total_pages)
        print(f"Открываем страницу {random_page}...")
        driver.get(base_url + str(random_page))
        time.sleep(2)

        try:
            ads = WebDriverWait(driver, 10).until(
                EC.presence_of_all_elements_located((By.CSS_SELECTOR, 'a[data-marker="item-title"]'))
            )
        except:
            print("На странице нет объявлений или произошла ошибка загрузки.")
            continue

        # Берем N кол-во случайных объявлений
        random.shuffle(ads)
        ads = ads[:2] if len(ads) >= 2 else ads

        for ad in ads:
            try:
                link = ad.get_attribute("href")
                if not link:
                    continue
                print(f"Парсим {link}...")
                data = parse_ad(driver, link)
                save_to_db(data)
            except StaleElementReferenceException:
                print("Элемент стал недоступен. Пропускаем этот элемент.")

        pages_parsed += 1  # Увеличиваем счетчик после обработки страницы

    driver.quit()


if __name__ == "__main__":
    create_database()
    main()
    print("Готово!")