import sqlite3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression
from IPython.display import display, Markdown

# Подключение к базе данных SQLite
conn = sqlite3.connect('real_estate.db')

# Запрос данных из базы данных (включаем все необходимые столбцы)
query = """
SELECT price, total_area, living_area, rooms, floor, total_floors, year, distance, balcony, condition, type, district, url
FROM properties
"""
df = pd.read_sql(query, conn)

# Общая информация о данных
df.info()
display(df.head(), df.tail())
print(df.describe())

# Вычисление статистик только для числовых столбцов
numeric_columns = df.select_dtypes(include=[np.number])

print("Среднее значение (только для числовых столбцов):")
print(numeric_columns.mean())

print("Медиана (только для числовых столбцов):")
print(numeric_columns.median())

print("Размах выборки (max - min) для 'price':", df['price'].max() - df['price'].min())
print("Квартили для 'price':", df['price'].quantile([0.25, 0.5, 0.75]))

# Графики распределения
plt.figure(figsize=(10, 6))
df['price'].hist(bins=20)
plt.xlabel('Цена в тыс.р.')
plt.ylabel('Частота')
plt.title('Распределение цен на квартиры')
plt.show()

plt.figure(figsize=(8, 6))
df['price'].plot(kind='box')
plt.title('Диаграмма размаха цен на квартиры')
plt.show()

# Диаграмма рассеяния
plt.figure(figsize=(8, 6))
sns.scatterplot(x='total_area', y='price', data=df)
plt.xlabel('Площадь')
plt.ylabel('Цена')
plt.title('Диаграмма рассеяния цены и площади')
plt.show()

# Корреляционный анализ
corr_matrix = df[['price', 'floor', 'distance', 'total_area', 'rooms']].corr()
plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=True, cmap='YlOrRd')
plt.title('Матрица корреляций')
plt.show()

corr, p_value = pearsonr(df['price'], df['floor'])
print(f"Коэффициент корреляции Пирсона: {corr:.2f}")
print(f"p-значение: {p_value:.4f}")
if p_value < 0.05:
    print("Корреляция статистически значима на уровне 5%")
else:
    print("Корреляция не является статистически значимой на уровне 5%")

# Множественная линейная регрессия с добавлением большего числа факторов
X = df[['total_area', 'living_area', 'rooms', 'floor', 'total_floors', 'distance']]  # добавлены новые факторы
y = df['price']
model = LinearRegression()
model.fit(X, y)
X_const = sm.add_constant(X)
results = sm.OLS(y, X_const).fit()

# Вывод уравнения регрессии
equation = f'y = {results.params.iloc[0]:.2f}'
for i, coef in enumerate(results.params.iloc[1:]):
    factor_name = X.columns[i]
    equation += f' + {coef:.2f} * {factor_name}'
display(Markdown(f'**Уравнение регрессии:**\n{equation}'))

# Проверка значимости коэффициентов
results_df = pd.DataFrame({
    'Фактор': ['Свободный член'] + list(X.columns),
    'Коэффициент': results.params.values,
    'Стандартная ошибка': results.bse.values,
    't-статистика': results.tvalues.values,
    'p-value': results.pvalues.values
})
display(results_df)

for i, p_value in enumerate(results_df['p-value'][1:]):
    if p_value > 0.05:
        print(f"Коэффициент при {results_df.loc[i+1, 'Фактор']} статистически незначим")

# Визуализация прогнозов vs. фактических значений
plt.figure(figsize=(8, 6))
plt.scatter(y, model.predict(X))
plt.plot([min(y), max(y)], [min(y), max(y)], 'r--')
plt.xlabel('Фактические значения')
plt.ylabel('Прогнозные значения')
plt.title('Прогнозные vs. Фактические значения')
plt.grid(True)
plt.show()

# Гистограмма ошибок
residuals = y - model.predict(X)
plt.figure(figsize=(8, 6))
plt.hist(residuals, bins=20)
plt.xlabel('Ошибка')
plt.ylabel('Частота')
plt.title('Гистограмма распределения ошибок')
plt.grid(True)
plt.show()

# Оценка модели
r_squared = model.score(X, y)
adjusted_r_squared = 1 - (1 - r_squared) * (len(y) - 1) / (len(y) - X.shape[1] - 1)
print(f"R-squared: {r_squared:.3f}")
print(f"Adjusted R-squared: {adjusted_r_squared:.3f}")