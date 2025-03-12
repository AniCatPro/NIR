import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from scipy.stats import pearsonr
from IPython.display import display, Markdown

# Функция для извлечения данных из базы данных
def get_data_from_db(db_name="real.db"):
    connection = sqlite3.connect(db_name)
    query = "SELECT price, total_area AS square, floor, distance FROM properties"
    df = pd.read_sql(query, connection)
    connection.close()
    return df

# Получаем данные
df = get_data_from_db()

# Логарифмическое преобразование переменной distance
df['log_distance'] = np.log1p(df['distance'])

# Информация о данных
df.info()

# Первые и последние строки
print(df.head())
print(df.tail())

# Описательная статистика
print(df.describe())

# Среднее для указанных столбцов
print(df[['price', 'floor']].mean())

# Медиана для указанных столбцов
print(df[['price', 'floor']].median())

# Стандартное отклонение для цены
print(df['price'].std())

# Гистограмма распределения цен
df['price'].hist(bins=20, figsize=(10, 6))
plt.xlabel('в тыс.р.')
plt.ylabel('Частота')
plt.title('Распределение цен на квартиры')
plt.show()

# Диаграмма размаха цен
df['price'].plot(kind='box', figsize=(8, 6))
plt.title('Диаграмма размаха цен на квартиры')
plt.show()

# Шесть диаграмм рассеяния
plt.figure(figsize=(14, 10))

plt.subplot(3, 2, 1)
sns.scatterplot(x='square', y='price', data=df)
plt.xlabel('Площадь')
plt.ylabel('Цена')
plt.title('Цена и Площадь')

plt.subplot(3, 2, 2)
sns.scatterplot(x='floor', y='price', data=df)
plt.xlabel('Этаж')
plt.ylabel('Цена')
plt.title('Цена и Этаж')

plt.subplot(3, 2, 3)
sns.scatterplot(x='log_distance', y='price', data=df)
plt.xlabel('Логарифм расстояния')
plt.ylabel('Цена')
plt.title('Цена и Логарифм расстояния')

plt.subplot(3, 2, 4)
sns.scatterplot(x='floor', y='square', data=df)
plt.xlabel('Этаж')
plt.ylabel('Площадь')
plt.title('Площадь и Этаж')

plt.subplot(3, 2, 5)
sns.scatterplot(x='log_distance', y='square', data=df)
plt.xlabel('Логарифм расстояния')
plt.ylabel('Площадь')
plt.title('Площадь и Логарифм расстояния')

plt.subplot(3, 2, 6)
sns.scatterplot(x='log_distance', y='floor', data=df)
plt.xlabel('Логарифм расстояния')
plt.ylabel('Этаж')
plt.title('Этаж и Логарифм расстояния')

plt.tight_layout()
plt.show()

# Анализ статистической значимости корреляции
features = ['square', 'floor', 'log_distance']
for feature in features:
    corr, p_value = pearsonr(df['price'], df[feature])
    print(f"Корреляция (цена и {feature}): {corr:.2f}, p-значение: {p_value:.4f}")
    if p_value < 0.05:
        print("Значимая зависимость на уровне 5%.")
    else:
        print("Нет значимой зависимости на уровне 5%.")

# Матрица корреляций
corr_matrix = df[['price', 'floor', 'log_distance']].corr()
plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=True, cmap='YlOrRd')
plt.title('Матрица корреляций')
plt.show()

# Определяем y
y = df['price']

# Используем Model 1 для анализа: [square, floor]
X = df[['square', 'floor']]
X_const = sm.add_constant(X)

# Обучаем Model 1
model = sm.OLS(y, X_const).fit()
print(model.summary())

# Гистограмма остатков для Model 1
residuals = y - model.predict(X_const)
plt.figure(figsize=(8, 6))
plt.hist(residuals, bins=20)
plt.title('Гистограмма распределения ошибок (Model 1)')
plt.xlabel('Ошибка')
plt.ylabel('Частота')
plt.grid(True)
plt.show()

# Визуализация прогнозных и фактических значений
plt.figure(figsize=(8, 6))
plt.scatter(y, model.predict(X_const))
plt.plot([min(y), max(y)], [min(y), max(y)], 'r--')
plt.title('Прогнозные vs. Фактические значения (Model 1)')
plt.xlabel('Фактические значения')
plt.ylabel('Прогнозные значения')
plt.grid(True)
plt.show()

# Полиномиальные признаки (если нужно)
poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(df[['square', 'floor', 'log_distance']])
model_poly = LinearRegression()
model_poly.fit(X_poly, y)

# Оценка с использованием полиномиальных признаков
X_const_poly = sm.add_constant(X_poly)
results_poly = sm.OLS(y, X_const_poly).fit()

# DataFrame с результатами
results_df = pd.DataFrame({
    'Фактор': ['Свободный член'] + list(results_poly.params.index[1:]),
    'Коэффициент': results_poly.params.values,
    'Стандартная ошибка': results_poly.bse.values,
    't-статистика': results_poly.tvalues.values,
    'p-value': results_poly.pvalues.values
})

# Уравнение регрессии
equation = f'y = {results_poly.params.iloc[0]:.2f}'
for i, coef in enumerate(results_poly.params.iloc[1:]):
    factor_name = results_df.loc[i+1, 'Фактор']
    equation += f' + {coef:.2f} * {factor_name}'
display(Markdown(f'**Уравнение регрессии:**\n{equation}'))

# Вывод результатов
print(results_df)
for i, p_value in enumerate(results_df['p-value'][1:]):
    factor_name = results_df.loc[i+1, 'Фактор']
    if p_value > 0.05:
        print(f"Коэффициент при {factor_name} является статистически незначимым при 5% уровне значимости.")