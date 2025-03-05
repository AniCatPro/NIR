import sqlite3
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
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

# Диаграмма рассеяния площади и цены
plt.figure(figsize=(8, 6))
sns.scatterplot(x='square', y='price', data=df)
plt.xlabel('Площадь')
plt.ylabel('Цена')
plt.title('Диаграмма рассеяния цены и площади')
plt.show()

# Матрица корреляций
corr_matrix = df[['price', 'floor', 'log_distance']].corr()
plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=True, cmap='YlOrRd')
plt.title('Матрица корреляций')
plt.show()

# Коэффициент корреляции Пирсона
corr, p_value = pearsonr(df['price'], df['floor'])
print(f"Коэффициент корреляции Пирсона: {corr:.2f}")
print(f"p-значение: {p_value:.4f}")
if p_value < 0.05:
    print("Корреляция статистически значима на уровне 5%")
else:
    print("Корреляция не является статистически значимой на уровне 5%")

# Полиномиальные признаки
poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(df[['square', 'floor', 'log_distance']])
y = df['price']

# Линейная регрессия
model = LinearRegression()
model.fit(X_poly, y)

# Получение значений коэффициентов
X_const = sm.add_constant(X_poly)
results = sm.OLS(y, X_const).fit()

# DataFrame с результатами
results_df = pd.DataFrame({
    'Фактор': ['Свободный член'] + list(results.params.index[1:]),
    'Коэффициент': results.params.values,
    'Стандартная ошибка': results.bse.values,
    't-статистика': results.tvalues.values,
    'p-value': results.pvalues.values
})

# Уравнение регрессии
equation = f'y = {results.params.iloc[0]:.2f}'
for i, coef in enumerate(results.params.iloc[1:]):
    factor_name = results_df.loc[i+1, 'Фактор']
    equation += f' + {coef:.2f} * {factor_name}'
display(Markdown(f'**Уравнение регрессии:**\n{equation}'))

# Вывод результатов
print(results_df)
for i, p_value in enumerate(results_df['p-value'][1:]):
    factor_name = results_df.loc[i+1, 'Фактор']
    if p_value > 0.05:
        print(f"Коэффициент при {factor_name} является статистически незначимым при 5% уровне значимости.")

# Прогнозные и фактические значения
plt.figure(figsize=(8, 6))
plt.scatter(y, model.predict(X_poly))
plt.plot([min(y), max(y)], [min(y), max(y)], 'r--')
plt.title('Прогнозные vs. Фактические значения')
plt.xlabel('Фактические значения')
plt.ylabel('Прогнозные значения')
plt.grid(True)
plt.show()

# Гистограмма ошибок
plt.figure(figsize=(8, 6))
residuals = y - model.predict(X_poly)
plt.hist(residuals, bins=20)
plt.title('Гистограмма распределения ошибок')
plt.xlabel('Ошибка')
plt.ylabel('Частота')
plt.grid(True)
plt.show()

# Метрики модели
r_squared = model.score(X_poly, y)
n = len(y)
p = X_poly.shape[1]
adjusted_r_squared = 1 - (1 - r_squared) * (n - 1) / (n - p - 1)

print(f"R-squared: {r_squared:.3f}")
print(f"Adjusted R-squared: {adjusted_r_squared:.3f}")