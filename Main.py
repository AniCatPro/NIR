###     !!!КОД ИЗ МЕТОДИЧКИ!!!
#Код анализирует данные на основе CSV и моделирует зависимость цены
#недвижимости от различных факторов

import numpy as np
from sklearn.linear_model import LinearRegression
import pandas as pd
from scipy.stats import t
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display, Markdown
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor


##6.Расчет описательной статистики и ее визуализация

# Импортируем библиотеку pandas
import pandas as pd

# Считываем данные из файла CSV
df = pd.read_csv('File_name.csv' , sep=';')

# Получаем информацию о количестве строк и столбцов, типах данных в каждом столбце, а также количество непропущенных значений
df.info()

# Выводим первые 5 строк DataFrame
df.head()

# Выводим последние 5 строк DataFrame
df.tail()

# Получаем основную статистическую информацию о числовых столбцах
df.describe()

# Среднее арифметическое для всех столбцов
df.mean()

# Среднее арифметическое для конкретного набора столбцов (например, цены квартиры и этажа)
df['price','floor'].mean()

# Медиана для всех столбцов
df.median()

# Медиана для конкретного набора столбцов (например, цены квартиры и этажа)
df['price','floor'].median()

# Мода для конкретного набора столбцов (например, цены квартиры и этажа)
df['price','floor'].moda()

# Стандартное отклонение для конкретного столбца (например, цены квартиры)
df['price'].std()

# Максимальное значение для конкретного столбца (например, цены квартиры)
df['price'].max()

# Минимальное значение для конкретного столбца (например, цены квартиры)
df['price'].min()

# Размах выборки для данных конкретного столбца (например, цены квартиры)
df['price'].max() - df['price'].min()

# Квартили Q1- Q3 для данных конкретного столбца (например, цены квартиры)
df['price'].quantile([0.25, 0.5, 0.75])

# Относительная частота для данных конкретного столбца (например, цены квартиры)
df['price'].value_counts(normalize=True)

# Абсолютная частота для данных конкретного столбца (например, цены квартиры)
df[' price '].value_counts()

# Строим гистограмму распределения данных (на примере цен на квартиры)
df['price'].hist(bins=20, figsize=(10, 6))
plt.xlabel('в тыс.р.')
plt.ylabel('Частота')
plt.title('Распределение цен на квартиры')
plt.show()

# Строим диаграмму распределения данных (на примере цен на квартиры)
df['price'].plot(kind='box', figsize=(8, 6))
plt.xlabel('в тыс.р. ')
plt.title('Диаграмма размаха цен на квартиры')
plt.show()

# Строим диаграмму рассеяния (scatter plot) между двумя столбцами (например, цена и площадь:
plt.figure(figsize=(8, 6))
sns.scatterplot(x='square', y='price', data=df)
plt.xlabel('Площадь')
plt.ylabel('Цена')
plt.title('Диаграмма рассеяния цены и площади')
plt.show()


##7. Проведение корреляционного анализа и его визуализация

# Вычисляем матрицу корреляций (например, между тремя заданными столбцами):
df[['price', 'floor', 'dist']].corr()

# Визуализируем матрицу корреляций с помощью тепловой карты (heatmap):
corr_matrix= df[['price', 'floor', 'dist']].corr()
plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=True, cmap='YlOrRd')
plt.title('Матрица корреляций')
plt.show()

# Расссчитываем коэффициент корреляции между двумя переменными
corr, p_value = pearsonr(df['price'], df['floor'])
print(f"Коэффициент корреляции Пирсона: {corr:.2f}")
print(f"p-значение: {p_value:.4f}")
if p_value < 0.05:
    print("Корреляция статистически значима на уровне 5%")
else:
    print("Корреляция не является статистически значимой на уровне 5%")


##8. Построение регрессионной моделей стоимости жилья

# Выделяем независимую переменную y (price) и объясняющие переменные X (для примера взяты площадь, этаж и расстояние до центра):
X = df[['sq', 'floor', 'dist']]
y = df['price']

# Создаем и обучаем модель множественной линейной регрессии
model = LinearRegression()
model.fit(X, y)

# Получаем значения коэффициентов объясняющих переменных множественной линейной регрессии
X_const = sm.add_constant(X)
results = sm.OLS(y, X_const).fit()
coefficients = model.coef_
intercept = model.intercept_

# Создаем DataFrame с результатами
results_df = pd.DataFrame({
    'Фактор': ['Свободный член'] + list(X.columns),
    'Коэффициент': [results.params[0]] + list(results.params[1:]),
    'Стандартная ошибка': [results.bse[0]] + list(results.bse[1:]),
    't-статистика': [results.tvalues[0]] + list(results.tvalues[1:]),
    'p-value': [results.pvalues[0]] + list(results.pvalues[1:])
})

# Выводим уравнение множественной линейной регрессии
equation = f'y = {results.params[0]:.2f}'
for i, coef in enumerate(results.params[1:]):
    factor_name = results_df.loc[i+1, 'Фактор']
    equation += f' + {coef:.2f} * {factor_name}'
    display(Markdown(f'**Уравнение регрессии:**\n{equation}')
)

# Вычисляем t-статистику и p-значения для коэффициентов объясняющих переменных множественной линейной регрессии
results_df = pd.DataFrame({
    'Фактор': ['Свободный член'] + list(X.columns),
    'Коэффициент': [results.params[0]] + list(results.params[1:]),
    'Стандартная ошибка': [results.bse[0]] + list(results.bse[1:]),
    't-статистика': [results.tvalues[0]] + list(results.tvalues[1:]),
    'p-value': [results.pvalues[0]] + list(results.pvalues[1:])
})

# Выводим информацию о статистической значимости коэффициентов объясняющих переменных множественной линейной регрессии
print(results_df)
for i, p_value in enumerate(results_df['p-value'][1:]):
    factor_name = results_df.loc[i+1, 'Фактор']
    if p_value > 0.05:
        print(f"Коэффициент при {factor_name} является статистически незначимым при 5% уровне значимости.")


##9. Отбор наилучшей регрессионной модели

# Выводим график
plt.figure(figsize=(8, 6))
plt.scatter(y, model.predict(X))
plt.plot([min(y), max(y)], [min(y), max(y)], 'r--')
plt.title('Прогнозные vs. Фактические значения')
plt.xlabel('Фактические значения')
plt.ylabel('Прогнозные значения')
plt.grid(True)
plt.show()

# Выводим гистограмму
plt.figure(figsize=(8, 6))
residuals = y - model.predict(X)
plt.hist(residuals, bins=20)
plt.title('Гистограмма распределения ошибок')
plt.xlabel('Ошибка')
plt.ylabel('Частота')
plt.grid(True)
plt.show()

# Рассчитываем R-квадрат
r_squared = model.score(X, y)

# Рассчитываем скорректированный R-квадрат
n = len(y)
p = X.shape[1]
adjusted_r_squared = 1 - (1 - r_squared) * (n - 1) / (n - p - 1)

# Выводим результаты
print(f"R-squared: {r_squared:.3f}")
print(f"Adjusted R-squared: {adjusted_r_squared:.3f}")
