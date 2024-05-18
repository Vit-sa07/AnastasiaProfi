import time

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LassoCV, RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score


# 1. Реализуйте функцию, возвращающую максимальный элемент в векторе x среди элементов, перед которыми стоит нулевой.
def max_element(arr):
    """
    Функция возвращает максимальный элемент в векторе arr среди элементов, перед которыми стоит нулевой.
    """
    zero_indices = np.where(arr == 0)[0]  # находим индексы всех нулевых элементов
    valid_indices = zero_indices[zero_indices + 1 < len(arr)]  # исключаем индексы, которые выходят за пределы массива
    if len(valid_indices) == 0:  # если нет допустимых нулевых элементов
        return None
    candidates = arr[valid_indices + 1]  # берем элементы, стоящие после нулевых
    return np.max(candidates)  # возвращаем максимальный из них

# Функция для записи результатов в файл
def write_results(text):
    with open("results.txt", "a", encoding="utf-8") as f:
        f.write(text + "\n")

# Пример использования:
x = np.array([6, 2, 0, 3, 0, 0, 5, 7, 0])
print('Задание 1')
print(max_element(x))
write_results('Задание 1')
write_results(f"Результат: {max_element(x)}")


# 2. Реализуйте функцию, принимающую на вход матрицу и некоторое число и возвращающую ближайший к числу элемент матрицы.
def nearest_value(X, v):
    """
    Функция возвращает элемент матрицы X, ближайший к числу v.
    """
    return X.flat[np.abs(X - v).argmin()]  # находим индекс минимального расстояния и возвращаем элемент


# Пример использования:
X = np.arange(0, 10).reshape((2, 5))
v = 3.6
print('Задание 2')
print(nearest_value(X, v))
write_results('Задание 2')
write_results(f"Результат: {nearest_value(X, v)}")


# 3. Реализуйте функцию scale(X), которая принимает на вход матрицу и масштабирует каждый ее столбец.
def scale(X):
    """
    Функция масштабирует каждый столбец матрицы X (вычитает выборочное среднее и делит на стандартное отклонение).
    """
    return (X - np.mean(X, axis=0)) / np.std(X, axis=0, ddof=1)


# Пример использования:
X = np.random.randint(1, 10, size=(5, 3))
print('Задание 3')
print(scale(X))
write_results('Задание 3')
write_results(f"Результат:\n{scale(X)}")

# 4. Реализуйте функцию, которая для заданной матрицы находит определитель, след, наименьший и наибольший элементы, норму Фробениуса, собственные числа и обратную матрицу.
def get_stats(X):
    """
    Функция вычисляет и возвращает определитель, след, наименьший и наибольший элементы, норму Фробениуса, собственные числа и обратную матрицу для заданной матрицы X.
    """
    stats = {
        'determinant': np.linalg.det(X),
        'trace': np.trace(X),
        'min_element': np.min(X),
        'max_element': np.max(X),
        'frobenius_norm': np.linalg.norm(X, 'fro'),
        'eigenvalues': np.linalg.eigvals(X),
        'inverse': np.linalg.inv(X) if np.linalg.det(X) != 0 else None
    }
    return stats


# Пример использования:
X = np.random.normal(10, 1, (3, 3))
print('Задание 4')
print(get_stats(X))
write_results('Задание 4')
write_results(f"Результат:\n{get_stats(X)}")

# 5. Повторите 100 раз следующий эксперимент: сгенерируйте две матрицы размера 10×10 из стандартного нормального распределения, перемножьте их и найдите максимальный элемент.
max_elements = []
for exp_num in range(100):
    A = np.random.randn(10, 10)
    B = np.random.randn(10, 10)
    C = np.dot(A, B)
    max_elements.append(np.max(C))

mean_max_element = np.mean(max_elements)
quantile_95 = np.percentile(max_elements, 95)
print('Задание 5')
print(f"Среднее значение максимальных элементов: {mean_max_element}")
print(f"95-процентная квантиль: {quantile_95}")
write_results('Задание 5')
write_results(f"Среднее значение максимальных элементов: {mean_max_element}")
write_results(f"95-процентная квантиль: {quantile_95}")

# 6. Какая из причин отмены рейса (CancellationCode) была самой частой?
df = pd.read_csv('2008.csv')
most_common_cancellation = df['CancellationCode'].value_counts().idxmax()
print('Задание 6')
print(f"Самая частая причина отмены рейса: {most_common_cancellation}")
write_results('Задание 6')
write_results(f"Самая частая причина отмены рейса: {most_common_cancellation}")

# 7. Найдите среднее, минимальное и максимальное расстояние, пройденное самолетом.
mean_distance = df['Distance'].mean()
min_distance = df['Distance'].min()
max_distance = df['Distance'].max()
print('Задание 7')
print(
    f"Среднее расстояние: {mean_distance}, Минимальное расстояние: {min_distance}, Максимальное расстояние: {max_distance}")
write_results('Задание 7')
write_results(f"Среднее расстояние: {mean_distance}, Минимальное расстояние: {min_distance}, Максимальное расстояние: {max_distance}")

# 8. Не выглядит ли подозрительным минимальное пройденное расстояние? В какие дни и на каких рейсах оно было? Какое расстояние было пройдено этими же рейсами в другие дни?
write_results('Задание 8')

suspicious_flights = df[df['Distance'] == min_distance]
write_results("Подозрительные рейсы:\n" + suspicious_flights[['Year', 'Month', 'DayofMonth', 'FlightNum', 'Distance', 'Origin', 'Dest']].to_string())

# Найдем расстояние, пройденное этими же рейсами в другие дни
other_days_flights = df[(df['FlightNum'].isin(suspicious_flights['FlightNum'])) &
                        (~df['DayofMonth'].isin(suspicious_flights['DayofMonth']))]

# Разделение по номерам рейсов и запись в файл
for flight_num in suspicious_flights['FlightNum'].unique():
    same_flight_other_days = other_days_flights[other_days_flights['FlightNum'] == flight_num]
    write_results(f"Рейс номер {flight_num} в другие дни:\n" + same_flight_other_days[['Year', 'Month', 'DayofMonth', 'FlightNum', 'Distance', 'Origin', 'Dest']].to_string())

# 9. Из какого аэропорта было произведено больше всего вылетов? В каком городе он находится?
print('Задание 9')
most_departures = df['Origin'].value_counts().idxmax()
print(f"Аэропорт с наибольшим количеством вылетов: {most_departures}")
write_results('Задание 9')
write_results(f"Аэропорт с наибольшим количеством вылетов: {most_departures}")

# 10. Найдите для каждого аэропорта среднее время полета (AirTime) по всем вылетевшим из него рейсам. Какой аэропорт имеет наибольшее значение этого показателя?
print('Задание 10')
mean_airtime_per_airport = df.groupby('Origin')['AirTime'].mean().idxmax()
print(f"Аэропорт с наибольшим средним временем полета: {mean_airtime_per_airport}")
write_results('Задание 10')
write_results(f"Аэропорт с наибольшим средним временем полета: {mean_airtime_per_airport}")

# 11. Найдите аэропорт, у которого наибольшая доля задержанных (DepDelay > 0) рейсов. Исключите при этом из рассмотрения аэропорты, из которых было отправлено меньше 1000 рейсов.
print('Задание 11')
airport_delays = df[df['DepDelay'] > 0].groupby('Origin').filter(lambda x: len(x) > 1000)
airport_with_most_delays = airport_delays.groupby('Origin').size().idxmax()
print(f"Аэропорт с наибольшей долей задержанных рейсов: {airport_with_most_delays}")
write_results('Задание 11')
write_results(f"Аэропорт с наибольшей долей задержанных рейсов: {airport_with_most_delays}")

# 12. Считайте выборку из файла при помощи функции pd.read_csv и ответьте на вопросы о пропущенных значениях.
missing_values = df.isnull().sum().sum()
rows_with_missing_values = df.isnull().any(axis=1).sum()
columns_with_missing_values = df.isnull().any(axis=0).sum()
print('Задание 12')
print(f"Всего пропущенных значений: {missing_values}")
print(f"Количество объектов с хотя бы одним пропуском: {rows_with_missing_values}")
print(f"Количество признаков с хотя бы одним пропущенным значением: {columns_with_missing_values}")
write_results('Задание 12')
write_results(f"Всего пропущенных значений: {missing_values}")
write_results(f"Количество объектов с хотя бы одним пропуском: {rows_with_missing_values}")
write_results(f"Количество признаков с хотя бы одним пропущенным значением: {columns_with_missing_values}")

# Исключите объекты с пропущенными значениями целевой переменной и целевые переменные равные 0
df = df.dropna(subset=['DepDelay'])
df = df[df['DepDelay'] != 0]
y = df['DepDelay']
X = df.drop(columns=['DepDelay'])

# 13. Преобразуйте каждый признак из указанного в пару новых признаков FeatureName_Hour, FeatureName_Minute.
for col in ['DepTime', 'CRSDepTime', 'ArrTime', 'CRSArrTime']:
    if col in df.columns:
        df[col + '_Hour'] = df[col] // 100
        df[col + '_Minute'] = df[col] % 100
        df = df.drop(columns=[col])
    else:
        print(f"Column {col} not found in DataFrame")
        write_results(f"Column {col} not found in DataFrame")

# Вывод результатов для проверки
print('Задание 13')
for col in ['DepTime_Hour', 'DepTime_Minute', 'CRSDepTime_Hour', 'CRSDepTime_Minute', 'ArrTime_Hour', 'ArrTime_Minute', 'CRSArrTime_Hour', 'CRSArrTime_Minute']:
    if col in df.columns:
        print(f"Столбец {col} успешно добавлен в DataFrame.")
        write_results(f"Столбец {col} успешно добавлен в DataFrame.")
    else:
        print(f"Столбец {col} отсутствует в DataFrame.")
        write_results(f"Столбец {col} отсутствует в DataFrame.")

# Задание 14. Изучите описание датасета и исключите признаки, сильно коррелирующие с ответами. Исключите признаки TailNum и Year.
# Сначала исключим категориальные переменные
numeric_df = df.select_dtypes(include=[np.number])

# Посчитаем корреляцию признаков с целевой переменной
correlation_matrix = numeric_df.corr()
target_corr = correlation_matrix['DepDelay'].abs().sort_values(ascending=False)
print('Задание 14')
# Выведем корреляцию с целевой переменной
print(target_corr)
write_results('Задание 14')
write_results(str(target_corr))

# Определим порог для корреляции
threshold = 0.5

# Исключим признаки с корреляцией выше порога
high_corr_features = target_corr[target_corr > threshold].index.tolist()
print(f"Признаки с высокой корреляцией: {high_corr_features}")
write_results(f"Признаки с высокой корреляцией: {high_corr_features}")

# Исключим признаки TailNum и Year только если они существуют в DataFrame
for col in ['TailNum', 'Year']:
    if col in df.columns:
        high_corr_features.append(col)

# Исключаем эти признаки из данных
df = df.drop(columns=[col for col in high_corr_features if col in df.columns])

print(f"Обновленные данные после исключения признаков: {df.columns.tolist()}")
write_results("Признаки с высокой корреляцией с целевой переменной могут содержать избыточную информацию, что может привести к переобучению модели. Исключение таких признаков помогает уменьшить избыточность данных, улучшить обобщающую способность модели и избежать проблем мультиколлинеарности.")
write_results(f"Обновленные данные после исключения признаков: {df.columns.tolist()}")

# 15. Приведем данные к виду, пригодному для обучения линейных моделей.
def transform_data(data):
    """
    Функция преобразует данные:
    - Заполняет пропущенные значения медианой для вещественных признаков и строкой 'nan' для категориальных
    - Масштабирует вещественные признаки
    - Применяет one-hot-кодирование к категориальным признакам
    """
    # Заполнение пропущенных значений
    data = data.copy()
    for col in data.select_dtypes(include=[np.number]).columns:
        data[col] = data[col].fillna(data[col].median())
    for col in data.select_dtypes(include=[object]).columns:
        data[col] = data[col].fillna('nan')

    # Масштабирование вещественных признаков
    scaler = StandardScaler()
    data_scaled = pd.DataFrame(scaler.fit_transform(data.select_dtypes(include=[np.number])),
                               columns=data.select_dtypes(include=[np.number]).columns)

    # One-hot-кодирование категориальных признаков
    data = data_scaled.join(pd.get_dummies(data.select_dtypes(exclude=[np.number])))
    return data


start_time = time.time()
X_transformed = transform_data(X)
end_time = time.time()
n_features = X_transformed.shape[1]

print('Задание 15')
print(f"Количество признаков после преобразования: {n_features}")
write_results('Задание 15')
write_results(f"Количество признаков после преобразования: {n_features}")
write_results(f"Время выполнения преобразования данных: {end_time - start_time} секунд")

# 16. Разбейте выборку и вектор целевой переменной на обучение и контроль в отношении 70/30.

# Разделяем данные на обучающую и контрольную выборки в отношении 70/30
X_train, X_test, y_train, y_test = train_test_split(X_transformed, y, test_size=0.3, random_state=42)
print('Задание 16')
print(f"Размер обучающей выборки: {X_train.shape[0]}")
print(f"Размер контрольной выборки: {X_test.shape[0]}")
write_results('Задание 16')
write_results(f"Размер обучающей выборки: {X_train.shape[0]}")
write_results(f"Размер контрольной выборки: {X_test.shape[0]}")

# 17. Обучите линейную регрессию на 1000 объектах из обучающей выборки и выведите значения MSE и R^2

# Убедимся, что данные масштабированы и не содержат пропущенных значений
X_train = X_train.fillna(X_train.median())
X_test = X_test.fillna(X_test.median())

# Создаем и обучаем модель линейной регрессии на первых 1000 объектах обучающей выборки
linear_model = LinearRegression(n_jobs=-1)
linear_model.fit(X_train[:1000], y_train[:1000])

# Делаем прогнозы на обучающей и контрольной выборках
y_train_pred = linear_model.predict(X_train[:1000])
y_test_pred = linear_model.predict(X_test)

# Вычисляем значения MSE и R^2 на обучающей и контрольной выборках
train_mse = mean_squared_error(y_train[:1000], y_train_pred)
test_mse = mean_squared_error(y_test, y_test_pred)
train_r2 = r2_score(y_train[:1000], y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)
print('Задание 17')
print(f"Линейная регрессия:\nTrain MSE: {train_mse}, Test MSE: {test_mse}\nTrain R2: {train_r2}, Test R2: {test_r2}")
print(f"Коэффициенты модели: {linear_model.coef_}")
write_results('Задание 17')
write_results(f"Линейная регрессия:\nTrain MSE: {train_mse}, Test MSE: {test_mse}\nTrain R2: {train_r2}, Test R2: {test_r2}")
write_results(f"Коэффициенты модели: {linear_model.coef_}")

# 18. Обучите линейные регрессии с L1- и L2-регуляризатором, подобрав лучшее значение параметра регуляризации

# Определяем сетку значений для параметра регуляризации
alphas = np.logspace(-4, 4, 50)

# Убедимся, что данные масштабированы
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train[:1000])
X_test_scaled = scaler.transform(X_test)

# Создаем и обучаем модели Lasso и Ridge с кросс-валидацией, используя параллельную обработку
lasso = LassoCV(alphas=alphas, cv=5, max_iter=10000, tol=1e-3, n_jobs=-1).fit(X_train_scaled, y_train[:1000])
ridge = RidgeCV(alphas=alphas, cv=5).fit(X_train_scaled, y_train[:1000])

# Делаем прогнозы на обучающей и контрольной выборках для моделей Lasso и Ridge
y_train_pred_lasso = lasso.predict(X_train_scaled)
y_test_pred_lasso = lasso.predict(X_test_scaled)
y_train_pred_ridge = ridge.predict(X_train_scaled)
y_test_pred_ridge = ridge.predict(X_test_scaled)

# Вычисляем значения MSE и R^2 на обучающей и контрольной выборках для моделей Lasso и Ridge
train_mse_lasso = mean_squared_error(y_train[:1000], y_train_pred_lasso)
test_mse_lasso = mean_squared_error(y_test, y_test_pred_lasso)
train_r2_lasso = r2_score(y_train[:1000], y_train_pred_lasso)
test_r2_lasso = r2_score(y_test, y_test_pred_lasso)

train_mse_ridge = mean_squared_error(y_train[:1000], y_train_pred_ridge)
test_mse_ridge = mean_squared_error(y_test, y_test_pred_ridge)
train_r2_ridge = r2_score(y_train[:1000], y_train_pred_ridge)
test_r2_ridge = r2_score(y_test, y_test_pred_ridge)
print('Задание 18')
print(
    f"Регрессия Лассо:\nTrain MSE: {train_mse_lasso}, Test MSE: {test_mse_lasso}\nTrain R2: {train_r2_lasso}, Test R2: {test_r2_lasso}")
print(
    f"Ridge Регрессия:\nTrain MSE: {train_mse_ridge}, Test MSE: {test_mse_ridge}\nTrain R2: {train_r2_ridge}, Test R2: {test_r2_ridge}")
write_results('Задание 18')
write_results(f"Регрессия Лассо:\nTrain MSE: {train_mse_lasso}, Test MSE: {test_mse_lasso}\nTrain R2: {train_r2_lasso}, Test R2: {test_r2_lasso}")
write_results(f"Ridge Регрессия:\nTrain MSE: {train_mse_ridge}, Test MSE: {test_mse_ridge}\nTrain R2: {train_r2_ridge}, Test R2: {test_r2_ridge}")