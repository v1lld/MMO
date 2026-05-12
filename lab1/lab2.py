# %% [markdown]
# # Лабораторная работа: Обработка признаков (часть 2)
# 
# ## Цель работы
# Изучение продвинутых способов предварительной обработки данных для дальнейшего формирования моделей.
# 
# ## Задание
# 1. Масштабирование признаков (не менее трёх способов).
# 2. Обработка выбросов для числовых признаков (удаление и замена выбросов).
# 3. Обработка нестандартного признака (не числового и не категориального).
# 4. Отбор признаков: один метод фильтрации, один метод обёртывания, один метод вложений.
# 
# Для демонстрации использован датасет **Auto** (данные об автомобилях), который содержит числовые и текстовые признаки.

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.feature_selection import mutual_info_regression, SelectKBest, RFE, SelectFromModel
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings('ignore')

# Загрузка данных
url = "https://raw.githubusercontent.com/selva86/datasets/master/Auto.csv"
df = pd.read_csv(url)

print("Размер данных:", df.shape)
df.head()

# %%
# Информация о пропусках и типах
df.info()

# %% [markdown]
# ### 2. Предварительная обработка данных

# %%
# 2.1 Обработка пропусков (столбец 'horsepower' может содержать '?')
df['horsepower'] = pd.to_numeric(df['horsepower'], errors='coerce')
print("Пропуски после приведения:", df.isnull().sum())

# Заполним пропуски медианой
df['horsepower'].fillna(df['horsepower'].median(), inplace=True)

# 2.2 Обработка нестандартного признака 'name' – извлечение марки автомобиля
def extract_brand(name):
    # Бренд – первое слово
    return name.split()[0]

df['brand'] = df['name'].apply(extract_brand)
# Можно сгруппировать редкие марки в 'Other' (для наглядности, но не обязательно)
df['brand'] = df['brand'].astype('category')

# Удаляем исходный столбец 'name' (теперь не нужен)
df.drop('name', axis=1, inplace=True)

# 2.3 Преобразуем категориальные признаки в числовые (origin, brand)
df = pd.get_dummies(df, columns=['origin', 'brand'], drop_first=True)

# Целевой признак – mpg (расход топлива)
y = df['mpg']
X = df.drop('mpg', axis=1)

print("Признаки после обработки:")
print(X.columns.tolist())

# %% [markdown]
# ### 3. Масштабирование числовых признаков (3 метода)

# %%
# Выделим числовые признаки
numeric_features = ['cylinders', 'displacement', 'horsepower', 'weight', 'acceleration', 'year']
X_num = X[numeric_features]
X_cat = X.drop(columns=numeric_features)

# Разделение на обучающую и тестовую выборки
X_train_num, X_test_num, X_train_cat, X_test_cat, y_train, y_test = train_test_split(
    X_num, X_cat, y, test_size=0.2, random_state=42
)

# Функция для визуализации распределений
def plot_scaling(original, scaled, method_name):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    original.hist(ax=axes[0], bins=30, edgecolor='black')
    axes[0].set_title(f'{method_name}: Исходное')
    scaled.hist(ax=axes[1], bins=30, edgecolor='black')
    axes[1].set_title(f'{method_name}: Масштабированное')
    plt.tight_layout()
    plt.show()

# 3.1 StandardScaler
scaler_std = StandardScaler()
X_train_num_std = scaler_std.fit_transform(X_train_num)
X_test_num_std = scaler_std.transform(X_test_num)
plot_scaling(X_train_num, pd.DataFrame(X_train_num_std, columns=numeric_features), 'StandardScaler')

# 3.2 MinMaxScaler
scaler_minmax = MinMaxScaler()
X_train_num_minmax = scaler_minmax.fit_transform(X_train_num)
X_test_num_minmax = scaler_minmax.transform(X_test_num)
plot_scaling(X_train_num, pd.DataFrame(X_train_num_minmax, columns=numeric_features), 'MinMaxScaler')

# 3.3 RobustScaler
scaler_robust = RobustScaler()
X_train_num_robust = scaler_robust.fit_transform(X_train_num)
X_test_num_robust = scaler_robust.transform(X_test_num)
plot_scaling(X_train_num, pd.DataFrame(X_train_num_robust, columns=numeric_features), 'RobustScaler')

# Объединяем масштабированные числовые с категориальными
X_train_std = np.hstack([X_train_num_std, X_train_cat.values])
X_test_std = np.hstack([X_test_num_std, X_test_cat.values])

X_train_minmax = np.hstack([X_train_num_minmax, X_train_cat.values])
X_test_minmax = np.hstack([X_test_num_minmax, X_test_cat.values])

X_train_robust = np.hstack([X_train_num_robust, X_train_cat.values])
X_test_robust = np.hstack([X_test_num_robust, X_test_cat.values])

# %% [markdown]
# ### 4. Обработка выбросов (на примере признака 'horsepower')

# %%
plt.figure(figsize=(10, 4))
plt.subplot(1, 3, 1)
sns.boxplot(y=df['horsepower'])
plt.title('Исходные данные (horsepower)')

# IQR
Q1 = df['horsepower'].quantile(0.25)
Q3 = df['horsepower'].quantile(0.75)
IQR = Q3 - Q1
lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR

# Удаление выбросов
df_no_outliers = df[(df['horsepower'] >= lower) & (df['horsepower'] <= upper)]
plt.subplot(1, 3, 2)
sns.boxplot(y=df_no_outliers['horsepower'])
plt.title('После удаления выбросов (IQR)')

# Каппинг (замена выбросов границами)
df_capped = df.copy()
df_capped['horsepower'] = df_capped['horsepower'].clip(lower=lower, upper=upper)
plt.subplot(1, 3, 3)
sns.boxplot(y=df_capped['horsepower'])
plt.title('После каппинга')
plt.tight_layout()
plt.show()

print("Количество удалённых строк:", len(df) - len(df_no_outliers))
print("Количество изменённых значений при каппинге:", (df['horsepower'] != df_capped['horsepower']).sum())

# %% [markdown]
# ### 5. Отбор признаков (feature selection)
# Используем масштабированные данные (StandardScaler) для единообразия.

# %%
X_train = X_train_std
X_test = X_test_std

# 5.1 Метод фильтрации: взаимная информация (mutual information)
mi_scores = mutual_info_regression(X_train, y_train, random_state=42)
mi_scores = pd.Series(mi_scores, index=numeric_features + list(X_cat.columns)).sort_values(ascending=False)
print("Взаимная информация по признакам:")
print(mi_scores)

# Выберем 5 лучших признаков
k = 5
selector_mi = SelectKBest(mutual_info_regression, k=k)
X_train_mi = selector_mi.fit_transform(X_train, y_train)
X_test_mi = selector_mi.transform(X_test)
selected_mi = selector_mi.get_support(indices=True)
print("\nВыбранные признаки (filter):", [list(numeric_features + list(X_cat.columns))[i] for i in selected_mi])

# 5.2 Метод обёртывания: RFE (Recursive Feature Elimination)
model = LinearRegression()
selector_rfe = RFE(model, n_features_to_select=k)
X_train_rfe = selector_rfe.fit_transform(X_train, y_train)
X_test_rfe = selector_rfe.transform(X_test)
selected_rfe = selector_rfe.get_support(indices=True)
print("\nВыбранные признаки (wrapper):", [list(numeric_features + list(X_cat.columns))[i] for i in selected_rfe])

# 5.3 Метод вложений: SelectFromModel с L1-регуляризацией (Lasso)
model_l1 = Lasso(alpha=0.1, max_iter=10000)
selector_embedded = SelectFromModel(model_l1, threshold='mean')
X_train_emb = selector_embedded.fit_transform(X_train, y_train)
X_test_emb = selector_embedded.transform(X_test)
selected_emb = selector_embedded.get_support(indices=True)
print("\nВыбранные признаки (embedded):", [list(numeric_features + list(X_cat.columns))[i] for i in selected_emb])

# %% [markdown]
# ### 6. Оценка качества моделей после отбора признаков
# Сравним среднеквадратичную ошибку на тестовой выборке.

# %%
def evaluate_regressor(X_train, X_test, y_train, y_test):
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return mean_squared_error(y_test, y_pred)

print("MSE на полном наборе:", evaluate_regressor(X_train, X_test, y_train, y_test))
print("MSE после фильтрации (MI):", evaluate_regressor(X_train_mi, X_test_mi, y_train, y_test))
print("MSE после RFE:", evaluate_regressor(X_train_rfe, X_test_rfe, y_train, y_test))
print("MSE после SelectFromModel (Lasso):", evaluate_regressor(X_train_emb, X_test_emb, y_train, y_test))

# %% [markdown]
# ### 7. Выводы
# - Масштабирование признаков (StandardScaler, MinMaxScaler, RobustScaler) показало, как можно нормализовать данные для последующего моделирования.
# - Обработка выбросов (удаление и каппинг) на признаке 'horsepower' продемонстрировала два подхода к работе с аномалиями.
# - Нестандартный признак 'name' был преобразован в категорию 'brand', что позволило учесть влияние марки автомобиля на расход топлива.
# - Отбор признаков с помощью фильтрации, обёртывания и вложений позволил сократить размерность, при этом качество модели (MSE) осталось на приемлемом уровне, а в некоторых случаях даже улучшилось.