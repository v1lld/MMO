import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats

from sklearn.impute import SimpleImputer, MissingIndicator
from sklearn.preprocessing import OneHotEncoder

# -----------------------------
# 1. ЗАГРУЗКА ДАННЫХ
# -----------------------------
url = "https://raw.githubusercontent.com/selva86/datasets/master/Auto.csv"
df = pd.read_csv(url)

print("Первые строки:")
print(df.head())

print("\nИнформация:")
print(df.info())

# -----------------------------
# 2. ПРЕДОБРАБОТКА
# -----------------------------
df.replace('?', np.nan, inplace=True)
df['horsepower'] = pd.to_numeric(df['horsepower'], errors='coerce')
df['origin'] = df['origin'].astype('category')

# -----------------------------
# 3. СОЗДАНИЕ ПРОПУСКОВ (ДЛЯ ДЕМОНСТРАЦИИ)
# -----------------------------
df_missing = df.copy()
df_missing.loc[df_missing.sample(frac=0.1, random_state=42).index, 'horsepower'] = np.nan

print("\nПропуски ДО:")
print(df_missing.isnull().sum())

# -----------------------------
# 4. ИМПЬЮТАЦИЯ (РАЗНЫЕ МЕТОДЫ)
# -----------------------------
# среднее
mean_imputer = SimpleImputer(strategy='mean')
df_mean = df_missing.copy()
df_mean['horsepower'] = mean_imputer.fit_transform(df_mean[['horsepower']])

# медиана
median_imputer = SimpleImputer(strategy='median')
df_median = df_missing.copy()
df_median['horsepower'] = median_imputer.fit_transform(df_median[['horsepower']])

# константа
const_imputer = SimpleImputer(strategy='constant', fill_value=0)
df_const = df_missing.copy()
df_const['horsepower'] = const_imputer.fit_transform(df_const[['horsepower']])

print("\nПропуски ПОСЛЕ (mean):")
print(df_mean.isnull().sum())

# -----------------------------
# 5. ФЛАГ ПРОПУСКОВ
# -----------------------------
indicator = MissingIndicator()
missing_flag = indicator.fit_transform(df_missing[['horsepower']])

df_missing['horsepower_missing'] = missing_flag.astype(int)

print("\nФлаг пропусков:")
print(df_missing[['horsepower', 'horsepower_missing']].head())

# -----------------------------
# 6. ВИЗУАЛИЗАЦИЯ ИМПЬЮТАЦИИ
# -----------------------------
plt.figure(figsize=(10,5))
sns.kdeplot(df_mean['horsepower'], label='mean')
sns.kdeplot(df_median['horsepower'], label='median')
sns.kdeplot(df_const['horsepower'], label='constant')
plt.legend()
plt.title("Сравнение методов заполнения")
plt.show()

# -----------------------------
# 7. КОДИРОВАНИЕ
# -----------------------------
df_encoded = df_mean.copy()

cat_cols = df_encoded.select_dtypes(include=['object', 'category']).columns

encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
encoded = encoder.fit_transform(df_encoded[cat_cols])

encoded_df = pd.DataFrame(
    encoded,
    columns=encoder.get_feature_names_out(cat_cols)
)

df_encoded = df_encoded.drop(columns=cat_cols)
df_encoded = pd.concat([df_encoded.reset_index(drop=True), encoded_df], axis=1)

print("\nПосле кодирования:")
print(df_encoded.head())

# -----------------------------
# 8. НОРМАЛИЗАЦИЯ (КАК В ЛЕКЦИИ)
# -----------------------------
def diagnostic_plots(df, variable):
    plt.figure(figsize=(12,5))

    plt.subplot(1, 2, 1)
    df[variable].hist(bins=30)
    plt.title(f'Histogram: {variable}')

    plt.subplot(1, 2, 2)
    stats.probplot(df[variable], dist="norm", plot=plt)

    plt.show()

feature = 'mpg'

print("\nИсходное распределение:")
diagnostic_plots(df_encoded, feature)

# логарифм
df_encoded['mpg_log'] = np.log(df_encoded[feature])
print("\nЛогарифм:")
diagnostic_plots(df_encoded, 'mpg_log')

# корень
df_encoded['mpg_sqrt'] = np.sqrt(df_encoded[feature])
print("\nКвадратный корень:")
diagnostic_plots(df_encoded, 'mpg_sqrt')

# Box-Cox
df_encoded['mpg_boxcox'], param = stats.boxcox(df_encoded[feature])
print(f"\nBox-Cox λ = {param}")
diagnostic_plots(df_encoded, 'mpg_boxcox')

# Yeo-Johnson
df_encoded['mpg_yeojohnson'], param = stats.yeojohnson(df_encoded[feature])
print(f"\nYeo-Johnson λ = {param}")
diagnostic_plots(df_encoded, 'mpg_yeojohnson')

# -----------------------------
# 9. ФИНАЛ
# -----------------------------
print("\nРазмер итогового датасета:", df_encoded.shape)