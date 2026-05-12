import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Установим стиль графиков
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("Set2")

# Создаем набор данных о квартирах в Москве
np.random.seed(42)
n_samples = 200

data = {
    'Площадь_кв_м': np.random.normal(55, 20, n_samples),
    'Комнаты': np.random.choice([1, 2, 3, 4], n_samples, p=[0.3, 0.35, 0.25, 0.1]),
    'Этаж': np.random.randint(1, 25, n_samples),
    'Цена_млн_руб': np.random.normal(12, 5, n_samples),
    'Расстояние_до_метро_мин': np.random.normal(10, 5, n_samples)
}

df = pd.DataFrame(data)

# Добавляем реалистичности: площадь связана с количеством комнат
df['Площадь_кв_м'] = df['Площадь_кв_м'] + df['Комнаты'] * 15

# Вносим пропуски в признак "Площадь_кв_м" (~12% значений)
missing_indices = np.random.choice(df.index, size=int(n_samples * 0.12), replace=False)
df.loc[missing_indices, 'Площадь_кв_м'] = np.nan

# Добавляем несколько выбросов (очень большие и очень маленькие квартиры)
outlier_indices = np.random.choice(df.dropna().index, size=5, replace=False)
df.loc[outlier_indices[:3], 'Площадь_кв_м'] = [200, 250, 300]  # пентхаусы
df.loc[outlier_indices[3:], 'Площадь_кв_м'] = [8, 10]  # микроквартиры

# Добавляем колонку с районом для контекста
df['Район'] = np.random.choice(['ЦАО', 'САО', 'ЗАО', 'ЮАО', 'ВАО', 'СЗАО'], n_samples)

print("=" * 60)
print("ДАННЫЕ О КВАРТИРАХ В МОСКВЕ")
print("=" * 60)
print(f"Всего записей: {len(df)}")
print(f"Признаков: {df.shape[1]}")
print(f"\nПервые 10 строк:")
print(df.head(10).to_string(index=False))

print(f"\nИнформация о пропусках:")
print(df.isnull().sum())
print(f"\nПроцент пропусков в 'Площадь_кв_м': {df['Площадь_кв_м'].isnull().mean()*100:.2f}%")
print("\n" + "=" * 60)
print("ЗАДАЧА №7: УСТРАНЕНИЕ ПРОПУСКОВ МЕДИАНОЙ")
print("=" * 60)

# Рассчитываем медиану для признака "Площадь_кв_м"
median_area = df['Площадь_кв_м'].median()

print(f"Медиана площади квартиры: {median_area:.2f} кв.м")
print(f"Количество пропусков до заполнения: {df['Площадь_кв_м'].isnull().sum()}")

# Создаем копию данных и заполняем пропуски медианой
df['Площадь_заполненная'] = df['Площадь_кв_м'].fillna(median_area)

print(f"Количество пропусков после заполнения: {df['Площадь_заполненная'].isnull().sum()}")

# Визуализация
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# График 1: До заполнения
axes[0].hist(df['Площадь_кв_м'].dropna(), bins=25, edgecolor='white', 
             color='steelblue', alpha=0.8)
axes[0].axvline(median_area, color='red', linestyle='--', linewidth=2, 
                label=f'Медиана = {median_area:.1f} кв.м')
axes[0].set_title('Распределение площади квартир\n(до заполнения пропусков)', 
                  fontsize=13, fontweight='bold')
axes[0].set_xlabel('Площадь (кв.м)')
axes[0].set_ylabel('Количество квартир')
axes[0].legend()

# График 2: После заполнения
axes[1].hist(df['Площадь_заполненная'], bins=25, edgecolor='white', 
             color='seagreen', alpha=0.8)
axes[1].axvline(median_area, color='red', linestyle='--', linewidth=2, 
                label=f'Медиана = {median_area:.1f} кв.м')
axes[1].set_title('Распределение площади квартир\n(после заполнения медианой)', 
                  fontsize=13, fontweight='bold')
axes[1].set_xlabel('Площадь (кв.м)')
axes[1].set_ylabel('Количество квартир')
axes[1].legend()

plt.tight_layout()
plt.show()

# Статистический анализ изменений
print("\nСтатистики ДО заполнения пропусков:")
print(df['Площадь_кв_м'].describe().to_string())
print(f"\nСтатистики ПОСЛЕ заполнения пропусков медианой:")
print(df['Площадь_заполненная'].describe().to_string())

# Показываем, какие значения были заполнены
filled_rows = df[df['Площадь_кв_м'].isnull()]
print(f"\nПримеры заполненных значений (первые 5):")
print(filled_rows[['Площадь_кв_м', 'Площадь_заполненная']].head())
print("\n" + "=" * 60)
print("ЗАДАЧА №27: ОБНАРУЖЕНИЕ И ЗАМЕНА ВЫБРОСОВ")
print("=" * 60)

# Используем заполненные данные для чистоты анализа
area_data = df['Площадь_заполненная'].copy()

# Расчет 5% и 95% квантилей
q5 = area_data.quantile(0.05)
q95 = area_data.quantile(0.95)

print(f"5% квантиль (нижняя граница): {q5:.2f} кв.м")
print(f"95% квантиль (верхняя граница): {q95:.2f} кв.м")

# Обнаружение выбросов
lower_outliers = area_data < q5
upper_outliers = area_data > q95
all_outliers = lower_outliers | upper_outliers

print(f"\nОбнаружено выбросов:")
print(f"  - Ниже 5% квантиля: {lower_outliers.sum()} квартир")
print(f"  - Выше 95% квантиля: {upper_outliers.sum()} квартир")
print(f"  - Всего выбросов: {all_outliers.sum()} ({all_outliers.mean()*100:.1f}%)")

print("\nВыбросы (площадь ниже нижней границы):")
print(area_data[lower_outliers].values)

print("\nВыбросы (площадь выше верхней границы):")
print(area_data[upper_outliers].values)

# Замена выбросов на граничные значения (винзоризация)
area_winsorized = area_data.copy()
area_winsorized[area_winsorized < q5] = q5
area_winsorized[area_winsorized > q95] = q95

print(f"\nЗначения после замены выбросов:")
print(f"  - Минимум был: {area_data.min():.1f}, стал: {area_winsorized.min():.1f}")
print(f"  - Максимум был: {area_data.max():.1f}, стал: {area_winsorized.max():.1f}")

# Визуализация результатов
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# 1. Ящик с усами (boxplot) до обработки
axes[0, 0].boxplot(area_data, vert=True, patch_artist=True, 
                    boxprops=dict(facecolor='lightcoral', alpha=0.6))
axes[0, 0].set_title('Ящик с усами ДО обработки выбросов', 
                     fontsize=12, fontweight='bold')
axes[0, 0].set_ylabel('Площадь (кв.м)')
axes[0, 0].set_xticklabels(['Площадь'])

# 2. Ящик с усами (boxplot) после обработки
axes[0, 1].boxplot(area_winsorized, vert=True, patch_artist=True, 
                    boxprops=dict(facecolor='lightgreen', alpha=0.6))
axes[0, 1].set_title('Ящик с усами ПОСЛЕ замены выбросов', 
                     fontsize=12, fontweight='bold')
axes[0, 1].set_ylabel('Площадь (кв.м)')
axes[0, 1].set_xticklabels(['Площадь'])

# 3. Гистограмма до обработки с границами
axes[1, 0].hist(area_data, bins=30, edgecolor='white', color='lightcoral', alpha=0.8)
axes[1, 0].axvline(q5, color='darkred', linestyle='--', linewidth=2, 
                   label=f'5% квантиль = {q5:.1f}')
axes[1, 0].axvline(q95, color='darkred', linestyle='--', linewidth=2, 
                   label=f'95% квантиль = {q95:.1f}')
axes[1, 0].set_title('Гистограмма ДО обработки выбросов', 
                     fontsize=12, fontweight='bold')
axes[1, 0].set_xlabel('Площадь (кв.м)')
axes[1, 0].set_ylabel('Количество квартир')
axes[1, 0].legend()

# 4. Гистограмма после обработки
axes[1, 1].hist(area_winsorized, bins=30, edgecolor='white', color='lightgreen', alpha=0.8)
axes[1, 1].axvline(q5, color='darkred', linestyle='--', linewidth=2, 
                   label=f'5% квантиль = {q5:.1f}')
axes[1, 1].axvline(q95, color='darkred', linestyle='--', linewidth=2, 
                   label=f'95% квантиль = {q95:.1f}')
axes[1, 1].set_title('Гистограмма ПОСЛЕ замены выбросов', 
                     fontsize=12, fontweight='bold')
axes[1, 1].set_xlabel('Площадь (кв.м)')
axes[1, 1].set_ylabel('Количество квартир')
axes[1, 1].legend()

plt.tight_layout()
plt.show()

# Сравнение статистик
comparison_df = pd.DataFrame({
    'До обработки': area_data.describe(),
    'После обработки': area_winsorized.describe()
})

print("\nСравнение статистик до и после обработки выбросов:")
print(comparison_df.round(2).to_string())

# Показываем примеры замененных значений
changed_mask = area_data != area_winsorized
if changed_mask.any():
    changes_df = pd.DataFrame({
        'Исходное значение': area_data[changed_mask],
        'Заменено на': area_winsorized[changed_mask],
        'Тип выброса': ['Ниже нижней границы' if x < q5 else 'Выше верхней границы' 
                        for x in area_data[changed_mask]]
    })
    print("\nПримеры замененных выбросов:")
    print(changes_df.head(10).to_string(index=False))

# Дополнительно: анализ влияния на другие характеристики
print(f"\nАнализ влияния обработки выбросов:")
print(f"Среднее: {area_data.mean():.2f} → {area_winsorized.mean():.2f} "
      f"({(area_winsorized.mean() - area_data.mean()):+.2f})")
print(f"СКО: {area_data.std():.2f} → {area_winsorized.std():.2f} "
      f"({(area_winsorized.std() - area_data.std()):+.2f})")
print(f"Межквартильный размах: {area_data.quantile(0.75) - area_data.quantile(0.25):.2f} → "
      f"{area_winsorized.quantile(0.75) - area_winsorized.quantile(0.25):.2f}")
print("\n" + "=" * 60)
print("ИТОГОВОЕ СРАВНЕНИЕ ВСЕХ ЭТАПОВ ОБРАБОТКИ")
print("=" * 60)

summary_df = pd.DataFrame({
    'Исходные данные\n(с пропусками)': df['Площадь_кв_м'],
    'После заполнения\nмедианой': df['Площадь_заполненная'],
    'После замены\nвыбросов': area_winsorized
})

print(summary_df.describe().round(2).to_string())

# Финальная визуализация всех трех состояний
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

states = [
    (df['Площадь_кв_м'].dropna(), 'Исходные данные\n(с пропусками)', 'steelblue'),
    (df['Площадь_заполненная'], 'После заполнения\nмедианой', 'seagreen'),
    (area_winsorized, 'После замены\nвыбросов', 'coral')
]

for ax, (data_state, title, color) in zip(axes, states):
    ax.hist(data_state, bins=25, edgecolor='white', color=color, alpha=0.8)
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.set_xlabel('Площадь (кв.м)')
    ax.set_ylabel('Количество квартир')
    ax.axvline(data_state.median(), color='red', linestyle='--', linewidth=1.5)

plt.tight_layout()
plt.show()

print("\n✅ Обработка данных завершена успешно!")
print("📊 Данные готовы к дальнейшему анализу и построению моделей.")