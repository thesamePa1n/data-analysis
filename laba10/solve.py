import numpy as np
import pandas as pd

i = 11  
print("="*80)
print(f"ЛАБОРАТОРНАЯ РАБОТА №10: ДВУХФАКТОРНЫЙ ДИСПЕРСИОННЫЙ АНАЛИЗ")
print(f"Вариант: i = {i}")
print("="*80)

data_matrix = np.array([
    [9.3 + 0.1*i, 8.9 + 0.1*i, 6.5 + 0.1*i],  
    [9.4 + 0.1*i, 9.1 + 0.1*i, 7.1 + 0.1*i],  
    [12.6 + 0.1*i, 9.8 + 0.1*i, 8.0 + 0.1*i] 
])

print(f"\n1. МАТРИЦА ДАННЫХ (i={i}):")
df = pd.DataFrame(data_matrix, 
                  index=['1001-1500 см³', '1501-2000 см³', 'Более 2000 см³'],
                  columns=['Бензин со свинцом', 'Бензин без свинца', 'Дизельное топливо'])
print(df.round(2).to_string())

n = 4  # количество автомобилей
np.random.seed(42)

car_data = np.zeros((n, 3, 3))

print(f"\n2. ГЕНЕРАЦИЯ ДАННЫХ ДЛЯ {n} АВТОМОБИЛЕЙ:")
print("Базовые значения + индивидуальные отклонения + случайная ошибка")

for car in range(n):
    for vol in range(3):
        for fuel in range(3):
            base = data_matrix[vol, fuel]
            individual_effect = np.random.normal(0, 0.3)
            error = np.random.normal(0, 0.2)
            car_data[car, vol, fuel] = base + individual_effect + error

print("Данные сгенерированы")

print("\n" + "="*80)
print("3. ВЫЧИСЛЕНИЕ ВСЕХ НЕОБХОДИМЫХ СУММ")
print("="*80)

a = 3 
b = 3 
N = n * a * b  

print(f"\nПараметры:")
print(f"n (автомобили) = {n}")
print(f"a (уровни A) = {a}")
print(f"b (уровни B) = {b}")
print(f"N = n × a × b = {n} × {a} × {b} = {N}")

# 1. Суммы по фактору A (по всем автомобилям и видам топлива)
sums_A = car_data.sum(axis=(0, 2))
ΣT2_A = np.sum(sums_A**2)
print(f"\n4.1. ΣT²_A (сумма квадратов сумм по фактору A):")
print(f"    Суммы по объемам: {sums_A}")
print(f"    Квадраты: {[x**2 for x in sums_A]}")
print(f"    ΣT²_A = {sums_A[0]**2:.2f} + {sums_A[1]**2:.2f} + {sums_A[2]**2:.2f} = {ΣT2_A:.2f}")

# 2. Суммы по фактору B (по всем автомобилям и объемам)
sums_B = car_data.sum(axis=(0, 1))
ΣT2_B = np.sum(sums_B**2)
print(f"\n4.2. ΣT²_B (сумма квадратов сумм по фактору B):")
print(f"    Суммы по топливу: {sums_B}")
print(f"    Квадраты: {[x**2 for x in sums_B]}")
print(f"    ΣT²_B = {sums_B[0]**2:.2f} + {sums_B[1]**2:.2f} + {sums_B[2]**2:.2f} = {ΣT2_B:.2f}")

# 3. Суммы по ячейкам AB (по всем автомобилям)
sums_AB = car_data.sum(axis=0)
ΣT2_AB = np.sum(sums_AB**2)
print(f"\n4.3. ΣT²_AB (сумма квадратов сумм по ячейкам A×B):")
print(f"    Матрица сумм AB (3×3):")
for i in range(3):
    print(f"    Строка {i+1}: {sums_AB[i]} → квадраты: {[x**2 for x in sums_AB[i]]}")
print(f"    ΣT²_AB = сумма всех 9 квадратов = {ΣT2_AB:.2f}")

# 4. Индивидуальные суммы по автомобилям
sums_H = car_data.sum(axis=(1, 2))
ΣT2_H = np.sum(sums_H**2)
print(f"\n4.4. ΣT²_H (сумма квадратов индивидуальных сумм):")
print(f"    Индивидуальные суммы по автомобилям: {sums_H}")
print(f"    Квадраты: {[x**2 for x in sums_H]}")
print(f"    ΣT²_H = {sums_H[0]**2:.2f} + {sums_H[1]**2:.2f} + {sums_H[2]**2:.2f} + {sums_H[3]**2:.2f} = {ΣT2_H:.2f}")

# 5. ΣT²_AH (автомобиль × объем)
sums_AH = np.zeros((n, a))
for car in range(n):
    for level_a in range(a):
        sums_AH[car, level_a] = np.sum(car_data[car, level_a, :])
ΣT2_AH = np.sum(sums_AH**2)
print(f"\n4.5. ΣT²_AH (сумма квадратов сумм по сочетаниям автомобиль×объем):")
print(f"    Матрица AH ({n}×{a}):")
for car in range(n):
    print(f"    Автомобиль {car+1}: {sums_AH[car]} → квадраты: {[x**2 for x in sums_AH[car]]}")
print(f"    ΣT²_AH = сумма всех {n*a} квадратов = {ΣT2_AH:.2f}")

# 6. ΣT²_BH (автомобиль × топливо)
sums_BH = np.zeros((n, b))
for car in range(n):
    for level_b in range(b):
        sums_BH[car, level_b] = np.sum(car_data[car, :, level_b])
ΣT2_BH = np.sum(sums_BH**2)
print(f"\n4.6. ΣT²_BH (сумма квадратов сумм по сочетаниям автомобиль×топливо):")
print(f"    Матрица BH ({n}×{b}):")
for car in range(n):
    print(f"    Автомобиль {car+1}: {sums_BH[car]} → квадраты: {[x**2 for x in sums_BH[car]]}")
print(f"    ΣT²_BH = сумма всех {n*b} квадратов = {ΣT2_BH:.2f}")

# 7. Общая сумма и константа C
total_sum = car_data.sum()
Σx = total_sum
C = (Σx**2) / N
print(f"\n4.7. Общая сумма и константа C:")
print(f"    Σx (общая сумма всех значений) = {Σx:.2f}")
print(f"    C = (Σx)²/N = ({Σx:.2f})²/{N} = {Σx**2:.2f}/{N} = {C:.2f}")

# 8. Σx² (сумма квадратов всех значений)
Σx2 = np.sum(car_data**2)
print(f"\n4.8. Σx² (сумма квадратов всех значений):")
print(f"    Σx² = {Σx2:.2f}")

print("\n" + "="*80)
print("5. РАСЧЕТ СУММ КВАДРАТОВ (SS) - КАЖДАЯ ФОРМУЛА")
print("="*80)

# 1. SSA
SSA = ΣT2_A / (n * b) - C
print(f"\n5.1. SSA (фактор A - объем двигателя):")
print(f"    SSA = ΣT²_A/(n×b) - C")
print(f"        = {ΣT2_A:.2f}/({n}×{b}) - {C:.2f}")
print(f"        = {ΣT2_A:.2f}/{n*b} - {C:.2f}")
print(f"        = {ΣT2_A/(n*b):.4f} - {C:.4f}")
print(f"        = {SSA:.4f}")

# 2. SSB
SSB = ΣT2_B / (n * a) - C
print(f"\n5.2. SSB (фактор B - вид топлива):")
print(f"    SSB = ΣT²_B/(n×a) - C")
print(f"        = {ΣT2_B:.2f}/({n}×{a}) - {C:.2f}")
print(f"        = {ΣT2_B:.2f}/{n*a} - {C:.2f}")
print(f"        = {ΣT2_B/(n*a):.4f} - {C:.4f}")
print(f"        = {SSB:.4f}")

# 3. SSH
SSH = ΣT2_H / (a * b) - C
print(f"\n5.3. SSH (индивидуальные различия):")
print(f"    SSH = ΣT²_H/(a×b) - C")
print(f"        = {ΣT2_H:.2f}/({a}×{b}) - {C:.2f}")
print(f"        = {ΣT2_H:.2f}/{a*b} - {C:.2f}")
print(f"        = {ΣT2_H/(a*b):.4f} - {C:.4f}")
print(f"        = {SSH:.4f}")

# 4. SSAB
SSAB = ΣT2_AB / n - C - SSA - SSB
print(f"\n5.4. SSAB (взаимодействие A×B):")
print(f"    SSAB = ΣT²_AB/n - C - SSA - SSB")
print(f"         = {ΣT2_AB:.2f}/{n} - {C:.2f} - {SSA:.4f} - {SSB:.4f}")
print(f"         = {ΣT2_AB/n:.4f} - {C:.4f} - {SSA:.4f} - {SSB:.4f}")
print(f"         = {ΣT2_AB/n - C:.4f} - {SSA:.4f} - {SSB:.4f}")
print(f"         = {SSAB:.4f}")

# 5. SSAH
SSAH = ΣT2_AH / b - C - SSA - SSH
print(f"\n5.5. SSAH (взаимодействие A×H):")
print(f"    SSAH = ΣT²_AH/b - C - SSA - SSH")
print(f"         = {ΣT2_AH:.2f}/{b} - {C:.2f} - {SSA:.4f} - {SSH:.4f}")
print(f"         = {ΣT2_AH/b:.4f} - {C:.4f} - {SSA:.4f} - {SSH:.4f}")
print(f"         = {ΣT2_AH/b - C:.4f} - {SSA:.4f} - {SSH:.4f}")
print(f"         = {SSAH:.4f}")

# 6. SSBH
SSBH = ΣT2_BH / a - C - SSB - SSH
print(f"\n5.6. SSBH (взаимодействие B×H):")
print(f"    SSBH = ΣT²_BH/a - C - SSB - SSH")
print(f"         = {ΣT2_BH:.2f}/{a} - {C:.2f} - {SSB:.4f} - {SSH:.4f}")
print(f"         = {ΣT2_BH/a:.4f} - {C:.4f} - {SSB:.4f} - {SSH:.4f}")
print(f"         = {ΣT2_BH/a - C:.4f} - {SSB:.4f} - {SSH:.4f}")
print(f"         = {SSBH:.4f}")

# 7. SS_total
SS_total = Σx2 - C
print(f"\n5.7. SS_total (общая сумма квадратов):")
print(f"    SS_total = Σx² - C")
print(f"             = {Σx2:.2f} - {C:.2f}")
print(f"             = {SS_total:.4f}")

# 8. SSABH (остаток)
SSABH = SS_total - SSA - SSB - SSH - SSAB - SSAH - SSBH
print(f"\n5.8. SSABH (взаимодействие A×B×H - остаток):")
print(f"    SSABH = SS_total - SSA - SSB - SSH - SSAB - SSAH - SSBH")
print(f"          = {SS_total:.4f} - {SSA:.4f} - {SSB:.4f} - {SSH:.4f} - {SSAB:.4f} - {SSAH:.4f} - {SSBH:.4f}")
print(f"          = {SSABH:.4f}")

# Проверка
SS_sum = SSA + SSB + SSH + SSAB + SSAH + SSBH + SSABH
print(f"\nПроверка: SSA + SSB + SSH + SSAB + SSAH + SSBH + SSABH = {SS_sum:.6f}")
print(f"         SS_total = {SS_total:.6f}")
print(f"         Разница = {abs(SS_total - SS_sum):.10f}")

print("\n" + "="*80)
print("6. РАСЧЕТ СТЕПЕНЕЙ СВОБОДЫ (df)")
print("="*80)

df_A = a - 1
df_B = b - 1
df_H = n - 1
df_AB = (a - 1) * (b - 1)
df_AH = (a - 1) * (n - 1)
df_BH = (b - 1) * (n - 1)
df_ABH = (a - 1) * (b - 1) * (n - 1)
df_total = N - 1

print(f"df_A = a - 1 = {a} - 1 = {df_A}")
print(f"df_B = b - 1 = {b} - 1 = {df_B}")
print(f"df_H = n - 1 = {n} - 1 = {df_H}")
print(f"df_AB = (a-1)×(b-1) = ({a}-1)×({b}-1) = {a-1}×{b-1} = {df_AB}")
print(f"df_AH = (a-1)×(n-1) = ({a}-1)×({n}-1) = {a-1}×{n-1} = {df_AH}")
print(f"df_BH = (b-1)×(n-1) = ({b}-1)×({n}-1) = {b-1}×{n-1} = {df_BH}")
print(f"df_ABH = (a-1)×(b-1)×(n-1) = ({a}-1)×({b}-1)×({n}-1) = {a-1}×{b-1}×{n-1} = {df_ABH}")
print(f"df_total = N - 1 = {N} - 1 = {df_total}")

print("\n" + "="*80)
print("7. РАСЧЕТ СРЕДНИХ КВАДРАТОВ (MS)")
print("="*80)

MSA = SSA / df_A
MSB = SSB / df_B
MSH = SSH / df_H
MSAB = SSAB / df_AB
MSAH = SSAH / df_AH
MSBH = SSBH / df_BH
MSABH = SSABH / df_ABH

print(f"MSA = SSA / df_A = {SSA:.4f} / {df_A} = {MSA:.4f}")
print(f"MSB = SSB / df_B = {SSB:.4f} / {df_B} = {MSB:.4f}")
print(f"MSH = SSH / df_H = {SSH:.4f} / {df_H} = {MSH:.4f}")
print(f"MSAB = SSAB / df_AB = {SSAB:.4f} / {df_AB} = {MSAB:.4f}")
print(f"MSAH = SSAH / df_AH = {SSAH:.4f} / {df_AH} = {MSAH:.4f}")
print(f"MSBH = SSBH / df_BH = {SSBH:.4f} / {df_BH} = {MSBH:.4f}")
print(f"MSABH = SSABH / df_ABH = {SSABH:.4f} / {df_ABH} = {MSABH:.4f}")

print("\n" + "="*80)
print("8. РАСЧЕТ F-КРИТЕРИЕВ")
print("="*80)

FA = MSA / MSAH
FB = MSB / MSBH
FH = MSH / MSABH
FAB = MSAB / MSABH

print(f"FA = MSA / MSAH = {MSA:.4f} / {MSAH:.4f} = {FA:.4f}")
print(f"FB = MSB / MSBH = {MSB:.4f} / {MSBH:.4f} = {FB:.4f}")
print(f"FH = MSH / MSABH = {MSH:.4f} / {MSABH:.4f} = {FH:.4f}")
print(f"FAB = MSAB / MSABH = {MSAB:.4f} / {MSABH:.4f} = {FAB:.4f}")

print("\n" + "="*90)
print("9. ИТОГОВАЯ ТАБЛИЦА ДИСПЕРСИОННОГО АНАЛИЗА (ANOVA)")
print("="*90)

print(f"{'Источник вариации':<25} {'SS':<12} {'df':<6} {'MS':<12} {'F':<12}")
print("-"*90)

print(f"{'A (объем двигателя)':<25} {SSA:<12.4f} {df_A:<6} {MSA:<12.4f} {FA:<12.4f}")
print(f"{'B (вид топлива)':<25} {SSB:<12.4f} {df_B:<6} {MSB:<12.4f} {FB:<12.4f}")
print(f"{'H (автомобили)':<25} {SSH:<12.4f} {df_H:<6} {MSH:<12.4f} {FH:<12.4f}")
print(f"{'A×B':<25} {SSAB:<12.4f} {df_AB:<6} {MSAB:<12.4f} {FAB:<12.4f}")
print(f"{'A×H (ошибка A)':<25} {SSAH:<12.4f} {df_AH:<6} {MSAH:<12.4f} {'-':<12}")
print(f"{'B×H (ошибка B)':<25} {SSBH:<12.4f} {df_BH:<6} {MSBH:<12.4f} {'-':<12}")
print(f"{'A×B×H (остаток)':<25} {SSABH:<12.4f} {df_ABH:<6} {MSABH:<12.4f} {'-':<12}")
print(f"{'Общее':<25} {SS_total:<12.4f} {df_total:<6} {'-':<12} {'-':<12}")
print("="*90)

print("\n" + "="*80)
print("10. СТАТИСТИЧЕСКИЕ ВЫВОДЫ")
print("="*80)

from scipy import stats

alpha = 0.05
F_crit_A = stats.f.ppf(1 - alpha, df_A, df_AH)
F_crit_B = stats.f.ppf(1 - alpha, df_B, df_BH)
F_crit_H = stats.f.ppf(1 - alpha, df_H, df_ABH)
F_crit_AB = stats.f.ppf(1 - alpha, df_AB, df_ABH)

print(f"Критические значения F (α={alpha}):")
print(f"  Fкрит({df_A},{df_AH}) = {F_crit_A:.4f}")
print(f"  Fкрит({df_B},{df_BH}) = {F_crit_B:.4f}")
print(f"  Fкрит({df_H},{df_ABH}) = {F_crit_H:.4f}")
print(f"  Fкрит({df_AB},{df_ABH}) = {F_crit_AB:.4f}")

print(f"\nВыводы о статистической значимости:")
print(f"1. Фактор A (Объем): FA={FA:.4f} {'>' if FA > F_crit_A else '<'} {F_crit_A:.4f} → {'ЗНАЧИМ' if FA > F_crit_A else 'НЕЗНАЧИМ'}")
print(f"2. Фактор B (Топливо): FB={FB:.4f} {'>' if FB > F_crit_B else '<'} {F_crit_B:.4f} → {'ЗНАЧИМ' if FB > F_crit_B else 'НЕЗНАЧИМ'}")
print(f"3. Индив. различия: FH={FH:.4f} {'>' if FH > F_crit_H else '<'} {F_crit_H:.4f} → {'ЗНАЧИМ' if FH > F_crit_H else 'НЕЗНАЧИМ'}")
print(f"4. Взаимодействие: FAB={FAB:.4f} {'>' if FAB > F_crit_AB else '<'} {F_crit_AB:.4f} → {'ЗНАЧИМ' if FAB > F_crit_AB else 'НЕЗНАЧИМ'}")

print("\n" + "="*80)
print("11. ИТОГОВОЕ ЗАКЛЮЧЕНИЕ")
print("="*80)

print("""
РЕЗУЛЬТАТЫ ДИСПЕРСИОННОГО АНАЛИЗА:

На основе проведенного двухфакторного дисперсионного анализа 
для связанных выборок можно сделать следующие выводы:

1. ФАКТОР A (ОБЪЕМ ДВИГАТЕЛЯ):
   - F({},{}) = {:.4f}, p {} 0.05
   - {} влияет на потребление топлива
   - Автомобили с большим объемом двигателя (>2000 см³) 
     потребляют значительно больше топлива

2. ФАКТОР B (ВИД ТОПЛИВА):
   - F({},{}) = {:.4f}, p {} 0.05
   - {} влияет на потребление топлива
   - Дизельное топливо является наиболее экономичным

3. ИНДИВИДУАЛЬНЫЕ РАЗЛИЧИЯ:
   - F({},{}) = {:.4f}, p {} 0.05
   - Автомобили {} различаются по потреблению топлива

4. ВЗАИМОДЕЙСТВИЕ A×B:
   - F({},{}) = {:.4f}, p {} 0.05
   - Взаимодействие {} значимо
   - Эффект вида топлива {} для разных объемов

ПРАКТИЧЕСКИЕ РЕКОМЕНДАЦИИ:
1. Для минимизации расхода топлива:
   - Выбирать автомобили с объемом 1001-1500 см³
   - Использовать дизельное топливо
2. При наличии значимого взаимодействия необходимо
   подбирать оптимальные комбинации объема и топлива
""".format(
    df_A, df_AH, FA, 
    '<' if FA > F_crit_A else '>',
    'ЗНАЧИМО' if FA > F_crit_A else 'НЕЗНАЧИМО',
    
    df_B, df_BH, FB,
    '<' if FB > F_crit_B else '>',
    'ЗНАЧИМО' if FB > F_crit_B else 'НЕЗНАЧИМО',
    
    df_H, df_ABH, FH,
    '<' if FH > F_crit_H else '>',
    'ЗНАЧИМО' if FH > F_crit_H else 'НЕЗНАЧИМЫ',
    
    df_AB, df_ABH, FAB,
    '<' if FAB > F_crit_AB else '>',
    'статистически' if FAB > F_crit_AB else 'статистически не',
    'различается' if FAB > F_crit_AB else 'одинаков'
))

print("="*80)