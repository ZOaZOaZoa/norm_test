import norm_test
import scipy.stats as sps
from random import random
from tqdm import tqdm

def test(data_generator):
    correct_pierson = 0
    correct_ks = 0
    total = 100
    for i in tqdm(range(total), leave=False):
        data = data_generator()
        statistic, pvalue = norm_test.chisquare_test(data, plot=False)
        if pvalue >= 0.95:
            correct_pierson += 1
        
        statistic, pvalue = norm_test.ks_test(data)
        if pvalue >= 0.05:
            correct_ks += 1

    print(f'{"Критерий Пирсона:":<35}{correct_pierson}/{total} раз критерий выполняется')
    print(f'{"Критерий Колмогорова-Смирнова:":<35}{correct_ks}/{total} раз критерий выполняется')

#1 Нормальные распределения
print('1. Нормальные распределения. Объём выборки 100')
def norm_generator():
    mu = 0
    sigma = 1
    return sps.norm.rvs(size=100, loc=mu, scale=sigma)
test(norm_generator)
print()

#2 Экспоненциальные распределения
print('2. Экспоненциальные распределения. Объём выборки 100')
def expon_generator():
    mu = (random() - 0.5) * 30
    sigma = 1 + random() * 30
    return sps.expon.rvs(size=100, loc=mu, scale=sigma)
test(expon_generator)
print()

#3 Экспоненциальные распределения 2
print('3. Экспоненциальные распределения. Объём выборки 1000')
def expon_generator2():
    mu = (random() - 0.5) * 30
    sigma = 1 + random() * 30
    return sps.expon.rvs(size=1000, loc=mu, scale=sigma)
test(expon_generator2)
print()

#4 Равномерные распределения 
print('4. Равномерные распределения. Объём выборки 100')
def uniform_generator():
    a = (random() - 0.5) * 30
    b = max([a + random() * 30, 1])
    return sps.expon.rvs(size=100, loc=a, scale=b)
test(uniform_generator)
print()

#5 Равномерные распределения 2
print('5. Равномерные распределения. Объём выборки 1000')
def uniform_generator():
    a = (random() - 0.5) * 30
    b = max([a + random() * 30, 1])
    return sps.expon.rvs(size=1000, loc=a, scale=b)
test(uniform_generator)
print()

#6 Распределение Коши 
print('6. Распределение Коши. Объём выборки 100')
def cauchy_generator():
    mu = (random() - 0.5) * 30
    sigma = 1 + random() * 30
    return sps.cauchy.rvs(size=100, loc=mu, scale=sigma)
test(cauchy_generator)
print()


""" Типовой вывод. Видно, что во всех экспериментах используемая реализация теста Колмогорова-Смирнова (Лилиефорса) показывает себя лучше.
1. Нормальные распределения. Объём выборки 100
Критерий Пирсона:                  100/100 раз критерий выполняется
Критерий Колмогорова-Смирнова:     100/100 раз критерий выполняется

2. Экспоненциальные распределения. Объём выборки 100
Критерий Пирсона:                  76/100 раз критерий выполняется
Критерий Колмогорова-Смирнова:     4/100 раз критерий выполняется

3. Экспоненциальные распределения. Объём выборки 1000
Критерий Пирсона:                  26/100 раз критерий выполняется
Критерий Колмогорова-Смирнова:     0/100 раз критерий выполняется

4. Равномерные распределения. Объём выборки 100
Критерий Пирсона:                  70/100 раз критерий выполняется
Критерий Колмогорова-Смирнова:     6/100 раз критерий выполняется

5. Равномерные распределения. Объём выборки 1000
Критерий Пирсона:                  20/100 раз критерий выполняется
Критерий Колмогорова-Смирнова:     0/100 раз критерий выполняется

6. Распределение Коши. Объём выборки 100
Критерий Пирсона:                  15/100 раз критерий выполняется
Критерий Колмогорова-Смирнова:     0/100 раз критерий выполняется
"""