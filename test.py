import norm_test
import scipy.stats as sps
from random import random
from tqdm import tqdm

def test(data_generator):
    correct = 0
    total = 100
    for i in tqdm(range(total), leave=False):
        data = data_generator()
        statistic, pvalue = norm_test.chisquare_test(data, plot=False)
        if pvalue >= 0.95:
            correct += 1

    print(f'{correct}/{total} раз критерий выполняется')

#1 Нормальные распределения
print('1. Нормальные распределения. Объём выборки 100')
def norm_generator():
    mu = (random() - 0.5) * 30
    sigma = 0.5 + random() * 30
    return sps.norm.rvs(size=100, loc=mu, scale=sigma)
test(norm_generator)

#2 Экспоненциальные распределения
print('2. Экспоненциальные распределения. Объём выборки 100')
def expon_generator():
    mu = (random() - 0.5) * 30
    sigma = 1 + random() * 30
    return sps.expon.rvs(size=100, loc=mu, scale=sigma)
test(expon_generator)

#3 Экспоненциальные распределения 2
print('3. Экспоненциальные распределения. Объём выборки 1000')
def expon_generator2():
    mu = (random() - 0.5) * 30
    sigma = 1 + random() * 30
    return sps.expon.rvs(size=1000, loc=mu, scale=sigma)
test(expon_generator2)

#4 Равномерные распределения 
print('4. Равномерные распределения. Объём выборки 100')
def uniform_generator():
    a = (random() - 0.5) * 30
    b = max([a + random() * 30, 1])
    return sps.expon.rvs(size=100, loc=a, scale=b)
test(uniform_generator)

#5 Равномерные распределения 2
print('5. Равномерные распределения. Объём выборки 1000')
def uniform_generator():
    a = (random() - 0.5) * 30
    b = max([a + random() * 30, 1])
    return sps.expon.rvs(size=1000, loc=a, scale=b)
test(uniform_generator)