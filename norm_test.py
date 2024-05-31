import scipy.stats as sps
import numpy as np
import matplotlib.pyplot as plt

def chisquare_test(data, plot=True):
    data_mean = np.mean(data)
    data_std = np.std(data)

    #Теоретическое распределение
    if plot:
        x = np.linspace(min(data), max(data), 100)
        plt.plot(x, sps.norm.pdf(x, loc=data_mean, scale=data_std), color='k', label='theoretical')
    
    #Диаграмма накопленных частот
    (observed, bins, _) = plt.hist(data, bins = round(1 + 3.2*np.log10(len(data))), density=True, color='g', label='observed')

    if not plot:
        plt.close()

    bin_centers = [ (bins[i] + bins[i+1])/2 for i in range(len(bins) - 1) ]
    bin_width = bins[1] - bins[0]

    #Ожидаемая диаграмма накопленных частот
    expected = np.array([ sps.norm.pdf(bin_c, loc=data_mean, scale=data_std) for bin_c in bin_centers ])
    #Корректировка, чтобы суммы накопленных частот были равны (требуется для теста)
    expected *= np.sum(observed)/np.sum(expected)
    
    #Критерий согласия Пирсона (хи-квадрат)
    statistic, pvalue = sps.chisquare(observed, expected, ddof=2)

    if plot:
        plt.bar(bin_centers, height=expected, width=0.4*bin_width, color='orange', label='expected')
        plt.legend()
        plt.title(f'Statistic: {statistic:.2f}, pvalue: {pvalue:.2f}')
        plt.show()
    
    return statistic, pvalue
    

def main():
    #data = sps.norm.rvs(size=100, loc=12, scale=13)   
    #data = sps.norm.rvs(size=10000)
    data = sps.norm.rvs(size=1000)
    statistic, pvalue = chisquare_test(data, plot=False)
    print(f'Statistic: {statistic:.2f}, pvalue: {pvalue:.2f}')

if __name__ == '__main__':
    main()
