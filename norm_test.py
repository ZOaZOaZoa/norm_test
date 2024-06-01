import scipy.stats as sps
import numpy as np
import matplotlib.pyplot as plt

def plot_fit(data, title="", isSubplot=False):
    data_mean = np.mean(data)
    data_std = np.std(data)

    #Теоретическое распределение
    x = np.linspace(min(data), max(data), 100)
    plt.plot(x, sps.norm.pdf(x, loc=data_mean, scale=data_std), 'k--', label='Теоретическая\nподгонка')
    
    #Диаграмма накопленных частот
    (observed, bins, _) = plt.hist(data, bins = round(1 + 3.2*np.log10(len(data))), density=True, label='Наблюдения')

    bin_centers = [ (bins[i] + bins[i+1])/2 for i in range(len(bins) - 1) ]
    bin_width = bins[1] - bins[0]

    #Ожидаемая диаграмма накопленных частот
    expected = np.array([ sps.norm.pdf(bin_c, loc=data_mean, scale=data_std) for bin_c in bin_centers ])

    _, chisquare_pvalue = chisquare_test(data)
    _, ks_pvalue = ks_test(data)
    _, beta_ks_pvalue, alpha, beta = beta_ks_test(data)
    plt.plot(x, sps.beta.pdf(x, alpha, beta), 'r--', label='Теоретическая\nподгонка $B(\\alpha, \\beta)$')

    plt.bar(bin_centers, height=expected, width=0.4*bin_width, color='orange', label='Ожидаемое')
    plt.legend()
    plt.title(f'{title}\nКритерий Пирсона pvalue - {chisquare_pvalue:.3f}\nКритерий К-С Нормальное-pvalue - {ks_pvalue:.3f} Бета-pvalue - {beta_ks_pvalue:.3f}')
    if not isSubplot:
        plt.show()
    
    return (chisquare_pvalue, ks_pvalue, beta_ks_pvalue)

def chisquare_test(data):
    return sps.normaltest(data)

def ks_test(data):
    mu = np.mean(data)
    std = np.std(data)
    scaled_data = (data - mu)/std
    return sps.kstest(scaled_data, sps.norm.cdf)

def beta_ks_test(data):
    mu = np.mean(data)
    var = np.var(data, ddof=1)
    r = -(mu ** 2 - mu + var)/var
    alpha = mu * r
    beta = (1-mu) * r
    return sps.kstest(data, 'beta', [alpha, beta])[:2]+ (alpha, beta)

def main():
    #data = sps.norm.rvs(size=100, loc=12, scale=13)   
    #data = sps.norm.rvs(size=10000)
    data = sps.beta.rvs(44, 3.5, size=1000)
    plot_fit(data)


if __name__ == '__main__':
    main()
