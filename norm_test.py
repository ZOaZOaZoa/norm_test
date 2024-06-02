import scipy.stats as sps
import numpy as np
import matplotlib.pyplot as plt


class PDFNormal:
    @staticmethod
    def estimate_params(data):
        return (np.mean(data), np.std(data, ddof=1))
    
    def __init__(self, data):
        self.mu, self.std = PDFNormal.estimate_params(data)
    
    def pdf(self, x):
        return sps.norm.pdf(x, loc=self.mu, scale=self.std)

class PDFBeta:
    @staticmethod
    def estimate_params(data):
        mu = np.mean(data)
        var = np.var(data, ddof=1)
        r = -(mu ** 2 - mu + var)/var
        alpha = mu * r
        beta = (1-mu) * r
        return (alpha, beta)
    
    def __init__(self, data):
        self.a, self.b = PDFBeta.estimate_params(data)

    def pdf(self, x):
        return sps.beta.pdf(x, self.a, self.b)

class DataContainer:
    def __init__(self, data):
        self.data = data
        self.pvalues = dict()

    def testHypotheses(self, normal: PDFNormal, beta: PDFBeta) -> None:
        _, self.pvalues['chisquare'] = sps.normaltest(self.data)
        _, self.pvalues['norm_ks'] = sps.kstest(self.data, 'norm', [normal.mu, normal.std])
        _, self.pvalues['beta_ks'] = sps.kstest(self.data, 'beta', [beta.a, beta.b])
        
    def pvals_info(self, label: str):
        if len(self.pvalues) == 0:
            return ''
        
        names = {
            'chisquare': 'Критерий Пирсона',
            'norm_ks': 'Критерий К-С Нормальное распр.',
            'beta_ks': 'Критерий К-С Бета-распр.'
        }
        res = f'{label:<30} '
        for test_name, pvalue in self.pvalues.items():
            res += f'{names[test_name]}: {pvalue:<8.3f} '
        
        return res


def plot_fit(data, title="", isSubplot=False):
    stat_box = DataContainer(data)
    expected_normal = PDFNormal(data)
    expected_beta = PDFBeta(data)

    stat_box.testHypotheses(expected_normal, expected_beta)

    #Теоретическое распределение
    x = np.linspace(min(data), max(data), 100)
    plt.plot(x, expected_normal.pdf(x), 'k--', label='Теоретическая\nподгонка')
    
    #Диаграмма накопленных частот
    (observed, bins, _) = plt.hist(data, bins = round(1 + 3.2*np.log10(len(data))), density=True, label='Наблюдения')

    bin_centers = [ (bins[i] + bins[i+1])/2 for i in range(len(bins) - 1) ]
    bin_width = bins[1] - bins[0]

    #Ожидаемая диаграмма накопленных частот
    expected_hist = np.array([ expected_normal.pdf(bin_c) for bin_c in bin_centers ])
    plt.plot(x, expected_beta.pdf(x), 'r--', label='Теоретическая\nподгонка $B(\\alpha, \\beta)$')
    plt.bar(bin_centers, height=expected_hist, width=0.4*bin_width, color='orange', label='Ожидаемое')
    plt.legend()
    plt.title(f"{title}\nКритерий Пирсона pvalue - {stat_box.pvalues['chisquare']:.3f}\nКритерий К-С Нормальное-pvalue - {stat_box.pvalues['norm_ks']:.3f} Бета-pvalue - {stat_box.pvalues['beta_ks']:.3f}")
    if not isSubplot:
        plt.show()
    
    return stat_box


def main():
    #data = sps.norm.rvs(size=100, loc=12, scale=13)   
    #data = sps.norm.rvs(size=10000)
    data = sps.beta.rvs(44, 3.5, size=1000)
    plot_fit(data)


if __name__ == '__main__':
    main()
