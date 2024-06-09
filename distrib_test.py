import scipy.stats as sps
import numpy as np
import matplotlib.pyplot as plt
from seaborn import histplot


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

    
    #Диаграмма накопленных частот
    histplot(data, bins = round(1 + 3.2*np.log10(len(data))), stat='density', label='Наблюдения')

    #Теоретические подгонки
    x = np.linspace(min(data), max(data), 100)
    plt.plot(x, expected_normal.pdf(x), 'k--', label=f'Теоретическая\nподгонка $N({expected_normal.mu:.2f}, {expected_normal.std:.2f})$')
    plt.plot(x, expected_beta.pdf(x), 'r--', label=f'Теоретическая\nподгонка $B({expected_beta.a:.2f}, {expected_beta.b:.2f})$')

    plt.legend()
    plt.title(f"{title}\nКритерий Пирсона pvalue - {stat_box.pvalues['chisquare']:.3f}\nКритерий К-С Нормальное-pvalue - {stat_box.pvalues['norm_ks']:.3f} Бета-pvalue - {stat_box.pvalues['beta_ks']:.3f}")
    if not isSubplot:
        plt.show()
    
    return stat_box