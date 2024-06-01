import norm_test
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from math import ceil

def get_col_no_blank(df, col_name):
    k = pd.Series(df[df[col_name].apply(lambda x: type(x) in [int, np.int64, float, np.float64])][col_name]).values.astype(np.float64)
    k = k[~np.isnan(k)]
    k = k[k!=0]

    return k

def main():
    while True:
        print('Введите название (или путь) файла Excel с данными: ', end='')
        file = input()
        if os.path.isfile(file):
            print(f'\nОткрываем файл {file}')
            break

        print(f'Файла {file} не существует')


    xl = pd.ExcelFile(file)
    sheets = xl.sheet_names

    if len(sheets) == 0:
        print('Не найдено листов в файле')
        exit()

    while True:
        print('Найдены следующие листы: ', end='')
        print(*sheets, sep='; ')
        if len(sheets) == 1:
            print(f'Выбран лист {sheets[0]}\n')
            sheet = sheets[0]
            break

        print('Введите название листа, откуда брать данные: ', end='')
        sheet = input()
        if sheet in sheets:
            print(f'Выбран лист {sheets[0]}\n')
            break

        print(f'Листа {sheet} среди найденных листов нет')

    print('Парсим данные')
    df = xl.parse(sheet)
    
    print('Найдены следующие столбцы:', *df.columns)
    print('Строим графики. Если pvalue > 0.05, значит критерий на нормальность выполняется')
    col_chunks = np.array_split(df.columns, ceil(len(df.columns) / 4))
    for k, chunk in enumerate(col_chunks):
        print(f'{k+1} график из {len(col_chunks)}')
        for i, col in enumerate(chunk):
            k = get_col_no_blank(df, col)
            plt.subplot(2,2, i+1)
            chisquare_pvalue, ks_pvalue, beta_ks_pvalue = norm_test.plot_fit(k, col, isSubplot=True)
            print(f'{col:<30} Критерий пирсона: {chisquare_pvalue:<8.3f} Критерий К-С: Нормальное - {ks_pvalue:<8.3f}, Бета - {beta_ks_pvalue:<8.3f}')
        plt.show()
        print()

if __name__ == '__main__':
    main()