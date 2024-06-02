import norm_test
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from math import ceil
from datetime import datetime

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
    print('Строим графики. Если pvalue > 0.05, значит критерий согласия выполняется')
    col_chunks = np.array_split(df.columns, ceil(len(df.columns) / 4))

    saveFolder = 'png\\' + str(datetime.now().strftime("%d.%m.%Y %H-%M-%S"))
    if not os.path.isdir('png'):
        os.mkdir('png')
    if not os.path.isdir(saveFolder):
        os.mkdir(saveFolder)

    plt.rcParams.update({'font.size': 14})

    for k, chunk in enumerate(col_chunks):
        print(f'{k+1} график из {len(col_chunks)}')
        for i, col in enumerate(chunk):
            vals = get_col_no_blank(df, col)
            plt.subplot(2,2, i+1)
            stat_box = norm_test.plot_fit(vals, col, isSubplot=True)
            print(stat_box.pvals_info(col))
        plt.subplots_adjust(hspace=0.3, left=0.2, right=0.8)
        fig = plt.gcf()
        fig.set_size_inches((25, 12), forward=False)
        fig.savefig(f'{saveFolder}\\Figure_{k+1}.png', bbox_inches='tight', dpi=150)
        plt.close()
    
    print(f'Построенные графики сохранены в: {os.getcwd()}\\{saveFolder}')
    input('Чтобы выйти из программы нажмите ENTER...')

if __name__ == '__main__':
    main()