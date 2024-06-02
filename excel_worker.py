import norm_test
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from math import ceil
from datetime import datetime

def get_col_no_blank(df, col_name):
    '''Убирает из серии NaN и нулевые значения'''
    k = pd.Series(df[df[col_name].apply(lambda x: type(x) in [int, np.int64, float, np.float64])][col_name]).values.astype(np.float64)
    k = k[~np.isnan(k)]
    k = k[k!=0]

    return k

def main():
    #Открытыие Excel файла
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

    cur_datetime = datetime.now().strftime("%d.%m.%Y %H-%M-%S")
    saveFolder = 'png\\' + str(cur_datetime)
    if not os.path.isdir('png'):
        os.mkdir('png')
    if not os.path.isdir(saveFolder):
        os.mkdir(saveFolder)
    col_chunks = np.array_split(df.columns, ceil(len(df.columns) / 4))

    protocol_columns = ['Выборка', 'Пирсон pvalue', 'КС нормальное pvalue', 'КС бета pvalue']
    protocol_rows = []

    #Построение графиков и проверка критериев согласия
    print('Строим графики. Если pvalue > 0.05, значит критерий согласия выполняется')
    plt.rcParams.update({'font.size': 14})
    for k, chunk in enumerate(col_chunks):
        print(f'{k+1} график из {len(col_chunks)}')
        for i, col in enumerate(chunk):
            vals = get_col_no_blank(df, col) #фильтрация NaN и нулей
            plt.subplot(2,2, i+1)
            stat_box = norm_test.plot_fit(vals, col, isSubplot=True) #Построение графика и получение результатов
            protocol_rows += [(col, stat_box.pvalues['chisquare'], stat_box.pvalues['norm_ks'], stat_box.pvalues['beta_ks']),]
            print(stat_box.pvals_info(col)) #Протоколирование в консоли
        plt.subplots_adjust(hspace=0.3, left=0.2, right=0.8)
        fig = plt.gcf()
        fig.set_size_inches((25, 12), forward=False)
        fig.savefig(f'{saveFolder}\\Figure_{k+1}.png', bbox_inches='tight', dpi=150)
        plt.close()
    
    protocol = pd.DataFrame(protocol_rows, columns=protocol_columns)
    if not os.path.isdir('protocol'):
        os.mkdir('protocol')
    protocol.to_excel(f'protocol\\{cur_datetime}.xlsx')
    
    print(f'Построенные графики сохранены в: {os.getcwd()}\\{saveFolder}')
    print(f'Протокол сохранен в: {os.getcwd()}\\protocol')

if __name__ == '__main__':
    main()