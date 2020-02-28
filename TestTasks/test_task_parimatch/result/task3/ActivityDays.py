import pandas as pd
from datetime import date as dt
import sys
import os


def activity():


    # если путь к файлу не был передан аргументом из командной строки, предпологаем, что он лежит рядом со скриптом
    try:
        source_file = sys.argv[1]
    except ValueError:
        print('Argument wasn\'t sent. Use Activity_Days.csv by default')
    finally:
        source_file = 'Activity_Days.csv'

    # если файл не существует, вызываем исключение
    if not os.path.exists(source_file):
        raise FileExistsError(f"File {source_file} doesn't exist.")

    if not os.path.isdir('D:/Send_All'):  # проверяем, существует ли конечная дериктория. Если нет - создаем.
        try:
            os.mkdir('D:/Send_All')
        except FileExistsError:
            print('Directory D:/Send_All doesn\'t exist.')

    try:
        df = pd.read_csv(source_file, sep=';')
        df['Date'] = pd.to_datetime(df.Date, format='%d--%m--%Y')
    except ValueError:
        print('Format of the file is different')


    buffer = []
    for date in pd.date_range('2018-01-01', '2019-12-31'):  # можно указать dt.today()

        # Tuesday - 1, Friday - 4
        if dt.weekday(date) == 1 or dt.weekday(date) == 4:
            if buffer:
                with open(f'D:/Send_All/Send_{date.date().strftime("%d-%m-%y")}.csv', 'w') as f:
                    f.write(','.join(map(str, buffer)) + ',' + date.date().strftime("%d-%m-%y"))

                buffer.clear()

        if df[df.Date == str(date)]['id'].tolist():
            buffer += df[df.Date == str(date)]['id'].tolist()


if __name__ == '__main__':
    activity()