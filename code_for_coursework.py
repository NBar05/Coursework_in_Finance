# Код для анализа методов машинного обучения
# в оценивании моделей ценообразования активов

import os
os.getcwd() # "/Users/nikitabaramiya/"
os.chdir("/Users/nikitabaramiya/Desktop/coursework2")

# для анализа времени работы алгоритмов
import time
from datetime import datetime

# для импорта (парсинга) и обработки данных
import re
import json
import numpy as np
import pandas as pd
from functools import reduce
from urllib.request import urlopen
from pandas_datareader.data import DataReader

# для визуализации данных
import seaborn as sns
import matplotlib.pyplot as plt

# для отображения распредления и нормировки данных
from scipy.stats import norm
from scipy.stats import zscore

# для проверки стационарности рядов
from statsmodels.tsa.stattools import adfuller

# для построения и оценивания моделей
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
# базовые оценщики
from sklearn.linear_model import ElasticNetCV
from sklearn.tree import DecisionTreeRegressor
# ансамбли
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
# ускоренная версия градиентного бустинга
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingRegressor


# Импорт финансовой информации
# Извлекаем символы и названия ценных бумаг из S&P500
sp500 = pd.read_excel("S&P500.xlsx")
symbols = sp500.Symbol.values
security_names = sp500.Security.values

# Рисунок 2 -- распределение компаний по секторам
plt.style.available
plt.style.use('seaborn-colorblind')
plt.figure(figsize = (15, 8))
p = sns.countplot(x = sp500['GICS Sector'], palette = 'Set2')
p.set_xticklabels(p.get_xticklabels(), rotation = 45, horizontalalignment = 'right')
p.set_xlabel('', fontsize = 10)
p.set_ylabel('', fontsize = 10)
p.set_title('Количество компаний по секторам', fontsize = 20)
plt.show()


## Парсинг данных с разных источников
# Извлекаем данные c yahoo, также фиксируем ценные бумаги, чьи данные не удалось извлечь
bad_names = []
stock_prices = pd.DataFrame()

# Настраиваем временной промежуток извлекаемых данных
now = datetime.now()
start = datetime(now.year - 15, 1, 1)
end = datetime(now.year, 3, 1)

timer = datetime.now()
for symbol in symbols:
    try:
        print(symbol)
        stock_df = DataReader(symbol, 'yahoo', start, end)
        stock_df['Name'] = symbol
        stock_df['Date'] = stock_df.index
        stock_prices = pd.concat([stock_prices, stock_df], ignore_index=True)
    except:
        bad_names.append(symbol)
        print('bad: %s' % (symbol))
    print(datetime.now() - timer)

# Проверяем на наличие неизвлечённых данных
print(bad_names) # ['ARNC'], не захотел

# Сбор данных длился около 10-ти минут
len(np.unique(stock_prices.Name)) # 504 ценной бумаги
stock_prices.shape # (1787812, 8)

# Сохраняем в csv
# name = f"S&P500_data_{start:%B-%d-%Y}_{end:%B-%d-%Y}.csv"
# stock_prices.to_csv(name, index = False)


# Функция для импорта данных с сайта:
# https://financialmodelingprep.com/developer/docs/company-financial-statement-growth-api/
def get_jsonparsed_data(url):
    # Receive the content of 'url', parse it as JSON and return the object
    response = urlopen(url)
    data = response.read().decode("utf-8")
    return json.loads(data)

# Функция для обработки содержимого каждой отдельной ссылки
def process_json_data(url):
    parse_data = get_jsonparsed_data(url)
    key, value = parse_data.items()
    # заносим в таблицу содержимое
    target_data = pd.DataFrame(value[1])
    target_data.rename(columns = {'date': 'Date'}, inplace = True)
    target_data['Date'] = target_data['Date'].apply(lambda x: \
    re.findall(r"\d{4}-\d{2}", str(x))[0])
    # возвращаем подготовленную таблицу
    return target_data

# Извлекаем всевозможные данные о финансовых отчётностях компаниях и их приложениях
bad_companies = []
company_profiles = pd.DataFrame()

timer = datetime.now()
for company in d:
    try:
        print(company)
        # отчёты о прибыли и убытках (31)
        url1 = f"https://financialmodelingprep.com/api/v3/financials/income-statement/{company}?period=quarter"
        income_statement = process_json_data(url1)
        # отчёты о балансе (29)
        url2 = f"https://financialmodelingprep.com/api/v3/financials/balance-sheet-statement/{company}?period=quarter"
        balance_sheet = process_json_data(url2)
        # отчёты о движении денежных средств (15)
        url3 = f"https://financialmodelingprep.com/api/v3/financials/cash-flow-statement/{company}?period=quarter"
        cash_flow = process_json_data(url3)
        # стоимости предприятий (6)
        url4 = f"https://financialmodelingprep.com/api/v3/enterprise-value/{company}?period=quarter"
        enterprise = process_json_data(url4)
        # ключевые метрики (48)
        url5 = f"https://financialmodelingprep.com/api/v3/company-key-metrics/{company}?period=quarter"
        metrics = process_json_data(url5)
        # параметры роста из отчёта о доходе (18)
        url6 = f"https://financialmodelingprep.com/api/v3/financial-statement-growth/{company}?period=quarter"
        growth = process_json_data(url6)
        # объединяем данные
        data_frames = [income_statement, balance_sheet, cash_flow, enterprise, metrics, growth]
        data_merged = reduce(lambda left, right: pd.merge(left, right, \
        on = ['Date'], how = 'outer'), data_frames)
        # добавляем имя и заносим в таблицу
        data_merged['Name'] = company
        company_profiles = pd.concat([company_profiles, data_merged])
    except:
        bad_companies.append(company)
        print('bad: %s' % (company))
    print(datetime.now() - timer)

# Проверяем на наличие неизвлечённых данных
print(bad_companies) # ['BRK-B', 'BF-B', 'J', 'NLOK', 'TFC']

# Сбор данных длился около 1-го часа
len(np.unique(company_profiles.Name)) # 500 ценных бумаг
company_profiles.shape # (21997, 149)

# Сохраняем в csv
# name = "data_financialmodelingprep.csv"
# company_profiles.to_csv(name, index = False)


## Работа с данными

# Открываем данные о деятельности фирм S&P500
company_profiles = pd.read_csv("data_financialmodelingprep.csv")
company_profiles.shape # (21997, 149)

# (company_profiles.dropna().Date.value_counts()).tail(41) # проблемы с датами
# нужно оставить только даты с месяцами 03, 06, 09, 12
boolean = company_profiles['Date'].apply(lambda x: \
bool(re.search(r"-(?:03|06|09|12)", x)))
# применяем фильтрование
company_profiles = company_profiles[boolean]
# убираем лишние года
boolean_years = company_profiles['Date'].apply(lambda x: \
re.findall(r"(\d{4})", str(x))[0]).astype(int) > 2008
# применяем фильтрование
company_profiles = company_profiles[boolean_years]
company_profiles.shape # (18091, 149)


# Реплицируем переменные из работы
profiles = company_profiles.sort_values(['Name', 'Date'])
spec_chars = profiles.loc[:, ['Name', 'Date']]

for symbol in company_profiles.Name.unique():
    print(symbol)
    filter = spec_chars.Name == symbol
    spec_chars.loc[filter, 'Stock Price'] = profiles.loc[filter, 'Stock Price']
    spec_chars.loc[filter, 'Dividend per Share'] = profiles.loc[filter, 'Dividend per Share']
    spec_chars.loc[filter, 'A2M'] = (profiles.loc[filter, 'Total assets'] / profiles.loc[filter, 'Market Cap']).shift(1)
    spec_chars.loc[filter, 'AC'] = (profiles.loc[filter, 'Working Capital'].diff() / profiles.loc[filter, 'Total shareholders equity']).shift(1)
    spec_chars.loc[filter, 'AT'] = (profiles.loc[filter, 'Total assets']).shift(1)
    spec_chars.loc[filter, 'NOA'] = ((profiles.loc[filter, 'Total assets'] - profiles.loc[filter, 'Cash and cash equivalents'] - \
    profiles.loc[filter, 'Cash and short-term investments'] - profiles.loc[filter, 'Long-term investments'] - \
    profiles.loc[filter, 'Total liabilities'] + profiles.loc[filter, 'Short-term debt'] + profiles.loc[filter, 'Long-term debt'] + \
    profiles.loc[filter, 'Interest Expense'] + profiles.loc[filter, 'Preferred Dividends'] + profiles.loc[filter, 'Total shareholders equity']) / \
    profiles.loc[filter, 'Total assets'].shift(1)).shift(1)
    spec_chars.loc[filter, 'ATO'] = (profiles.loc[filter, 'Revenue'] / spec_chars.loc[filter, 'NOA'].shift(1)).shift(1)
    spec_chars.loc[filter, 'B2M'] = (profiles.loc[filter, 'Book Value per Share'] * profiles.loc[filter, 'Number of Shares'] / \
    profiles.loc[filter, 'Market Cap']).shift(1)
    spec_chars.loc[filter, 'C'] = ((profiles.loc[filter, 'Cash and cash equivalents'] + profiles.loc[filter, 'Short-term investments']) / \
    profiles.loc[filter, 'Total assets']).shift(1)
    spec_chars.loc[filter, 'CF2B'] = (profiles.loc[filter, 'Free Cash Flow per Share'] / profiles.loc[filter, 'Book Value per Share']).shift(1)
    spec_chars.loc[filter, 'CF2P'] = (profiles.loc[filter, 'Free Cash Flow per Share'] / profiles.loc[filter, 'Stock Price']).shift(1)
    spec_chars.loc[filter, 'CTO'] = (profiles.loc[filter, 'Revenue'] / profiles.loc[filter, 'Total assets'].shift(1)).shift(1)
    spec_chars.loc[filter, 'D2A'] = (profiles.loc[filter, 'Depreciation & Amortization'] / profiles.loc[filter, 'Total assets']).shift(1)
    spec_chars.loc[filter, 'D2P'] = profiles.loc[filter, 'Dividend Yield'].shift(1)
    spec_chars.loc[filter, 'DPI2A'] = ((profiles.loc[filter, 'Property, Plant & Equipment Net'].diff() + profiles.loc[filter, 'Inventories'].diff()) / \
    profiles.loc[filter, 'Total assets'].shift(1)).shift(1)
    spec_chars.loc[filter, 'E2P'] = (profiles.loc[filter, 'EPS'] / profiles.loc[filter, 'Stock Price']).shift(1)
    spec_chars.loc[filter, 'FC2R'] = ((profiles.loc[filter, 'R&D Expenses'] + profiles.loc[filter, 'SG&A Expense']) / \
    profiles.loc[filter, 'Revenue']).shift(1)
    spec_chars.loc[filter, 'OC2R'] = (profiles.loc[filter, 'Operating Expenses'] / profiles.loc[filter, 'Revenue']).shift(1)
    spec_chars.loc[filter, 'I'] = profiles.loc[filter, 'Total assets'].pct_change().shift(1)
    spec_chars.loc[filter, 'Lev'] = ((profiles.loc[filter, 'Short-term debt'] + profiles.loc[filter, 'Long-term debt']) / \
    (profiles.loc[filter, 'Short-term debt'] + profiles.loc[filter, 'Long-term debt'] + profiles.loc[filter, 'Total shareholders equity'])).shift(1)
    spec_chars.loc[filter, 'MCC'] = profiles.loc[filter, 'Market Cap'].pct_change().shift(1)
    spec_chars.loc[filter, 'NSI'] = np.log(profiles.loc[filter, 'Number of Shares']).diff().shift(1)
    spec_chars.loc[filter, 'OA'] = (profiles.loc[filter, 'Working Capital'] / profiles.loc[filter, 'Total assets'].shift(1)).shift(1)
    spec_chars.loc[filter, 'OL'] = ((profiles.loc[filter, 'Cost of Revenue'] + profiles.loc[filter, 'SG&A Expense']) / \
    profiles.loc[filter, 'Total assets']).shift(1)
    spec_chars.loc[filter, 'OP'] = ((profiles.loc[filter, 'Revenue'] - profiles.loc[filter, 'Cost of Revenue'] - profiles.loc[filter, 'Interest Expense'] -\
    profiles.loc[filter, 'SG&A Expense']) / (profiles.loc[filter, 'Book Value per Share'] * profiles.loc[filter, 'Number of Shares'])).shift(1)
    spec_chars.loc[filter, 'PCM'] = ((profiles.loc[filter, 'Revenue'] - profiles.loc[filter, 'Cost of Revenue']) / profiles.loc[filter, 'Revenue']).shift(1)
    spec_chars.loc[filter, 'PM'] = profiles.loc[filter, 'Profit Margin'].shift(1)
    spec_chars.loc[filter, 'PROF'] = (profiles.loc[filter, 'Gross Profit'] / (profiles.loc[filter, 'Book Value per Share'] * \
    profiles.loc[filter, 'Number of Shares'])).shift(1)
    spec_chars.loc[filter, 'Q'] = ((profiles.loc[filter, 'Total assets'] + profiles.loc[filter, 'Market Cap'] - \
    profiles.loc[filter, 'Cash and cash equivalents'] - profiles.loc[filter, 'Short-term investments'] - profiles.loc[filter, 'Tax Liabilities']) / \
    profiles.loc[filter, 'Total assets']).shift(1)
    spec_chars.loc[filter, 'RNA'] = ((profiles.loc[filter, 'Operating Income'] - profiles.loc[filter, 'Depreciation & Amortization']) / \
    spec_chars.loc[filter, 'NOA'].shift(1)).shift(1)
    spec_chars.loc[filter, 'ROA'] = (profiles.loc[filter, 'Net Income'] / profiles.loc[filter, 'Total assets'].shift(1)).shift(1)
    spec_chars.loc[filter, 'ROE'] = (profiles.loc[filter, 'Net Income'] / (profiles.loc[filter, 'Book Value per Share'] * \
    profiles.loc[filter, 'Number of Shares']).shift(1)).shift(1)
    spec_chars.loc[filter, 'S2P'] = (profiles.loc[filter, 'Revenue'] / profiles.loc[filter, 'Market Cap']).shift(1)
    spec_chars.loc[filter, 'SGA2S'] = (profiles.loc[filter, 'SG&A Expense'] / profiles.loc[filter, 'Revenue']).shift(1)


# и сцепляем с переменными роста и другими метриками
spec_chars = pd.concat([spec_chars, profiles.loc[:, 'PE ratio':'Stock-based compensation to Revenue']], axis = 1)
spec_chars = pd.concat([spec_chars, profiles.loc[:, 'Gross Profit Growth':'SG&A Expenses Growth']], axis = 1)
spec_chars.shape # (18091, 81)


# Открываем информацию о стоимости акций
stock_prices = pd.read_csv("S&P500_data_January-01-2005_March-01-2020.csv", \
parse_dates = ['Date'], index_col = 'Date')

# оставляем только полезные строки и столбцы
stock_prices = stock_prices.loc['2009-03-01':'2020-03-01', ['Close', 'Adj Close', 'Name']]
stock_prices.reset_index(level = 0, inplace = True)
# находим концы кварталов и отсекаем дни
# stock_prices_q = stock_prices[stock_prices.Date.dt.is_quarter_end] # проблемы с выходными днями
boolean = stock_prices['Date'].apply(lambda x: \
bool(re.search(r"-(?:03|06|09|12)-", str(x))))
# применяем фильтрование
stock_prices = stock_prices[boolean]
# группируем по месяцу, году и компании, находим наибольшую имеющуюся дату
st = stock_prices.groupby(by = [stock_prices.Date.dt.month, stock_prices.Date.dt.year, stock_prices.Name])
stock_prices_q = st.apply(lambda x: x[x.Date == x.Date.max()])
stock_prices_q = stock_prices_q.reset_index(drop = True)
# приводим даты к формату YYYY-MM
stock_prices_q['Date'] = stock_prices_q['Date'].apply(lambda x: \
re.findall(r"\d{4}-\d{2}", str(x))[0])
stock_prices_q.shape # (21162, 4)


# Сцепим две данные таблицы
big = pd.merge(stock_prices_q, spec_chars, on = ['Date', 'Name'], how = 'inner')
big.shape # (17545, 72)

# смотрим, есть ли проблема со стоимостями акций
b = big.dropna(subset = ['Stock Price', 'Close'])
r = b['Stock Price'] - b['Close']
# Рисунок 3 -- будем тестить обе переменные при наличии необходимости
plt.style.use('seaborn-colorblind')
f, axes = plt.subplots(1, 2, figsize = (20, 10), sharex = False)
sns.distplot(r, hist = False, fit = norm, kde = False, ax = axes[0])
axes[0].set_title('Гистограмма разницы между 2-мя переменными, отражающими стоимость акций', fontsize = 10)
sns.boxplot(r, ax = axes[1])
axes[1].set_title('Боксплот разницы между 2-мя переменными, отражающими стоимость акций', fontsize = 10)
plt.show()


# Создаём бинарные переменные секторов
sector_cols = pd.get_dummies(sp500['GICS Sector'])
sectors = pd.concat([sp500.Symbol, sector_cols], axis = 1)
sectors.rename(columns = {'Symbol': 'Name'}, inplace = True)

# сцепляем big таблицу с секторами
bigbig = pd.merge(big, sectors, on = ['Name'], how = 'inner')
bigbig.shape # (17545, 94)


# Тест Дики-Фуллера для проверки рядов на стационарность
def adf_test(timeseries):
    dftest = adfuller(timeseries, autolag = 'AIC')
    dfoutput = pd.Series(dftest[0:4], index = ['Test Statistic', 'p-value', \
    '#Lags Used', 'Number of Observations Used'])
    for key,value in dftest[4].items():
       dfoutput['Critical Value (%s)' % key] = value
    return dfoutput["p-value"]


# Открываем собранные макроэкономические данные
macro = pd.read_csv("data_macro_2020-04.csv", parse_dates = ['sasdate'], index_col = 'sasdate')

# оставляем интересующие нас годы
macro_data = macro.loc['2008-12-01':'2020-03-01', :]
macro_data.reset_index(level = 0, inplace = True)
# переименуем столбец и отсекаем лишнее
macro_data.rename(columns = {'sasdate': 'Date'}, inplace = True)
macro_data['Date'] = macro_data['Date'].apply(lambda x: \
re.findall(r"\d{4}-\d{2}", str(x))[0])

# Выделим столбцы, которые понадобятся потом для риск премий и бетт акций
mini_macro = macro_data.loc[:, ['Date', 'GS10', 'S&P 500']]
# Все стобцы сдвинем, чтобы использовать их, как предсказателей
big_macro = macro_data
big_macro.iloc[:, 1:] = big_macro.iloc[:, 1:].shift(1)

# алгоритм для приведения к стационарности
non_stationaries = []
dict_of_transforms = {}
for column in big_macro.columns.to_list()[1:]:
    print(column)
    a = big_macro[column]
    b = big_macro[column].pct_change()
    c = np.log(big_macro[column]).pct_change()
    #d = (np.log(big_macro[column]) - np.log(big_macro[column].shift(1))).diff()
    if adf_test(a.replace([np.inf, -np.inf], np.nan).dropna()) < 0.01:
        dict_of_transforms[column] = 'no_transform'
    elif adf_test(b.replace([np.inf, -np.inf], np.nan).dropna()) < 0.01:
        big_macro[column] = b
        dict_of_transforms[column] = 'pct_change'
    elif adf_test(c.replace([np.inf, -np.inf], np.nan).dropna()) < 0.01:
        big_macro[column] = c
        dict_of_transforms[column] = 'log_change'
    #elif adf_test(d.replace([np.inf, -np.inf], np.nan).dropna()) < 0.05:
    #    big_macro[column] = d
    #    dict_of_transforms[column] = 'diff_log_change'  изменение лог разницы крадёт много наблюдениий
    else:
        non_stationaries.append(column)

# посмотрим, что получилось
pd.DataFrame(dict_of_transforms.items())[1].value_counts()
len(non_stationaries) # 46

# оставим только стационарные ряды
big_macro = big_macro.drop(columns = non_stationaries) # осталось 217 колонок
# объединяем даннные по макро переменным
macro_data = pd.merge(mini_macro, big_macro, on = ['Date'], how = 'outer')

# Сцепляем всё в единое целое
big_data_1 = pd.merge(bigbig, macro_data, on = ['Date'], how = 'inner')
big_data_1.shape # (17545, 298)

# Убираем NA по объясняемой переменной и сортируем
big_data_1.dropna(subset = ['Close'], inplace = True) # не повлияло
big_data_1.sort_values(['Name', 'Date'], inplace = True)


# Второй вариант: соединяем всё, кроме данных с yahoo
big = pd.merge(spec_chars, sectors, on = ['Name'], how = 'inner') # (18091, 92)
big_data_2 = pd.merge(big, macro_data, on = ['Date'], how = 'inner') # (18091, 296)
# Убираем NA по объясняемой переменной и сортируем
big_data_2.dropna(subset = ['Stock Price'], inplace = True)
big_data_2.sort_values(['Name', 'Date'], inplace = True)
big_data_2.shape # (17671, 296)


# Посмотрим, сколько акций в каждой таблице
names_1_set = set(big_data_1.Name)
names_2_set = set(big_data_2.Name)
names_1_set - names_2_set # пусто
names_2_set - names_1_set # 'ARNC': во второй таблице на одну компанию больше

# Сконструрируем риск-премии для акций, но прежде проверим, везде ли есть дивиденды
# Если пропусков мало, заменим на нули и используем
big_data_1.loc[:, 'Dividend per Share'].isna().sum() # 19
big_data_2.loc[:, 'Dividend per Share'].isna().sum() # 9
big_data_1.loc[:, 'Dividend per Share'].fillna(0, inplace = True)
big_data_2.loc[:, 'Dividend per Share'].fillna(0, inplace = True)

# Создаём ERP: риск премию за акцию
y1 = pd.DataFrame()
y2 = pd.DataFrame()
for symbol in big_data_2.Name.unique():
    if symbol == 'ARNC':
        print(symbol)
        r2 = pd.DataFrame()
        filter_2 = big_data_2.Name == symbol
        # добавляем стобцы с датой и обозначением акции
        r2['Date'] = big_data_2[filter_2].Date
        r2['Name'] = symbol
        # считаем доходность актива
        r2['Asset_return'] = big_data_2[filter_2]['Stock Price'].pct_change().multiply(100) + \
        + big_data_2[filter_2]['Dividend per Share'].multiply(100) / \
        big_data_2[filter_2]['Stock Price'].shift(-1)
        # считаем премию за риск
        r2['ERP'] = r2['Asset_return'] - big_data_2[filter_2].GS10_x
        # beta * market return: премия за риск рынка
        r2['Market_return'] = big_data_2[filter_2]['S&P 500_x'].pct_change().multiply(100)
        r2['beta'] = r2.loc[:, ['Asset_return', 'Market_return']].cov().iloc[0, 0] / r2.loc[:, 'Market_return'].var()
        r2['MRP'] = r2['beta'] * r2['Market_return']
        # сцепляем
        y2 = pd.concat([y2, r2])
    else:
        print(symbol)
        r1 = pd.DataFrame()
        r2 = pd.DataFrame()
        # аналогичные действия
        filter_1 = big_data_1.Name == symbol
        r1['Date'] = big_data_1[filter_1].Date
        r1['Name'] = symbol
        r1['Asset_return'] = big_data_1[filter_1]['Close'].pct_change().multiply(100) + \
        + big_data_1[filter_1]['Dividend per Share'].multiply(100) / \
        big_data_1[filter_1]['Close'].shift(-1)
        r1['ERP'] = r1['Asset_return'] - big_data_1[filter_1].GS10_x
        r1['Market_return'] = big_data_1[filter_1]['S&P 500_x'].pct_change().multiply(100)
        r1['beta'] = r1.loc[:, ['Asset_return', 'Market_return']].cov().iloc[0, 0] / r1.loc[:, 'Market_return'].var()
        r1['MRP'] = r1['beta'] * r1['Market_return']
        y1 = pd.concat([y1, r1])
        filter_2 = big_data_2.Name == symbol
        r2['Date'] = big_data_2[filter_2].Date
        r2['Name'] = symbol
        r2['Asset_return'] = big_data_2[filter_2]['Stock Price'].pct_change().multiply(100) + \
        + big_data_2[filter_2]['Dividend per Share'].multiply(100) / \
        big_data_2[filter_2]['Stock Price'].shift(-1)
        r2['ERP'] = r2['Asset_return'] - big_data_2[filter_2].GS10_x
        r2['Market_return'] = big_data_2[filter_2]['S&P 500_x'].pct_change().multiply(100)
        r2['beta'] = r2.loc[:, ['Asset_return', 'Market_return']].cov().iloc[0, 0] / r2.loc[:, 'Market_return'].var()
        r2['MRP'] = r2['beta'] * r2['Market_return']
        y2 = pd.concat([y2, r2])

# объединяем
big_data_1 = pd.concat([big_data_1, y1.loc[:, ['beta', 'ERP']]], axis = 1) # 300 cols
big_data_2 = pd.concat([big_data_2, y2.loc[:, ['beta', 'ERP']]], axis = 1) # 298 cols


# Рисунок 4 -- Преобразование динамики ряда
apple = big_data_1.loc[big_data_1.Name == 'AAPL', ['Date', 'Close', 'ERP']]
plt.style.use('seaborn-colorblind')
f, axes = plt.subplots(1, 2, figsize = (20, 10), sharex = False)

sns.lineplot(data = apple, x = 'Date', y = 'Close', ax = axes[0])
axes[0].set_title('Динамика стоимости акции Apple', fontsize = 15)
axes[0].tick_params(axis = "x", labelsize = 8)
for tick in axes[0].get_xticklabels():
    tick.set_rotation(90)

sns.lineplot(data = apple, x = 'Date', y = 'ERP', ax = axes[1])
axes[1].set_title('Динамика риск премии акции Apple', fontsize = 15)
axes[1].tick_params(axis = "x", labelsize = 8)
for tick in axes[1].get_xticklabels():
    tick.set_rotation(90)

plt.show()


# удаляем использованные столбцы
big_data_1.drop(columns = ['Close', 'Adj Close', 'Stock Price', 'Dividend per Share', 'Utilities', 'GS10_x', 'S&P 500_x'], inplace = True)
big_data_2.drop(columns = ['Stock Price', 'Dividend per Share', 'Utilities', 'GS10_x', 'S&P 500_x'], inplace = True)

# удаляем NA и infы
big_clear_data_1 = big_data_1.replace([np.inf, -np.inf], np.nan).dropna() # (11505, 293)
big_clear_data_2 = big_data_2.replace([np.inf, -np.inf], np.nan).dropna() # (11663, 293)


# Рисунок 5 -- разброс доходностей
plt.style.use('seaborn-colorblind')
f, axes = plt.subplots(1, 2, figsize = (20, 10), sharex = False)
sns.boxplot(big_clear_data_1.ERP, ax = axes[0])
axes[0].set_title('Боксплот доходностей до', fontsize = 15)
sns.boxplot(big_clear_data_1.ERP[np.abs(big_clear_data_1.ERP) < 40], ax = axes[1])
axes[1].set_title('Боксплот доходностей после', fontsize = 15)
plt.show()

# Удаляем выбросы по риск премиям
big_clear_data_1 = big_clear_data_1.loc[np.abs(big_clear_data_1.ERP) < 40, :] # 11275
big_clear_data_2 = big_clear_data_2.loc[np.abs(big_clear_data_2.ERP) < 40, :] # 11462

# а также с помощью нормировки пройдёмся по отчётностям
big_clear_data_1 = big_clear_data_1[(np.abs(zscore(big_clear_data_1.loc[:, 'A2M':'SG&A Expenses Growth'])) < 3).all(axis = 1)] #  9906
big_clear_data_2 = big_clear_data_2[(np.abs(zscore(big_clear_data_2.loc[:, 'A2M':'SG&A Expenses Growth'])) < 3).all(axis = 1)] # 10069

# переместим даты и названия в индексы, удалим все пустышки
big_clear_data_1.set_index(['Date', 'Name'], inplace = True)
big_clear_data_2.set_index(['Date', 'Name'], inplace = True)

# Сохраняем в csv
# name_1 = "working_data_1.csv"
# big_clear_data_1.to_csv(name_1, index = True)
# name_2 = "working_data_2.csv"
# big_clear_data_2.to_csv(name_2, index = True)


## Тестирование моделей

# Открываем созданные данные
big_clear_data_1 = pd.read_csv("working_data_1.csv") #  9906 rows x 291 columns
big_clear_data_2 = pd.read_csv("working_data_2.csv") # 10069 rows x 291 columns

# Разделяем на объясняемую и объясняющие, делим на тренировочную и тестовую
X = big_clear_data_2.loc[:, 'A2M':'beta']
y = big_clear_data_2.loc[:, 'ERP']

# Разбиваем данные на тренировочные и тестовые выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 42)


# Оцениваем несколько вариантов оценщиков
# Алгоритм №1: Эластичная сеть
reg1 = ElasticNetCV(l1_ratio = [.1, .5, .7, .9, .95, .99, 1], random_state = 42)
reg1.fit(X_train, y_train)

train_score_1 = reg1.score(X_train, y_train)
test_score_1 = reg1.score(X_test, y_test)

print('Score on train: ' + str(train_score_1)) # околонулевой коэффициент детерминации на обеих датасетах
print('Score on test: ' + str(test_score_1)) # отрицательный коэффициент детерминации обеих датасетах


# Алгоритм №2: Дерево решений
# Переберём глубины дерева от 1 до 50
all_time = datetime.now()

kfold = KFold(n_splits = 5, shuffle = True, random_state = 241)
tree_scores = {}
for i in range(1, 51):
    print(i)
    tree_scores[str(i)] = round(np.mean(cross_val_score(DecisionTreeRegressor(max_depth = i, max_features = 'auto', random_state = 42), \
    X_train, y_train, cv = kfold, scoring = 'r2')), 3)

print('Whole time: ' + str(datetime.now() - all_time)) # 4 минуты

tree_scores # {'1': 0.03, '2': 0.06, '3': 0.081, '4': 0.082, '5': 0.081, '6': 0.071, '7': 0.029, .......}
best2 = max(tree_scores, key = tree_scores.get)
print(f"Best depth: {best2}") # Best depth: 4

# получим оценку от лучшего по кросс-валидации решающего дерева
reg2 = DecisionTreeRegressor(max_depth = int(best2), max_features = 'auto', random_state = 42)
reg2.fit(X_train, y_train)

train_score_2 = reg2.score(X_train, y_train)
test_score_2 = reg2.score(X_test, y_test)

print('Score on train: ' + str(train_score_2)) # Score on train: 0.131
print('Score on test: ' + str(test_score_2)) # Score on test: 0.080


# Алгоритм №3: Адаптивный бустинг с решающими деревьями
# Пройдёмся по всем более менее хорошим глубинам дерева из прошлого пункта
# (первыф прогон показал, что нужно проверить также и более большую глубину, ибо наблюдался постоянный рост качества)
all_time = datetime.now()

ada_scores = {}
for i in range(2, 11):
    timer = datetime.now()
    print(i)
    reg3 = AdaBoostRegressor(DecisionTreeRegressor(max_depth = i, max_features = 'auto', random_state = 42), \
    n_estimators = 500, random_state = 42)
    reg3.fit(X_train, y_train)
    ada_scores[str(i)] = round(reg3.score(X_test, y_test), 3)
    print('Time: ' + str(datetime.now() - timer))

print('Whole time: ' + str(datetime.now() - all_time)) # 30 минут

ada_scores # {'2': 0.078, '3': 0.194, '4': 0.26, '5': 0.347, '6': 0.41, '7': 0.436, '8': 0.454, '9': 0.441, '10': 0.429}
best3 = max(ada_scores, key = ada_scores.get)
print(f"Best depth: {best3}") # Best depth: 8

# Рисунок 6 -- изменение качества адаптивного бустинга
sorted_ada_scores = pd.DataFrame(ada_scores.items())
depths, scores = sorted_ada_scores[0].astype(int), sorted_ada_scores[1]

plt.style.use('seaborn-colorblind')
p = sns.lineplot(depths, scores)
p.set_xlabel('Глубина решающего дерева (базового оценщика)', fontsize = 10)
p.set_ylabel('Коэффициент детерминации модели на тренировочной выборке', fontsize = 10)
p.set_title('Пик при глубине дерева, равной 8', fontsize = 15)
plt.axvline(8, color = 'red')

plt.show()

# Оценим лучший вариант adaboost алгоритма, время и качество:
timer = datetime.now()

reg3 = AdaBoostRegressor(DecisionTreeRegressor(max_depth = int(best3), max_features = 'auto', random_state = 42), n_estimators = 500, random_state = 42)
reg3.fit(X_train, y_train)
train_score_3 = reg3.score(X_train, y_train)
test_score_3 = reg3.score(X_test, y_test)

print('Score on train: ' + str(train_score_3)) # Score on train: 0.807
print('Score on test: ' + str(test_score_3)) # Score on test: 0.454

print('Time: ' + str(datetime.now() - timer)) # 5 минут


# Алгоритм №4: Случайный лес
# Сначала дадим алгоритму самому выбрать глубину
reg4 = RandomForestRegressor(n_estimators = 500, max_features = 'auto', random_state = 42)
reg4.fit(X_train, y_train)

train_score_4 = reg4.score(X_train, y_train)
test_score_4 = reg4.score(X_test, y_test)

print('Score on train: ' + str(train_score_4)) # Score on train: 0.899
print('Score on test: ' + str(test_score_4)) # Score on test: 0.281

# Аналогично пройдёмся по всем более менее хорошим глубинам дерева из прошлого пункта
all_time = datetime.now()

rf_scores = {}
for i in range(2, 11):
    timer = datetime.now()
    print(i)
    reg4 = RandomForestRegressor(max_depth = i, n_estimators = 500, max_features = 'auto', random_state = 42)
    reg4.fit(X_train, y_train)
    rf_scores[str(i)] = round(reg4.score(X_test, y_test), 3)
    print('Time: ' + str(datetime.now() - timer))

print('Whole time: ' + str(datetime.now() - all_time)) # 22 минуты

rf_scores # {'2': 0.075, '3': 0.108, '4': 0.135, '5': 0.158, '6': 0.178, '7': 0.196, '8': 0.212, '9': 0.226, '10': 0.239}
best4 = max(rf_scores, key = rf_scores.get)
print(f"Best depth: {best4}") # Best depth: 10, можно лучше, но слишком долгое обучение

all_time = datetime.now()

# Рисунок 7 -- изменение качества случайного леса
sorted_rf_scores = pd.DataFrame(rf_scores.items())
depths, scores = sorted_rf_scores[0].astype(int), sorted_rf_scores[1]

plt.style.use('seaborn-colorblind')
p = sns.lineplot(depths, scores)
p.set_xlabel('Глубина решающего дерева (базового оценщика)', fontsize = 10)
p.set_ylabel('Коэффициент детерминации модели на тренировочной выборке', fontsize = 10)
p.set_title('Рост качества с увеличением глубины', fontsize = 15)

plt.show()


# Алгоритм №5: градиентный бустинг
reg5 = GradientBoostingRegressor(n_estimators = 500, max_features = 'auto', random_state = 42)
reg5.fit(X_train, y_train)

train_score_5 = reg5.score(X_train, y_train)
test_score_5 = reg5.score(X_test, y_test)

print('Score on train: ' + str(train_score_5)) # Score on train: 0.843
print('Score on test: ' + str(test_score_5)) # Score on test: 0.606


# Алгоритм №6: скоростной градиентный бустинг
reg6 = HistGradientBoostingRegressor(max_iter = 500, random_state = 42)
reg6.fit(X_train, y_train)

train_score_6 = reg6.score(X_train, y_train)
test_score_6 = reg6.score(X_test, y_test)

print('Score on train: ' + str(train_score_6)) # Score on train: 0.993
print('Score on test: ' + str(test_score_6)) # Score on test: 0.674


# Пройдёмся по разным глубинам дерева решений градиентного бустинга
all_time = datetime.now()

hgb_scores = {}
for i in range(2, 31):
    timer = datetime.now()
    print(i)
    reg6 = HistGradientBoostingRegressor(max_depth = i, max_iter = 500, random_state = 42)
    reg6.fit(X_train, y_train)
    hgb_scores[str(i)] = [round(reg6.score(X_train, y_train), 3), round(reg6.score(X_test, y_test), 3)]
    print('Time: ' + str(datetime.now() - timer))

print('Whole time: ' + str(datetime.now() - all_time)) # 11 минут

hgb_scores # {'2': [0.573, 0.432], '3': [0.834, 0.625], '4': [0.934, 0.668], '5': [0.977, 0.66], '6': [0.991, 0.672], '7': [0.995, 0.654],
# '8': [0.997, 0.676], '9': [0.997, 0.661], '10': [0.997, 0.682], '11': [0.997, 0.676], '12': [0.996, 0.681], '13': [0.995, 0.673],
# '14': [0.994, 0.675], '15': [0.994, 0.677], '16': [0.993, 0.672], '17': [0.993, 0.68], '18': [0.993, 0.675], '19': [0.993, 0.675],
# '20': [0.993, 0.681], '21': [0.993, 0.674], '22': [0.993, 0.674], '23': [0.993, 0.674], '24': [0.993, 0.674], '25': [0.993, 0.674],
# '26': [0.993, 0.674], '27': [0.993, 0.674], '28': [0.993, 0.674], '29': [0.993, 0.674], '30': [0.993, 0.674]}


# Рисунок 8 -- изменение качества градиентного бустинга
sorted_hgb_scores = pd.DataFrame(hgb_scores.items())
sorted_hgb_scores_part = pd.DataFrame(sorted_hgb_scores[1].to_list())
hgb_scores = pd.concat([sorted_hgb_scores[0], sorted_hgb_scores_part], axis = 1)
depths, scores_train, scores_test = hgb_scores.iloc[:, 0].astype(int), hgb_scores.iloc[:, 1], hgb_scores.iloc[:, 2]

plt.style.use('seaborn-colorblind')
p = sns.lineplot(depths, scores_train, label = 'train')
p = sns.lineplot(depths, scores_test, label = 'test')
p.set_xlabel('Глубина решающего дерева (базового оценщика)', fontsize = 10)
p.set_ylabel('Коэффициент детерминации модели', fontsize = 10)
p.set_title('Качество стабилизируется с увеличением глубины', fontsize = 15)

plt.show()


# Изучим результаты лучшего оценщика подробнее
reg6 = HistGradientBoostingRegressor(max_depth = 10, max_iter = 500, random_state = 42)
reg6.fit(X_train, y_train)

# Рисунок 9 -- отклонение наших предсказаний от фактических данных, и не только
y_pred6 = reg6.predict(X_test)
diff6 = y_test - y_pred6

plt.style.use('seaborn-colorblind')
f, axes = plt.subplots(1, 2, figsize = (20, 10), sharex = False)
sns.distplot(y_test, hist = True, bins = 100, kde = False, color = 'Green', label = 'Actual', ax = axes[0])
sns.distplot(y_pred6, hist = True, bins = 100, kde = False, color = 'Red', label = 'Predicted', ax = axes[0])
axes[0].legend(loc = 'upper left', frameon = False)
axes[0].set_xlabel('', fontsize = 10)
axes[0].set_title('Гистограммы доходностей акций', fontsize = 15)
sns.distplot(diff6, hist = True, bins = 50, fit = norm, kde = False, color = 'Blue', ax = axes[1])
axes[1].set_xlabel('', fontsize = 10)
axes[1].set_title('Остатки модели', fontsize = 15)

plt.show()


# Посчитаем, сколько наблюдений предсказаны с точностью в 1%, 5% и 10%, а также изучим квантили
(np.abs(diff6) < 1).sum() # 485 из 2518
(np.abs(diff6) < 5).sum() # 1742 из 2518
(np.abs(diff6) < 10).sum() # 2244 из 2518

np.abs(diff6).quantile([.05, .25, .5, .75, .95])


# The end :)
