# Код для анализа методов машинного обучения
# в оценивании моделей ценообразования активов

import os
os.getcwd() # "/Users/nikitabaramiya/"
os.chdir("/Users/nikitabaramiya/Desktop/coursework2")

# для анализа времени
import time
from datetime import datetime

# для импорта и обработки данных
import re
import json
import talib
import scipy
import numpy as np
import pandas as pd
import QuantLib as ql
from functools import reduce
from urllib.request import urlopen
from pandas_datareader.data import DataReader

# для визуализации данных
import seaborn as sns
import matplotlib.pyplot as plt

# для отсеивания переменных
from sklearn.linear_model import Lasso
from sklearn.feature_selection import VarianceThreshold

# для построения и оценивания моделей
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split

from sklearn.svm import SVR
from sklearn.linear_model import RidgeCV
from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import RandomForestRegressor

from sklearn.ensemble import VotingRegressor
from sklearn.ensemble import StackingRegressor


# Импорт финансовой информации
# Извлекаем символы и названия ценных бумаг из S&P500
sp500 = pd.read_excel("S&P500.xlsx")
symbols = sp500.Symbol.values
security_names = sp500.Security.values

# Настраиваем временной промежуток извлекаемых данных
now = datetime.now()
start = datetime(now.year - 15, 1, 1)
end = datetime(now.year, 3, 1)

# Извлекаем данные, также фиксируем ценные бумаги, чьи данные не удалось извлечь
timer = datetime.now()

bad_names = []
stock_prices = pd.DataFrame()

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

def process_json_data(url):
    parse_data = get_jsonparsed_data(url)
    key, value = parse_data.items()

    target_data = pd.DataFrame(value[1])
    target_data.rename(columns = {'date': 'Date'}, inplace = True)
    target_data['Date'] = target_data['Date'].apply(lambda x: \
    re.findall(r"\d{4}-\d{2}", str(x))[0])

    return target_data

# Извлекаем всевозможные данные о компаниях
timer = datetime.now()

bad_companies = []
company_profiles = pd.DataFrame()

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


## Feature selection

# correlation selector
def corr_selector(data, target, criteria = 0.5):
    corr_list = []

    features = X.columns.to_list()
    for f in features:
        corr_f = np.corrcoef(X[f], y)[0, 1]
        corr_list.append(corr_f)

    corr_bool = np.abs(corr_list) > criteria
    corr_score = np.round(corr_list, 3)
    dict = {'features': features, 'correlation': corr_score, 'selected_c': corr_bool}
    corr_dataframe = pd.DataFrame(data = dict)
    return corr_dataframe

# lasso selector
def lasso_selector(X, y, alpha = 1):
    clf = Lasso(alpha = 1, max_iter = 10000)
    clf.fit(X, y)

    features = X.columns.to_list()
    lasso_bool = np.abs(clf.coef_) > 0
    dict = {'features': features, 'selected_l': lasso_bool}
    lasso_dataframe = pd.DataFrame(data = dict)
    return lasso_dataframe

# variance selector
def var_selector(X, thres = 0):
    selector = VarianceThreshold(threshold = thres)
    selector.fit(X)

    features = X.columns.to_list()
    var_bool = selector.get_support()
    dict = {'features': features, 'selected_v': var_bool}
    var_dataframe = pd.DataFrame(data = dict)
    return var_dataframe

# meta selector
def meta_feature_selector(data, target, thres = 0):
    data.dropna(inplace = True)
    X = data.drop(target, axis = 1)
    y = data[target]

    corr_data = corr_selector(X, y)
    lasso_data = lasso_selector(X, y)
    var_data = var_selector(X)

    data_frames = [corr_data, lasso_data, var_data]
    data_merged = reduce(lambda left, right: pd.merge(left, right, \
    on = ['features'], how = 'outer'), data_frames)

    data_merged['selection_score'] = data_merged['selected_c'].astype(int) + \
    data_merged['selected_l'].astype(int) + data_merged['selected_v'].astype(int)

    data_merged.sort_values(by = ['selection_score'], ascending = False, inplace = True)
    return(data_merged)


## Работа с данными
# Открываем собранные данные
macro = pd.read_csv("macro.csv").iloc[1:,:].drop('ACOGNO', axis = 1)

company_profiles = pd.read_csv("data_financialmodelingprep.csv")

# (company_profiles.dropna().Date.value_counts()).head(41).sum() # 14392 observations of interest
stock_prices = pd.read_csv("S&P500_data_January-01-2005_March-01-2020.csv", parse_dates=['Date'])
# находим концы кварталов и отсекаем дни
stock_prices_q = stock_prices[stock_prices.Date.dt.is_quarter_end]
stock_prices_q['Date'] = stock_prices_q['Date'].apply(lambda x: \
re.findall(r"\d{4}-\d{2}", str(x))[0])
# Оставляем только полезные столбцы
stock_prices_q = stock_prices_q.loc[:, 'Close':'Date']


d = meta_feature_selector(data, target)
X = macro.loc[:, d[d.loc[:, 'selection_score'] == 3].features.values]


#

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 42)

estimators = [('lr', RidgeCV()), ('svr', SVR(random_state = 42)), \
('dtr', DecisionTreeRegressor(random_state=42))]

reg = StackingRegressor(estimators = estimators, \
final_estimator = RandomForestRegressor(n_estimators = 50, random_state = 42))

reg.fit(X_train, y_train)
score = reg.score(X_test, y_test)


# Итоговое время
print('Total time: ' + str(datetime.now() - main_start_time))
