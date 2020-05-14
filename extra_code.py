import talib
import QuantLib as ql
from fbprophet import Prophet
from datetime import timedelta
from sklearn.decomposition import PCA

# Prophet
data = pd.read_csv("S&P500_data_12-01-01_20-03-08.csv")

df = data[data.Name == "T"]
df = df[["Date", "Close"]]
df = df.rename(columns = {"Date": "ds", "Close": "y"})

m = Prophet()
m.fit(df)
future = m.make_future_dataframe(periods=365)
forecast = m.predict(future)

fig1 = m.plot(forecast)
fig2 = m.plot_components(forecast)

plt.show()

# delta
delta = timedelta(days = 1)
macro_data.loc[:, 'sasdate'] = macro_data['sasdate'].apply(lambda x: x - delta)

# PCA part
#macro_1.iloc[:, 2:] = macro_1.iloc[:, 2:].shift(1)

#pca = PCA(n_components = 5)
#macro_1_pca = pd.DataFrame(pca.fit_transform(macro_1.iloc[:, 2:].dropna()))
#pca.explained_variance_ratio_

#macro_1_pca = pd.concat([macro_1.Date[1:], macro_1_pca], axis = 1)
#macro_1_pca.iloc[:, 1:] = macro_1_pca.iloc[:, 1:].shift(1)
#macro_1_pca.dropna(inplace = True)
#macro_1_pca.rename(columns = {0: 'pca_1', 1: 'pca_2', 2: 'pca_3', \
#3: 'pca_4', 4: 'pca_5'}, inplace = True)

#macro_1.iloc[:, 2:] = macro_1.iloc[:, 2:].pct_change().multiply(100)

#pca = PCA(n_components = 10)
#macro_2_pca = pd.DataFrame(pca.fit_transform(macro_1.iloc[:, 2:].dropna()))
#pca.explained_variance_ratio_

#macro_2_pca = pd.concat([macro_1.Date[2:], macro_2_pca], axis = 1)
#macro_2_pca.iloc[:, 1:] = macro_2_pca.iloc[:, 1:].shift(2)
#macro_2_pca.dropna(inplace = True)
#macro_2_pca.rename(columns = {0: 'pca_1_ch', 1: 'pca_2_ch', 2: 'pca_3_ch', \
#3: 'pca_4_ch', 4: 'pca_5_ch', 5: 'pca_6_ch', 6: 'pca_7_ch', 7: 'pca_8_ch', \
#8: 'pca_9_ch', 9: 'pca_10_ch'}, inplace = True)

#m = pd.merge(macro_1, macro_1_pca, on = ['Date'], how = 'outer')
#macro_data = pd.merge(m, macro_2_pca, on = ['Date'], how = 'outer') # (45, 58)


non_stationaries_1 = []
dict_of_transforms_1 = {}
for column in big_profiles.columns.to_list()[2:]:
    print(column)
    a = big_profiles.loc[big_profiles.Name == 'AAPL', column]
    b = a.pct_change()
    c = np.log(a) - np.log(a.shift(1))
    d = (np.log(a) - np.log(a.shift(1))).diff()
    try:
        if adf_test(a.replace([np.inf, -np.inf], np.nan).dropna()) < 0.05:
            dict_of_transforms_1[column] = 'no_transform'
            for symbol in big_profiles.Name.unique():
                big_profiles.loc[big_profiles.Name == symbol, column] = \
                big_profiles.loc[big_profiles.Name == symbol, column].shift(1)
        elif adf_test(b.replace([np.inf, -np.inf], np.nan).dropna()) < 0.05:
            dict_of_transforms_1[column] = 'pct_change'
            for symbol in big_profiles.Name.unique():
                big_profiles.loc[big_profiles.Name == symbol, column] = \
                big_profiles.loc[big_profiles.Name == symbol, column].pct_change().shift(1)
        elif adf_test(c.replace([np.inf, -np.inf], np.nan).dropna()) < 0.05:
            dict_of_transforms_1[column] = 'log_change'
            for symbol in big_profiles.Name.unique():
                big_profiles.loc[big_profiles.Name == symbol, column] = \
                (np.log(big_profiles.loc[big_profiles.Name == symbol, column]) - \
                np.log(big_profiles.loc[big_profiles.Name == symbol, column]).shift(1)).shift(1)
        elif adf_test(d.replace([np.inf, -np.inf], np.nan).dropna()) < 0.05:
            dict_of_transforms_1[column] = 'diff_log_change'
            for symbol in big_profiles.Name.unique():
                big_profiles.loc[big_profiles.Name == symbol, column] = \
                (np.log(big_profiles.loc[big_profiles.Name == symbol, column]) - \
                np.log(big_profiles.loc[big_profiles.Name == symbol, column]).shift(1)).diff().shift(1)
        else:
            non_stationaries_1.append(column)
    except:
        non_stationaries_1.append(column)

pd.DataFrame(dict_of_transforms_1.items()).sort_values(1).head(27)

b = big_profiles.drop(columns = non_stationaries_1)
b = b.drop(columns = pd.DataFrame(dict_of_transforms_1.items()).sort_values(1).head(12)[0].values)

# сдвинем параметры на единичку
for symbol in big_data_2.Name.unique():
    if symbol == 'ARNC':
        print(symbol)
        big_data_2.loc[big_data_2.Name == symbol, 'Revenue per Share':'SG&A Expenses Growth'] = \
        big_data_2.loc[big_data_2.Name == symbol, 'Revenue per Share':'SG&A Expenses Growth'].shift(1)
    else:
        print(symbol)
        big_data_1.loc[big_data_1.Name == symbol, 'Revenue per Share':'SG&A Expenses Growth'] = \
        big_data_1.loc[big_data_1.Name == symbol, 'Revenue per Share':'SG&A Expenses Growth'].shift(1)
        big_data_2.loc[big_data_2.Name == symbol, 'Revenue per Share':'SG&A Expenses Growth'] = \
        big_data_2.loc[big_data_2.Name == symbol, 'Revenue per Share':'SG&A Expenses Growth'].shift(1)



## Feature selection

# correlation selector
def corr_selector(X, y, criteria = 0.5):
    corr_list = []
    # считаем корреляции и добавляем в лист
    features = X.columns.to_list()
    for f in features:
        corr_f = np.corrcoef(X[f], y)[0, 1]
        corr_list.append(corr_f)
    # смотрим, какие корреляции больше заданного критерия
    corr_bool = np.abs(corr_list) > criteria
    corr_score = np.round(corr_list, 3)
    dict = {'features': features, 'correlation': corr_score, 'selected_c': corr_bool}
    corr_dataframe = pd.DataFrame(data = dict)
    return corr_dataframe

# lasso selector
def lasso_selector(X, y, alpha = 1):
    clf = Lasso(alpha = 1, max_iter = 10000)
    clf.fit(X, y)
    # смотрим, коэффициенты каких переменных больше нуля
    features = X.columns.to_list()
    lasso_bool = np.abs(clf.coef_) > 0
    dict = {'features': features, 'selected_l': lasso_bool}
    lasso_dataframe = pd.DataFrame(data = dict)
    return lasso_dataframe

# variance selector
def var_selector(X, thres = 0):
    selector = VarianceThreshold(threshold = thres)
    selector.fit(X)
    # проверка на наличие неменяющихся переменных
    features = X.columns.to_list()
    var_bool = selector.get_support()
    dict = {'features': features, 'selected_v': var_bool}
    var_dataframe = pd.DataFrame(data = dict)
    return var_dataframe

# meta selector
def meta_feature_selector(X, y, criteria = 0.5, alpha = 1, thres = 0):
    corr_data = corr_selector(X, y, criteria = criteria)
    lasso_data = lasso_selector(X, y, alpha = alpha)
    var_data = var_selector(X, thres = thres)
    # собираем все селекторы воедино
    data_frames = [corr_data, lasso_data, var_data]
    data_merged = reduce(lambda left, right: pd.merge(left, right, \
    on = ['features'], how = 'outer'), data_frames)
    # получаем сводную оценку
    data_merged['selection_score'] = data_merged['selected_c'].astype(int) + \
    data_merged['selected_l'].astype(int) + data_merged['selected_v'].astype(int)
    # сортируем по оценке
    data_merged.sort_values(by = ['selection_score'], ascending = False, inplace = True)
    return(data_merged)



# Experimental block

# функция, проверяющая все переменные в заданной таблице
#def check_series(data):
#    non_stationaries = []
#    columns = data.columns.tolist()
#    for i in columns:
#        print(i)
#        if adf_test(data[i].replace([np.inf, -np.inf], np.nan).dropna()) > 0.05:
#            non_stationaries.append(i)
#    return non_stationaries

# non_stat = check_series(data = big_clear_data_1.loc[big_clear_data_1.Name == 'AAPL', 'A2M':])
# non_stat

# Удаление выбросов с помощью правила межквартильного размаха
# IR = big_clear_data_1.quantile(.75) - big_clear_data_1.quantile(.25)
# ci_1 = big_clear_data_1.quantile(.25) - 1.5 * IR
# ci_2 = big_clear_data_1.quantile(.75) + 1.5 * IR

# (big_clear_data_1 < ci_1).sum().sort_values().tail(20)
# (big_clear_data_1 > ci_2).sum().sort_values().tail(20)

# Смотрим на корреляции с риск премией (предварительный отбор)
# meta_selection = meta_feature_selector(X, y)
# meta_selection.sort_values('correlation').head(20)
# meta_selection.sort_values('correlation').tail(20)
# X = X.loc[:, meta_selection.loc[meta_selection.selected_l == True, 'features'].values]

# Попробуем сделать предварительный отбор признаков
# selected = meta_feature_selector(X, y, criteria = 0.1, alpha = 1, thres = 0)

# Оставим столбцы, отобранные селектором
# X_1 = X.loc[:, selected[selected.selection_score > 1].features.values]

#rf_scores_extra = {}
#for i in range(11, 20):
#    timer = datetime.now()
#    print(i)
#    reg4 = RandomForestRegressor(max_depth = i, n_estimators = 500, max_features = 'auto', random_state = 42)
#    reg4.fit(X_train, y_train)
#    rf_scores_extra[str(i)] = round(reg4.score(X_test, y_test), 3)
#    print('Time: ' + str(datetime.now() - timer))

#print('Whole time: ' + str(datetime.now() - all_time)) #  минуты

#rf_scores_extra
#best4 = max(rf_scores_extra, key = rf_scores_extra.get)
#print(f"Best depth: {best4}") # Best depth:

#reg4 = RandomForestRegressor(max_depth = int(best4), n_estimators = 500, max_features = 'auto', random_state = 42)
#reg4.fit(X_train, y_train)

#train_score_4 = reg4.score(X_train, y_train)
#test_score_4 = reg4.score(X_test, y_test)

#print('Score on train: ' + str(train_score_4))
#print('Score on test: ' + str(test_score_4))


'Revenue', 'Revenue Growth', 'Cost of Revenue', 'Gross Profit', 'R&D Expenses', 'SG&A Expense', 'Operating Expenses', 'Operating Income',
'Interest Expense', 'Earnings before Tax', 'Income Tax Expense', 'Net Income - Non-Controlling int', 'Net Income - Discontinued ops', 'Net Income',
'Preferred Dividends', 'Net Income Com', 'EPS', 'EPS Diluted', 'Weighted Average Shs Out', 'Weighted Average Shs Out (Dil)', 'Dividend per Share',
'Gross Margin', 'EBITDA Margin', 'EBIT Margin', 'Profit Margin', 'Free Cash Flow margin', 'EBITDA', 'EBIT', 'Consolidated Income',
'Earnings Before Tax Margin', 'Net Profit Margin', 'Cash and cash equivalents', 'Short-term investments', 'Cash and short-term investments',
'Receivables', 'Inventories', 'Total current assets', 'Property, Plant & Equipment Net', 'Goodwill and Intangible Assets', 'Long-term investments',
'Tax assets', 'Total non-current assets', 'Total assets', 'Payables', 'Short-term debt', 'Total current liabilities', 'Long-term debt', 'Total debt',
'Deferred revenue', 'Tax Liabilities', 'Deposit Liabilities', 'Total non-current liabilities', 'Total liabilities', 'Other comprehensive income',
'Retained earnings (deficit)', 'Total shareholders equity', 'Investments', 'Net Debt', 'Other Assets', 'Other Liabilities', 'Depreciation & Amortization',
'Stock-based compensation', 'Operating Cash Flow', 'Capital Expenditure', 'Acquisitions and disposals', 'Investment purchases and sales',
'Investing Cash flow', 'Issuance (repayment) of debt', 'Issuance (buybacks) of shares', 'Dividend payments', 'Financing Cash Flow',
'Effect of forex changes on cash', 'Net cash flow / Change in cash', 'Free Cash Flow', 'Net Cash/Marketcap', 'Stock Price', 'Number of Shares',
'Market Capitalization', '- Cash & Cash Equivalents', '+ Total Debt', 'Enterprise Value_x', 'Revenue per Share', 'Net Income per Share',
'Operating Cash Flow per Share', 'Free Cash Flow per Share', 'Cash per Share', 'Book Value per Share', 'Tangible Book Value per Share',
'Shareholders Equity per Share', 'Interest Debt per Share', 'Market Cap', 'Enterprise Value_y', 'PE ratio', 'Price to Sales Ratio', 'POCF ratio',
'PFCF ratio', 'PB ratio', 'PTB ratio', 'EV to Sales', 'Enterprise Value over EBITDA', 'EV to Operating cash flow', 'EV to Free cash flow',
'Earnings Yield', 'Free Cash Flow Yield', 'Debt to Equity', 'Debt to Assets', 'Net Debt to EBITDA', 'Current ratio', 'Interest Coverage',
'Income Quality', 'Dividend Yield', 'Payout Ratio', 'SG&A to Revenue', 'R&D to Revenue', 'Intangibles to Total Assets', 'Capex to Operating Cash Flow',
'Capex to Revenue', 'Capex to Depreciation', 'Stock-based compensation to Revenue', 'Graham Number', 'Graham Net-Net', 'Working Capital',
'Tangible Asset Value', 'Net Current Asset Value', 'Invested Capital', 'Average Receivables', 'Average Payables', 'Average Inventory', 'Capex per Share',
'Gross Profit Growth', 'EBIT Growth', 'Operating Income Growth', 'Net Income Growth', 'EPS Growth', 'EPS Diluted Growth', 'Weighted Average Shares Growth',
'Weighted Average Shares Diluted Growth', 'Dividends per Share Growth', 'Operating Cash Flow growth', 'Free Cash Flow growth', 'Receivables growth',
'Inventory Growth', 'Asset Growth', 'Book Value per Share Growth', 'Debt Growth', 'R&D Expense Growth', 'SG&A Expenses Growth'
