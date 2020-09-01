from datetime import datetime, timedelta
import gc
import numpy as np
import pandas as pd
import lightgbm as lgb
import time
import sklearn.preprocessing as pre_processing
from sklearn.linear_model import LinearRegression
# m=LinearRegression()
# m.fit(df.iloc[:,0].values.reshape(-1, 1),df.iloc[:,6])
start = time.time()

pd.options.display.max_columns = 50

wins = [7,28]
lags = [7,28]
lags2=[1,2]
start_col = 1
pred_date = datetime(2016, 5, 23)
end_col = 1919+28
pred_col = [f"d_{i}" for i in range(1914+28, 1942+28)]
split_ratio = 0.2

path = 'D:/Kaggle/M5 Forecasting - Accuracy/'
print('convert wide data format to long format')
numcols = [f"d_{day}" for day in range(1, end_col - 5)]
dtype = {numcol: "int16" for numcol in numcols}
df = pd.read_csv(path+"sales_train_evaluation.csv", dtype=dtype,
                 usecols=list(range(6)) + list(range(start_col + 5, end_col)))
for i in pred_col:
    df[i] = np.nan
catcols = ['id', 'item_id', 'dept_id', 'store_id', 'cat_id', 'state_id']
df = pd.melt(df, id_vars=catcols, var_name="d", value_name="sales")
cal = pd.read_csv(path+"calendar.csv",
                  dtype={'wm_yr_wk': 'int16', 'snap_CA': 'int8', 'snap_TX': 'int8', 'snap_WI': 'int8'})
cal["date"] = pd.to_datetime(cal["date"])

print('merge df and cal')
df = df.merge(cal, on="d", copy=False)
#del cal
price = pd.read_csv(path+"sell_prices.csv",
                    dtype={'sell_price': 'float32', 'wm_yr_wk': 'int16'})
print('merge df and price')
df = df.merge(price, on=["store_id", "item_id", "wm_yr_wk"], how='left',copy=False)  # no price on some date for some item
#mean_val = df.groupby('id')['sell_price'].mean()
fill_mean = lambda g: g.fillna(g.mean())
df['sell_price']=df.groupby('id')['sell_price'].apply(fill_mean)
#df['sell_price']=df.groupby('id')['sell_price'].fillna(method='bfill')
del price
df['pq']=df['sell_price']*df['sales']
dummies = ['item_id', 'dept_id', 'store_id', 'cat_id', 'state_id', 'event_name_1', 'event_type_1',
           'event_name_2', 'event_type_2']
df.drop(columns=['weekday', "wm_yr_wk"], inplace=True)
df['quarter'] = getattr(df["date"].dt, 'quarter')
df['week'] = getattr(df["date"].dt, 'weekofyear')
df['mday'] = getattr(df["date"].dt, 'day')
label = pre_processing.LabelEncoder()
for i in dummies:
    df[i] = df[i].astype(str)
    df[i] = label.fit_transform(df[i])

print('feature engineer')


def creat_f(df):
    lag_cols2 = [f"lag_{lag}" for lag in lags2]
    for lag, lag_col in zip(lags2, lag_cols2):
        df[lag_col] = df[["id", "sales"]].groupby("id")["sales"].shift(lag)
    lag_cols = [f"lag_{lag}" for lag in lags]
    for lag, lag_col in zip(lags, lag_cols):
        df[lag_col] = df[["id", "sales"]].groupby("id")["sales"].shift(lag)

        # df.dropna(subset=[lag_col], inplace=True)
    for win in wins:
        for lag, lag_col in zip(lags, lag_cols):
            df[f"rmean_{lag}_{win}"] = df[["id", lag_col]].groupby("id")[lag_col].transform(
                lambda x: x.rolling(win).mean())

    #

def creat_f2(df):
    lag_cols2 = [f"lagall_{lag}" for lag in lags2]
    for lag, lag_col in zip(lags2, lag_cols2):
        df[lag_col] = df[["id", "sales"]].groupby("id")["sales"].shift(lag)
    lag_cols = [f"lagall_{lag}" for lag in lags]
    for lag, lag_col in zip(lags, lag_cols):
        df[lag_col] = df[["id", "sales"]].groupby("id")["sales"].shift(lag)
        # df.dropna(subset=[lag_col], inplace=True)
    for win in wins:
        for lag, lag_col in zip(lags, lag_cols):
            df[f"rmeanall_{lag}_{win}"] = df[["id", lag_col]].groupby("id")[lag_col].transform(
                lambda x: x.rolling(win).mean())
    #


#creat_f2(df)
for i in range(1,29):
    day = pred_date + timedelta(days=i - 1)
    inter_date = cal[cal.date <= day]['date']
    inter=range(len(inter_date)-1,-1,-i)
    train_date=inter_date[sorted(list(inter))]
    x_train = df[df.date.isin(train_date)]
    pre_fea = set(x_train.columns) - {'id', 'd', 'sales', 'date','pq'}
    print(pre_fea)
    pre_x = x_train[x_train.date ==day]
    pre_x=pre_x[pre_fea]
    x_train = x_train[x_train.date < day]
    y_train = x_train["sales"]
    x_train = x_train[pre_fea]
    x_train.dropna(inplace=True)
    print(i,x_train.shape)
    #print(x_train)
    print(i,'split training and testing data')
    np.random.seed(1)
    test_size = int(split_ratio * len(x_train))
    test_inx = np.random.choice(x_train.index.values, test_size, replace=False)
    print(i,'test size',len(test_inx))
    train_inx = np.setdiff1d(x_train.index.values, test_inx)
    print(i, 'train size', len(train_inx))
    train_data = lgb.Dataset(x_train.loc[train_inx], label=y_train.loc[train_inx], categorical_feature=dummies,
                             free_raw_data=False)
    test_data = lgb.Dataset(x_train.loc[test_inx], label=y_train.loc[test_inx], categorical_feature=dummies,
                            free_raw_data=False)
    del x_train, y_train, train_inx, test_inx
    gc.collect()
    print(i,'modelling')
    params = {
        "objective": "poisson",
        "metric": "rmse",
        "force_row_wise": True,
        "learning_rate": 0.075,
        #         "sub_feature" : 0.8,
        "sub_row": 0.75,
        "bagging_freq": 1,
        "lambda_l2": 0.1,
        #         "nthread" : 4
        #'device': 'gpu',
        'verbosity': 1,
        'num_iterations': 1200,
        'num_leaves': 128,
        "min_data_in_leaf": 100,
        "seed":1
    }
    model = lgb.train(params, train_data, valid_sets=[test_data], verbose_eval=20)

    print('predict',i, day)
    #print(pre_x)
    df.loc[df.date == day, 'sales'] = model.predict(pre_x)
df = df[['id', 'd', 'sales']]
df = df.pivot(index='id', columns='d', values='sales')
col2 = [f"F{i}" for i in range(1, 29)]
col_rep = {key: value for key, value in zip(pred_col, col2)}
df.rename(columns=col_rep, inplace=True)
df = df[col2]
df.to_csv(path+'my_lgb_interval_eval_0625.csv')
end = time.time()
print(end - start)
