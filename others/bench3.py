#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 21:53:04 2020

@author: lisa
"""


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 12 16:21:07 2020

@author: lisa
"""
#https://blog.csdn.net/qq_42859317/article/details/102594096 install cuda cuddn

#https://www.kaggle.com/kneroma/m5-first-public-notebook-under-0-50

from  datetime import datetime, timedelta
import gc
import numpy as np, pandas as pd
import lightgbm as lgb
import pickle


def create_dt(first_day,tr_last):
    prices = pd.read_csv("m5-forecasting-accuracy/sell_prices.csv", dtype = PRICE_DTYPES)
    for col, col_dtype in PRICE_DTYPES.items():
        if col_dtype == "category":
            prices[col] = prices[col].cat.codes.astype("int16")
            prices[col] -= prices[col].min()
    test_name1=[i for i in range(11608,11618)]   
    prices_test=prices[prices['wm_yr_wk'].isin(test_name1)]
    cal = pd.read_csv("m5-forecasting-accuracy/calendar.csv", dtype = CAL_DTYPES)
    cal["date"] = pd.to_datetime(cal["date"])
    for col, col_dtype in CAL_DTYPES.items():
        if col_dtype == "category":
            cal[col] = cal[col].cat.codes.astype("int16")
            cal[col] -= cal[col].min()
    test_name=[f"d_{day}" for day in range(1880,1942)]
    cal_test=cal[cal['d'].isin(test_name)]
    
    start_day = max(1,first_day)
    #start_day = max(1 if is_train  else tr_last-max_lags, first_day)
    numcols = [f"d_{day}" for day in range(start_day,tr_last+1)]
    catcols = ['id', 'item_id', 'dept_id','store_id', 'cat_id', 'state_id']
    dtype = {numcol:"float32" for numcol in numcols} 
    dtype.update({col: "category" for col in catcols if col != "id"})
    dt = pd.read_csv("m5-forecasting-accuracy/sales_train_validation.csv", 
                     nrows = None, usecols = catcols + numcols, dtype = dtype)
    for col in catcols:
        if col != "id":
            dt[col] = dt[col].cat.codes.astype("int16")
            dt[col] -= dt[col].min()
            
    for day in range(tr_last+1, tr_last+ 28 +1):
            dt[f"d_{day}"] = np.nan
    dt=dt.drop(["d_1427","d_1792"],axis=1)
    dt = pd.melt(dt,
                  id_vars = catcols,
                  value_vars = [col for col in dt.columns if col.startswith("d_")],
                  var_name = "d",
                  value_name = "sales")
    dt_test=dt[dt['d'].isin(test_name)]
    
    return prices,cal,dt,prices_test,cal_test,dt_test

def comb(prices,cal,dt):
    dt = dt.merge(cal, on= "d",how='left',copy = False)
    dt_out = dt.merge(prices, on = ["store_id", "item_id", "wm_yr_wk"],how='left',copy = False)
    return dt_out


def create_fea(dt):
    lags = [1,7, 28]
    lag_cols = [f"lag_{lag}" for lag in lags ]
    for lag, lag_col in zip(lags, lag_cols):
        dt[lag_col] = dt[["id","sales"]].groupby("id")["sales"].shift(lag)

    wins = [7,28]
    for win in wins :
        for lag,lag_col in zip(lags, lag_cols):
            dt[f"rmean_{lag}_{win}"] = dt[["id", lag_col]].groupby("id")[lag_col].transform(lambda x : x.rolling(win).mean())
            #dt[f"rstd_{lag}_{win}"] = dt[["id", lag_col]].groupby("id")[lag_col].transform(lambda x : x.rolling(win).std())
            #dt[f"rsum_{lag}_{win}"] = dt[["id", lag_col]].groupby("id")[lag_col].transform(lambda x : x.rolling(win).sum())
          
            
    g_1= dt[["d","store_id","lag_7"]].groupby(["d","store_id"])["lag_7"].sum()
    g_1=g_1.reset_index()
    g_1.columns=["d","store_id","store_total"]
    #g_2= dt[["d","item_id","lag_7"]].groupby(["d","item_id"])["lag_7"].sum()
    g_3= dt[["d","store_id","dept_id","lag_7"]].groupby(["d","store_id","dept_id"])["lag_7"].sum()
    g_3=g_3.reset_index()
    g_3.columns=["d","store_id","dept_id","dept_total"]
    dt=pd.merge(dt,g_1,on=["d","store_id"],how="left") 
    #dt=pd.merge(dt,g_2,on=["d","item_id"],how="left") 
    dt=pd.merge(dt,g_3,on=["d","store_id","dept_id",],how="left") 
    dt['per']=dt["dept_total"]/dt["store_total"]
    
    
    date_features = {
        
        "wday": "weekday",
        "week": "weekofyear",
        "month": "month",
        "quarter": "quarter",
        "year": "year",
        "mday": "day",
#         "ime": "is_month_end",
#         "ims": "is_month_start",
    }
    
#     dt.drop(["d", "wm_yr_wk", "weekday"], axis=1, inplace = True)
    
    for date_feat_name, date_feat_func in date_features.items():
        if date_feat_name in dt.columns:
            dt[date_feat_name] = dt[date_feat_name].astype("int16")
        else:
            dt[date_feat_name] = getattr(dt["date"].dt, date_feat_func).astype("int16")
            
            





if __name__ == "__main__":
    # dt = pd.read_csv("m5-forecasting-accuracy/sales_train_validation.csv")
    # dt=dt.iloc[:3,:12]
    # dt = pd.melt(dt,
    #               id_vars =["store_id"],
    #               var_name = "d",
    #         value_name = "sales") 
    # g_1= dt[["d","store_id","sales"]].groupby(["d","store_id"])["sales"].sum()
    # g_1=g_1.reset_index()
    # print(g_1.head(5))
    # print(g_1.columns)
    
    
    CAL_DTYPES={"event_name_1": "category", "event_name_2": "category", "event_type_1": "category", 
         "event_type_2": "category", "weekday": "category", 'wm_yr_wk': 'int16', "wday": "int16",
        "month": "int16", "year": "int16", "snap_CA": "float32", 'snap_TX': 'float32', 'snap_WI': 'float32' }
    PRICE_DTYPES = {"store_id": "category", "item_id": "category", "wm_yr_wk": "int16","sell_price":"float32" }
    pd.options.display.max_columns = 100
    h = 28 
    max_lags = 28
    tr_last = 1913
    fday = datetime(2016,4, 25) 
    FIRST_DAY = 1200 # If you want to load all the data set it to '1' -->  Great  memory overflow  risk !
    # melt数据
    prices,cal,dt,prices_test,cal_test,dt_test= create_dt(first_day= FIRST_DAY,tr_last=tr_last)
    df=comb(prices,cal,dt)
    print('data is done')
    
    create_fea(df)
    print('processing')
    
    cat_feats = ['item_id', 'dept_id','store_id', 'cat_id', 'state_id'] + ["event_name_1", "event_name_2", "event_type_1", "event_type_2"]
    useless_cols = ["id", "date", "sales","d", "wm_yr_wk", "weekday"]
    train_cols = df.columns[~df.columns.isin(useless_cols)]
    X_train = df[train_cols]
    y_train = df["sales"]
    print('end')
    
    # train_data = lgb.Dataset(X_train, label = y_train, categorical_feature=cat_feats, free_raw_data=False)
    # fake_valid_inds = np.random.choice(len(X_train), 1000000, replace = False)
    # fake_valid_data = lgb.Dataset(X_train.iloc[fake_valid_inds], label = y_train.iloc[fake_valid_inds],categorical_feature=cat_feats,
    #                              free_raw_data=False)   # This is just a subsample of the training set, not a real validation set !
    
    
    
    np.random.seed(777)
    fake_valid_inds = np.random.choice(X_train.index.values, 2_000_000, replace = False)
    train_inds = np.setdiff1d(X_train.index.values, fake_valid_inds)
    train_data=lgb.Dataset
    train_data = lgb.Dataset(X_train.loc[train_inds] , label = y_train.loc[train_inds], 
                          categorical_feature=cat_feats, free_raw_data=False)
    fake_valid_data = lgb.Dataset(X_train.loc[fake_valid_inds], label = y_train.loc[fake_valid_inds],
                              categorical_feature=cat_feats,
                  free_raw_data=False)# This is a random sample, we're not gonna apply any time series train-test-split tricks here!
    del df, X_train, y_train, fake_valid_inds,train_inds ; gc.collect()
    params = {
        "objective" : "poisson",
        "metric" :"rmse",
        "force_row_wise" : True,
        "learning_rate" : 0.08,
#         "sub_feature" : 0.8,
        "sub_row" : 0.75,
        "bagging_freq" : 1,
        "lambda_l2" : 0.1,
#         "nthread" : 4
        "metric": ["rmse"],
    'verbosity': 1,
    'num_iterations' : 800,
    'num_leaves': 128,
    "min_data_in_leaf": 100,
}
    m_lgb = lgb.train(params, train_data, valid_sets = [fake_valid_data], verbose_eval=20) 
    m_lgb.save_model("model3.lgb")
    print('training is over')
    
    
    
     #模型读取
    #f2=open('model3.lgb','rb')
    #s2=f2.read()
    #m_lgb=pickle.loads(s2)
    te=comb(prices_test,cal_test,dt_test)
    alphas = [1.028]
    #alphas = [1.028, 1.023, 1.018]
    weights = [1/len(alphas)]*len(alphas)
    sub = 0.
    for icount, (alpha, weight) in enumerate(zip(alphas, weights)):
        #te = create_dt(False) #每次重新读取数据很慢
        cols = [f"F{i}" for i in range(1,29)]
        for tdelta in range(0, 28):
            day = fday + timedelta(days=tdelta)
            print(tdelta, day)
            tst = te[(te.date >= day - timedelta(days=max_lags)) & (te.date <= day)].copy()
            create_fea(tst)
            #train_cols=tst.columns[~tst.columns.isin(useless_cols)]
            tst = tst.loc[tst.date == day , train_cols]
            te.loc[te.date == day, "sales"] = alpha*m_lgb.predict(tst) # magic multiplier by kyakovlev
        te_sub = te.loc[te.date >= fday, ["id", "sales"]].copy()
        #     te_sub.loc[te.date >= fday+ timedelta(days=h), "id"] = te_sub.loc[te.date >= fday+timedelta(days=h), 
#    
        te_sub["F"] = [f"F{rank}" for rank in te_sub.groupby("id")["id"].cumcount()+1]
        te_sub = te_sub.set_index(["id", "F" ]).unstack()["sales"][cols].reset_index()
        te_sub.fillna(0., inplace = True)
        te_sub.sort_values("id", inplace = True)
        te_sub.reset_index(drop=True, inplace = True)
        te_sub.to_csv(f"submission_{icount}.csv",index=False)
        if icount == 0 :
            sub = te_sub
            sub[cols] *= weight
        else:
            sub[cols] += te_sub[cols]*weight
        print(icount, alpha, weight)
    sub2 = sub.copy()
    sub2["id"] = sub2["id"].str.replace("validation$", "evaluation")
    sub = pd.concat([sub, sub2], axis=0, sort=False)
    sub.to_csv("submission4.csv",index=False)
    sub.id.nunique(), sub["id"].str.contains("validation$").sum()