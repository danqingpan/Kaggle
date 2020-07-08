# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 15:11:22 2020

@author: 81701
"""

from  datetime import datetime, timedelta
import gc
import numpy as np, pandas as pd
import lightgbm as lgb


h = 28 
max_lag = 0
tr_last = 1913
fday = datetime(2016,4, 25) 

CAL_DTYPES={"event_name_1": "category", "event_name_2": "category", "event_type_1": "category", 
         "event_type_2": "category", "weekday": "category", 'wm_yr_wk': 'int16', "wday": "int16",
        "month": "int16", "year": "int16", "snap_CA": "float32", 'snap_TX': 'float32', 'snap_WI': 'float32' }
PRICE_DTYPES = {"store_id": "category", "item_id": "category", "wm_yr_wk": "int16","sell_price":"float32" }



catcols = ['id','item_id','dept_id','store_id','cat_id','state_id']
numcols = [f"d_{day}" for day in range(1, tr_last + 1)]

dtype = {numcol: "float32" for numcol in numcols}
dtype.update({col:"category" for col in catcols if col != 'id'})

dt = pd.read_csv("kaggle/input/m5-forecasting-accuracy/sales_train_validation.csv",
                     usecols = catcols + numcols, dtype = dtype)


# =============================================================================
def check_zero_rate():
    count = 0
    for i in range(30490):
        sample = dt.iloc[i][6:]
        if len(np.where(sample == 0)[0])/sample.shape[0] < 0.1:
            count += 1

    print(count)
# =============================================================================


# =============================================================================
def check_zero_dic_gen():
    maxcount = np.array(dt.shape[0]*[0])
    
    continuousCount = []
     
    for i in range(dt.shape[0]):
        oncount = False
        count = []
        index = []
        sample = dt.iloc[i][6:]
        for j in range(0,sample.shape[0]):
            if sample[j] == 0:
                if oncount == True:
                    count[-1] =  count[-1]+1
                
                else: 
                    oncount = True
                    count.append(1)
                    index.append(j)
            else:
                oncount = False
    
        countdic = {index[i]:count[i] for i in range(len(index))}
    
        continuousCount.append(countdic)
    
    np.save('countdic.npy',np.array(continuousCount))
    
    np.save('maxcount.npy',maxcount)
# =============================================================================

def data_preprocess():
    all_string_of_days = [f'd_{i}' for i in range(1,tr_last+1)]
    countdic = np.load('countdic.npy')
    
    # compute initial zero
    begining_zero_index = []
    for i in range(countdic.shape[0]):
        if list(countdic[i].keys())[0] == 0:
            begining_zero_index.append(i)
    
    begining_zero_id = dt.iloc[begining_zero_index]["id"]
    begining_zero_id = np.array(begining_zero_id)
    
    begining_zero_length = [list(countdic[i].values())[0] for i in begining_zero_index]
    
    begining_zero_string = []
    for i in range(len(begining_zero_length)):
        begining_zero_string.append(all_string_of_days[:begining_zero_length[i]])
    
        
    dic_without_init_zero = countdic.copy()
    
    for i in range(dic_without_init_zero.shape[0]):
        if list(dic_without_init_zero[i].keys())[0] == 0:
            dic_without_init_zero[i].pop(0)
    
    #print(dic_without_init_zero.shape)
    # create off shelf infomation list
    
    #[(index,[(off_time,off_len),(off_time,off_len)]),()...]
    # make off_shelf_list
    off_shelf_list = []
    
    for idx in range(len(dic_without_init_zero)):
        sorteddic = sorted(dic_without_init_zero[idx].items(), key = lambda kv:(kv[1], kv[0]),reverse=True)    
        diff = [sorteddic[i][1] - sorteddic[i+1][1] for i in range(len(sorteddic) - 1)]
        if len(diff) > 3:
            for i in [2,1,0]:
                if diff[i] > 30 and np.percentile(diff,95) < 2:
                    off_shelf_list.append((idx,[(sorteddic[j][0],sorteddic[j][1]) for j in range(i+1) if sorteddic[j][0] + sorteddic[j][1] < tr_last-max_lag]))
                    break
    
    off_shelf_list_temp = []
    for item in off_shelf_list:
        if item[1] != []:
            off_shelf_list_temp.append(item)
        
    off_shelf_list = off_shelf_list_temp
    # finish off_shelf_list
    
    off_shelf_string = []
    for i in range(len(off_shelf_list)):
        for j in range(len(off_shelf_list[i][1])):
            if j == 0:
                off_shelf_string.append(all_string_of_days[off_shelf_list[i][1][j][0]:off_shelf_list[i][1][j][0]+off_shelf_list[i][1][j][1]])
            else:
                off_shelf_string[i] += all_string_of_days[off_shelf_list[i][1][j][0]:off_shelf_list[i][1][j][0]+off_shelf_list[i][1][j][1]]
            
    off_shelf_index = [off_shelf_list[i][0] for i in range(len(off_shelf_list))]
    off_shelf_id = dt.iloc[off_shelf_index]["id"]
    off_shelf_id = np.array(off_shelf_id)
    
        
    # mark data begining with 0
    for i,thisid in enumerate(begining_zero_id):
        print(i)
        dt.loc[thisid,begining_zero_string[i]] = np.nan
    
    # mark off shelf data
    for i,thisid in enumerate(off_shelf_id):
        print(i)
        dt.loc[thisid,off_shelf_string[i]] = -1
    
    dt.to_csv('dt_-1.csv',index = None)


# =============================================================================
# last_is_zero = []
# for i in range(countdic.shape[0]):
#     if list(countdic[i].keys())[-1] + list(countdic[i].values())[-1] == tr_last and list(countdic[i].keys())[-1] < tr_last - max_lag:
#         last_is_zero.append(i)
# 
# last_is_zero_id = np.array(dt.iloc[last_is_zero]["id"])
# 
# # save data with 57 days more off shelf
# dt_last_zero = dt[dt.id.isin(last_is_zero_id)]
# #dt_last_zero.to_csv('data/last_zero_57.csv')
# 
# dt_non_last_zero = dt[~dt.id.isin(last_is_zero_id)]
# =============================================================================


def create_dt(is_train = True, nrows = None):
    prices = pd.read_csv("kaggle/input/m5-forecasting-accuracy/sell_prices.csv", dtype = PRICE_DTYPES)
    for col, col_dtype in PRICE_DTYPES.items():
        if col_dtype == "category":
            prices[col] = prices[col].cat.codes.astype("int16")
            prices[col] -= prices[col].min()
            
    cal = pd.read_csv("kaggle/input/m5-forecasting-accuracy/calendar.csv", dtype = CAL_DTYPES)
    cal["date"] = pd.to_datetime(cal["date"])
    for col, col_dtype in CAL_DTYPES.items():
        if col_dtype == "category":
            cal[col] = cal[col].cat.codes.astype("int16")
            cal[col] -= cal[col].min()
    
    numcols = [f"d_{day}" for day in range(350,tr_last+1)]
    catcols = ['id', 'item_id', 'dept_id','store_id', 'cat_id', 'state_id']
    dtype = {numcol:"float32" for numcol in numcols} 
    dtype.update({col: "category" for col in catcols if col != "id"})
    dt = pd.read_csv("kaggle/input/m5-forecasting-accuracy/new_dt.csv", 
                     nrows = nrows, usecols = catcols + numcols, dtype = dtype)
    
    for col in catcols:
        if col != "id":
            dt[col] = dt[col].cat.codes.astype("int16")
            dt[col] -= dt[col].min()
    
    if not is_train:
        for day in range(tr_last+1, tr_last+ 28 +1):
            dt[f"d_{day}"] = np.nan
    
    dt = pd.melt(dt,
                  id_vars = catcols,
                  value_vars = [col for col in dt.columns if col.startswith("d_")],
                  var_name = "d",
                  value_name = "sales")
    
    dt = dt.merge(cal, on= "d", copy = False)
    dt = dt.merge(prices, on = ["store_id", "item_id", "wm_yr_wk"], copy = False)
    
    
    return dt

def create_fea(dt):
    lags = [7, 14]
    lag_cols = [f"lag_{lag}" for lag in lags ]
    for lag, lag_col in zip(lags, lag_cols):
        dt[lag_col] = dt[["id","sales"]].groupby("id")["sales"].shift(lag)

    wins = [7, 14, 28]
    for win in wins :
        for lag,lag_col in zip(lags, lag_cols):
            dt[f"rmean_{lag}_{win}"] = dt[["id", lag_col]].groupby("id")[lag_col].transform(lambda x : x.rolling(win).mean())

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


max_lag = 43        
df = create_dt(is_train = True, nrows = None)
create_fea(df)

#df = df.dropna(axis = 0, how='any')
df.dropna(inplace=True)

cat_feats = ['item_id', 'dept_id','store_id', 'cat_id', 'state_id'] + ["event_name_1", "event_name_2", "event_type_1", "event_type_2"]
useless_cols = ["id", "date", "sales","d", "wm_yr_wk", "weekday"]
train_cols = df.columns[~df.columns.isin(useless_cols)]
X_train = df[train_cols]
y_train = df["sales"]

np.random.seed(777)

fake_valid_inds = np.random.choice(X_train.index.values, 2_000_000, replace = False)
train_inds = np.setdiff1d(X_train.index.values, fake_valid_inds)
train_data = lgb.Dataset(X_train.loc[train_inds] , label = y_train.loc[train_inds], 
                         categorical_feature=cat_feats, free_raw_data=False)
fake_valid_data = lgb.Dataset(X_train.loc[fake_valid_inds], label = y_train.loc[fake_valid_inds],
                              categorical_feature=cat_feats,
                 free_raw_data=False)


del df, X_train, y_train, fake_valid_inds,train_inds ; gc.collect()

lgb_params = {
                    'boosting_type': 'gbdt',
                    'objective': 'tweedie',
                    'tweedie_variance_power': 1.1,
                    'metric': 'rmse',
                    'subsample': 0.5,
                    'subsample_freq': 1,
                    'learning_rate': 0.03,
                    'num_leaves': 2**11-1,
                    'min_data_in_leaf': 2**12-1,
                    'feature_fraction': 0.5,
                    'max_bin': 100,
                    'n_estimators': 1400,
                    'boost_from_average': False,
                    'num_iterations' : 1200,
                    'verbose': -1,
                }

m_lgb = lgb.train(lgb_params, train_data, valid_sets = [fake_valid_data], verbose_eval=20) 
m_lgb.save_model("model.lgb")

alphas = [1]
weights = [1/len(alphas)]*len(alphas)
sub = 0.

for icount, (alpha, weight) in enumerate(zip(alphas, weights)):

    te = create_dt(False)
    cols = [f"F{i}" for i in range(1,29)]

    for tdelta in range(0, 28):
        day = fday + timedelta(days=tdelta)
        print(tdelta, day)
        tst = te[(te.date >= day - timedelta(days=max_lag)) & (te.date <= day)].copy()
        create_fea(tst)
        tst = tst.loc[tst.date == day , train_cols]
        te.loc[te.date == day, "sales"] = alpha*m_lgb.predict(tst) # magic multiplier by kyakovlev



    te_sub = te.loc[te.date >= fday, ["id", "sales"]].copy()
#     te_sub.loc[te.date >= fday+ timedelta(days=h), "id"] = te_sub.loc[te.date >= fday+timedelta(days=h), 
#                                                                           "id"].str.replace("validation$", "evaluation")
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
sub.to_csv("result/submission.csv",index=False)


