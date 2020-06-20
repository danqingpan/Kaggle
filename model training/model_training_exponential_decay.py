from  datetime import datetime, timedelta
import gc
import numpy as np, pandas as pd
import lightgbm as lgb
import xgboost as xgb

CAL_DTYPES={"event_name_1": "category", "event_name_2": "category", "event_type_1": "category", 
         "event_type_2": "category", "weekday": "category", 'wm_yr_wk': 'int16', "wday": "int16",
        "month": "int16", "year": "int16", "snap_CA": "float32", 'snap_TX': 'float32', 'snap_WI': 'float32' }

PRICE_DTYPES = {"store_id": "category", "item_id": "category", "wm_yr_wk": "int16","sell_price":"float32" }

h = 28 
max_lags = 43
tr_last = 1913
fday = datetime(2016,4,25) 

#store_number = 8

def create_dt(is_train = True, nrows = None, first_day = 1200):
    
    # prices
    prices = pd.read_csv("kaggle/input/m5-forecasting-accuracy/sell_prices.csv", dtype = PRICE_DTYPES)


    for col, col_dtype in PRICE_DTYPES.items():
        if col_dtype == "category":
            prices[col] = prices[col].cat.codes.astype("int16")
            prices[col] -= prices[col].min()
    
    # calender        
    cal = pd.read_csv("kaggle/input/m5-forecasting-accuracy/calendar.csv", dtype = CAL_DTYPES)
    cal["date"] = pd.to_datetime(cal["date"])
    
    for col, col_dtype in CAL_DTYPES.items():
        if col_dtype == "category":
            cal[col] = cal[col].cat.codes.astype("int16")
            cal[col] -= cal[col].min()
    
    start_day = max(1 if is_train  else tr_last - max_lags, first_day)
    numcols = [f"d_{day}" for day in range(start_day,tr_last + 1)]
    
    catcols = ['id', 'item_id', 'dept_id','store_id', 'cat_id', 'state_id']
    dtype = {numcol:"float32" for numcol in numcols} 
    dtype.update({col: "category" for col in catcols if col != "id"})
   
    dt = pd.read_csv("kaggle/input/m5-forecasting-accuracy/sales_train_validation.csv", 
                     nrows = nrows, usecols = catcols + numcols, dtype = dtype)
    

    #dt = dt[dt['store_id'] == 'CA_1']
    
    # continue
    for col in catcols:
        if col != "id":
            dt[col] = dt[col].cat.codes.astype("int16")
            dt[col] -= dt[col].min()
    
    if is_train == False:
        for day in range(tr_last + 1, tr_last + 28 + 1):
            dt[f"d_{day}"] = np.nan
            
    
    dt = pd.melt(dt,
                  id_vars = catcols,
                  value_vars = [col for col in dt.columns if col.startswith("d_")],
                  var_name = "d",
                  value_name = "sales")
    
    dt = dt.merge(cal, on= "d", copy = False)
    dt = dt.merge(prices, on = ["store_id", "item_id", "wm_yr_wk"], copy = False)
    
    class_mapping = {}
    for i in range(1,tr_last+56):
        class_mapping['d_'+str(i)] = i
    dt['d'] = dt['d'].map(class_mapping)
    
    return dt


def create_fea(dt):
    
    lags_1 = [1, 2, 3, 7, 14, 21]
    
    lag_cols = [f"lag_{lag}" for lag in lags_1 ]
    for lag, lag_col in zip(lags_1, lag_cols):
        dt[lag_col] = dt[["id","sales"]].groupby("id")["sales"].shift(lag)
        
    lags_2 = [1, 2, 3, 7, 14]
    wins = [7, 28]
    for win in wins :
        for lag,lag_col in zip(lags_2, lag_cols):
            dt[f"rmean_{lag}_{win}"] = dt[["id", lag_col]].groupby("id")[lag_col].transform(lambda x : x.rolling(win).mean())
    
 
    dt['price_day_avg_shop'] = dt.groupby(['store_id','sell_price','d'])['sell_price'].transform('mean')
    #prices['price_nunique'] = prices.groupby(['store_id','item_id'])['sell_price'].transform('nunique')
    
    #lag_price = [7,28]
    #for lag in lag_price:
        #dt[f"lag_{lag}_price"] = dt[["id","sell_price"]].groupby("id")["sell_price"].shift(lag)
    
    
    #for lag in [7, 28]: # shift 1 day and std
    #    dt['lag_1_' + 'std_' + str(lag)]  = dt.groupby(['id'])['sales'].transform(lambda x: x.shift(1).rolling(lag).std()).astype(np.float16)
    
    #for lag in lags_1[:3]: # shop mean 
    #    dt['shop_lag_'+ str(lag) + '_mean' ]  = dt.groupby(['store_id','d'])['lag_' + str(lag)].transform('mean').astype(np.float16)
        
        
    date_features = {
        
        "wday": "weekday",
        "week": "weekofyear",
        "month": "month",
        "quarter": "quarter",
        "year": "year",
        "mday": "day",
    }
    
#     dt.drop(["d", "wm_yr_wk", "weekday"], axis=1, inplace = True)
    
    for date_feat_name, date_feat_func in date_features.items():
        if date_feat_name in dt.columns:
            dt[date_feat_name] = dt[date_feat_name].astype("int16")
        else:
            dt[date_feat_name] = getattr(dt["date"].dt, date_feat_func).astype("int16")
    
    dt['year'] =  dt['year']  - 2011



FIRST_DAY = 800
load = 0

df = None
if load:
    df = pd.read_pickle('data/thisdata.pkl')
else:    
    df = create_dt(is_train=True, first_day= FIRST_DAY)
    print(df.shape)
    
    create_fea(df)
    print(df.shape)
    
    
    df.dropna(inplace = True)
    print(df.shape)
    
    df.to_pickle('data/thisdata.pkl')
#print(df['year'].unique())



df['d'] = df['d'].apply(np.log)
#df['d'] = df['d']
#print(df['d'].unique())
w = df['d']/df['d'].max()

#print(valid_data_last_days.shape)

cat_feats = ['item_id', 'dept_id','store_id', 'cat_id', 'state_id'] + ["event_name_1", "event_name_2", "event_type_1", "event_type_2"]
useless_cols = ["id", "date", "sales","d", "wm_yr_wk", "weekday"]
train_cols = df.columns[~df.columns.isin(useless_cols)]

#df.to_pickle('data/train_data.pkl')


X_train = df[train_cols]
y_train = df["sales"]



#valid_data_last_days_train = valid_data_last_days[train_cols]
#valid_data_last_days_ytrain = valid_data_last_days['sales']

######################
#use_model = 'lgb'
######################

np.random.seed(777)
#num_rounds = 1000

fake_valid_inds = np.random.choice(X_train.index.values, 4000000, replace = False)
train_inds = np.setdiff1d(X_train.index.values, fake_valid_inds)



# =============================================================================
# if use_model == 'xgb':
#     
# 
#     train_data = xgb.DMatrix(X_train.loc[train_inds], y_train.loc[train_inds],weight = w.loc[train_inds])
#     fake_valid_data = xgb.DMatrix(X_train.loc[fake_valid_inds], y_train.loc[fake_valid_inds], weight = w.loc[fake_valid_inds])
#     print(X_train.loc[train_inds].shape)
#     print(y_train.loc[train_inds].shape)
#     
# 
#     params = {
#         #'booster':'gbtree',
#         'objective':'reg:tweedie',
#         'tweedie_variance_power':1.2,
#         'eval_metric': 'rmse',
#         #'gpu_id': 0,
#         #'tree_method':'gpu_hist',
#         #'num_class':3,
#         'gamma':0.1,
#         'max_depth':6,
#         'lambda':2,
#         'subsample':0.7,
#         'colsample_bytree':0.7,
#         'min_child_weight':3,
#         #'slient':1,s
#         'eta':0.03,
#         'seed':1000,
#         'nthread':4,
#     }
#     
#     evallist = [(fake_valid_data, 'eval'), (train_data, 'train')]
#     
#     plst = list(params.items())
#     model = xgb.train(plst,train_data,num_rounds,evallist)
# =============================================================================
   

#if use_model == 'lgb':


train_data = lgb.Dataset(X_train.loc[train_inds] , label = y_train.loc[train_inds], 
                         categorical_feature=cat_feats, free_raw_data=False,
                         #weight = w.loc[train_inds]
                         )
fake_valid_data = lgb.Dataset(X_train.loc[fake_valid_inds], label = y_train.loc[fake_valid_inds],
                              categorical_feature=cat_feats,#weight = w.loc[fake_valid_inds],
                              free_raw_data = False)# This is a random sample, we're not gonna apply any time series train-test-split tricks here!

del df, X_train, y_train, fake_valid_inds,train_inds ; gc.collect()


params = {
        #"objective" : "poisson",
        'objective': 'tweedie',
        'tweedie_variance_power': 1.2,
        "metric" :"rmse",
        "force_row_wise" : True,
        "learning_rate" : 0.04,
        "sub_feature" : 0.8,
        "sub_row" : 0.75,
        "bagging_freq" : 1,
        "lambda_l2" : 0.1,
        "nthread" : 4,
        'verbosity': 1,
        'num_iterations' : 1400,
        'num_leaves': 384,
        "min_data_in_leaf": 128,
        #'min_child_weight': 5
}



m_lgb = lgb.train(params, train_data, 
                  valid_sets = [fake_valid_data],
                  verbose_eval=20) 



model_name = 'lgb_model_tw12_lr004_n1200_ex'

m_lgb.save_model("lgb_models/" + model_name + ".lgb")
print('model saved')



##############################################################################

m_lgb = lgb.Booster(model_file="lgb_models/" + model_name + ".lgb")
print('model loaded')

#alphas = [1.00]
#weights = [1/len(alphas)]*len(alphas)
#sub = 0.

#for icount, (alpha, weight) in enumerate(zip(alphas, weights)):

temp_df = create_dt(is_train = False)
cols = [f"F{i}" for i in range(1,29)]

for tdelta in range(0, 28):
    day = fday + timedelta(days=tdelta)
    print(tdelta, day)
    test_df = temp_df[(temp_df.date >= day - timedelta(days=max_lags)) & (temp_df.date <= day)].copy()
    create_fea(test_df)
    test_df = test_df.loc[test_df.date == day , train_cols]
    temp_df.loc[temp_df.date == day, "sales"] = m_lgb.predict(test_df) # magic multiplier by kyakovlev


sub = temp_df.loc[temp_df.date >= fday, ["id", "sales"]].copy()
#     te_sub.loc[te.date >= fday+ timedelta(days=h), "id"] = te_sub.loc[te.date >= fday+timedelta(days=h), 
#                                                                           "id"].str.replace("validation$", "evaluation")
sub["F"] = [f"F{rank}" for rank in sub.groupby("id")["id"].cumcount()+1]
sub = sub.set_index(["id", "F" ]).unstack()["sales"][cols].reset_index()
sub.fillna(0., inplace = True)
sub.sort_values("id", inplace = True)
sub.reset_index(drop=True, inplace = True)
sub.to_csv(f"submission_{0}.csv",index=False)
    
    #if icount == 0 :
    #    sub = te_sub
    #    sub[cols] *= weight
    #else:
    #    sub[cols] += te_sub[cols]*weight
    #print(icount, alpha, weight)

sub2 = sub.copy()
sub2["id"] = sub2["id"].str.replace("validation$", "evaluation")
sub = pd.concat([sub, sub2], axis=0, sort=False)
sub.to_csv("submission_new.csv",index=False)


