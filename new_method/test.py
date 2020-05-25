import numpy as np
import pandas as pd
import os, sys, gc, time, warnings, pickle, random
import lightgbm as lgb

pd.options.display.max_columns = 100
VER = 1  # Our model version
SEED = 42  # We want all things

# LIMITS and const
TARGET = 'sales'  # Our target
START_TRAIN = 0  # We can skip some rows (Nans/faster training)
END_TRAIN = 1913  # End day of our train set
P_HORIZON = 28  # Prediction horizon
USE_AUX = False  # Use or not pretrained models

root = 'kaggle/input/m5-forecasting-accuracy/'
days = ['d_'+str(i) for i in range(1,END_TRAIN+1)]

# FEATURES to remove
## These features lead to overfit
## or values not present in test set
remove_features = ['id', 'state_id', 'store_id',
                   'date', 'wm_yr_wk', 'd', 'sales', 'sell_price']

BASE = root + 'grid_part_1.pkl'
PRICE = root + 'grid_part_2.pkl'
CALENDAR = root + 'grid_part_3.pkl'
LAGS = root + 'lags_df.pkl'

# STORES ids
STORES_IDS = pd.read_csv(root + 'sales_train_validation.csv')['store_id']
STORES_IDS = list(STORES_IDS.unique())

def seed_everything(seed = 0):
    random.seed(seed)
    np.random.seed(seed)

# Read data
def get_data_by_store(store):
    # Read and contact basic feature
    df = pd.concat([pd.read_pickle(BASE),
                    pd.read_pickle(PRICE).iloc[:, 2:],
                    pd.read_pickle(CALENDAR).iloc[:, 2:]],
                   axis=1)
    #print(list(df))
    # Leave only relevant store
    df = df[df['store_id'] == store]

    df2 = pd.read_pickle(LAGS).iloc[:, 4:]
    df2 = df2[df2.index.isin(df.index)]
    
    df = pd.concat([df, df2], axis=1)
    del df2  # to not reach memory limit

    # Create features list
    features = [col for col in list(df) if col not in remove_features]
    df = df[['id', 'd', 'sales'] + features]
    df = df[df['d'].isin(days)]
    #df.info()

    # Skipping first n rows
    # df = df[df['d'] >= START_TRAIN].reset_index(drop=True)
    # print(list(df))

    return df, features

#for store in STORES_IDS:
#get_data_by_store(STORES_IDS[0])
df,features = get_data_by_store(STORES_IDS[0])
print(df['d'].unique())
#df.info()
df.to_csv(STORES_IDS[0]+'.csv')


"""
#data = pd.read_pickle(CALENDAR)
#print(list(data))

# Recombine Test set after training
def get_base_test():
    
    base_test = pd.DataFrame()

    for store_id in STORES_IDS:
        temp_df = pd.read_pickle('test_' + store_id + '.pkl')
        temp_df['store_id'] = store_id
        base_test = pd.concat([base_test, temp_df]).reset_index(drop=True)

    return base_test


########################### Helper to make dynamic rolling lags
#################################################################################

if __name__ == "__main__":
    
    seed_everything(SEED)  # to be as deterministic

    lgb_params = {
        'boosting_type': 'gbdt',
        'objective': 'tweedie',
        'tweedie_variance_power': 1.1,
        'metric': 'rmse',
        'subsample': 0.5,
        'subsample_freq': 1,
        'learning_rate': 0.03,
        'num_leaves': 2 ** 11 - 1,
        'min_data_in_leaf': 2 ** 12 - 1,
        'feature_fraction': 0.5,
        'max_bin': 100,
        'n_estimators': 1400,
        'boost_from_average': False,
        'verbose': -1,
    }
    lgb_params['seed'] = SEED  # as possible
    #N_CORES = psutil.cpu_count()  # Available CPU cores

    # SPLITS for lags creation
    SHIFT_DAY = 28
    N_LAGS = 15
    LAGS_SPLIT = [col for col in range(SHIFT_DAY, SHIFT_DAY + N_LAGS)]
    ROLS_SPLIT = []
    for i in [1, 7, 14]:
        for j in [7, 14, 30, 60]:
            ROLS_SPLIT.append([i, j])
    ########################### Aux Models

    print('begin')
    #################################################################################
    
    grid_df, MODEL_FEATURES = get_data_by_store('CA_1')
    
    #按store_id分别训练
    for store_id in STORES_IDS:
        
        #print('Train', store_id)
        # Get grid for current store
        grid_df, features_columns = get_data_by_store(store_id)
        #grid_df：输入样本
        
        # Masks for
        # Train (All data less than 1913)
        # "Validation" (Last 28 days - not real validatio set)
        # Test (All data greater than 1913 day,
        #       with some gap for recursive features)
        train_mask = grid_df['d'] <= END_TRAIN
        valid_mask = train_mask & (grid_df['d'] > (END_TRAIN - P_HORIZON))
        preds_mask = grid_df['d'] > (END_TRAIN - 100)
        # Apply masks and save lgb dataset as bin
        # to reduce memory spikes during dtype convertations
        # https://github.com/Microsoft/LightGBM/issues/1032
        # "To avoid any conversions, you should always use np.float32"
        # or save to bin before start training
        # https://www.kaggle.com/c/talkingdata-adtracking-fraud-detection/discussion/53773
        train_data = lgb.Dataset(grid_df[train_mask][features_columns],
                                 label=grid_df[train_mask][TARGET])
        train_data.save_binary('train_data.bin')
        train_data = lgb.Dataset('train_data.bin')
        valid_data = lgb.Dataset(grid_df[valid_mask][features_columns],
                                 label=grid_df[valid_mask][TARGET])
        # Saving part of the dataset for later predictions
        # Removing features that we need to calculate recursively
        grid_df = grid_df[preds_mask].reset_index(drop=True)
        keep_cols = [col for col in list(grid_df) if '_tmp_' not in col]
        grid_df = grid_df[keep_cols]
        grid_df.to_pickle('test_' + store_id + '.pkl')
        del grid_df
        # Launch seeder again to make lgb training 100% deterministic
        # with each "code line" np.random "evolves"
        # so we need (may want) to "reset" it
        seed_everything(SEED)
        estimator = lgb.train(lgb_params,
                              train_data,
                              valid_sets=[valid_data],
                              verbose_eval=100,
                              )
        # Save model - it's not real '.bin' but a pickle file
        # estimator = lgb.Booster(model_file='model.txt')
        # can only predict with the best iteration (or the saving iteration)
        # pickle.dump gives us more flexibility
        # like estimator.predict(TEST, num_iteration=100)
        # num_iteration - number of iteration want to predict with,
        # NULL or <= 0 means use best iteration
        model_name = 'lgb_model_' + store_id + '_v' + str(VER) + '.bin'
        pickle.dump(estimator, open(model_name, 'wb'))
        # Remove temporary files and objects
        # to free some hdd space and ram memory
        del train_data, valid_data, estimator
        gc.collect()
        # "Keep" models features for predictions
        
    
    grid_df, features_columns = get_data_by_store(STORES_IDS[-1])
    MODEL_FEATURES = features_columns
        
    ########################### Predict
    #################################################################################
    
    # Create Dummy DataFrame to store predictions

    all_preds = pd.DataFrame()

    # Join back the Test dataset with
    # a small part of the training data
    # to make recursive features

    base_test = get_base_test()

    # Timer to measure predictions time
    main_time = time.time()
    
    # Loop over each prediction day
    # As rolling lags are the most timeconsuming
    # we will calculate it for whole day
    for PREDICT_DAY in range(1, 29):
        
        print('Predict | Day:', PREDICT_DAY)
        start_time = time.time()

        # Make temporary grid to calculate rolling lags
        grid_df = base_test.copy()
        grid_df = pd.concat([grid_df, make_lag_roll(ROLS_SPLIT,grid_df)], axis=1)
        for store_id in STORES_IDS:

            # Read all our models and make predictions
            # for each day/store pairs
            model_path = 'lgb_model_' + store_id + '_v' + str(VER) + '.bin'
            #if USE_AUX:
            #    model_path = AUX_MODELS + model_path

            estimator = pickle.load(open(model_path, 'rb'))

            day_mask = base_test['d'] == (END_TRAIN + PREDICT_DAY)
            store_mask = base_test['store_id'] == store_id

            mask = (day_mask) & (store_mask)
            base_test[TARGET][mask] = estimator.predict(grid_df[mask][MODEL_FEATURES])

        # Make good column naming and add
        # to all_preds DataFrame
        temp_df = base_test[day_mask][['id', TARGET]]
        temp_df.columns = ['id', 'F' + str(PREDICT_DAY)]
        if 'id' in list(all_preds):
            all_preds = all_preds.merge(temp_df, on=['id'], how='left')
        else:
            all_preds = temp_df.copy()

        print('#' * 10, ' %0.2f min round |' % ((time.time() - start_time) / 60),
              ' %0.2f min total |' % ((time.time() - main_time) / 60),
              ' %0.2f day sales |' % (temp_df['F' + str(PREDICT_DAY)].sum()))
        del temp_df

    all_preds = all_preds.reset_index(drop=True)
    #################################################################################

    submission = pd.read_csv(ORIGINAL + 'sample_submission.csv')[['id']]
    submission = submission.merge(all_preds, on=['id'], how='left').fillna(0)
    submission.to_csv('submission_v' + str(VER) + '.csv', index=False)
"""