import numpy as np
import pandas as pd
import os, sys, gc, time, warnings, pickle, psutil, random

from math import ceil

warnings.filterwarnings('ignore')

def get_memory_usage():
    return np.round(psutil.Process(os.getpid()).memory_info()[0]/2.**30, 2) 
        
def sizeof_fmt(num, suffix='B'):
    for unit in ['','Ki','Mi','Gi','Ti','Pi','Ei','Zi']:
        if abs(num) < 1024.0:
            return "%3.1f%s%s" % (num, unit, suffix)
        num /= 1024.0
    return "%.1f%s%s" % (num, 'Yi', suffix)

def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2    
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                       df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)    
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df


def merge_by_concat(df1, df2, merge_on):
    merged_gf = df1[merge_on]
    merged_gf = merged_gf.merge(df2, on=merge_on, how='left')
    new_columns = [col for col in list(merged_gf) if col not in merge_on]
    df1 = pd.concat([df1, merged_gf[new_columns]], axis=1)
    return df1


TARGET = 'sales'         # Our main target
END_TRAIN = 1913         # Last day in train set
MAIN_INDEX = ['id','d']  # We can identify item by these columns

print('Load Main Data')

root = 'kaggle/input/m5-forecasting-accuracy/'

##############################################################################
train_df = pd.read_csv(root + 'sales_train_validation.csv')
prices_df = pd.read_csv(root + 'sell_prices.csv')
calendar_df = pd.read_csv(root + 'calendar.csv')


index_columns = ['id','item_id','dept_id','cat_id','store_id','state_id']
grid_df = pd.melt(train_df, id_vars = index_columns, var_name = 'd', value_name = TARGET)


print('Train rows:', len(train_df), len(grid_df))



# To be able to make predictions
# we need to add "test set" to our grid
add_grid = pd.DataFrame()

for i in range(1,29):
    temp_df = train_df[index_columns]
    temp_df = temp_df.drop_duplicates()
    temp_df['d'] = 'd_'+ str(END_TRAIN + i)
    temp_df[TARGET] = np.nan
    add_grid = pd.concat([add_grid,temp_df])

grid_df = pd.concat([grid_df,add_grid])
grid_df = grid_df.reset_index(drop=True)

del temp_df, add_grid
del train_df


for col in index_columns:
    grid_df[col] = grid_df[col].astype('category')

# engage release time
release_df = prices_df.groupby(['store_id','item_id'])['wm_yr_wk'].agg(['min']).reset_index()
release_df.columns = ['store_id','item_id','release']

grid_df = merge_by_concat(grid_df, release_df, ['store_id','item_id'])
del release_df

grid_df = merge_by_concat(grid_df, calendar_df[['wm_yr_wk','d']], ['d'])
grid_df = grid_df[grid_df['wm_yr_wk'] >= grid_df['release']]
grid_df = grid_df.reset_index(drop=True)

grid_df['release'] = grid_df['release'] - grid_df['release'].min()
grid_df['release'] = grid_df['release'].astype(np.int16)

# save 1st file
grid_df.to_pickle(root + 'grid_part_1.pkl')

print('Size:', grid_df.shape)
print('Prices')

##############################################################################
# Handle price
prices_df['price_max'] = prices_df.groupby(['store_id','item_id'])['sell_price'].transform('max')
prices_df['price_norm'] = prices_df['sell_price']/prices_df['price_max']
del prices_df['price_max']

prices_df['price_avg'] = prices_df.groupby(['store_id','item_id'])['sell_price'].transform('mean')

prices_df['price_nunique'] = prices_df.groupby(['store_id','item_id'])['sell_price'].transform('nunique')

calendar_prices = calendar_df[['wm_yr_wk','month','year']]
calendar_prices = calendar_prices.drop_duplicates(subset=['wm_yr_wk'])
prices_df = prices_df.merge(calendar_prices[['wm_yr_wk','month','year']], on=['wm_yr_wk'], how='left')

del calendar_prices
del prices_df['month'], prices_df['year']

print('Merge prices and save part 2')

# Merge Prices
original_columns = list(grid_df)
grid_df = grid_df.merge(prices_df, on=['store_id','item_id','wm_yr_wk'], how='left')
keep_columns = [col for col in list(grid_df) if col not in original_columns]
grid_df = grid_df[MAIN_INDEX + keep_columns]
grid_df = reduce_mem_usage(grid_df)

# Safe part 2
grid_df.to_pickle(root + 'grid_part_2.pkl')
print('Size:', grid_df.shape)

del prices_df

##############################################################################
# handle date
grid_df = pd.read_pickle(root + 'grid_part_1.pkl')
grid_df = grid_df[MAIN_INDEX]

# Merge calendar partly
icols = ['date','d','event_name_1','event_type_1','event_name_2',
         'event_type_2','snap_CA','snap_TX','snap_WI']

grid_df = grid_df.merge(calendar_df[icols], on=['d'], how='left')

# Minify data
# 'snap_' columns we can convert to bool or int8
icols = ['event_name_1','event_type_1','event_name_2','event_type_2',
         'snap_CA','snap_TX','snap_WI']

for col in icols:
    grid_df[col] = grid_df[col].astype('category')

# Convert to DateTime
grid_df['date'] = pd.to_datetime(grid_df['date'])

# Make some features from date
grid_df['tm_d'] = grid_df['date'].dt.day.astype(np.int8) # day in a month
grid_df['tm_m'] = grid_df['date'].dt.month.astype(np.int8) # month in a year
grid_df['tm_y'] = grid_df['date'].dt.year # year
grid_df['tm_y'] = (grid_df['tm_y'] - grid_df['tm_y'].min()).astype(np.int8) # normalized year

grid_df['tm_wm'] = grid_df['tm_d'].apply(lambda x: ceil(x/7)).astype(np.int8) # week in a month

grid_df['tm_dw'] = grid_df['date'].dt.dayofweek.astype(np.int8) # day in a week
grid_df['tm_w_end'] = (grid_df['tm_dw'] >= 5).astype(np.int8) # is weekend or not

# Remove date
del grid_df['date']
del grid_df['tm_d']

print('Save part 3')

# Safe part 3
grid_df.to_pickle(root + 'grid_part_3.pkl')
print('Size:', grid_df.shape)

# We don't need calendar_df anymore
del calendar_df
del grid_df

# grid_df = pd.read_pickle('grid_part_1.pkl')
# grid_df['d'] = grid_df['d'].apply(lambda x: x[2:]).astype(np.int16)

# Remove 'wm_yr_wk'
# as test values are not in train set
# del grid_df['wm_yr_wk']
#grid_df.to_pickle('grid_part_1.pkl')

#del grid_df



