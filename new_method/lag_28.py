import numpy as np
import pandas as pd
import os, warnings


warnings.filterwarnings('ignore')

root = 'kaggle/input/m5-forecasting-accuracy/'

print('Load Data')
grid_df = pd.read_pickle(root + 'grid_part_1.pkl')
grid_df = grid_df[['id','d','store_id','sales']]


# direct shift
for day_shift in list(range(1, 1 + 3)) + [7,14,28,56]:
    print('lag:' + str(day_shift))
    grid_df['sales_lag_'+ str(day_shift)] = grid_df.groupby(['id'])['sales'].transform(lambda x: x.shift(day_shift)).astype(np.float16)

# item-wise mean with different combinations
for day_shift in [1,7,28]: # shift and mean
    print('mean:', day_shift)
    for day_window in [7,28]:
        col_name = 'lag_'+ str(day_shift) + '_mean_' + str(day_window)
        grid_df[col_name] = grid_df.groupby(['id'])['sales'].transform(lambda x: x.shift(day_shift).rolling(day_window).mean()).astype(np.float16)

# std(1 day lag)
for day_shift in [7,28]: # shift 1 day and std
    print('std:', day_shift)
    grid_df['lag_1_' + 'std_' + str(day_shift)]  = grid_df.groupby(['id'])['sales'].transform(lambda x: x.shift(1).rolling(day_shift).std()).astype(np.float16)

# shop-wise mean (different lag)
for day_shift in list(range(1, 1 + 3)) + [7]: # shop mean 
    print('shop-wise mean:' + str(day_shift))
    grid_df['shop_lag_'+ str(day_shift) + '_mean' ]  = grid_df.groupby(['store_id','d'])['sales_lag_' + str(day_shift)].transform('mean').astype(np.float16)

    
print('Save lags and rollings')
grid_df.to_pickle(root + 'lags_df.pkl')

grid_df.info()
