# Kaggle Contest: M5 forcasting - accuracy
## Introduction
This is a kaggle contest for predicting sales of Walmart, the world's largest company.
Contestants are given Walmart sales data for previous 1913 days for 3049 products in 10 shops, 
and they are required to predict the daily sales in the next 28 days for each product.
## Dataset
There are three datasets used for sales prediction.The main dataset includes the daily sales records for all products.
Another two datasets includes,
* product prices (weekly)
* date, holiday and event

The main dataset has the following format:

product  | day_1  | day_2 | ... | day_1913
 ---- | ----- | ------  | ------ | ------
food_1   | 12     | 15    | ... | 22 
food_2   | 5      | 4     | ... | 11
...      | ...    | ...   | ... | ...  
house_1  | 2      | 3     | 2   | 1
house_2  | 5      | 2     | 5   | 7
...      | ...    | ...   | ... | ...  
hobby_1  | 6      | 6     | 2   | 8
hobby_2  | 4      | 2     | 8   | 7
...      | ...    | ...   | ... | ...  

Products can be categorized into food, household and hobby. Our task is to predict sales from day 1913 to 1941.

### augment dataset
In this contest, it is free to prepare outside dataset by contestants themselves. 
For example, contestants can use weather and temperature data. Intuitively, these factors have influences on daily sales.
Or share price may also have an influence on Walmart's sales.
However, collecting date faces difficulties. For example, there might exists unknown timelapse between share price and sales.
Also, the precise positions of these Walmart shops are not provided which makes it difficult to know the correct local weather.

## Data Analysis
![aggregate](https://github.com/danqingpan/Kaggle/blob/master/plots/aggregate_sales.png "aggregate sales")  
This figure shows the trend for total product sales.
Generally, the total sales rise with time. However, at the final stage, it seems that at the final stage, the rising speed accelerated.
Seasons also play a key role for the total sales amount.
Generally, there are ten shops and each shop have different sales performance.

## Methods/Models
Many supervised learning models can be used to predict sales amount.Here are some of them.
### ensemble trees
Some ensemble tree models are quite popular in kaggle. It seems that ensembles trees are more frequently used in kaggle contests.
Also in real situations, this types of models are quite useful.  
**XGBoost** is a model introduced by Tianqi Chen, etc in 2016. XGBoost is a model developed based on GBDT.
Similar with GBDT, XGBoost gives predicition by using the sum in the ensemble trees. 
The diffenence between XGBoost and GBDT lies in their objectives. XGBoost has some features,  
a. It generates new trees to fit the residual error.  
b. When predicting a observation, XGBoost decide its score by each tree and add them together.  
c. The final score is the prediction.  
**Light GBM** is a gradient boost framework that uses tree based learning algorithms. Compared with other tree ensembles, light GBM
uses leaf-wise growing method. It choose the leaf with max delta loss to grow. Light GBM is much faster compared with XGBoost while these two models have similar performance.  
In this contest, to fit 16 GB memory limit, many contestants choose to use light GBM as main classifier.


