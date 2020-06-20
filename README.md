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

* The main dataset has the following format:

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

### augmented datasets
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
* **XGBoost** is a model introduced by Tianqi Chen, etc in 2016. XGBoost is a model developed based on GBDT.
Similar with GBDT, XGBoost gives predicition by using the sum in the ensemble trees. 
The diffenence between XGBoost and GBDT lies in their objectives. XGBoost has some features,  
**a**. It generates new trees to fit the residual error.  
**b**. When predicting a observation, XGBoost decide its score by each tree and add them together.  
**c**. The final score is the prediction.  
* **Light GBM** is a gradient boost framework that uses tree based learning algorithms. Compared with other tree ensembles, light GBM
uses leaf-wise growing method. It choose the leaf with max delta loss to grow. Light GBM is much faster compared with XGBoost while these two models have similar performance.  
In this contest, to fit 16 GB memory limit, many contestants choose to use light GBM as main classifier.  
* other typical tree ensemble models includes **random forest**, **ADAboost** etc.  

### problem with tree ensembles
It might seems to be unnatural to use a tree ensemble here since daily sales is time series data. A basic solution here is to use time delay data to predict current sales. E.g. we can use sales 7 days ago to predict the sales of the current day. Intuitively, today's sales amount might be similar to the sales amount 7 days ago. We can further use other time delay such as 1 day, 2 days , 14 days as features.  
Since the task is to predict sales for the last 28 days. Intuitively, data close to day 1913 will be more useful since it reflects the current trend. To emphasize this property, we can put different weight on training data and make close 'close' data more important. Also, we can see that seasons have an large influence on the sales, data in the same season should also be emphasized. 

### Deep models
We also have some choices for deep learning models. Deep learning models generally have a better performance when there are a lot of data.  
* **LSTM** or RNN are classic choices for time series data. They can 'remember' previous training samples. Based on this property, they are wildly used in natural language processing area. It seems to be natural to use LSTM as a predictor in this case.  
### problem with LSTM
Some difficulities using LSTM includes:  
* hard to use outside data. Here outside data refer to non-sales data such as holiday information.  
* can only be trained for one product.  
* training is slow because this is a deep model.The training basically requires GPU to accelerate.  

## Choosing a model
Considering the computing resources provided by kaggle, it seem better to choose Light GBM model as the predictor.There are several reasons.  
**a**. Kaggle provides freely 16GB memory and 8 core CPU for this contest. Since there is no GPU, it will be difficult to use LSTM.  
**b**. 16GB is too small for XGBoost in this contest unless we only choose to use partial data for training. It will be possible if we divide dataset into shops (totally 10 shops) and train independent models.  
**c**. Even if we can use XGBoost, light GBM is still faster to train.  
**d**. Generally, the performance between Light GBM and XGBoost is similar.  
Based on these reasons, it seems to be better to start the training with light GBM model.  

## Feature engineering
As described before, we can use time lag data as features. We use the model to fit the sales of current day.  
* input: time lag (sales for previous days), holiday, category, date , etc
* output: sales of the day  

some features are listed here:
* normal lag: e.g. 1,2,7,14 ... days' sale lag  
* shop average lag: the average sale of the shop/category with lag  
* week/month average lag: the average sale for the previous week/month  
* price  
* price average  
* date  
* holiday  
* event  

It is possible to extract more features. However, it may cause problems.  
**a**, Too many features may lead to overfitting.  
**b**, Training dataset might be too large to fit into the memory.  

## Training

