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


