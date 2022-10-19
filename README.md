# kaggle-recruit-9th-place-solution

## Competition
https://www.kaggle.com/competitions/recruit-restaurant-visitor-forecasting


## Dataset
Datasets are provided in the competition's official page:
https://www.kaggle.com/competitions/recruit-restaurant-visitor-forecasting/data

Beside those, external dataset was available (below is a post from the competition organizer.)
https://www.kaggle.com/competitions/recruit-restaurant-visitor-forecasting/discussion/45012

Downloaded dataset is located in `./data`.

## Overview of 9th place solution


### 1. Data preparation

The script is saved in `./scr/1_data_preparation`.

In this competition, contestants are provided a time-series forecasting problem. The dataset includes the number of visitors to each restaurant on specific dates and we were required to forecast the number of visitors in the future period from 2017-04-23 to 2017-05-31 (39 days in total).
To prevent data leakage and properly validate the model, I created features and target lables of the training dataset by mimicing the problem set as follows: 

Features - using information available until 2017-03-14
Labels - the number of visitors from 2017-03-15 to 2017-04-22 (39 days)

To increase the size of the training dataset, the data period was slided by another 39 days. And the same procedures were executed nine times.




### 2. 1st stage model

The scripts for the 1st stage models are saved in `./src/2_first_stage_model`.

I explored a variety of modeling libraries (xgboost, lightgbm, catboost, keras) and different sets of features, and evaluated local cross validation and public LB scores. Initially 15 models were selected as candidates for the ensemble model.




### 3. 2nd stage model - ensemble
- I tried stacking with xgboost as well as weighted average of 1st stage model predictions. The public LB scores were similar between those two, but local CV scores are slightly better for weighted average and I chose weighted average in the end.
- I tried different sets of single model predictions among 15 models and selected the below 4 based on local CV and public LB score.
- Public LB is 0.467 and Private LB 0.46780. I placed at 9th place!

| Single Model     | CV in train1 | CV in train2 | Public LB |
| ---------------- | ------------ | ------------ | --------- |
| 03_17bb_xgb      | 0.4978       | 0.5120       | 0.469     |
| 03_19al_xgb      | 0.4999       | 0.5165       | NA        |
| 05_19ac_catboost | 0.4985       | 0.5151       | 0.469     |
| 05_19acb_catboost| NA           | NA           | NA        |

| Ensemble Model   | CV in train1 | CV in train2 | Public LB | Private LB|
| ---------------- | ------------ | ------------ | --------- | --------- |
| 22_19ae_ensemble | 0.4956       | 0.5101       | 0.467     | 0.46780   |

