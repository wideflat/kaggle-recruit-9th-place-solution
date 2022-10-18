#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta, date
from collections import defaultdict
from sklearn import *
from sklearn.metrics import mean_squared_error
from sklearn.cluster import KMeans
from sklearn.cross_validation import KFold


# In[2]:


input_path = "C:/Users/Kohei/Documents/Kaggle/Recruit/00_input/"


# ### Function

# In[3]:


def dummy(df, cats, inplace_): # inplace=True will delete original feature
    for feat in cats:
        print('Creating dummy variables for {}'.format(feat))
        df_dummy = pd.get_dummies(df[feat], drop_first=True, sparse=True)
        df_dummy = df_dummy.rename(columns=lambda x: feat+'_'+ str(x))
        df.drop(([feat]), axis=1, inplace=inplace_)
        df = pd.merge(df, df_dummy, left_index=True, right_index=True)
    return df


# In[4]:


lbl = preprocessing.LabelEncoder()


# In[5]:


def reserve_calc(df):
    df['visit_datetime'] = pd.to_datetime(df['visit_datetime'])
    df['visit_date'] = df['visit_datetime'].dt.date
    df['visit_hour'] = df['visit_datetime'].dt.hour
    df['visit_date'] = pd.to_datetime(df['visit_date'])
    df['reserve_datetime'] = pd.to_datetime(df['reserve_datetime'])
    df['reserve_date'] = df['reserve_datetime'].dt.date
    df['reserve_date'] = pd.to_datetime(df['reserve_date'])
    df['hr_dif'] = df.apply(lambda r: (r['visit_datetime'] - r['reserve_datetime']).seconds/3600, axis=1)
    df['hr_dif_24mod'] = df['hr_dif'] % 24
    return df


# ### date

# In[6]:


df_date = pd.read_csv(os.path.join(input_path, 'date_info.csv')).rename(columns={'calendar_date':'visit_date'})

# date modification and add next days
df_date.drop(['day_of_week'],axis=1, inplace=True)
df_date['visit_date'] = pd.to_datetime(df_date['visit_date'])

# add next and previous days
date1 = [datetime(2017, 6, 1),date(2017, 6, 2)]
holiday1 = [0,0]
add1 = pd.DataFrame({'visit_date':date1,'holiday_flg':holiday1})

date2 = [datetime(2015, 12, 30),date(2015, 12, 31)]
holiday2 = [1,1]
add2 = pd.DataFrame({'visit_date':date2,'holiday_flg':holiday2})

df_date = pd.concat([add2,df_date,add1])
df_date['dow'] = df_date['visit_date'].dt.dayofweek
df_date['month'] = df_date['visit_date'].dt.month

df_date['visit_date+1d'] = (df_date['visit_date'] + timedelta(days=1))


# previos 1-6 days
for d in range(1,2):
    df_date['visit_date-'+str(d)+'d'] = (df_date['visit_date'] - timedelta(days=d*1))


# next day's holiday flag
tmp = df_date.iloc[:,:2]
tmp = tmp.rename(columns={'visit_date':'visit_date+1d','holiday_flg':'holiday_flg+1d'})

df_date = pd.merge(df_date, tmp, how='left', on='visit_date+1d')
df_date['holiday_flg+1d'] = df_date['holiday_flg+1d'].fillna(0)
df_date.drop(['visit_date+1d'],axis=1, inplace=True)

# previous day's holiday flag
tmp = df_date.iloc[:,:2]
tmp = tmp.rename(columns={'visit_date':'visit_date-1d','holiday_flg':'holiday_flg-1d'})

df_date = pd.merge(df_date, tmp, how='left', on='visit_date-1d')
df_date['holiday_flg-1d'] = df_date['holiday_flg-1d'].fillna(0)
df_date.drop(['visit_date-1d'],axis=1, inplace=True)


# revising holidays
# holidays on weekends are not holidays
wkend_holidays = df_date.apply((lambda x:(x.dow=='Sunday' or x.dow=='Saturday') and x.holiday_flg==1), axis=1)
df_date['holiday_flg_rev'] = df_date['holiday_flg']
df_date.loc[wkend_holidays, 'holiday_flg_rev'] = 0

# weight
df_date['weight'] = (df_date.index + 1) / len(df_date)**5

# one-hot-encoding
cats = ['dow','month']
df_date = dummy(df_date, cats, False)
df_date.head(10)


# ### visitors

# In[7]:


visitors = pd.read_csv(os.path.join(input_path, 'air_visit_data.csv'))
visitors['visit_date'] = pd.to_datetime(visitors['visit_date'])
visitors.head()


# ### air_store

# In[8]:


air_str = pd.read_csv(os.path.join(input_path, 'air_store_info_rev.csv'))

air_str['var_max_lat']  = air_str['latitude'].max() - air_str['latitude']
air_str['var_max_long'] = air_str['longitude'].max() - air_str['longitude']

kmeans = KMeans(n_clusters=12, random_state=0).fit(air_str[['latitude','longitude']])
air_str['km_latlong'] = pd.DataFrame(kmeans.predict(air_str[['latitude','longitude']]))


air_str.rename(columns={'air_genre_name':'air_genre'}, inplace=True)
air_str.rename(columns={'air_area_name':'air_area'}, inplace=True)
air_str['air_areaL1'] = air_str['air_area'].apply(lambda x: ' '.join(x.split(' ')[:1]))
air_str['air_areaL2'] = air_str['air_area'].apply(lambda x: ' '.join(x.split(' ')[1]))
air_str['air_areaL3'] = air_str['air_area'].apply(lambda x: ' '.join(x.split(' ')[2]))

air_str['air_genre'] = lbl.fit_transform(air_str['air_genre'])
air_str['air_areaL1_lbl'] = lbl.fit_transform(air_str['air_areaL1'])
air_str['air_areaL2_lbl'] = lbl.fit_transform(air_str['air_areaL2'])
air_str['air_areaL3_lbl'] = lbl.fit_transform(air_str['air_areaL3'])

air_str.drop(['air_area','air_areaL2','air_areaL3'],axis=1, inplace=True)

cats = ['air_areaL3_lbl']
air_str = dummy(air_str, cats, False)
air_str.head()


# ### hpg_store

# In[9]:


hpg_str = pd.read_csv(os.path.join(input_path, 'hpg_store_info.csv'))

kmeans = KMeans(n_clusters=12, random_state=0).fit(hpg_str[['latitude','longitude']])
hpg_str['km_hpg_latlong'] = pd.DataFrame(kmeans.predict(hpg_str[['latitude','longitude']]))

hpg_str.rename(columns={'hpg_genre_name':'hpg_genre'}, inplace=True)
hpg_str.rename(columns={'hpg_area_name':'hpg_areaL3'}, inplace=True)

hpg_str['hpg_genre'] = lbl.fit_transform(hpg_str['hpg_genre'])

hpg_str.drop(['latitude','longitude'],axis=1, inplace=True)

store = pd.read_csv(os.path.join(input_path, 'store_id_relation.csv'))
hpg_str = pd.merge(hpg_str, store, how='left', on=['hpg_store_id'])
hpg_str.head()


# ### air_reserve

# In[10]:


air_res = pd.read_csv(os.path.join(input_path, 'air_reserve.csv'))
air_res = reserve_calc(air_res)
air_res['dow'] = air_res['visit_date'].dt.dayofweek

air_res.head()


# ### hpg_reserve

# In[11]:


hpg_res = pd.read_csv(os.path.join(input_path, 'hpg_reserve.csv'))
hpg_res = reserve_calc(hpg_res)
hpg_res['dow'] = hpg_res['visit_date'].dt.dayofweek

store = pd.read_csv(os.path.join(input_path, 'store_id_relation.csv'))
hpg_res = pd.merge(hpg_res, store, how='left', on=['hpg_store_id'])
hpg_res.head()


# ### Weather

# In[12]:


jma = pd.read_csv(os.path.join(input_path, 'jma/jma.csv'))
jma['visit_date'] = pd.to_datetime(jma['date'])
jma.drop('date', axis=1, inplace=True)

jma['weather_daytime'] = lbl.fit_transform(jma['weather_daytime'])
jma['weather_nighttime'] = lbl.fit_transform(jma['weather_nighttime'])

cats = ['weather_daytime','weather_nighttime']
jma = dummy(jma, cats, False)
jma.head()


# ### Train and Test

# In[13]:


train = pd.read_csv(os.path.join(input_path, 'air_visit_data.csv'))
train['visit_date'] = pd.to_datetime(train['visit_date'])

test = pd.read_csv(os.path.join(input_path, 'sample_submission.csv'))
test.drop(['visitors'],axis=1, inplace=True)

test['air_store_id'] = test['id'].apply(lambda x: '_'.join(x.split('_')[:2]))
test['visit_date'] = test['id'].apply(lambda x: x.split('_')[-1])
test['visit_date'] = pd.to_datetime(test['visit_date'])
test.drop(['id'],axis=1, inplace=True)


# In[15]:


# weighted mean + min,max,count
def visit_numeric(df, tmp, prefix, var, alt_var1, alt_var2):
    wmean = lambda x:((x.weight * x.y).sum()/x.weight.sum())
    df = df.merge(tmp.groupby(var).apply(wmean).reset_index(), on=var, how='left')
    df.rename(columns={0:str(prefix)+'_wmean'}, inplace=True)
    missing = df[str(prefix)+'_wmean'].isnull()
    df.loc[missing,str(prefix)+'_wmean'] = df[missing].merge(tmp.groupby(alt_var1).mean().reset_index(),how='left',on=alt_var1)['y'].values
    missing = df[str(prefix)+'_wmean'].isnull()
    df.loc[missing,str(prefix)+'_wmean'] = df[missing].merge(tmp.groupby(alt_var2).mean().reset_index(),how='left',on=alt_var2)['y'].values
    
    # max
    df.loc[:,str(prefix)+'_max'] = df.merge(tmp.groupby(var).max().reset_index(), on=var, how='left')['y'].values
    missing = df[str(prefix)+'_max'].isnull()
    df.loc[missing,str(prefix)+'_max'] = df[missing].merge(tmp.groupby(alt_var1).max().reset_index(),how='left',on=alt_var1)['y'].values
    missing = df[str(prefix)+'_max'].isnull()
    df.loc[missing,str(prefix)+'_max'] = df[missing].merge(tmp.groupby(alt_var2).max().reset_index(),how='left',on=alt_var2)['y'].values
    
    # min
    df.loc[:,str(prefix)+'_min'] = df.merge(tmp.groupby(var).min().reset_index(), on=var, how='left')['y'].values
    missing = df[str(prefix)+'_min'].isnull()
    df.loc[missing,str(prefix)+'_min'] = df[missing].merge(tmp.groupby(alt_var1).min().reset_index(),how='left',on=alt_var1)['y'].values
    missing = df[str(prefix)+'_min'].isnull()
    df.loc[missing,str(prefix)+'_min'] = df[missing].merge(tmp.groupby(alt_var2).min().reset_index(),how='left',on=alt_var2)['y'].values

    # med
    df.loc[:,str(prefix)+'_med'] = df.merge(tmp.groupby(var).median().reset_index(), on=var, how='left')['y'].values
    missing = df[str(prefix)+'_med'].isnull()
    df.loc[missing,str(prefix)+'_med'] = df[missing].merge(tmp.groupby(alt_var1).median().reset_index(),how='left',on=alt_var1)['y'].values
    missing = df[str(prefix)+'_med'].isnull()
    df.loc[missing,str(prefix)+'_med'] = df[missing].merge(tmp.groupby(alt_var2).median().reset_index(),how='left',on=alt_var2)['y'].values

    # count
    df.loc[:,str(prefix)+'_cnt'] = df.merge(tmp.groupby(var).count().reset_index(), on=var, how='left')['y'].values
    missing = df[str(prefix)+'_cnt'].isnull()
    df.loc[missing,str(prefix)+'_cnt'] = df[missing].merge(tmp.groupby(alt_var1).count().reset_index(),how='left',on=alt_var1)['y'].values
    missing = df[str(prefix)+'_cnt'].isnull()
    df.loc[missing,str(prefix)+'_cnt'] = df[missing].merge(tmp.groupby(alt_var2).count().reset_index(),how='left',on=alt_var2)['y'].values

    return df


# In[16]:


def create_data(data_flag, df, begin, end, gap):
    
    # main data
    df_out = df[df.visit_date>=begin]
    df_out = df_out[df_out.visit_date<=end]

    # data used to create features
    df_visitors = visitors[visitors.visit_date<(begin-timedelta(days=gap))]

    # prep data to calculate mean
    df_tmp = pd.merge(df_visitors, df_date, how='left', on=['visit_date'])
    df_tmp = pd.merge(df_tmp, air_str, how='left', on=['air_store_id'])
    df_tmp = df_tmp.rename(columns={'visitors':'y'})
    df_tmp['y'] = np.log1p(df_tmp['y'])

    # prepare main data
    df_out = pd.merge(df_out, df_date, how='left', on=['visit_date'])
    df_out = pd.merge(df_out, air_str, how='left', on=['air_store_id'])
    df_out = pd.merge(df_out, hpg_str[['air_store_id','km_hpg_latlong']], how='left', on=['air_store_id'])

    # days from first date
    df_out['days_from_first_date'] = df_out.apply(lambda r:(r['visit_date']-pd.to_datetime(begin)+timedelta(days=gap+1)).days, axis=1)
    
    # flag
    if data_flag=='train_all':
        df_out['flag'] = 0
    elif data_flag=='train1':
        df_out['flag'] = 1
    elif data_flag=='train2':
        df_out['flag'] = 2
    elif data_flag=='test':
        df_out['flag'] = df_out['days_from_first_date'].apply(lambda x: 1 if x<=6 else 2)
    
    # store x date
    df_out = visit_numeric(df_out, df_tmp, 'dow_all',['air_store_id','dow'],['air_store_id'],['air_store_id'])
    df_out = visit_numeric(df_out, df_tmp, 'dowhol_all',['air_store_id','dow','holiday_flg'],['air_store_id','dow'],['air_store_id'])

    # target encoding
    if data_flag=='test':
        tmp = df_tmp.groupby(['air_genre'])['y'].mean().reset_index().rename(columns={'y':'genre_mean'})
        df_out  = df_out.merge(tmp, on = ['air_genre'], how='left')
        
        tmp = df_tmp.groupby(['air_genre','dow'])['y'].mean().reset_index().rename(columns={'y':'genre_dow_mean'})
        df_out  = df_out.merge(tmp, on = ['air_genre','dow'], how='left')
        
        tmp = df_tmp.groupby(['air_areaL3_lbl'])['y'].mean().reset_index().rename(columns={'y':'areaL3_mean'})
        df_out  = df_out.merge(tmp, on = ['air_areaL3_lbl'], how='left')
        
        tmp = df_tmp.groupby(['air_areaL3_lbl','dow'])['y'].mean().reset_index().rename(columns={'y':'areaL3_dow_mean'})
        df_out  = df_out.merge(tmp, on = ['air_areaL3_lbl','dow'], how='left')
    else:
        kf = KFold(df_out.shape[0], n_folds=5, random_state=1234, shuffle=True)
        for i, (tr_index,vl_index) in enumerate(kf):
            tr, vl = df_out.loc[tr_index].copy(), df_out.loc[vl_index].copy()
            tr['y'] = np.log1p(tr['visitors'])
            tmp = tr.groupby(['air_genre'])['y'].mean().reset_index().rename(columns={'y':'genre_mean'})
            vl  = vl.merge(tmp, on = ['air_genre'], how='left')
            
            tmp = tr.groupby(['air_genre','dow'])['y'].mean().reset_index().rename(columns={'y':'genre_dow_mean'})
            vl  = vl.merge(tmp, on = ['air_genre','dow'], how='left')
            
            tmp = tr.groupby(['air_areaL3_lbl'])['y'].mean().reset_index().rename(columns={'y':'areaL3_mean'})
            vl  = vl.merge(tmp, on = ['air_areaL3_lbl'], how='left')
            
            tmp = tr.groupby(['air_areaL3_lbl','dow'])['y'].mean().reset_index().rename(columns={'y':'areaL3_dow_mean'})
            vl  = vl.merge(tmp, on = ['air_areaL3_lbl','dow'], how='left')
            if i==0:
                tr_all = vl
            else:
                tr_all = pd.concat([tr_all,vl])
        df_out = tr_all
        del tr_all,vl
        
        
    # air_reserve
    df_airres = air_res[air_res.reserve_date<(begin-timedelta(days=gap))]
    df_out.loc[:,'res_ttl'] = df_out.merge(df_airres.groupby(['air_store_id','visit_date']).sum().reset_index(), on=['air_store_id','visit_date'], how='left')['reserve_visitors'].values
    df_out.loc[:,'res_cnt'] = df_out.merge(df_airres.groupby(['air_store_id','visit_date']).count().reset_index(), on=['air_store_id','visit_date'], how='left')['reserve_visitors'].values
    df_out.loc[:,'res_mean'] = df_out.merge(df_airres.groupby(['air_store_id','visit_date']).mean().reset_index(), on=['air_store_id','visit_date'], how='left')['reserve_visitors'].values
    df_out.loc[:,'res_hr_std'] = df_out.merge(df_airres.groupby(['air_store_id','visit_date']).std().reset_index(), on=['air_store_id','visit_date'], how='left')['visit_hour'].values
    
    tmp = df_airres.groupby(['air_store_id','visit_date','dow'])['reserve_visitors'].mean().reset_index().rename(columns={'reserve_visitors':'res_ttl'})
    tmp = tmp.groupby(['air_store_id','dow'])['res_ttl'].mean().reset_index().rename(columns={'res_ttl':'res_ttl_dow_mean'})
    df_out = df_out.merge(tmp, on=['air_store_id','dow'], how='left')
    
    tmp = df_airres.groupby(['air_store_id','visit_date','dow'])['reserve_visitors'].count().reset_index().rename(columns={'reserve_visitors':'res_cnt'})
    tmp = tmp.groupby(['air_store_id','dow'])['res_cnt'].mean().reset_index().rename(columns={'res_cnt':'res_cnt_dow_mean'})
    df_out = df_out.merge(tmp, on=['air_store_id','dow'], how='left')

    # lagged    
    for d in [1,3,5,10,20,30]:
        df_out['visit_date-'+str(d)+'d'] = (df_out['visit_date']-timedelta(days=d*1))
        df_out.loc[:,'lag_'+str(d)+'d'] = df_out.merge(df_tmp[['air_store_id','visit_date','y']],left_on=['air_store_id','visit_date-'+str(d)+'d'],right_on=['air_store_id','visit_date'], how='left')['y'].values
        df_out.drop(['visit_date-'+str(d)+'d'],axis=1, inplace=True)
        
    for w in range(1,21):
        df_out['visit_date-'+str(w)+'w'] = (df_out['visit_date']-timedelta(days=w*7))
        df_out.loc[:,'lag_'+str(w)+'w'] = df_out.merge(df_tmp[['air_store_id','visit_date','y']],left_on=['air_store_id','visit_date-'+str(w)+'w'],right_on=['air_store_id','visit_date'], how='left')['y'].values
        df_out.loc[:,'lag_res_'+str(w)+'w'] = df_out.merge(df_airres.groupby(['air_store_id','visit_date']).sum().reset_index(),left_on=['air_store_id','visit_date-'+str(w)+'w'],right_on=['air_store_id','visit_date'], how='left')['reserve_visitors'].values
        df_out.drop(['visit_date-'+str(w)+'w'],axis=1, inplace=True)
    
    # moving avg
    for d in [5,10,20,30,50,100]:
        tmp = df_tmp[df_tmp.visit_date>=(begin-timedelta(days=gap)-timedelta(days=d))]
        
        tmp2 = tmp.groupby(['air_store_id'])['y'].mean().reset_index().rename(columns={'y':'mean_'+str(d)+'d'})
        tmp2['mean_'+str(d)+'d'] = tmp2['mean_'+str(d)+'d'].fillna(0)
        df_out = pd.merge(df_out, tmp2, on=['air_store_id'], how='left')
        
        tmp2 = tmp.groupby(['air_store_id'])['y'].max().reset_index().rename(columns={'y':'max_'+str(d)+'d'})
        df_out = pd.merge(df_out, tmp2, on=['air_store_id'], how='left')
        
        tmp2 = tmp.groupby(['air_store_id'])['y'].min().reset_index().rename(columns={'y':'min_'+str(d)+'d'})
        df_out = pd.merge(df_out, tmp2, on=['air_store_id'], how='left')
        
        tmp2 = tmp.groupby(['air_store_id'])['y'].std().reset_index().rename(columns={'y':'std_'+str(d)+'d'})
        df_out = pd.merge(df_out, tmp2, on=['air_store_id'], how='left')
        
        df_out['scale_to_maxmin_'+str(d)+'d'] =  (df_out['mean_'+str(d)+'d']-df_out['min_'+str(d)+'d'])/(df_out['max_'+str(d)+'d']-df_out['min_'+str(d)+'d'])        
        df_out['scale_to_std_'+str(d)+'d'] =  (df_out['mean_'+str(d)+'d'])/(df_out['std_'+str(d)+'d'])        

        del tmp, tmp2

    for w in [2,4,8]:
        tmp = df_tmp[df_tmp.visit_date>=(begin-timedelta(days=gap)-timedelta(days=w*7))]
        tmp2 = tmp.groupby(['air_store_id','dow'])['y'].mean().reset_index().rename(columns={'y':'mean_dow_'+str(w)+'w'})
        tmp2['mean_dow_'+str(w)+'w'] = tmp2['mean_dow_'+str(w)+'w'].fillna(0)
        df_out = pd.merge(df_out, tmp2, on=['air_store_id','dow'], how='left')
        
        del tmp, tmp2
        
    # weather
    df_out = df_out.merge(jma, left_on=['air_areaL1','visit_date'], right_on=['prefecture','visit_date'], how='left')
    df_out.drop(['prefecture','air_areaL1'],axis=1, inplace=True)
    df_out.drop(['wind_max_inst','rainfall_max1h','wind_avg','temperature_high','temperature_low','temperature_avg','snowfall_max','humidity_avg','daylight_hr'],axis=1, inplace=True)
    
    # drop
    df_out.drop('weight',axis=1, inplace=True)

    
    # fill na
    df_out['na_cnt'] = df_out.isnull().sum(axis=1)
    df_out = df_out.fillna(-1)
    
    return df_out


# In[17]:


# train_all - 39 days split, 0 days gap
ini_date = date(2017,4,23)
df_all = pd.DataFrame()
split_days = 39
skip_days = 39
gap_days = 0
for i in range(1,int(360/skip_days)+1):
    print('%i / %i ' %(i,int(360/skip_days)))
    df_out = create_data('train_all', train,                         ini_date-timedelta(days=i*skip_days),                         ini_date-timedelta(days=i*skip_days)+timedelta(days=(split_days-1)),                         gap_days)
    df_all = pd.concat([df_all,df_out])
df_all.to_csv('train_all.csv', index=False)


# In[18]:


# train1 - 6 days split, 0 days gap
ini_date = date(2017,4,23)
df_all = pd.DataFrame()
split_days = 6
skip_days = 6
gap_days = 0
for i in range(1,int(360/skip_days)+1):
    print('%i / %i ' %(i,int(360/skip_days)))
    df_out = create_data('train1', train,                         ini_date-timedelta(days=i*skip_days),                         ini_date-timedelta(days=i*skip_days)+timedelta(days=(split_days-1)),                         gap_days)
    df_all = pd.concat([df_all,df_out])
df_all.to_csv('train1.csv', index=False)


# In[19]:


# train2 - 33 days split, 6 days gap
ini_date = date(2017,4,23)
df_all = pd.DataFrame()
split_days = 33
skip_days = 33
gap_days = 6
for i in range(1,int(360/split_days)+1):
    print('%i / %i ' %(i,int(360/skip_days)))
    df_out = create_data('train2', train,                         ini_date-timedelta(days=i*skip_days),                         ini_date-timedelta(days=i*skip_days)+timedelta(days=(split_days-1)),                         gap_days)
    df_all = pd.concat([df_all,df_out])
df_all.to_csv('train2.csv', index=False)


# In[20]:


test_all = create_data('test', test, date(2017,4,23), date(2017,5,31), 0)
test_all.to_csv('test_all.csv', index=False)


# In[ ]:




