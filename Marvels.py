#!/usr/bin/env python
# coding: utf-8

# In[1]:
#pip install streamlit
#pip install scikit-learn-intelex
get_ipython().system('pip install scikit-learn-intelex')
from sklearnex import patch_sklearn
patch_sklearn()


# In[2]:


from IPython.display import Image
from IPython.core.display import HTML 
Image(url= "https://cdn-images-1.medium.com/max/1600/1*fUKUEUeqgIYU9xlxj4pwhw.png")


# Getting setup

# In[3]:


get_ipython().system('pip uninstall imbalanced-learn -y')


# In[4]:


#!pip install plotly 
#!pip install chart-studio 
#!pip install -U scikit-learn 

get_ipython().system('pip install imbalanced-learn')


# import required libraries

# Import the data from github

# In[5]:


#import chart_studio.plotly as py
#import plotly.graph_objs as go
#import plotly as plotly
import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import SMOTENC
from sklearn import metrics

from sklearn.preprocessing import LabelEncoder

import xgboost as xgb
from xgboost.sklearn import XGBClassifier

import types
import pandas as pd

def __iter__(self): return 0


# In[6]:


#Remove the data if you run this notebook more than once
get_ipython().system('rm equipment_failure_data_1.csv')


# In[7]:


#import first half from github
get_ipython().system('wget https://raw.githubusercontent.com/shadgriffin/machine_failure/master/equipment_failure_data_1.csv')


# In[8]:


data=pd.read_csv("dataset.csv")


# In[9]:


data


# In[10]:


data.head()


# In[11]:


data.shape


# In[12]:


DataFrame = pd.DataFrame(data.groupby(['ID']).agg(['count']))
DataFrame.shape


# In[13]:


DataFrame = pd.DataFrame(data.groupby(['DATE']).agg(['count']))
DataFrame.shape


# In[14]:


df_failure_thingy=data
df_failure_thingy=df_failure_thingy.drop_duplicates(subset=['ID','DATE'])
df_failure_thingy.shape


# In[15]:


data.isnull().sum()


# There are no null values in the data set

# In[16]:


DataFrame = pd.DataFrame(data.groupby(['EQUIPMENT_FAILURE'])['ID'].agg(['count']))
DataFrame


# In[17]:


data.describe()


# In[18]:


cor=data.corr( method='pearson')
cor=cor[['EQUIPMENT_FAILURE']]
cor['ABS_EQUIPMENT_FAILURE']=abs(cor['EQUIPMENT_FAILURE'])
cor=cor.sort_values(by=['ABS_EQUIPMENT_FAILURE'], ascending=[False])


# In[19]:


cor


# In[20]:


data['DATE'] = pd.to_datetime(data['DATE'])


# In[21]:


data=data.sort_values(by=['ID','DATE'], ascending=[True, True])

data['flipper'] = np.where((data.ID != data.ID.shift(1)), 1, 0)
data.head()


# #### Data transformations and Feature Engineering

# In[22]:


#Select the first record of each machine
feature_window=21
starter=data[data['flipper'] == 1]
starter=starter[['DATE','ID']]


# In[23]:


#rename date to start_date
starter=starter.rename(index=str, columns={"DATE": "START_DATE"})


# In[24]:


#convert START_DATE to date
starter['START_DATE'] = pd.to_datetime(starter['START_DATE'])


# In[25]:


#Merge START_DATE to the original data set
data=data.sort_values(by=['ID', 'DATE'], ascending=[True, True])
starter=starter.sort_values(by=['ID'], ascending=[True])
data =data.merge(starter, on=['ID'], how='left')


# In[26]:


# calculate the number of days since the beginning of each well. 
data['C'] = data['DATE'] - data['START_DATE']
data['TIME_SINCE_START'] = data['C'] / np.timedelta64(1, 'D')
data=data.drop(columns=['C'])
data['too_soon'] = np.where((data.TIME_SINCE_START < feature_window) , 1, 0)


# In[27]:


data.rename(columns={'Mass Air Flow Rate(g/s)':'S15'},inplace=True)


# In[28]:


data.rename(columns={'Fuel Pressure':'S16'},inplace=True)


# In[29]:


data.rename(columns={'OIL PRESSURE':'S17'},inplace=True)


# In[30]:


data.rename(columns={'BAROMTRIC PRESSURE(KPA)':'S18'},inplace=True)


# In[31]:


data.rename(columns={'AIR INTAKE TEMP':'S19'},inplace=True)


# In[32]:


#Crate a running mean, max, min, and median for the sensor variables
data['RPM_mean'] = np.where((data.too_soon == 0),(data['RPM'].rolling(min_periods=1, window=feature_window).mean()) , data.RPM)
data['RPM_median'] = np.where((data.too_soon == 0),(data['RPM'].rolling(min_periods=1, window=feature_window).median()) , data.RPM)
data['RPM_max'] = np.where((data.too_soon == 0),(data['RPM'].rolling(min_periods=1, window=feature_window).max()) , data.RPM)
data['RPM_min'] = np.where((data.too_soon == 0),(data['RPM'].rolling(min_periods=1, window=feature_window).min()) , data.RPM)


data['THROTTLE_mean'] = np.where((data.too_soon == 0),(data['THROTTLE'].rolling(min_periods=1, window=feature_window).mean()) , data.THROTTLE)
data['THROTTLE_median'] = np.where((data.too_soon == 0),(data['THROTTLE'].rolling(min_periods=1, window=feature_window).median()) , data.THROTTLE)
data['THROTTLE_max'] = np.where((data.too_soon == 0),(data['THROTTLE'].rolling(min_periods=1, window=feature_window).max()) , data.THROTTLE)
data['THROTTLE_min'] = np.where((data.too_soon == 0),(data['THROTTLE'].rolling(min_periods=1, window=feature_window).min()) , data.THROTTLE)

data['S15_mean'] = np.where((data.too_soon == 0),(data['S15'].rolling(min_periods=1, window=feature_window).mean()) , data.S15)
data['S15_median'] = np.where((data.too_soon == 0),(data['S15'].rolling(min_periods=1, window=feature_window).median()) , data.S15)
data['S15_max'] = np.where((data.too_soon == 0),(data['S15'].rolling(min_periods=1, window=feature_window).max()) , data.S15)
data['S15_min'] = np.where((data.too_soon == 0),(data['S15'].rolling(min_periods=1, window=feature_window).min()) , data.S15)

data['S16_mean'] = np.where((data.too_soon == 0),(data['S16'].rolling(min_periods=1, window=feature_window).mean()) , data.S16)
data['S16_median'] = np.where((data.too_soon == 0),(data['S16'].rolling(min_periods=1, window=feature_window).median()) , data.S16)
data['S16_max'] = np.where((data.too_soon == 0),(data['S16'].rolling(min_periods=1, window=feature_window).max()) , data.S16)
data['S16_min'] = np.where((data.too_soon == 0),(data['S16'].rolling(min_periods=1, window=feature_window).min()) , data.S16)


data['S17_mean'] = np.where((data.too_soon == 0),(data['S17'].rolling(min_periods=1, window=feature_window).mean()) , data.S17)
data['S17_median'] = np.where((data.too_soon == 0),(data['S17'].rolling(min_periods=1, window=feature_window).median()) , data.S17)
data['S17_max'] = np.where((data.too_soon == 0),(data['S17'].rolling(min_periods=1, window=feature_window).max()) , data.S17)
data['S17_min'] = np.where((data.too_soon == 0),(data['S17'].rolling(min_periods=1, window=feature_window).min()) , data.S17)

data['S18_mean'] = np.where((data.too_soon == 0),(data['S18'].rolling(min_periods=1, window=feature_window).mean()) , data.S18)
data['S18_median'] = np.where((data.too_soon == 0),(data['S18'].rolling(min_periods=1, window=feature_window).median()) , data.S18)
data['S18_max'] = np.where((data.too_soon == 0),(data['S18'].rolling(min_periods=1, window=feature_window).max()) , data.S18)
data['S18_min'] = np.where((data.too_soon == 0),(data['S18'].rolling(min_periods=1, window=feature_window).min()) , data.S18)



data['S19_mean'] = np.where((data.too_soon == 0),(data['S19'].rolling(min_periods=1, window=feature_window).mean()) , data.S19)
data['S19_median'] = np.where((data.too_soon == 0),(data['S19'].rolling(min_periods=1, window=feature_window).median()) , data.S19)
data['S19_max'] = np.where((data.too_soon == 0),(data['S19'].rolling(min_periods=1, window=feature_window).max()) , data.S19)
data['S19_min'] = np.where((data.too_soon == 0),(data['S19'].rolling(min_periods=1, window=feature_window).min()) , data.S19)


data.head()


# In[33]:


data['RPM_chg'] = np.where((data.RPM_mean == 0),0 , data.RPM/data.RPM_mean)


data['THROTTLE_chg'] = np.where((data.THROTTLE_mean == 0),0 , data.THROTTLE/data.THROTTLE_mean)

data['S15_chg'] = np.where((data.S15_mean==0),0 , data.S15/data.S15_mean)
data['S16_chg'] = np.where((data.S16_mean == 0),0 , data.S16/data.S16_mean)
data['S17_chg'] = np.where((data.S17_mean == 0),0 , data.S17/data.S17_mean)
data['S18_chg'] = np.where((data.S18_mean == 0),0 , data.S18/data.S18_mean)
data['S19_chg'] = np.where((data.S19_mean == 0),0 , data.S19/data.S19_mean)


# #### Dealing with the small number of failures.

# In[34]:


target_window=28
data=data.sort_values(by=['ID', 'DATE'], ascending=[True, True])
data.reset_index(level=0, inplace=True)


# In[35]:


df_failure_thingy=data[data['EQUIPMENT_FAILURE'] == 1]
df_failure_thingy=df_failure_thingy[['DATE','ID']]
df_failure_thingy=df_failure_thingy.rename(index=str, columns={"DATE": "FAILURE_DATE"})
data=data.sort_values(by=['ID'], ascending=[True])
df_failure_thingy=df_failure_thingy.sort_values(by=['ID'], ascending=[True])


# In[36]:


data =data.merge(df_failure_thingy, on=['ID'], how='left')


# In[37]:


data=data.sort_values(by=['ID','DATE'], ascending=[True, True])

data['FAILURE_DATE'] = pd.to_datetime(data['FAILURE_DATE'])
data['DATE'] = pd.to_datetime(data['DATE'])
data['C'] = data['FAILURE_DATE'] - data['DATE']

data['TIME_TO_FAILURE'] = data['C'] / np.timedelta64(1, 'D')


# In[38]:


data=data.drop(columns=['index'])


# In[39]:


data=data.sort_values(by=['ID', 'DATE'], ascending=[True, True])


# In[40]:


data.reset_index(inplace=True)


# In[41]:


data.head()


# In[42]:


data['FAILURE_TARGET'] = np.where(((data.TIME_TO_FAILURE < target_window) & ((data.TIME_TO_FAILURE>=0))), 1, 0)

data.head() #Failure target  is equal to 1 if the record proceeds a failure by "failure_window" days or less.


# In[43]:


tips_summed = data.groupby(['FAILURE_TARGET'])['RPM'].count()
tips_summed


# In[44]:


data['FAILURE_TARGET'].mean()


# #### Creating the Testing, Training and Validation Groupings

# In[45]:


#Getting a Unique List of All IDs 

aa=data
pd_id=aa.drop_duplicates(subset='ID')
pd_id=pd_id[['ID']]
pd_id.shape


# In[46]:


np.random.seed(42)


# In[47]:


pd_id['wookie'] = (np.random.randint(0, 10000, pd_id.shape[0]))/10000


# In[48]:


pd_id=pd_id[['ID', 'wookie']]


# In[49]:


pd_id['MODELING_GROUP'] = np.where(((pd_id.wookie <= 0.35)), 'TRAINING', np.where(((pd_id.wookie <= 0.65)), 'VALIDATION', 'TESTING'))


# In[50]:


tips_summed = pd_id.groupby(['MODELING_GROUP'])['wookie'].count()
tips_summed


# In[51]:


data=data.sort_values(by=['ID'], ascending=[True])
pd_id=pd_id.sort_values(by=['ID'], ascending=[True])


# In[52]:


data =data.merge(pd_id, on=['ID'], how='inner')

data.head()


# In[53]:


#Records in a group
tips_summed = data.groupby(['MODELING_GROUP'])['wookie'].count()
tips_summed


# In[54]:


#Failures in a group
tips_summed = data.groupby(['MODELING_GROUP'])['FAILURE_TARGET'].sum()
tips_summed


# In[55]:


#Building dataframe for training data
df_training=data[data['MODELING_GROUP'] == 'TRAINING']
df_training=df_training.drop(columns=['MODELING_GROUP','C','wookie','TIME_TO_FAILURE','flipper','START_DATE'])
df_training.shape


# In[56]:


#Building dataframe for training and testing data
df_train_test=data[data['MODELING_GROUP'] != 'VALIDATION']
df_train_test=df_train_test.drop(columns=['wookie','TIME_TO_FAILURE','flipper','START_DATE'])
df_train_test.shape


# In[57]:


#Building dataframe for all data
df_total=data.drop(columns=['C','wookie','TIME_TO_FAILURE','flipper','START_DATE'])
df_total.shape


# #### SMOTE the Training Data

# In[58]:


training_features=df_training[['REGION_CLUSTER','MAINTENANCE_VENDOR','MANUFACTURER','WELL_GROUP','AGE_OF_EQUIPMENT','S15','S17','THROTTLE','RPM',
 'S16','S19','S18','TYRE PRESSURE','RPM_mean','RPM_median','RPM_max','RPM_min','THROTTLE_mean','THROTTLE_median','THROTTLE_max','THROTTLE_min','S15_mean','S15_median',
 'S15_max','S15_min','S16_mean','S16_median','S16_max','S16_min','S17_mean','S17_median','S17_max','S17_min','S18_mean','S18_median','S18_max','S18_min','S19_mean','S19_median','S19_max','S19_min',
 'RPM_chg','THROTTLE_chg','S15_chg','S16_chg','S17_chg','S18_chg','S19_chg']]


# In[59]:


training_target=df_training[['FAILURE_TARGET']]


# In[60]:


from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import SMOTENC
smx = SMOTENC(random_state=12,  categorical_features=[0, 1, 2, 3])


# In[61]:


x_res, y_res = smx.fit_resample(training_features, training_target.values.ravel())


# In[63]:


#Formating independent variable
df_x=pd.DataFrame(x_res)
df_x.columns = [
'REGION_CLUSTER','MAINTENANCE_VENDOR','MANUFACTURER','WELL_GROUP','AGE_OF_EQUIPMENT','S15','S17','THROTTLE','RPM',
 'S16','S19','S18','TYRE PRESSURE','RPM_mean','RPM_median','RPM_max','RPM_min','THROTTLE_mean','THROTTLE_median','THROTTLE_max','THROTTLE_min','S15_mean','S15_median',
 'S15_max','S15_min','S16_mean','S16_median','S16_max','S16_min','S17_mean','S17_median','S17_max','S17_min','S18_mean','S18_median','S18_max','S18_min','S19_mean','S19_median','S19_max','S19_min',
 'RPM_chg','THROTTLE_chg','S15_chg','S16_chg','S17_chg','S18_chg','S19_chg']
df_x.head()


# In[64]:


#formating depndent variable
df_y=pd.DataFrame(y_res)
df_y.columns = ['FAILURE_TARGET']


# In[65]:


df_y.mean(axis = 0) 


# In[66]:


#merging dependent and independent variables into a dataframe
df_balanced = pd.concat([df_y, df_x], axis=1)
df_balanced.head()


# #### Data Tranformation and feature engineering

# In[67]:


#Converting categorical data into binary variables
df_dv = pd.get_dummies(df_balanced['REGION_CLUSTER'])
df_dv=df_dv.rename(columns={"A": "CLUSTER_A","B":"CLUSTER_B","C":"CLUSTER_C","D":"CLUSTER_D","E":"CLUSTER_E","F":"CLUSTER_F","G":"CLUSTER_G","H":"CLUSTER_H"})
df_balanced= pd.concat([df_balanced, df_dv], axis=1)
df_dv = pd.get_dummies(df_balanced['MAINTENANCE_VENDOR'])
df_dv=df_dv.rename(columns={"I": "MV_I","J":"MV_J","K":"MV_K","L":"MV_L","M":"MV_M","N":"MV_N","O":"MV_O","P":"MV_P"})
df_balanced = pd.concat([df_balanced, df_dv], axis=1)
df_dv = pd.get_dummies(df_balanced['MANUFACTURER'])
df_dv=df_dv.rename(columns={"Q": "MN_Q","R":"MN_R","S":"MN_S","T":"MN_T","U":"MN_U","V":"MN_V","W":"MN_W","X":"MN_X","Y":"MN_Y","Z":"MN_Z"})
df_balanced = pd.concat([df_balanced, df_dv], axis=1)
df_dv = pd.get_dummies(df_balanced['WELL_GROUP'])
df_dv=df_dv.rename(columns={1: "WG_1",2:"WG_2",3:"WG_3",4:"WG_4",5:"WG_5",6:"WG_6",7:"WG_7",8:"WG_8"})
df_balanced = pd.concat([df_balanced, df_dv], axis=1)


# In[68]:


df_dv = pd.get_dummies(df_train_test['REGION_CLUSTER'])
df_dv=df_dv.rename(columns={"A": "CLUSTER_A","B":"CLUSTER_B","C":"CLUSTER_C","D":"CLUSTER_D","E":"CLUSTER_E","F":"CLUSTER_F","G":"CLUSTER_G","H":"CLUSTER_H"})
df_train_test= pd.concat([df_train_test, df_dv], axis=1)
df_dv = pd.get_dummies(df_train_test['MAINTENANCE_VENDOR'])
df_dv=df_dv.rename(columns={"I": "MV_I","J":"MV_J","K":"MV_K","L":"MV_L","M":"MV_M","N":"MV_N","O":"MV_O","P":"MV_P"})
df_train_test = pd.concat([df_train_test, df_dv], axis=1)
df_dv = pd.get_dummies(df_train_test['MANUFACTURER'])
df_dv=df_dv.rename(columns={"Q": "MN_Q","R":"MN_R","S":"MN_S","T":"MN_T","U":"MN_U","V":"MN_V","W":"MN_W","X":"MN_X","Y":"MN_Y","Z":"MN_Z"})
df_train_test = pd.concat([df_train_test, df_dv], axis=1)
df_dv = pd.get_dummies(df_train_test['WELL_GROUP'])
df_dv=df_dv.rename(columns={1: "WG_1",2:"WG_2",3:"WG_3",4:"WG_4",5:"WG_5",6:"WG_6",7:"WG_7",8:"WG_8"})
df_train_test = pd.concat([df_train_test, df_dv], axis=1)


# In[69]:


f_dv = pd.get_dummies(df_total['REGION_CLUSTER'])
df_dv=df_dv.rename(columns={"A": "CLUSTER_A","B":"CLUSTER_B","C":"CLUSTER_C","D":"CLUSTER_D","E":"CLUSTER_E","F":"CLUSTER_F","G":"CLUSTER_G","H":"CLUSTER_H"})
df_total= pd.concat([df_total, df_dv], axis=1)
df_dv = pd.get_dummies(df_total['MAINTENANCE_VENDOR'])
df_dv=df_dv.rename(columns={"I": "MV_I","J":"MV_J","K":"MV_K","L":"MV_L","M":"MV_M","N":"MV_N","O":"MV_O","P":"MV_P"})
df_total = pd.concat([df_total, df_dv], axis=1)
df_dv = pd.get_dummies(df_total['MANUFACTURER'])
df_dv=df_dv.rename(columns={"Q": "MN_Q","R":"MN_R","S":"MN_S","T":"MN_T","U":"MN_U","V":"MN_V","W":"MN_W","X":"MN_X","Y":"MN_Y","Z":"MN_Z"})
df_total = pd.concat([df_total, df_dv], axis=1)
df_dv = pd.get_dummies(df_total['WELL_GROUP'])
df_dv=df_dv.rename(columns={1: "WG_1",2:"WG_2",3:"WG_3",4:"WG_4",5:"WG_5",6:"WG_6",7:"WG_7",8:"WG_8"})
df_total = pd.concat([df_total, df_dv], axis=1)


# #### Building the model on balance data

# In[70]:


df_balanced=df_balanced.drop(columns=['REGION_CLUSTER','MAINTENANCE_VENDOR','MANUFACTURER','WELL_GROUP'])


# In[71]:


#seperating the dependent and independent variable
features = [x for x in df_balanced.columns if x not in ['FAILURE_TARGET']]  
dependent=pd.DataFrame(df_balanced['FAILURE_TARGET'])
independent=df_balanced.drop(columns=['FAILURE_TARGET'])


# In[72]:


df_balanced.head()


# In[73]:


import matplotlib.pylab as plt
get_ipython().run_line_magic('matplotlib', 'inline')

def evaluate_model(alg, train, target, predictors,  early_stopping_rounds=1): 
#Fit the algorithm on the data
    alg.fit(train[predictors], target['FAILURE_TARGET'], eval_metric='error')
        
 #Predict training set:
    dtrain_predictions = alg.predict(train[predictors])
    dtrain_predprob = alg.predict_proba(train[predictors])[:,1]
    
    feat_imp = pd.Series(alg.get_booster().get_fscore()).sort_values(ascending=False) 
    feat_imp.plot(kind='bar', title='Feature Importance', color='g') 
    plt.ylabel('Feature Importance Score')
        
    #Print model report:
    print("\nModel Report")
    print("Accuracy : %.4g" % metrics.accuracy_score(target['FAILURE_TARGET'].values, dtrain_predictions))
    print("AUC Score (Balanced): %f" % metrics.roc_auc_score(target['FAILURE_TARGET'], dtrain_predprob))


# In[74]:


#Defining parameter values
estimator_vals=160
lr_vals = 0.8
md_vals = 12
mcw_vals = 0.5
gamma_vals =.1
subsample_vals = .5
c_bt_vals = 1
reg_lambda_vals = 1
reg_alpha_vals = 1


# In[75]:


#Defining the model
xgb0 = XGBClassifier(objective = 'binary:logistic',use_label_encoder=False,learning_rate = lr_vals,
n_estimators=estimator_vals,max_depth=md_vals,min_child_weight=mcw_vals,
gamma=gamma_vals,subsample=subsample_vals,colsample_bytree=c_bt_vals,
reg_lambda=reg_lambda_vals,reg_alpha=reg_alpha_vals);


# In[76]:


evaluate_model(xgb0, independent, dependent,features) 


# In[77]:


#Evaluating model with accuracy and AUC Score
df_testing=df_train_test[df_train_test['MODELING_GROUP'] == 'TESTING'].copy()
df_training=df_train_test[df_train_test['MODELING_GROUP'] != 'TESTING'].copy()


# In[78]:


#Evaluating the unbalanced data
df_training['P_FAIL']= xgb0.predict_proba(df_training[features])[:,1];
df_training['Y_FAIL'] = np.where(((df_training.P_FAIL <= .50)), 0, 1)
#Print model report:
print("Accuracy : %.4g" % metrics.accuracy_score(df_training['FAILURE_TARGET'].values, df_training['Y_FAIL']))
print("AUC Score (Train): %f" % metrics.roc_auc_score(df_training['FAILURE_TARGET'], df_training['P_FAIL']))


# In[79]:


df_testing['P_FAIL']= xgb0.predict_proba(df_testing[features])[:,1];
df_testing['Y_FAIL'] = np.where(((df_testing.P_FAIL <= .50)), 0, 1)
#Print model report:
print("Accuracy : %.4g" % metrics.accuracy_score(df_testing['FAILURE_TARGET'].values, df_testing['Y_FAIL']))
print("AUC Score (Train): %f" % metrics.roc_auc_score(df_testing['FAILURE_TARGET'], df_testing['P_FAIL']))


# In[80]:


#Evaluating the model with simple confusion matrix
print(pd.crosstab(df_testing.Y_FAIL, df_testing.EQUIPMENT_FAILURE, dropna=False))


# In[81]:


print(pd.crosstab(df_testing.Y_FAIL, df_testing.FAILURE_TARGET, dropna=False))


# In[82]:


forecast_window=90
cutoff=0.5
df=df_train_test
df['P_FAIL']= xgb0.predict_proba(df[features])[:,1];
df['Y_FAIL'] = np.where(((df.P_FAIL <= cutoff)), 0, 1)


# In[83]:


pd_id=df.drop_duplicates(subset='ID')
pd_id=pd_id[['ID']]
pd_id.shape


# In[84]:


pd_id=pd_id.reset_index(drop=True)
pd_id=pd_id.reset_index(drop=False)
pd_id=pd_id.rename(columns={"index": "IDs"})
pd_id['IDs']=pd_id['IDs']+1
pd_id.head()


# In[85]:


column = pd_id["IDs"]
max_value = column.max()+1
max_value


# In[86]:


df=df.sort_values(by=['ID'], ascending=[True])
pd_id=pd_id.sort_values(by=['ID'], ascending=[True])
df =df.merge(pd_id, on=['ID'], how='inner')
df.head()


# In[87]:


df=df.sort_values(by=['ID','DATE'], ascending=[True,True])

#reset index
df=df.reset_index(drop=True)


# In[88]:


#creating a null dataframe 
df_fred=df
df_fred['Y_FAIL_sumxx']=0
df_fred=df_fred[df_fred['IDs'] == max_value+1]
df_fred.shape

#sum the number of signals occuring over the last 90 days for each machine individually
for x in range(max_value):
        dffx=df[df['IDs'] ==x]
        dff=dffx.copy()
        dff['Y_FAIL_sumxx'] =(dff['Y_FAIL'].rolling(min_periods=1, window=(forecast_window)).sum())
        df_fred= pd.concat([df_fred,dff])
        
df=df_fred

# if a signal has occured in the last 90 days, the signal is 0.
df['Y_FAILZ']=np.where((df.Y_FAIL_sumxx>1), 0, df.Y_FAIL)


# In[89]:


df['SIGNAL_ID'] = df['Y_FAILZ'].cumsum()


# In[90]:


df_signals=df[df['Y_FAILZ'] == 1]
df_signal_date=df_signals[['SIGNAL_ID','DATE','ID']]
df_signal_date=df_signal_date.rename(index=str, columns={"DATE": "SIGNAL_DATE"})
df_signal_date=df_signal_date.rename(index=str, columns={"ID": "ID_OF_SIGNAL"})


# In[91]:


df_signal_date.shape


# In[92]:


df =df.merge(df_signal_date, on=['SIGNAL_ID'], how='outer') 
df.copy()


# In[93]:


df=df[['DATE', 'ID', 'EQUIPMENT_FAILURE', 'FAILURE_TARGET','FAILURE_DATE',
       'P_FAIL', 'Y_FAILZ','SIGNAL_ID',
       'SIGNAL_DATE','ID_OF_SIGNAL','MODELING_GROUP']]


# In[94]:


df['C'] = df['FAILURE_DATE'] - df['SIGNAL_DATE'].copy()
df['WARNING'] = df['C'] / np.timedelta64(1, 'D').copy()


# In[95]:


df['TRUE_POSITIVE'] = np.where(((df.EQUIPMENT_FAILURE == 1) & (df.WARNING<=forecast_window) &(df.WARNING>=0) & (df.ID_OF_SIGNAL==df.ID)), 1, 0)


# In[96]:


# define a false negative
df['FALSE_NEGATIVE'] = np.where((df.TRUE_POSITIVE==0) & (df.EQUIPMENT_FAILURE==1), 1, 0)


# In[97]:


# define a false positive
df['BAD_S']=np.where((df.WARNING<0) | (df.WARNING>=forecast_window | df.WARNING.isnull().values.any()), 1, 0)
df['FALSE_POSITIVE'] = np.where(((df.Y_FAILZ == 1) & (df.BAD_S==1) & (df.ID_OF_SIGNAL==df.ID)), 1, 0)
df['bootie']=1
df['CATEGORY']=np.where((df.FALSE_POSITIVE==1),'FALSE_POSITIVE',
                                      (np.where((df.FALSE_NEGATIVE==1),'FALSE_NEGATIVE',
                                                (np.where((df.TRUE_POSITIVE==1),'TRUE_POSITIVE','TRUE_NEGATIVE')))))


# In[98]:


df.head()


# In[99]:


pd.options.display.max_rows = 1000
dd=df[df['ID']==100001]
dd.head(1000)


# In[100]:


table = pd.pivot_table(df, values=['bootie'], index=['MODELING_GROUP'],columns=['CATEGORY'], aggfunc=np.sum)
table


# In[101]:


from IPython.display import Image
from IPython.core.display import HTML 
Image(url= "https://cdn-images-1.medium.com/max/1600/1*fUKUEUeqgIYU9xlxj4pwhw.png")


# In[102]:


df['TOTAL_COST']=df.FALSE_NEGATIVE*30000+df.FALSE_POSITIVE*1500+df.TRUE_POSITIVE*7500


# In[103]:


table = pd.pivot_table(df, values=['TOTAL_COST'],index=['MODELING_GROUP'], aggfunc=np.sum)
table


# In[104]:


wells=df[['ID','MODELING_GROUP']]
wells=wells.drop_duplicates(subset='ID')
wells = wells.groupby(['MODELING_GROUP'])['ID'].count()
wells=pd.DataFrame(wells)
wells=wells.rename(columns={"ID": "WELLS"})
wells


# In[105]:


tc = df.groupby(['MODELING_GROUP'])['TOTAL_COST'].sum()
tc=pd.DataFrame(tc)


# In[106]:


ac =tc.merge(wells, on=['MODELING_GROUP'], how='inner')


# In[107]:


ac['AVERAGE_COST']=ac.TOTAL_COST/ac.WELLS
ac['LIFT']=27948-ac.AVERAGE_COST
ac


# In[108]:


#Fine tuning
forecast_window=90
cutoff=0.02


# In[109]:


df=df_train_test


# In[110]:


df_testing['P_FAIL']= xgb0.predict_proba(df_testing[features])[:,1];
df_testing['Y_FAIL'] = np.where(((df_testing.P_FAIL <= .50)), 0, 1)
#create a unique list of machines
pd_id=df.drop_duplicates(subset='ID')
pd_id=pd_id[['ID']]


#label each machine with a sequential number
pd_id=pd_id.reset_index(drop=True)
pd_id=pd_id.reset_index(drop=False)
pd_id=pd_id.rename(columns={"index": "IDs"})
pd_id['IDs']=pd_id['IDs']+1
pd_id.head()

#grab the max number of machines +1

column = pd_id["IDs"]
max_value = column.max()+1
max_value

#append sequential number to main file.  Now each machine has a sequencial id.
df=df.sort_values(by=['ID'], ascending=[True])
pd_id=pd_id.sort_values(by=['ID'], ascending=[True])
df =df.merge(pd_id, on=['ID'], how='inner')
df.head()

#sort data
df=df.sort_values(by=['ID','DATE'], ascending=[True,True])

#reset index
df=df.reset_index(drop=True)

#create a null dataframe for the next step
df_fred=df
df_fred['Y_FAIL_sumxx']=0
df_fred=df_fred[df_fred['IDs'] == max_value+1]
df_fred.shape

#sum the number of signals occuring over the last 90 days for each machine individually

for x in range(max_value):
        dffx=df[df['IDs'] ==x]
        dff=dffx.copy()
        dff['Y_FAIL_sumxx'] =(dff['Y_FAIL'].rolling(min_periods=1, window=(forecast_window)).sum())
        df_fred= pd.concat([df_fred,dff])
        
        
df=df_fred

# if a signal has occured in the last 90 days, the signal is 0.
df['Y_FAILZ']=np.where((df.Y_FAIL_sumxx>1), 0, df.Y_FAIL)


#sort the data by id and date.

df=df.sort_values(by=['ID','DATE'], ascending=[True, True])
#create signal id with the cumsum function.
df['SIGNAL_ID'] = df['Y_FAILZ'].cumsum()


df_signals=df[df['Y_FAILZ'] == 1]
df_signal_date=df_signals[['SIGNAL_ID','DATE','ID']]
df_signal_date=df_signal_date.rename(index=str, columns={"DATE": "SIGNAL_DATE"})
df_signal_date=df_signal_date.rename(index=str, columns={"ID": "ID_OF_SIGNAL"})



df=df.merge(df_signal_date, on=['SIGNAL_ID'], how='outer') 


df=df[['DATE', 'ID', 'EQUIPMENT_FAILURE', 'FAILURE_TARGET','FAILURE_DATE',
       'P_FAIL', 'Y_FAILZ','SIGNAL_ID',
       'SIGNAL_DATE','ID_OF_SIGNAL','MODELING_GROUP']]



df['C'] = df['FAILURE_DATE'] - df['SIGNAL_DATE']
df['WARNING'] = df['C'] / np.timedelta64(1, 'D')


# In[111]:


# define a true positive
df['TRUE_POSITIVE'] = np.where(((df.EQUIPMENT_FAILURE == 1) & (df.WARNING<=forecast_window) &(df.WARNING>=0) & (df.ID_OF_SIGNAL==df.ID)), 1, 0)

# define a false negative
df['FALSE_NEGATIVE'] = np.where((df.TRUE_POSITIVE==0) & (df.EQUIPMENT_FAILURE==1), 1, 0)

# define a false positive
df['BAD_S']=np.where((df.WARNING<0) | (df.WARNING>=forecast_window), 1, 0)

df['FALSE_POSITIVE'] = np.where(((df.Y_FAILZ == 1) & (df.BAD_S==1) & (df.ID_OF_SIGNAL==df.ID)), 1, 0)

df['bootie']=1

df['CATEGORY']=np.where((df.FALSE_POSITIVE==1),'FALSE_POSITIVE',
                                      (np.where((df.FALSE_NEGATIVE==1),'FALSE_NEGATIVE',
                                                (np.where((df.TRUE_POSITIVE==1),'TRUE_POSITIVE','TRUE_NEGATIVE')))))


# In[112]:


table = pd.pivot_table(df, values=['bootie'], index=['MODELING_GROUP'],columns=['CATEGORY'], aggfunc=np.sum)
table


# In[113]:


#Now we can calculate the total cost.
df['TOTAL_COST']=df.FALSE_NEGATIVE*30000+df.FALSE_POSITIVE*1500+df.TRUE_POSITIVE*7500
#Aggregate the costs by modeling group.
table = pd.pivot_table(df, values=['TOTAL_COST'],index=['MODELING_GROUP'], aggfunc=np.sum)

#Calculate the number of machines in each modeling group.
wells=df[['ID','MODELING_GROUP']]
wells=wells.drop_duplicates(subset='ID')
wells = wells.groupby(['MODELING_GROUP'])['ID'].count()
wells=pd.DataFrame(wells)
wells=wells.rename(columns={"ID": "WELLS"})

tc = df.groupby(['MODELING_GROUP'])['TOTAL_COST'].sum()
tc=pd.DataFrame(tc)
#Merge the total costs and total machines into one data frame.
ac =tc.merge(wells, on=['MODELING_GROUP'], how='inner')
#Calculate the average cost per machine.

ac['AVERAGE_COST']=ac.TOTAL_COST/ac.WELLS
ac['LIFT']=27948-ac.AVERAGE_COST
ac


# In[119]:


df=df_total


# In[122]:


df_testing['P_FAIL']= xgb0.predict_proba(df_testing[features])[:,1];
df_testing['Y_FAIL'] = np.where(((df_testing.P_FAIL <= .50)), 0, 1)


# In[124]:



#grab the max number of machines +1

column = pd_id["IDs"]
max_value = column.max()+1
max_value

#append sequential number to main file.  Now each machine has a sequencial id.
df=df.sort_values(by=['ID'], ascending=[True])
pd_id=pd_id.sort_values(by=['ID'], ascending=[True])
df =df.merge(pd_id, on=['ID'], how='inner')
df.head()

#sort data
df=df.sort_values(by=['ID','DATE'], ascending=[True,True])

#reset index
df=df.reset_index(drop=True)

#create a null dataframe for the next step
df_fred=df
df_fred['Y_FAIL_sumxx']=0
df_fred=df_fred[df_fred['IDs'] == max_value+1]
df_fred.shape


# In[128]:


df


# In[130]:


#sum the number of signals occuring over the last 90 days for each machine individually
for x in range(max_value):
        dffx=df[df['IDs'] ==x]
        dff=dffx.copy()
        dff['Y_FAIL_sumxx'] =(dff['Y_FAIL_sumxx'].rolling(min_periods=1, window=(forecast_window)).sum())
        df_fred= pd.concat([df_fred,dff])
               
df=df_fred
# if a signal has occured in the last 90 days, the signal is 0.
df['Y_FAILZ']=np.where((df.Y_FAIL_sumxx>1), 0, df.Y_FAIL_sumxx)


# In[135]:


#create signal id with the cumsum function.
df['SIGNAL_ID'] = df['Y_FAILZ'].cumsum()
df_signals=df[df['Y_FAILZ'] == 1]
df_signal_date=df_signals[['SIGNAL_ID','DATE','ID']]
df_signal_date=df_signal_date.rename(index=str, columns={"DATE": "SIGNAL_DATE"})
df_signal_date=df_signal_date.rename(index=str, columns={"ID": "ID_OF_SIGNAL"})
df =df.merge(df_signal_date, on=['SIGNAL_ID'], how='outer') 
df=df[['DATE', 'ID', 'EQUIPMENT_FAILURE', 'FAILURE_TARGET','FAILURE_DATE', 'Y_FAILZ','SIGNAL_ID','MODELING_GROUP']]
df['C'] = df['FAILURE_DATE'] - df['DATE']
df['WARNING'] = df['C'] / np.timedelta64(1, 'D')


# In[139]:


# define a true positive
df['TRUE_POSITIVE'] = np.where(((df.EQUIPMENT_FAILURE == 1) & (df.WARNING<=forecast_window) &(df.WARNING>=0) & (df.ID)), 1, 0)
# define a false negative
df['FALSE_NEGATIVE'] = np.where((df.TRUE_POSITIVE==0) & (df.EQUIPMENT_FAILURE==1), 1, 0)
# define a false positive
df['BAD_S']=np.where((df.WARNING<0) | (df.WARNING>=forecast_window), 1, 0)
df['FALSE_POSITIVE'] = np.where(((df.Y_FAILZ == 1) & (df.BAD_S==1) & (df.ID)), 1, 0)
df['bootie']=1
df['CATEGORY']=np.where((df.FALSE_POSITIVE==1),'FALSE_POSITIVE',
                                      (np.where((df.FALSE_NEGATIVE==1),'FALSE_NEGATIVE',
                                                (np.where((df.TRUE_POSITIVE==1),'TRUE_POSITIVE','TRUE_NEGATIVE')))))


# In[140]:


#Now we can calculate the total cost.
df['TOTAL_COST']=df.FALSE_NEGATIVE*30000+df.FALSE_POSITIVE*1500+df.TRUE_POSITIVE*7500
#Aggregate the costs by modeling group.
table = pd.pivot_table(df, values=['TOTAL_COST'],index=['MODELING_GROUP'], aggfunc=np.sum)

#Calculate the number of machines in each modeling group.
wells=df[['ID','MODELING_GROUP']]
wells=wells.drop_duplicates(subset='ID')
wells = wells.groupby(['MODELING_GROUP'])['ID'].count()
wells=pd.DataFrame(wells)
wells=wells.rename(columns={"ID": "WELLS"})

tc = df.groupby(['MODELING_GROUP'])['TOTAL_COST'].sum()
tc=pd.DataFrame(tc)
#Merge the total costs and total machines into one data frame.
ac =tc.merge(wells, on=['MODELING_GROUP'], how='inner')
#Calculate the average cost per machine.

ac['AVERAGE_COST']=ac.TOTAL_COST/ac.WELLS
ac['LIFT']=27948-ac.AVERAGE_COST
ac


# In[ ]:




