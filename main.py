#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd


# In[2]:


###   raw url   ###

cases_2021_test_url  = 'https://raw.githubusercontent.com/shumanpng/CMPT459-D100-SPRING2022/main/dataset/cases_2021_test.csv'
cases_2021_train_url = 'https://raw.githubusercontent.com/shumanpng/CMPT459-D100-SPRING2022/main/dataset/cases_2021_train.csv'
location_2021_url    = 'https://raw.githubusercontent.com/shumanpng/CMPT459-D100-SPRING2022/main/dataset/location_2021.csv'

###   read dataset   ###

cases_2021_test = pd.read_csv(cases_2021_test_url)
cases_2021_train = pd.read_csv(cases_2021_train_url)
location_2021 = pd.read_csv(location_2021_url)


# ### 1.1 Cleaning messy outcome labels

# In[3]:


hospitalized = ['Discharged', 'Discharged from hospital', 'Hospitalized', 'critical condition', 'discharge', 'discharged']
nonhospitalized = ['Alive', 'Receiving Treatment', 'Stable', 'Under treatment', 'recovering at home 03.03.2020', 
                   'released from quarantine', 'stable', 'stable condition']
deceased = ['Dead', 'Death', 'Deceased', 'Died', 'death', 'died']
recovered = ['Recovered', 'recovered']

cases_2021_train.loc[cases_2021_train['outcome'].isin(hospitalized), 'outcome_group'] = 'hospitalized'
cases_2021_train.loc[cases_2021_train['outcome'].isin(nonhospitalized), 'outcome_group'] = 'nonhospitalized'
cases_2021_train.loc[cases_2021_train['outcome'].isin(deceased), 'outcome_group'] = 'deceased'
cases_2021_train.loc[cases_2021_train['outcome'].isin(recovered), 'outcome_group'] = 'recovered'
cases_2021_train = cases_2021_train.drop(['outcome'], axis=1)
cases_2021_train.head(5)


# ### 1.4 Data cleaning and imputing missing values

# Clean 'Age' in Test Data:

# In[4]:


cases_2021_test_dropNA = cases_2021_test.dropna(subset = ['age'])   # dataset without NA Age values

cases_2021_test_withoutDash = cases_2021_test_dropNA[cases_2021_test_dropNA["age"].str.contains("-")==False] # data that doesn't contains '-' in Age
cases_2021_test_withoutDash.age = cases_2021_test_withoutDash.age.astype(float).round()

cases_2021_test_withDash = cases_2021_test_dropNA[cases_2021_test_dropNA['age'].str.contains('-')] # data that contains '-' in Age
cases_2021_test_withDash['age'] = round(((cases_2021_test_withDash['age'].apply(lambda x: sum(map(float, x.split('-')))))/2)) # Average age values with '-' (eg. 20-30 => 25)

cases_2021_test_FormattedAge = pd.concat([cases_2021_test_withDash, cases_2021_test_withoutDash]) # combine dataframes

cases_2021_test_FormattedAge.head(5)


# Clean 'Age' in Train Data:

# In[5]:


cases_2021_train_dropNA = cases_2021_train.dropna(subset = ['age'])   # dataset without NA Age values

cases_2021_train_withoutDash = cases_2021_train_dropNA[cases_2021_train_dropNA["age"].str.contains("-")==False] # data that doesn't contains '-' in Age
cases_2021_train_withoutDash.age = cases_2021_train_withoutDash.age.astype(float).round()

cases_2021_train_withDash = cases_2021_train_dropNA[cases_2021_train_dropNA['age'].str.contains('-')] # data that contains '-' in Age
cases_2021_train_withDash['age'] = cases_2021_train_withDash['age'].str.replace('-','-0') 
cases_2021_train_withDash['age'] = round(((cases_2021_train_withDash['age'].apply(lambda x: sum(map(float, x.split('-')))))/2)) # Average age values with '-' (eg. 20-30 => 25)

cases_2021_train_FormattedAge = pd.concat([cases_2021_train_withDash, cases_2021_train_withoutDash]) # combine dataframes

cases_2021_train_FormattedAge.head(5)


# Impute other missing values:

# In[6]:


cases_2021_train_FormattedAge[['sex']] = cases_2021_train_FormattedAge[['sex']].fillna('Not Specified')
cases_2021_test_FormattedAge[['sex']] = cases_2021_test_FormattedAge[['sex']].fillna('Not Specified')

cases_2021_train_FormattedAge[['province']] = cases_2021_train_FormattedAge[['province']].fillna('Not Specified')
cases_2021_test_FormattedAge[['province']] = cases_2021_test_FormattedAge[['province']].fillna('Not Specified')


# ### 1.5 Dealing with outliers

# Cannot have a case fatality ratio of over 100%

# In[7]:


location_out = location_2021[location_2021['Case_Fatality_Ratio'] < 100]


# Still some strangely high Fatality Ratios exist. Only 28 records have > 10% fatality ratio, so we will remove them as this is extremely high for Covid.

# In[8]:


location_out = location_out[location_out['Case_Fatality_Ratio'] < 10]


# Also remove locations that list fatality rates of 0%, as this seems like an error for those with high cases, and misrepresentative for those with low cases.

# In[9]:


location_out = location_out[location_out['Case_Fatality_Ratio'] > 0]


# In[10]:


location_out.min()


# In[11]:


location_out.max()


# In[12]:


location_out.std()


# In[13]:


location_out.mean()


# In[14]:


location_out


# In[15]:


location_out['Last_Update'].unique()


# ### 1.6 Joining the cases and location dataset

# In[16]:


#   US & Korea fix   #
cases_2021_train_combine = cases_2021_train_FormattedAge.drop(columns=['latitude', 'longitude'], axis=1)
cases_2021_test_combine = cases_2021_test_FormattedAge.drop(columns=['latitude', 'longitude'], axis=1)
location_2021_combine = location_2021

cases_2021_train_combine = cases_2021_train_combine.replace(['United States'],'US')
cases_2021_test_combine = cases_2021_test_combine.replace(['United States'],'US')
cases_2021_train_combine = cases_2021_train_combine.replace(['South Korea'],'Korea, South')
cases_2021_test_combine = cases_2021_test_combine.replace(['South Korea'],'Korea, South')

location_2021_combine.fillna('Not Specified', inplace=True)
cases_2021_train_combine['province'].fillna('Not Specified', inplace=True)
cases_2021_test_combine['province'].fillna('Not Specified', inplace=True)

###  create combine key  ###
cases_2021_train_combine['Province_Country'] = cases_2021_train_combine['province'] + ', ' + cases_2021_train_combine['country']
cases_2021_test_combine['Province_Country'] = cases_2021_test_combine['province'] + ', ' + cases_2021_test_combine['country']
location_2021_combine['Province_Country'] = location_2021['Province_State'] + ', ' + location_2021['Country_Region']

###  merge with combine key  ###
cases_train_location_merged = pd.merge(location_2021_combine, cases_2021_train_combine, on='Province_Country')
cases_test_location_merged = pd.merge(location_2021_combine, cases_2021_test_combine, on='Province_Country')

###  output cvs  ###
cases_test_location_merged.to_csv(r'../results/cases_2021_test_processed.csv', encoding='utf-8', index=False)
cases_train_location_merged.to_csv(r'../results/cases_2021_train_processed.csv', encoding='utf-8', index=False)
location_2021_combine.to_csv(r'../results/location_2021_processed.csv', encoding='utf-8', index=False)


# 1.7 Data
cases_test_location_merged[["age", "chronic_disease_binary", "country", "Case_Fatality_Ratio", "date_confirmation"]].to_csv(r'../results/cases_2021_test_processed_features.csv', encoding='utf-8', index=False)
cases_train_location_merged[["age", "chronic_disease_binary", "country", "Case_Fatality_Ratio", "date_confirmation"]].to_csv(r'../results/cases_2021_train_processed_features.csv', encoding='utf-8', index=False)