#!/usr/bin/env python
# coding: utf-8

# In[1]:


import json
import pandas as pd
import numpy as np
from collections import Counter


# lucene with python

# In[2]:


business = []
for line in open('/Users/devanshimittal/Downloads/yelp_dataset/business.json', 'r'):
    business.append(json.loads(line))


# In[3]:


# user = []
# for line in open('/Users/devanshimittal/Downloads/yelp_dataset/user.json', 'r'):
#     user.append(json.loads(line))


# In[4]:


review = []
for line in open('/Users/devanshimittal/Downloads/yelp_dataset/review.json', 'r'):
    review.append(json.loads(line))


# In[5]:


tip = []
for line in open('/Users/devanshimittal/Downloads/yelp_dataset/tip.json', 'r'):
    tip.append(json.loads(line))


# In[6]:


review_df = pd.DataFrame.from_dict(review)
business_df = pd.DataFrame.from_dict(business)
user_df = pd.DataFrame.from_dict(user)


# In[7]:


tip_df = pd.DataFrame.from_dict(tip)


# In[33]:


pd.merge(business_df, tip_df, on = 'business_id')['categories']


# In[29]:


from collections import Counter
category = business_df['categories'].to_list()


# In[43]:


business_df.dropna(inplace=True)


# In[45]:


restaurant_data = business_df[business_df['categories'].str.contains("Restaurants")]


# In[51]:


rest_review = pd.merge(restaurant_data, review_df, on = 'business_id')
finaldata = pd.merge(rest_review , tip_df, on = ['business_id', 'user_id'])


# In[55]:


finaldata[['stars_y', 'text_x', 'text_y']].to_csv('finaldata.csv')


# In[ ]:





# In[ ]:





# In[ ]:





# In[33]:

#
#city_review_df = pd.merge(review_df, business_df[['business_id', 'city']], on='business_id')
#
#
## Picking a city that has the highest number of users that have reviewed more than 20 businesses.
#
## In[42]:
#
#
#dict_cities = {}
#for city in business_df['city'].unique():
#    df1 = city_review_df[city_review_df['city']==city]
#    df2 = df1.groupby('user_id')['business_id'].apply(lambda x:list(np.unique(x)))
#    df2['user_id'] = df2.index
#    users = []
#    for i in df2.index:
#        if len(df2[i]) >= 20:
#            #print(i, len(g[i]))
#            users.append(i)
#    dict_cities[city] = len(users)
#
#
## In[44]:
#
#
#Keymax = max(dict_cities, key=dict_cities.get)
#print(Keymax)
#
#
## So we take Las Vegas as our city and get the user ids who have rated more than 20 businesses for Las Vegas only.
#
## In[45]:
#
#
#df1 = city_review_df[city_review_df['city']=='Las Vegas']
#df2 = df1.groupby('user_id')['business_id'].apply(lambda x:list(np.unique(x)))
#df2['user_id'] = df2.index
#users = []
#for i in df2.index:
#    if len(df2[i]) >= 20:
#        users.append(i)
#
#
## Filtering review by the users list, the below dataframe is our main dataframe
#
## In[48]:
#
#
#filter_review = df1[df1['user_id'].isin(users)].reset_index()
#
#
## we could do the below steps if needed
#
## In[147]:
#
#
#pd.merge(filter_review, user_df, on='user_id')

