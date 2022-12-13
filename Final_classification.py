#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np 
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC


# In[2]:


#create classifiers 
lr = LogisticRegression(solver='lbfgs', max_iter=1000)


# In[3]:


svc = LinearSVC(C=1.0)
rfc = RandomForestClassifier(n_estimators=10000)


# In[ ]:





# In[ ]:





# In[4]:


df = pd.read_csv('Comp1801CourseworkData.csv')
df.head()


# In[5]:


df['Salary'] = np.where( df['Salary']> 35000, 1, 0)

df['Sex'] = df['Sex'].astype('category')
df['Sex'] = df['Sex'].cat.codes

df['Education'] = df['Education'].astype('category')
df['Education'] = df['Education'].cat.codes

df['WorkType'] = df['WorkType'].astype('category')
df['WorkType'] = df['WorkType'].cat.codes

df['Region'] = df['Region'].astype('category')
df['Region'] = df['Region'].cat.codes


# In[6]:


df


# In[7]:


from sklearn.model_selection import train_test_split
train, test = train_test_split(df, test_size=0.1)


# In[ ]:





# In[8]:


train


# In[ ]:





# In[9]:


test


# In[ ]:





# In[10]:


X = train.iloc[:,:8]
y = train["Salary"]


# In[11]:


y


# In[12]:


y.shape


# In[ ]:





# In[13]:


X.shape


# In[ ]:





# In[ ]:





# In[14]:


lr.fit(X, y)


# In[16]:


lr.score(X, y)


# In[17]:


lr.coef_


# In[ ]:





# In[ ]:





# In[ ]:




