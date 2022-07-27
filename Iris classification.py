#!/usr/bin/env python
# coding: utf-8

# In[1]:


#importing libraries
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn as sk
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


#reading the dataset
dataset=pd.read_csv(r'C:\Users\Aditi\Desktop\Projects\Iris\Iris.csv')


# In[3]:


dataset.shape


# In[4]:


dataset.info()


# In[5]:


#displaying the column names
dataset.columns[0:]


# In[6]:


#detailed info of the dataset
dataset.describe()


# In[7]:


#first few rows of the dataset
dataset.iloc[:10]


# In[8]:


#checking if any null values present
dataset.isnull().sum()


# In[9]:


data=dataset.iloc[:,1:]
data.groupby('Species').mean()


# In[10]:


data.groupby('Species').median()


# In[11]:


data.groupby('Species').agg(['mean','median'])


# In[12]:


#another way of implementing the above said thing -
data.groupby('Species').agg([np.mean,np.median])


# If we want to find the mean median for lengths and minimum and maximum for widths then -

# In[13]:


agg_dict={fields: ['mean','median'] for fields in data.columns if fields !='Species'}
agg_dict


# In[14]:


agg_dict['SepalWidthCm']=['max','min']
agg_dict['PetalWidthCm']=['max','min']
agg_dict


# In[15]:


data.groupby('Species').agg(agg_dict)


# In[16]:


#Visualizations
plt.plot(dataset.SepalLengthCm,dataset.SepalWidthCm,ls='',marker='x')
plt.title("Relation between Sepal Length and Sepal Width")
plt.xlabel("Sepal Length (cm)")
plt.ylabel("Sepal Width (cm)")
plt.show()


# In[17]:


#Making some styling changes in the visualization
plt.plot(dataset.SepalLengthCm,dataset.SepalWidthCm,ls='',marker='x')
plt.title("Relation between Sepal Length and Sepal Width",fontweight='bold',fontsize=16,fontname='Lucida Handwriting',color='White')
plt.xlabel("Sepal Length (cm)",fontweight='bold',fontsize=16,fontname='Lucida Handwriting',color='White')
plt.ylabel("Sepal Width (cm)",fontweight='bold',fontsize=16,fontname='Lucida Handwriting',color='White')
plt.show()


# In[18]:


#Doing for petals
plt.plot(dataset.PetalLengthCm,dataset.PetalWidthCm,ls='',marker='x')
plt.title("Relation between Petal Length and Petal Width",fontweight='bold',fontsize=16,fontname='Lucida Handwriting',color='White')
plt.xlabel("Petal Length (cm)",fontweight='bold',fontsize=16,fontname='Lucida Handwriting',color='White')
plt.ylabel("Petal Width (cm)",fontweight='bold',fontsize=16,fontname='Lucida Handwriting',color='White')
plt.show()


# In[19]:


plt.plot(dataset.PetalLengthCm,dataset.PetalWidthCm,ls='',marker='o',label='petal',color='blue')
plt.plot(dataset.SepalLengthCm,dataset.SepalWidthCm,ls='',marker='o',label='sepal',color='green')
plt.legend()


# In[20]:


dataset.groupby('Species').mean().plot(color=['red','blue','green','yellow'],figsize=(5,5),label=['Sepal Length','Sepal Width','Petal Length','Petal Width'],fontsize=10)
plt.legend()


# In[21]:


#displaying without id column
data.groupby('Species').mean().plot(color=['red','blue','green','yellow'],figsize=(5,5),label=['Sepal Length','Sepal Width','Petal Length','Petal Width'],fontsize=10)
plt.legend()


# In[22]:


sns.pairplot(data,hue='Species',size=5)
plt.legend()


# In[23]:


sns.jointplot(x=dataset.SepalLengthCm,y=dataset.SepalWidthCm,kind='hex',color='yellow')
#Drker colors shows where most values overlap


# In[24]:


plt.bar(dataset.Species,dataset.SepalLengthCm,color = "blue")
plt.xlabel("Species")
plt.ylabel("Sepal Length")
plt.show()


# In[25]:


plot=sns.FacetGrid(data,col='Species',margin_titles=True)
plot.map(plt.hist,'SepalWidthCm',color='green')


# In[26]:


plot=sns.FacetGrid(data,col='Species',margin_titles=False)
plot.map(plt.hist,'SepalLengthCm',color='orange')


# From the last 2 visualizations, it is visible that whether we assign the value to Margin-titles as True or False, itproduces the same result. Hence, we can concur that this option is experimental and may not work in all cases.

# In[27]:


sns.set_context('notebook')
data.plot.hist(bins=25,alpha=0.5)
plt.xlabel("Size (cm)")
plt.legend()


# In[28]:


data.boxplot(by='Species')


# In[29]:


#As the labels are overlapping, lets ncrease the size


# In[30]:


data.boxplot(by='Species',figsize=(10,10))


# In[34]:


data.boxplot(by=('SepalLengthCm'),figsize=(25,25))


# In[35]:


data.boxplot(by=('SepalWidthCm'),figsize=(25,25))


# In[36]:


data.boxplot(by=('PetalLengthCm'),figsize=(25,25))


# In[37]:


data.boxplot(by=('PetalWidthCm'),figsize=(25,25))


# In[38]:


#Correct Way
data.set_index('Species').stack().to_frame()


# In[39]:


data.set_index('Species').stack().to_frame().reset_index()


# In[40]:


data1=data.set_index('Species').stack().to_frame().reset_index().rename(columns={0:'Size','level_1':'Measurement'})
data1


# In[41]:


sns.boxplot(x='Measurement',y='Size',hue='Species',data=data1)
sns.set(rc={'figure.figsize':(10,10)})
plt.legend(bbox_to_anchor=(1,1))


# # Models -

# In[42]:


from sklearn.model_selection import train_test_split
train, test = train_test_split(data, test_size = 0.4, stratify = data['Species'], random_state = 42)


# In[43]:


x_train=train[['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']]
y_train=train[['Species']]
x_test=test[['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']]
y_test=test[['Species']]


# In[44]:


from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression


# # Decision Tree Classifier

# In[45]:


dt=DecisionTreeClassifier(max_depth = 3, random_state = 1)
dt.fit(x_train,y_train)


# In[46]:


dt.fit(x_test,y_test)


# In[47]:


dt.score(x_test,y_test)


# In[48]:


prediction=dt.predict(x_test)
print('The accuracy of the Decision Tree is','{:.3f}'.format(metrics.accuracy_score(prediction,y_test)))


# In[49]:


#To see the importance of each feature 
dt.feature_importances_


# In[ ]:


#Now it is clear that sepal measurements play no role in determining the type of iris the sample is.


# # Gaussian NB Classifier

# In[50]:


gnb=GaussianNB()
gnb.fit(x_train,y_train)


# In[51]:


gnb.fit(x_test,y_test)


# In[52]:


prediction=gnb.predict(x_test)


# In[56]:


gnb.score(x_test,y_test)


# # KNN - Classifier

# In[64]:


knn=KNeighborsClassifier(n_neighbors=3)
knn.fit(x_train,y_train)


# In[65]:


knn.fit(x_test,y_test)


# In[66]:


prediction=knn.predict(x_test)
print(prediction)


# In[67]:


knn.score(x_test,y_test)


# # SVC

# In[69]:


svc=SVC(kernel='linear') 
svc.fit(x_train, y_train)


# In[71]:


svc.fit(x_test,y_test)


# In[72]:


prediction=svc.predict(x_test)
prediction


# In[73]:


svc.score(x_test,y_test)


# # Logistic Regression

# In[74]:


lr=LogisticRegression()
lr.fit(x_train,y_train)


# In[75]:


lr.fit(x_test,y_test)


# In[77]:


prediction=lr.predict(x_test)
prediction


# In[78]:


lr.score(x_test,y_test)


# In[80]:


#Printing the accuracy for each model - 
print("Decision Tree Classifier - ",dt.score(x_test,y_test))
print("Gusian Naive Bayesian - ",gnb.score(x_test,y_test))
print("KNN - ",knn.score(x_test,y_test))
print("SVC - ",svc.score(x_test,y_test))
print("Logistic Regression - ",lr.score(x_test,y_test))


# Now it is clear that we are getting the maximum accuracy from the Decision Tree Classifier.

# In[86]:


from sklearn import tree
tree.plot_tree(dt,filled=True)

