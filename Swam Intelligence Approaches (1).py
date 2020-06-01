#!/usr/bin/env python
# coding: utf-8

# # Swarm Intelligence Approaches 

# Parameter Setting of Deep Learning Neural Network: Acase Study of Website Classification.
# 

# # 1. Introduction
# The problem of detecting phishing websites hasbeen addressed many times using various methodologies from conventional classifiers to more complex hybrid methods. Recentadvancements in deep learning approaches suggested that the classification of phishing websites using deep learning neural networksshould outperform the traditional machine learning algorithms.However, the results of utilizing deep neural networks heavily depend on the setting of different learning parameters. In this paper,we propose a swarm intelligence based approach to parameter setting of deep learning neural network.
# Therefore we import the libraries and tabular Dataset that we intend to use in developing physhing website classifier.

# In[8]:


#importing data for basic analysis-preprocessing.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from numpy.random import random
get_ipython().run_line_magic('matplotlib', 'inline')
#importing the libraries for data preprocessing
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score,KFold,StratifiedKFold, GridSearchCV
from sklearn.metrics import confusion_matrix,accuracy_score,f1_score,average_precision_score,recall_score,roc_auc_score
from sklearn.preprocessing import RobustScaler,StandardScaler,LabelEncoder,MinMaxScaler

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.feature_selection import SelectKBest,chi2

from keras.models import Sequential
from keras.layers import Activation,BatchNormalization
from keras.layers.core import Dense,Dropout
from keras.optimizers import Adam
from keras.metrics import categorical_crossentropy
from keras.callbacks import ReduceLROnPlateau,EarlyStopping
Data=pd.read_csv("//Users//nelsonotumaongaya//Desktop//Phishing.csv",header="infer")


# In[9]:


Data


# # Understanding the Data Sets We explore the data sets further. 

# We need to understand the data set indetail first.
# We develop a brief understanding of the dataset with which we will be working with. For example how many features are there in the dataset, how man unique label. How are they distributed or how are the labels distributed, different data types and quantities. 

# In[3]:


Data.head()# we explore the headers on the datasets to understand the features.


# In[4]:


Data.tail()# we explore the tail on the datasets to understand it. Shows the bottom Data in the set


# In[5]:


Data.tail().T# We explore the data sets to check on the headers conformity and understand it.


# In[6]:


Data.sample(10)


# In[7]:


Data.T


# In[8]:


Data.isnull().sum()#Check if there are any missing values


# # 2. We now Querry the data sets to obtain the reports.

# In[9]:


len(Data)#Shows how much data the Dataset contains:


# In[10]:


Data.head() #Shows the top Data in the set that you are exploring.


# In[11]:


Data.info() #This displays all columns and their data types,


# In[12]:


Data.describe()# This shows you some basic descriptive statistics for all numeric columns in the data set.which includes the count,mean,standard deviation,min and max


# Cleaning the dataset accordingly so that it is well suited for a Machine Learning Model.
# 

# Let us examine the
# feature "Sub-domain and multi sub-domain". A technique used by
# phishers to scam users is by adding a sub-domain to the URL so
# users may believe they are dealing with an authentic website.

# In[13]:


x=Data.iloc [:,1:-1]
x=x.values
y=Data.iloc[:,-1].values


# In[14]:


x


# In[15]:


y


# # 2. Building The Model Using Decision Tree

# In[16]:


from sklearn import preprocessing
x1= preprocessing.normalize(x)
#We normalize the data: refers to rescaling real valued numeric attributes into the range 0 and 1.
#It is useful to scale the input attributes for a model that relies on the magnitude of values


# In[17]:


x1


# We spilt the data and create training and test data sets 80% Training set and 20% is the Testing Set.

# In[18]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x1,y,test_size=0.20,random_state=42)


# In[19]:


print(x_train.shape, y_train.shape,x_test.shape, y_test.shape)


# In[20]:


#We now fit the model to determine the accuracy using Decision Trees
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn import tree
clf_gini=DecisionTreeClassifier(criterion="gini", random_state=100,max_depth=3, min_samples_leaf=5)
clf_gini.fit(x_train, y_train)
Y_pred=clf_gini.predict(x_test)
from sklearn import metrics
metrics.accuracy_score(y_test,Y_pred)*100 


# In[21]:


import pydotplus
from sklearn.tree import export_graphviz
from IPython.display import Image
dot_data=tree.export_graphviz(clf_gini)
graph=pydotplus.graph_from_dot_data(dot_data)
Image(graph.create_png())


# # Plotting the Data Graphically
# 

# if the value in the column of result is less than 1 then it is legit, if its more than >>1 then its phishy
# "Sub-domain and multi sub-domain". A technique used by
# phishers to scam users is by adding a sub-domain to the URL so
# users may believe they are dealing with an authentic website. therefore the dots in the domain part should be less than 1,if it is more than 1 or equals to zero then its not legitimate, hence suspicious else phishy

# In[22]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x1,y,test_size=0.20,random_state=42)
from sklearn.metrics import accuracy_score


# In[23]:


maxdepths=[3,4,5,6,7,8,9,10,11,15,20,25,30,35,40,45,50,60]# alist contains
trainAcc=np.zeros(len(maxdepths))
testAcc=np.zeros(len(maxdepths))


# In[24]:


trainAcc


# In[25]:


testAcc


# In[26]:


index=0
for depth in maxdepths:
    clf=tree.DecisionTreeClassifier(max_depth=depth)
    clf=clf.fit(x_train,y_train)
    y_predTrain=clf.predict(x_train)
    y_predTest=clf.predict(x_test)
    trainAcc[index]=accuracy_score(y_train,y_predTrain)
    testAcc[index]=accuracy_score(y_test,y_predTest)
    index +=1


# In[27]:


plt.plot(maxdepths,trainAcc,'ro-',maxdepths,testAcc,'bv--')
plt.legend(['Training Accuracy','Test Accuracy'])
plt.xlabel('Max Depth')
plt.ylabel('Accuracy')


# # Conclusion 

# The plot shows that the accuracy will continue to improve as the maximum depth of the tree increases,i.e the model becomes more complex.

# # (II)Building the Model using Multi Linear Regression (Swarm Intelligence Approach)

# Linear Regression is one of the most commonly used predictive modelling techniques. The aim of the modelling technique is to find a Mathematical equation for continuous response variable. The equation can be generalised as:y=B1+B2x+x. The coefficients are called regression coefficient(B1 and B2).
# We import the libraries for our data sets, the libraries will assist us in exploring the data sets.

# In[28]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#importing the libraries for our data sets.


# In[29]:


#we now load the dataset from the folder called Phishing 
Data=pd.read_csv("//Users//nelsonotumaongaya//Desktop//Phishing.csv",header="infer")


# In[30]:


Data# we now visualize our dataset 


# In[31]:


#We now check for our dataset information.
Data.info


# In[32]:


Data


# In[33]:


#We now check for number of columns in our dataset
Data.columns.values


# # Lets Get the Independent and Dependent Variables

# In[34]:


#Get dependent and independent variables.
# enables us select all rows / columns
# -1 is the index of last column in python
x = Data.iloc[:,:-1].values #independent variables
y = Data.iloc[:,-1].values #dependent variable


# In[35]:


#We now Display all values in x - independent variables
print(x)


# In[36]:


#Display all values in y - dependent variables
print(y)


# Lets Check for missing values in my dataset.
# True shows that the respective column have missing
# values

# In[37]:


Data.isnull()


# # Lets Check for the sum of missing values in our dataset.
# 

# In[38]:


Data.isnull().sum()


# From the results, the dataset is clean, there are no missing values

# # Splitting the Dataset into Train Set and Test Set

# In[39]:


#We now split the dataset into train and 
# 80% observation in Train and 20% in Test - since we have 50 observation
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.2, random_state = 0)


# In[40]:


print (x_train)


# In[41]:


print (y_train)


# In[42]:


#Display.max function in pandas enables us to display all values in our dataset.
pd.options.display.max_columns=None
print (x_test)


# In[43]:


print (y_test)


# # Fitting Multi Linear Regression to the Training Set.

# In[44]:


#We will import Linear Regression function from sklearn package
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train, y_train)


# In[45]:


print(x_train)


# In[46]:


print(y_test)


# # Predicting the Test Set Results

# In[47]:


#Predicting Test Results
y_pred = regressor.predict(x_test)


# In[48]:


print(y_pred)


# In[49]:


#Compare Predicted and Actuals
Data = pd.DataFrame({'Actual':y_test.flatten(),'Predicted':y_pred.flatten()})


# In[50]:


Data


# # We Now Visualize The Results

# In[51]:


#Visualize Actuals Vs Predicted
df1 = Data.head(40)
df1.plot(kind='bar',figsize=(16,10))
plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
plt.show()


# In[52]:


#Import Metrics from sklearn
from sklearn import metrics


# In[53]:


#Evaluate the Performance of the algorithm
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))


# # Conclusion 

# Therefore the accuracy of the model is 94%- This shows that the model is fit for data analytics.since the predicted results against the actual indicate close relationship of the data.
# 

# # 3. K-Nearest Neighbour Classifier (Swam Intelligence Approach)

# # Introduction

# In this approach the class label of a test instance is predicted based on the majority class of its k
# closest training instances. The number of nearest neighbors, k , is a hyperparameter that must be
# provided by the user, along with the distance metric. By default we can use Euclidean distance
# (equivalent to Minkoswki distance with an exponent factor equals to p=2)

# In[54]:


#we import the libraries for our data preprocessing
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from numpy.random import random
get_ipython().run_line_magic('matplotlib', 'inline')


# In[55]:


#we now load the dataset from the folder called Phishing 
Data=pd.read_csv("//Users//nelsonotumaongaya//Desktop//Phishing.csv",header="infer")


# In[56]:


Data


# # 2 Building The Model Using K-Nearest Neighbor Classifier

# We have already explored the same data in Decision tree and Regression Analysis
# therefore the data is clean-

# In[57]:


#We first split the dataset into training set comprising 80% and test set,comprising 20%
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.2,random_state = 1)


# In[58]:


x_train


# In[59]:


y_train


# In[60]:


#Display.max function in pandas enables us to display all values in our dataset.
pd.options.display.max_columns=None
x_test


# In[61]:


y_test


# In[62]:


#We import KNeighborsClassifier from sklearn
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
numNeighbours=[1,5,10,15,20,25,30]
trainAcc=[]
testAcc=[]


# In[63]:


#We fit the KNearest Neighbor model to the training set
for k in numNeighbours:
    clf=KNeighborsClassifier(n_neighbors=k, metric='minkowski',p=2)
    clf.fit(x_train,y_train)
    y_predTrain=clf.predict(x_train)
    y_predTest=clf.predict(x_test)
    trainAcc.append(accuracy_score(y_train,y_predTrain))
    testAcc.append(accuracy_score(y_test,y_predTest))


# In[64]:


plt.plot(numNeighbours,trainAcc,'ro-',numNeighbours,testAcc,'bv--')
plt.legend(['Training Accuracy','Test Accuracy'])
plt.xlabel('Number of Neighbours')
plt.ylabel('Accuracy')


# # 3 Conclusion

# Based on the training accuracy graph, it decreases with increase in depth. The same happens
# with test accuracy graph. It decreases sharply then increases slightly then it drops. From the test
# accuracy graph, the model performs best when it has a depth of 1 hence that is when it is best to
# deploy it (with this depth its accuracy level in detecting phishing sites is 96%).

# # 4. Polynomial Regression

# Explore the Dataset Again to Understand it.
# 

# In[65]:


#library import section
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import numpy as np # linear algebra
import matplotlib.pyplot as plt # plotting
import os # accessing directory structure
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression


# In[66]:


# Load the training data from the CSV file
Data=pd.read_csv("//Users//nelsonotumaongaya//Desktop//Phishing.csv",header="infer")
Data.dataframeName = 'Phishing.csv'
#determine data size
nRow, nCol = Data.shape
print(f'There are {nRow} rows and {nCol} columns')
print("\nLet's take a quick glance at what the data looks like:")
Data.head(5)


# Each row has records/inputs(the features/indepedent variables) collected for a particular webite;
# and the end result(output/dependent variable): if the website was used for phising or not

# In[67]:


print("\nKey statistical values:")
print(Data.describe())


# Distribution graphs (histogram/bar graph) of sampled columns:

# # 1.Loading and Understanging the Phishing Websites Dataset"

# "We need to understand the data set indetail first.\n",
#     "We develop a brief understanding of the dataset with which we will be working with. For example how many features are there in the dataset, how man unique label. How are they distributed or how are the labels distributed, different data types and quantities. "

# In[68]:


#importing data for basic analysis-preprocessing.\n",
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from numpy.random import random
get_ipython().run_line_magic('matplotlib', 'inline')


# # Getting useful information from the dataset"

# "Now, lets quickly find out how many classes in the dataset and how they the distributed in the dataset."

# In[69]:


Data=pd.read_csv("//Users//nelsonotumaongaya//Desktop//Phishing.csv",header="infer")
from collections import Counter


# In[70]:


classes =Counter(Data['Result'].values)


# In[71]:


classes.most_common()


# This information can be presented using a DataFrame which will produce a very good table

# In[72]:


class_dist=pd.DataFrame(classes.most_common(),columns=['Class','Num_Observation'])


# In[73]:


class_dist


# We could also use a plot to convey the information as well

# In[74]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[75]:


plt.style.use('ggplot')


# In[76]:


subplot=class_dist.groupby("Class")["Num_Observation"].sum().plot(kind="barh",width=0.2,figsize=(10,8))
subplot.set_xlabel("Number of Observations",fontsize=15)
subplot.set_ylabel("Class",fontsize=14)
for i in subplot.patches:
    subplot.text(i.get_width()+0.1,i.get_y()+0.1, str(i.get_width()),fontsize=11)


# In[ ]:





# "What is the range of the values present in the different columns, what are the unique values present in them and so on pandas we can use describe."

# In[77]:


Data.describe().T# We describe the data. further description of the dataset


# In[78]:


Data.info()# Gives more information about the data as null


# # 3. Cleaning the Class Labels and Inspecting for Missing Vaues

# The aim id clean the data and split it into two parts. Training and Testing

# # Introduction

# It is not good practice to create Machine Learning models using the labels with negative values. it affects the performance of the model hence we need to change the value -1 to be 0"

# In[79]:


Data.rename(columns={"Result":"Class"},inplace=True)
Data["Class"]=Data["Class"].map({-1:0,1:1})
Data["Class"].unique()


# # We now split the dataset into 80:20 ratio

# In[80]:


from sklearn.model_selection import train_test_split


# In[11]:


x=Data.iloc[:,0:30].values.astype(int)
y=Data.iloc[:,30].values.astype(int)
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=np.random.seed(7))


# In[12]:


x


# In[13]:


y


# In[14]:


len(y_train)


# In[15]:


8844+2211


# In[16]:


2211/11055*100# this how we have picked our training set


# Let's Serialize the splits as well. Remember that our splits are now nothing but an array of valus and can be serialized"

# In[17]:


# our data now does not have missing values
x_train


# In[18]:


x_train


# In[19]:


y_train


# # 4. Training Logistics Regression Model"¶
# 

# We instatiate Logistic Regression Model and fit it to the training data

# In[20]:


from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
from sklearn.linear_model import LogisticRegression
import wandb
import time


# # We define some utility function for training machine learning model, with code to measure its training time performance

# In[21]:


def train_eval_pipeline(model, train_data,test_data,name):
    #initialize weights and biases
    wandb.init(project="Phishing-websites-detection",name=name)
    #segragate the datasets
    (x_train, y_train)=train_data
    (x_test,y_test)=test_data
    #Train the model and keep the log of all the necessary metrics
    start=time.time()
    model.fit(x_train,y_train)
    end=time.time()-start
    prediction=model.predict(x_test)
    wandb.log({"accuracy": accuracy_score(y_test,prediction)*100.0,"precision": precision_recall_fscore_support(y_test,prediction, average="macro")[0],"recall":precision_recall_fscore_support(y_test,prediction,average='macro')[1],"training_time":end})
    print("Accuracy score of the Logistic Regression classifier with default hyperparameter values {0:.2f}%".format(accuracy_score(y_test,prediction)*100.))
    print("\n")
    print("---Classificatin Report of the Logistic Regression classifier with default hyperparameter values ----")
    print("\n")
    print(classification_report(y_test,prediction,target_names=["Phishing Websites","Normal Websites"]))
logreg=LogisticRegression()
train_eval_pipeline(logreg,(x_train,y_train),(x_test,y_test),"logistic_regression")


# # Improving the Model

# Can we improve this model? Agood way to start approaching this idea is tune the hyperparameters of the model.
# We want to look at which is the best parameter for our model. We define the grid of values for the hyperparameter we would like to tune. In this case we use random search for hyperparameters tuning.

# In[22]:


#import GridSearchCV if something goes outside the region we penalize it
from sklearn.model_selection import RandomizedSearchCV


# In[23]:


#We define the grid
penalty=["l1","l2"]
C=[0.8,0.9,1.0]
tol=[0.01,0.001,0.0001]#what we can tolerate-tolerant values
max_iter=[100,150,200,250]# maximum iteration


# In[26]:


#we create key value dist
param_grid=dict(penalty=penalty,C=C,tol=tol,max_iter=max_iter)


# Now with the grid, we work to find the best set of values of hyperparameters values.

# In[28]:


#Instanstiate RandomizedSearchCV with the required parameters.
param_grid=dict(penalty=penalty,C=C,tol=tol,max_iter=max_iter)
random_model=RandomizedSearchCV(estimator=logreg,param_distributions=param_grid, cv=5)


# In[29]:


#Instanstiate RandomizedSearchCV with the required parameters.
random_model=RandomizedSearchCV(estimator=logreg,param_distributions=param_grid, cv=5)
random_model_result=random_model.fit(x_train,y_train)


# In[30]:


#summary of the results
best_score, best_params=random_model_result.best_score_,random_model_result.best_params_


# In[31]:


#summary of the results
best_score, best_params=random_model_result.best_score_,random_model_result.best_params_
print("Best Score: %.2f using %s" %(best_score*100, best_params))


# Random search did not help much in boosting up the accuracy score.
# Just to ensure that lets take the hyperparameter values and train another logistic regression model with the same values.

# In[32]:


#log the hyperparameter values with which we are going to train our model.
config=wandb.config
config.tol=0.01
config.penalty="12"
config.C=1.0


# In[33]:


#Train the model
logreg=LogisticRegression(tol=config.tol,penalty=config.penalty,max_iter=250,C=config.C)
train_eval_pipeline(logreg,(x_train,y_train),(x_test,y_test),'Logistic-regression -random-search')


# # 8. Random Forest Classifier-RFC

# In[34]:


#print("Random Forest Classifier")
#forest_params = {"max_depth": list(range(10,50,1)),"n_estimators" : [350,400,450]}
#forest = GridSearchCV(RandomForestClassifier(), forest_params,n_jobs=-1,cv=10,scoring='roc_auc_ovo')
#forest.fit(x_train, y_train)
#random_forest = forest.best_estimator_
#print("Best Estimator")
#print(random_forest)


# In[35]:


#Parameters have been choosing based on GridSearchCV
random_forest = RandomForestClassifier(max_depth=10,n_estimators=350)
random_forest.fit(x_train,y_train)


# In[37]:


forest_score = cross_val_score(random_forest, x_train, y_train, cv=10,scoring='roc_auc_ovo')
forest_score_teste = cross_val_score(random_forest, x_test, y_test, cv=10,scoring='roc_auc_ovo')
print('Score RFC Train: ', round(forest_score.mean() * 100, 2).astype(str) + '%')
print('Score RFC Test: ', round(forest_score_teste.mean() * 100, 2).astype(str) + '%')


# In[38]:


y_pred_rf = random_forest.predict(x_test)


# In[39]:


cm_rf = confusion_matrix(y_test,y_pred_rf)


# In[51]:


acc_score_rf = accuracy_score(y_test,y_pred_rf)
f1_score_rf = f1_score(y_test,y_pred_rf)
pred_rf = average_precision_score(y_test,y_pred_rf)
recall_rf = recall_score(y_test,y_pred_rf)
roc_rf = roc_auc_score(y_test,y_pred_rf,multi_class='ovo')
print('Accuracy Random Forest ',round(acc_score_rf*100,2).astype(str)+'%')
print('Pred media Random Forest ',round(pred_rf*100,2).astype(str)+'%')
print('F1 Random Forest ',round(f1_score_rf*100,2).astype(str)+'%')
print('Recall Random Forest ',round(recall_rf*100,2).astype(str)+'%')
print('ROC Random Forest ',round(roc_rf*100,2).astype(str)+'%')


# The accuracy in Random Forest for Website Phishing is 96.38%

# In[55]:


fig, ax = plt.subplots(figsize=(10,6))
sns.heatmap(cm_rf, ax=ax, annot=True, cmap=plt.cm.copper)
ax.set_title("Random Forest \n Confusion Matrix", fontsize=14)
ax.set_xticklabels(['B', 'M'], fontsize=14, rotation=0)
ax.set_yticklabels(['B', 'M'], fontsize=14, rotation=360)


# # Ada Boost Classifier

# In[56]:


#print("Ada Boost Classifier")
#ada_params = {'n_estimators' : list(range(100,200))}
#grid_ada = GridSearchCV(AdaBoostClassifier(), ada_params,n_jobs=8,cv=10,scoring='roc_auc_ovo')
#grid_ada.fit(X_train, y_train)
#ada = grid_ada.best_estimator_
#print("Best Estimator")
#print(ada)


# In[57]:


#Parameters have been choosing based on GridSearchCV
ada = AdaBoostClassifier(n_estimators=102)
ada.fit(x_train,y_train)


# In[58]:


ada_score = cross_val_score(ada, x_train, y_train, cv=10,scoring='roc_auc_ovo')
ada_score_teste = cross_val_score(ada, x_test, y_test, cv=10,scoring='roc_auc_ovo')
print('Score AdaBoost Train: ', round(ada_score.mean() * 100, 2).astype(str) + '%')
print('Score AdaBoost Test: ', round(ada_score_teste.mean() * 100, 2).astype(str) + '%')


# In[59]:


y_pred_ada = ada.predict(x_test)


# In[60]:


cm_ada = confusion_matrix(y_test,y_pred_ada)


# In[64]:


acc_score_ada = accuracy_score(y_test,y_pred_ada)
f1_score_ada = f1_score(y_test,y_pred_ada)
precisao_ada = average_precision_score(y_test,y_pred_ada)
recall_ada = recall_score(y_test,y_pred_ada)
roc_ada = roc_auc_score(y_test,y_pred_ada,multi_class='ovo')
print('Acuracy ADA Boost ',round(acc_score_ada*100,2).astype(str)+'%')
print('Pred media Ada Boost ',round(precisao_ada*100,2).astype(str)+'%')
print('F1 Ada Boost ',round(f1_score_ada*100,2).astype(str)+'%')
print('Recall Ada Boost ',round(recall_ada*100,2).astype(str)+'%')
print('ROC Ada Boost ',round(roc_ada*100,2).astype(str)+'%')


# In[66]:


fig, ax = plt.subplots(figsize=(10,6))
sns.heatmap(cm_ada, ax=ax, annot=True, cmap=plt.cm.copper)
ax.set_title("Ada Boost \n Confusion Matrix", fontsize=14)
ax.set_xticklabels(['B', 'M'], fontsize=14, rotation=0)
ax.set_yticklabels(['B', 'M'], fontsize=14, rotation=360)


# # 9. Gradient Boost Classifier- GBC

# In[67]:


#print("Gradient Boost Classifier")
#grad_params = {'n_estimators' : [50,55,60,65,70,75,80,85,90],'max_depth' : list(range(3,11,1))}
#grad = GridSearchCV(GradientBoostingClassifier(), grad_params,n_jobs=-1,cv=10,scoring='roc_auc_ovo')
#grad.fit(X_train, y_train)
#grad_boost = grad.best_estimator_
#print("Best Estimator")
#print(grad_boost)


# In[68]:


#Parameters have been choosing based on GridSearchCV
grad_boost = GradientBoostingClassifier(n_estimators=65,max_depth=4)
grad_boost.fit(x_train, y_train)


# In[69]:


grad_score = cross_val_score(grad_boost, x_train, y_train, cv=10,scoring='roc_auc_ovo')
grad_score_teste = cross_val_score(grad_boost, x_test, y_test, cv=10,scoring='roc_auc_ovo')
print('Score GradBoost Train: ', round(grad_score.mean() * 100, 2).astype(str) + '%')
print('Score GradBoost Test: ', round(grad_score_teste.mean() * 100, 2).astype(str) + '%')


# In[70]:


y_pred_gb = grad_boost.predict(x_test)


# In[71]:


cm_gb = confusion_matrix(y_test,y_pred_gb)


# In[73]:


acc_score_gb = accuracy_score(y_test,y_pred_gb)
f1_score_gb = f1_score(y_test,y_pred_gb)
pred_gb = average_precision_score(y_test,y_pred_gb)
recall_gb = recall_score(y_test,y_pred_gb)
roc_gb = roc_auc_score(y_test,y_pred_gb,multi_class='ovo')
print('Acuracy Gradient Boosting ',round(acc_score_gb*100,2).astype(str)+'%')
print('Pred media Gradient Boosting  ',round(pred_gb*100,2).astype(str)+'%')
print('F1 Gradient Boosting  ',round(f1_score_gb*100,2).astype(str)+'%')
print('Recall Gradient Boosting  ',round(recall_gb*100,2).astype(str)+'%')
print('ROC Gradient Boosting ',round(roc_gb*100,2).astype(str)+'%')


# In[74]:


fig, ax = plt.subplots(figsize=(10,6))
sns.heatmap(cm_gb, ax=ax, annot=True, cmap=plt.cm.copper)
ax.set_title("Gradient Boosting  \n Confusion Matrix", fontsize=14)
ax.set_xticklabels(['B', 'M'], fontsize=14, rotation=0)
ax.set_yticklabels(['B', 'M'], fontsize=14, rotation=360)


# In[80]:


results = [ada_score,forest_score,grad_score]
results_test = [ada_score_teste,forest_score_teste,grad_score_teste]
name_model = ["AdaBoost","RFC","GradBoost"]


# In[81]:


fig,ax=plt.subplots(figsize=(10,5))
ax.boxplot(results)
ax.set_xticklabels(name_model)
plt.tight_layout()


# In[82]:


fig,ax=plt.subplots(figsize=(10,5))
ax.boxplot(results_test)
ax.set_xticklabels(name_model)
plt.tight_layout()


# In[ ]:




