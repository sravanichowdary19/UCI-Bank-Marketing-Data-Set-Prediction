#!/usr/bin/env python
# coding: utf-8

# There are 2 CSV files that were imported from "https://archive.ics.uci.edu/ml/machine-learning-databases/00222/" 
# 
# 1.bank-additional-full.csv
# 
# 2.bank-additional.csv
# 
# Required Output: we have to predict if the client will subscribe a term deposit or not

# # 1. Import the Libraries required for Exploratory Data analysis, Visualization, Models

# In[4]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 
from sklearn.model_selection import train_test_split
get_ipython().run_line_magic('matplotlib', 'inline')
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity="all"
from sklearn import metrics
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

from sklearn.linear_model import LogisticRegression
from statsmodels.stats.outliers_influence import variance_inflation_factor as vif
from sklearn.metrics import classification_report,roc_auc_score,roc_curve
from sklearn.svm import SVC
from sklearn.ensemble import BaggingClassifier,  GradientBoostingClassifier,RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
sns.set(color_codes=True)
from sklearn.naive_bayes import GaussianNB


# In[5]:


#Import the CSV files into the current directory as Dataframe
bank_addtional_full_data=pd.read_csv('bank-additional-full.csv',sep=';')
bank_full_data=pd.read_csv('bank-additional.csv',sep=';')
bank_addtional_full_data.shape
bank_full_data.shape


# # 2.Get the variable names, types, missing values , univariate and Bi variate distributions

# In[6]:


bank_addtional_full_data.columns


# In[7]:


bank_addtional_full_data.info()


# In[8]:


bank_addtional_full_data.describe().T


# There do not seem to be any missing values except for Previous but that must be fine based on the definition "previous: number of contacts performed before this campaign and for this client" could be zero.

# In[9]:


bank_addtional_full_data.isnull().any()


# In[10]:


# check if the outcome is balanced or quite imbalaced
sns.countplot(bank_addtional_full_data['y'])
bank_addtional_full_data['y'].value_counts()


# In[11]:


#plotting employment variation rate - quarterly indicator emp.var.rate
plt.rcParams['figure.figsize'] = (8, 6)
sns.countplot(x='emp.var.rate', hue='emp.var.rate', data=bank_addtional_full_data);


# In[12]:


#Get the dummy variable for dependent variable
y_n ={'yes' : 1, 'no' : 0}
bank_addtional_full_data['y_encode'] = bank_addtional_full_data['y'].map(lambda x: y_n[x])


# In[13]:


#getting marital status of groupby people
age_group_names = ['young', 'lower middle', 'middle', 'senior']
bank_addtional_full_data['age_binned'] = pd.qcut(bank_addtional_full_data['age'], 4, labels = age_group_names)
bank_addtional_full_data['age_binned'].value_counts()
bank_addtional_full_data['marital'].value_counts()
Distri_marital_age = bank_addtional_full_data['y_encode'].groupby([bank_addtional_full_data['marital'],bank_addtional_full_data['age_binned']] ) 
Distri_marital_age.mean()


# Senior people seem to be doing a term deposit irrespective of marital status where as "unknown:younger and middle groups" are an exception
# single younger group has 16% response

# Univariate for all the categorical variables

# In[14]:


def countplot_withd(label, dataset):
    plt.figure(figsize=(15,7))
    Y = dataset[label]
    total = len(Y)*1.
    ax=sns.countplot(x=label, data=dataset, hue="y")
    for p in ax.patches:
        ax.annotate('{:.1f}%'.format(100*p.get_height()/total), (p.get_x()+0.1, p.get_height()+5))

    plt.show()     


# In[15]:


countplot_withd('marital', bank_addtional_full_data)


# Married group has a higher respose rate on term deposits than the other categories but majority of the customers are married followed by single and Divorced

# In[16]:


# see the distribution of job categories
countplot_withd('job', bank_addtional_full_data)


# customers who have a job of admin have the highest rate of subscribing a term deposit, but they are also the highest when it comes to not subscribing. This is simply because we have more customers working as admin than any other profession

# In[17]:


#Default: Denotes if the customer has credit in default or not. The categories are yes, no and unknown.
countplot_withd('default', bank_addtional_full_data)


# In[18]:


#housing: Denotes if the customer has a housing loan. Three categories are ‘no’, ’yes’, ’unknown’.
countplot_withd('housing', bank_addtional_full_data) 


# we can see from the above plot, majority of the customers have a housing loan.

# In[19]:


#poutcome: This feature denotes the outcome of the previous marketing campaign
countplot_withd('poutcome', bank_addtional_full_data) 


# For most of the customers, the previous marketing campaign outcome does not exists.customers who had a successful outcome from the previous campaign, majority of those customers did subscribe for a term deposit. As it has the distribution of 2.2% for term deposit class, and 1.2% for non term deposit class. From this, we can make an assumption, that this feature may hold some value in predicting the target variable

# In[20]:


#day_of_week: This feature denotes the last contact day of the week (categorical: ‘mon’,’tue’,’wed’,’thu’,’fri’)
countplot_withd('day_of_week', bank_addtional_full_data) 


# all the days have the similar distribution for both the classes. ~17% not subscribing fro term deposit and ~2-3% subsribing for the same. As there is no much difference in the distribution we can conclude that this variable may not be helpful in prediction

# # Check the Bivariate Distribution

# In[21]:


sns.pairplot(bank_addtional_full_data,diag_kind='kde')


# initial examination of the plots do not show any striking correlation between the independent varaiables and most the independednt variables seem to be either skewed or multimodal distributions. However, we shall check correlation matrix and VIF for any multicollinearity 

# In[22]:


plt.figure(figsize=(10,8))
sns.heatmap(bank_addtional_full_data.corr(),
            annot=True,
            linewidths=.5,
            center=0,
            cbar=False,
            cmap="YlGnBu")
plt.show()


# We can see that Emp.Var.rate is highly correlated with Euribor and number of employed about 97% and 91%. Also, Consumer price index seem to be significantly correlated with the euribor 3M and nr. employed with >60%

# In[23]:


import copy
bank_addtional_full_data1=copy.deepcopy(bank_addtional_full_data)


# In[24]:


#Check the Multi collinearity ....create dummies for categorical variables

bank_addtional_full_data1['poutcome1'] = bank_addtional_full_data['poutcome'].map({'nonexistent':0, 'failure':-1,'success':1})
bank_addtional_full_data1['day_of_week1'] = bank_addtional_full_data['day_of_week'].map({'mon':1, 'tue':2,'wed':3,'thu':4,'fri':5})
bank_addtional_full_data1['month1'] = bank_addtional_full_data['month'].map({'jan':1, 'feb':2,'mar':3,'apr':4,'may':5,'jun':6, 'jul':7,'aug':8,'sep':9,'oct':10,'nov':11,'dec':12})
bank_addtional_full_data1['contact1'] = bank_addtional_full_data['contact'].map({'cellular':0,'telephone':1})
bank_addtional_full_data1['job1'] = bank_addtional_full_data['job'].map({'admin.':1, 'blue-collar':2,'entrepreneur':3,'housemaid':4,'management':5, 'unemployed':6,'retired':7,'self-employed':8,'services':9,'student':10,'technician':11,'unknown':12})
bank_addtional_full_data1['marital1'] = bank_addtional_full_data['marital'].map({'divorced':1, 'married':2,'single':3,'unknown':4})
bank_addtional_full_data1['education1'] = bank_addtional_full_data['education'].map({'basic.4y':1, 'basic.6y':2,'basic.9y':3,'high.school':4,'illiterate':5,'professional.course':6,'university.degree':7,'unknown':8})
bank_addtional_full_data1['default1'] = bank_addtional_full_data['default'].map({'no':0,'yes':1,'unknown':2})
bank_addtional_full_data1['housing1'] = bank_addtional_full_data['housing'].map({'no':0,'yes':1,'unknown':2})
bank_addtional_full_data1['loan1'] = bank_addtional_full_data['loan'].map({'no':0,'yes':1,'unknown':2})
 


# In[29]:


x=bank_addtional_full_data1.drop(['y','poutcome','day_of_week','month','contact','y_encode','age_binned','job','marital','education','default','housing','loan'],axis=1)


# In[30]:


x.info()


# In[31]:



vif_data=pd.DataFrame()
vif_data['feature']=x.columns
vif_data['vif']=[vif(x.values,i) for i in range(len(x.columns))]
print(vif_data)


# In[57]:


#Drop nr.employed
x1=x.drop(['nr.employed','cons.price.idx','euribor3m','pdays','cons.conf.idx'],axis=1)
vif_data=pd.DataFrame()
vif_data['feature']=x1.columns
vif_data['vif']=[vif(x1.values,i) for i in range(len(x1.columns))]
print(vif_data)


# In[63]:


bank_addtional_full_data=bank_addtional_full_data.drop_duplicates()
bank_addtional_full_data.shape


# In[64]:


bank_addtional_full_data_ind=bank_addtional_full_data.drop(['y','y_encode','age_binned'],axis=1)
bank_addtional_full_data_dep=bank_addtional_full_data['y'].map({'yes':1,'no':0})


# In[65]:


bank_addtional_full_data_ind.head()


# In[66]:


bank_addtional_full_data_ind1=bank_addtional_full_data_ind.drop(['age','nr.employed','duration','cons.price.idx','pdays','cons.conf.idx'],axis=1)


# In[67]:


bank_addtional_full_data_dep.value_counts()


# In[68]:


df_ind=pd.get_dummies(bank_addtional_full_data_ind1,columns= ['poutcome','day_of_week','month','contact','job','marital','education','default','housing','loan'])


# In[69]:


df_ind.head()


# In[70]:


xtrain, xtest, ytrain, ytest=train_test_split(df_ind,bank_addtional_full_data_dep, test_size=0.3, random_state=5)
xtrain.shape
xtest.shape


# # Apply Logistic regression

# In[71]:


model_lr =LogisticRegression(solver="liblinear")
model_lr.fit(xtrain,ytrain)
ypred_lr=model_lr.predict(xtest)
coeff_df_lr=pd.DataFrame(model_lr.coef_)
coeff_df_lr['intercept']=model_lr.intercept_
coefficients = pd.concat([pd.DataFrame(xtrain.columns),pd.DataFrame(np.transpose(model_lr.coef_))], axis =1)
print(coefficients)


# In[72]:


import statsmodels.api as sm
# building the model and fitting the data
log_reg = sm.Logit(ytrain, xtrain).fit()
print(log_reg.summary())


# In[73]:


cm_lr=metrics.confusion_matrix(ytest,ypred_lr, labels=[1,0])
df_cm=pd.DataFrame(cm_lr,index=[i for i in ["1","0"]], columns=[i for i in ["predict1", "predict0"]])
plt.figure(figsize=(8,5))
sns.heatmap(df_cm, annot=True)


# In[74]:


prob=model_lr.fit(xtrain,ytrain).predict_proba(xtest)
fpr1,tpr1, thresholds1=roc_curve(ytest,prob[:,1])
auc = metrics.roc_auc_score(ytest,model_lr.predict(xtest))
auc
plt.plot(fpr1, tpr1, label='Logistic (area = %0.2f)' % auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('1-Specificity(False Positive Rate)')
plt.ylabel('Sensitivity(True Positive Rate)')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")


# In[75]:


ytrain_pred_lr=model_lr.predict(xtrain)
print("model accuracy for test dataset for Logistic: {0:0.4f}%".format (metrics.accuracy_score(ytest,ypred_lr)*100))
print(metrics.classification_report(ytest, ypred_lr ,labels=[0,1]))


# The training data set accuracy score is ~70% and for test it is 90% which is higher. However, the recall for term dep subscription is 0.5 which is almost random and f1 score is quite less ~0.50 . the less popular class having a lesser recall

# # Naive Bayes

# In[76]:


model_nb=GaussianNB()
model_nb.fit(xtrain,ytrain)
ytrain_pred_nb=model_nb.predict(xtrain)
print("model_accuracy for train data using naive bayes: {0:0.4f}%".format (metrics.accuracy_score(ytrain,ytrain_pred_nb)*100))
ytest_pred_nb=model_nb.predict(xtest)
print("model accuracy for test data using naive bayes: {0: 0.4f}%" .format (metrics.accuracy_score(ytest,ytest_pred_nb)*100))


# In[77]:


NB_roc_auc = metrics.roc_auc_score(ytest, model_nb.predict(xtest))
fpr, tpr, thresholds = roc_curve(ytest, model_nb.predict_proba(xtest)[:,1])
plt.figure()
plt.plot(fpr, tpr, label='Naive Bayes (area = %0.2f)' % NB_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.savefig('Log_ROC')
plt.show()


# # Train standard Ensemble model - Random forest

# In[78]:


rf =RandomForestClassifier(n_estimators=50,random_state=1,max_features =6,max_depth=7)
rf=rf.fit(xtrain,ytrain)
rfy_pred=rf.predict(xtest)
cm=metrics.confusion_matrix(ytest,rfy_pred,labels=[0,1])


# In[79]:


df_cm_rf=pd.DataFrame(cm,index=[i for i in ["1","0"]], columns=[i for i in ["pred1","pred0"]])
plt.figure(figsize=(8,6))
sns.heatmap(df_cm_rf, annot=True)
print(metrics.classification_report(ytest,rfy_pred))


# In[80]:


ytest_pred_rf=rf.predict(xtest)
ytrain_pred_rf=rf.predict(xtrain)
print("train dataset accuracy for Randomforest {0:0.4f}%".format (metrics.accuracy_score(ytrain,ytrain_pred_rf)*100) )
print("test dataset accuracy for Randomforest {0:0.4f}%".format (metrics.accuracy_score(ytest,ytest_pred_rf)*100) )


# In[82]:


rf_roc_auc = metrics.roc_auc_score(ytest, rf.predict(xtest))
fpr, tpr, thresholds = roc_curve(ytest, rf.predict_proba(xtest)[:,1])
plt.figure()
plt.plot(fpr, tpr, label='Random Forest (area = %0.2f)' % rf_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.savefig('Log_ROC')
plt.show()


# In[83]:


#feature importance
feature_imp = pd.Series(rf.feature_importances_,index=df_ind.columns).sort_values(ascending=False)
feature_imp


# # Conclusion

# Logistic regression gave an accuracy of ~70% as a baseline where as naive bayes and random forest gave a better accuracy 
# of 86% and 90% respectively.
# However, there is a scope of improvement to do upsampling using SMOTE which is evident from the class F1 score and recall numbers. Couldn't take that route due to time constraint
# Also, we can have more features derived based on the business inputs and marketing teams
# 

# In[ ]:




