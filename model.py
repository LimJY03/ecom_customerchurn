import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
import joblib

"""# Data Acquisition

"""**Read Dataset**"""
ecomdf= pd.read_csv('ECommerceDataset.csv')
ecomdf.head()

"""# Data Exploration"""

ecomdf.shape

ecomdf.dtypes

"""Looking at the data types, it doesn't look like there's any anomaly in the values in the dataset. """

ecomdf.duplicated().sum()

ecomdf.isnull().sum()

"""From the preliminary Data set we can see that there are certain missing values. There are no duplicate entries.

There are 5630 rows and 20 columns.
"""

print('Total number of rows of missing values:', ecomdf.isnull().any(axis=1).sum())

"""In Pandas, True is internally represented as a 1, while False as a 0. Axis=1 means we are looking at the row. Therefore, taking the sum of this Series will return the number of rows with at least one missing value, which is total of 1856 rows.

**Check the value counts of each of the categorical variables:**
"""

for i in ecomdf.columns:
    if ecomdf[i].dtypes == 'object':
        print(i)
        print()
        print('Counts:')
        print(ecomdf[i].value_counts())
        print()
        print()

ecomdf.describe().transpose()

"""# Data Preparation/Data Cleaning

**1. Handling Missing Values**
"""

# Copying the data frame into another one
ecomdfNew= ecomdf.copy()

# Set inplace true to reflect the change on original dataframe
ecomdf.drop(['CustomerID'],axis=1, inplace=True)

"""**Check variables where there are missing values:**"""

for i in ecomdf.columns:
    if ecomdf[i].isnull().sum() > 0:
        print(i)
        print('Total null values:', ecomdf[i].isnull().sum())
        print()

"""**Filling the missing values by the mean values**"""

for i in ecomdf.columns:
    if ecomdf[i].isnull().sum() > 0:
        ecomdf[i].fillna(ecomdf[i].mean(),inplace=True)

ecomdf.isnull().sum()

"""**2. Standardize naming convention for data**"""

def replace(col_name: str, convert_pair):
    for initial, final in convert_pair:
        ecomdf.loc[ecomdf[col_name] == initial, col_name] = final

replace(col_name='PreferredLoginDevice', convert_pair=[('Mobile Phone', 'Phone')])
replace(col_name='PreferredPaymentMode', convert_pair=[('CC', 'Credit Card'), ('COD', 'Cash on Delivery')])
replace(col_name='PreferedOrderCat', convert_pair=[('Mobile Phone', 'Phone'), ('Mobile','Phone')])

ecomdf.head()

"""**3. Outlier Treatment**

**Converting the Churn variable to object**
"""

ecomdf['Churn'] = ecomdf['Churn'].astype('object')

plt.figure(figsize=(80,20))
sns.boxplot(data=ecomdf)
plt.title('The boxplot to study outliers', fontsize=50)
plt.xlabel('Variables', fontsize= 50)
plt.ylabel('Values', fontsize=50)
plt.xticks(rotation=90, fontsize=30)
plt.yticks(fontsize=40)

"""From the boxplot, we can see that there are quite a lot of outliers in almost all of the variables.

**Calculate total number of outliers**
"""

cat = ecomdf.select_dtypes(include='object').columns #object type columns
num = list(ecomdf.select_dtypes(exclude='object').columns) #numerical type columns

sumNum = 0

for cols in num:
    Q1 = ecomdf[cols].quantile(0.25)
    Q3 = ecomdf[cols].quantile(0.75)
    IQR=Q3-Q1
    lr= Q1-(1.5 * IQR)
    ur= Q3+(1.5 * IQR)
    ((ecomdf[cols] < (Q1 - 1.5 * IQR)) | (ecomdf[cols] > (Q3 + 1.5 * IQR))).sum()
    numOfOut= ((ecomdf[cols] < (Q1 - 1.5 * IQR)) | (ecomdf[cols] > (Q3 + 1.5 * IQR))).sum()
    print(numOfOut)
    ecomdf[cols] = ecomdf[cols].mask(ecomdf[cols]<lr, lr, )
    ecomdf[cols] = ecomdf[cols].mask(ecomdf[cols]>ur, ur, )
    sumNum = sumNum + numOfOut

print()
print('The total number of outliers:', sumNum)

"""There are total 2287 outliers for our data.

Thus, we will now treat outliers. For this we will define the lower range and upper range which is going to be at a distance of 1.5 times the Interquartile range from the respective whiskers.
"""

def remove_outlier(col):
    sorted(col)
    Q1,Q3=np.percentile(col,[25,75])
    IQR=Q3-Q1
    lr= Q1-(1.5 * IQR)
    ur= Q3+(1.5 * IQR)
    return lr, ur

for column in ecomdf.columns:
    if ecomdf[column].dtype != 'object':
        lr,ur=remove_outlier(ecomdf[column])
        ecomdf[column]=np.where(ecomdf[column]>ur,ur,ecomdf[column])
        ecomdf[column]=np.where(ecomdf[column]<lr,lr,ecomdf[column])

plt.figure(figsize=(80,20))
sns.boxplot(data=ecomdf)
plt.title('The boxplot to study outliers', fontsize=50)
plt.xlabel('Variables', fontsize= 50)
plt.ylabel('Values', fontsize=50)
plt.xticks(rotation=90, fontsize=30)
plt.yticks(fontsize=40)

sumNum = 0

for cols in num:
    Q1 = ecomdf[cols].quantile(0.25)
    Q3 = ecomdf[cols].quantile(0.75)
    IQR=Q3-Q1
    lr= Q1-(1.5 * IQR)
    ur= Q3+(1.5 * IQR)
    ((ecomdf[cols] < (Q1 - 1.5 * IQR)) | (ecomdf[cols] > (Q3 + 1.5 * IQR))).sum()
    numOfOut= ((ecomdf[cols] < (Q1 - 1.5 * IQR)) | (ecomdf[cols] > (Q3 + 1.5 * IQR))).sum()
    print(numOfOut)
    ecomdf[cols] = ecomdf[cols].mask(ecomdf[cols]<lr, lr, )
    ecomdf[cols] = ecomdf[cols].mask(ecomdf[cols]>ur, ur, )
    sumNum = sumNum + numOfOut

print()
print('The total number of outliers:', sumNum)

"""Here we can see that we are free from outliers. The outliers are now replaced with their corresponding upper range or lower range values.

# Exploratory Data Analysis

**1. Analyse Churn and Not Churn**

**Bar chart of Not Churn(0) vs Churn(1)**
"""

plt.figure(figsize = (10,5))
ax=sns.countplot(x='Churn', data=ecomdf)
plt.xlabel("Churn Type")
plt.ylabel('Number')
plt.title("Not Churn vs Churn")
plt.show()

"""**Bar chart of Not Churn(0) vs Churn(1) With Percentage**"""

plt.figure(figsize = (10,5))
ax = sns.countplot(x='Churn', data = ecomdf)
for a in ax.patches:
    ax.annotate(format((a.get_height()/5630)*100,'.2f'), (a.get_x() + a.get_width()/2., a.get_height()),\
                ha='center',va='center',size=12,xytext=(0, -10),textcoords='offset points')
plt.xlabel("Churn Type")
plt.ylabel('Number')
plt.title("Not Churn vs Churn")
plt.show()

"""**2. Analyse Customer Behaviour and Pattern**

**Distribution of Tenure of the Customers**
"""

plt.figure(figsize = (15,10))
sns.displot(x='Tenure', kde=True, data=ecomdf)
plt.title("Distribution of Tenure of the Customers on the platform")
plt.show()

"""**Distribution on Total Number of Orders in Last Month**"""

plt.figure(figsize = (15,10))
sns.displot(x='OrderCount', kde = True, data = ecomdf)
plt.title("Distribution of Number of Ordercount")
plt.show()

"""**Distribution of Recency of the Customers**"""

plt.figure(figsize = (15,10))
sns.displot(x='DaySinceLastOrder', kde = True, data = ecomdf)
plt.title("Distribution of Recency of the Customers")
plt.show()

"""**Distribution of Percentage of Customer Increase In Orders**"""

plt.figure(figsize = (15,10))
sns.displot(x='OrderAmountHikeFromlastYear', kde = True, data = ecomdf)
plt.title('Distribution of Percentage of Customer Increase in Orders')
plt.show()

"""Distribution of Hours Spent on Mobile Application or Website by Customers"""

plt.figure(figsize = (15,5))
axx = sns.countplot(x='HourSpendOnApp', data=ecomdf)
for a in axx.patches:
    axx.annotate(format((a.get_height()/5630)*100,'.2f'), (a.get_x() + a.get_width()/2., a.get_height()),\
                ha='center',va='center',size=12,xytext=(0, 5),textcoords='offset points')
plt.title("Distribution of hours spent on the app by the customers")
plt.show()

"""**3. Analyse Differences between Churn and Not Churn Customer**

**Distribution of Satisfaction score between Churned and Retained Customers**
"""

plt.figure(figsize = (10,5))
sns.countplot(x='SatisfactionScore', hue = 'Churn', palette ='magma', data = ecomdf)
plt.title("Distribution of Satisfaction Score for Churned and Retained customers")
plt.show()

"""**Distribution of Gender for Churned and Retained Customers**"""

plt.figure(figsize = (10,5))
sns.countplot(x='Gender', hue = 'Churn', palette ='mako', data = ecomdf)
plt.title("Distribution of Gender for Churned and Retained customers")
plt.show()

"""**Distribution of Complain between Churn and Not Churn Customer**"""

plt.figure(figsize = (10,5))
sns.countplot(x='Complain', hue = 'Churn', palette = 'viridis', data = ecomdf)
plt.title("Distribution of Complain between Churn and Retained Customer")
plt.show()

"""**Analysing Churn by Each Variable**

Categorical Encoding (Converting the categorical values to numerical values)
"""

from sklearn.preprocessing import LabelEncoder
enc = LabelEncoder()
for col in ecomdf.select_dtypes(include='object'):
    ecomdf[col]=enc.fit_transform(ecomdf[col])

"""Heatmap to show relationship between Churn or Not Churn with Each Variable"""

plt.figure(figsize = (20,10))
sns.heatmap(ecomdf.corr(), annot = True, cmap=plt.cm.Reds)
plt.title('Correlation Matrix for the Customer Dataset')
plt.show()

"""Through the correlation matrix, we will able to observe which variable more correlate with Churn.

# Modelling

Import libraries for modelling
"""

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score, ConfusionMatrixDisplay
import pickle

X = ecomdf.drop(['Churn'],axis=1)
y = ecomdf['Churn']
X.columns.values

y.value_counts()

from imblearn.combine import SMOTEENN
smtn = SMOTEENN(random_state = 0)

# Making samples
X, y = smtn.fit_resample(X,y)
y.value_counts()

# Training the model
smtn.fit(X,y)

"""Split Data - Train and Test Data"""

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=0)

"""GridsearchCV for hyperparameter tuning"""

from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler, MinMaxScaler, Normalizer

class my_classifier(BaseEstimator,):
    def __init__(self, estimator=None):
        self.estimator = estimator
    def fit(self, X, y=None):
        self.estimator.fit(X,y)
        return self
    def predict(self, X, y=None):
        return self.estimator.predict(X,y)
    def predict_proba(self, X):
        return self.estimator.predict_proba(X)
    def score(self, X, y):
        return self.estimator.score(X, y)

pipe = Pipeline([('scaler', StandardScaler()), ('clf', my_classifier())])

parameters = [
              {'clf':[LogisticRegression(max_iter=1000)],
               'clf__C':[0.001,0.01,.1,1],
               'clf__solver':['lbfgs','liblinear']
               },
             {'clf':[RandomForestClassifier()],
                'clf__criterion':['gini','entropy']
             },
             {
               'clf':[LinearDiscriminantAnalysis()],
               'clf__solver':['svd','lsqr','eigen']
             },
             {
              'clf':[XGBClassifier()],
                'clf__learning_rate':[0.01,0.1,0.2,0.3],
                'clf__reg_lambda':[0.01,0.1,1],
                'clf__reg_alpha': [0.01,0.1,0,1],
             }]

grid = GridSearchCV(pipe, parameters, cv=5, verbose=True)
grid.fit(X_train,y_train)

gs=grid.best_estimator_

grid.best_score_

y_pred = grid.predict(X_test,)

confusionMatrix = confusion_matrix(y_test, y_pred)
print(confusionMatrix)

fig, ax = plt.subplots(figsize=(5, 5))
im = ax.imshow(confusionMatrix,cmap="copper")
ax.xaxis.set(ticks=(0, 1), ticklabels=('Predicted 0s', 'Predicted 1s'))
ax.yaxis.set(ticks=(0, 1), ticklabels=('Actual 0s', 'Actual 1s'))
ax.set_ylim(1.5, -0.5)
for i in range(2):
    for j in range(2):
        ax.text(j, i, confusionMatrix[i, j], ha='center', va='center', color='red')
cbar = plt.colorbar(im)
cbar.set_label("Colorbar")
plt.show()

print(classification_report(y_test, y_pred))

feature_array = grid.best_estimator_[-1].feature_importances_
importance = dict(zip(ecomdf.drop('Churn',axis=1).columns,feature_array))
importance = dict(sorted(importance.items(), key= lambda item:item[1],reverse = True) )
fig, ax = plt.subplots(figsize=(20,5))
sns.barplot(x=list(importance.keys()), y=list(importance.values()))
plt.tick_params(axis='x', labelrotation=90)
plt.show()

from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
logit_roc_auc = roc_auc_score(y_test, grid.predict(X_test))
fpr, tpr, thresholds = roc_curve(y_test, grid.predict_proba(X_test)[:,1])
plt.figure()
plt.plot(fpr, tpr, label='Random Forest (area = %0.2f)' % logit_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.savefig('Log_ROC')
plt.show()

joblib.dump(gs, 'best_model.pkl')
