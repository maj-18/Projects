#import libraries
import numpy as np 
import pandas as pd 
from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

import warnings 
warnings.filterwarnings('ignore')


#reading the dataset
data = pd.read_csv('ElectionData.csv')
#filling missing values 
data['pre.votersPercentage'].fillna(data['pre.votersPercentage'].mean(),inplace=True)
data['pre.totalVoters'].fillna(data['pre.totalVoters'].median(),inplace=True)
data['validVotesPercentage'].fillna(data['validVotesPercentage'].median(),inplace=True)
data['Votes'].fillna(data['Votes'].median(),inplace=True)

# Dropping columns from correlation heatmap with ultiple observations of high correlation coefficients(>0.90) to avoid multicollinearity
data=data.drop(['totalMandates','numParishesApproved','blankVotes','nullVotes','subscribedVoters','totalVoters','pre.blankVotes','pre.nullVotes','pre.subscribedVoters','pre.totalVoters','Percentage','Mandates','pre.blankVotesPercentage','pre.votersPercentage'],axis=1)

#OUTLIER DETECTION
#1. availableMandates
Q1=np.percentile(data['availableMandates'],25,interpolation='midpoint')
Q3=np.percentile(data['availableMandates'],75,interpolation='midpoint')
IQR=Q3-Q1
low=Q1-1.5*IQR
up=Q3+1.5*IQR
#Replacing outliers using median
data.loc[((data['availableMandates'] > up) | (data['availableMandates'] < low)),'availableMandates'] =data['availableMandates'].median()

#2. numParishes
Q1=np.percentile(data['numParishes'],25,interpolation='midpoint')
Q3=np.percentile(data['numParishes'],75,interpolation='midpoint')
IQR=Q3-Q1
low=Q1-1.5*IQR
up=Q3+1.5*IQR
#Replacing outliers using median
data.loc[((data['numParishes'] > up) | (data['numParishes'] < low)),'numParishes'] =data['numParishes'].median()

#3. votersPercentage
Q1=np.percentile(data['votersPercentage'],25,interpolation='midpoint')
Q3=np.percentile(data['votersPercentage'],75,interpolation='midpoint')
IQR=Q3-Q1
low=Q1-1.5*IQR
up=Q3+1.5*IQR
#Replacing outliers using median
data.loc[((data['votersPercentage'] > up) | (data['votersPercentage'] < low)),'votersPercentage'] =data['votersPercentage'].median()

#4. pre.nullVotesPercentage
Q1=np.percentile(data['pre.nullVotesPercentage'],25,interpolation='midpoint')
Q3=np.percentile(data['pre.nullVotesPercentage'],75,interpolation='midpoint')
IQR=Q3-Q1
low=Q1-1.5*IQR
up=Q3+1.5*IQR
#Replacing outliers using median
data.loc[((data['pre.nullVotesPercentage'] > up) | (data['pre.nullVotesPercentage'] < low)),'pre.nullVotesPercentage'] =data['pre.nullVotesPercentage'].median()

#5. validVotesPercentage
Q1=np.percentile(data['validVotesPercentage'],25,interpolation='midpoint')
Q3=np.percentile(data['validVotesPercentage'],75,interpolation='midpoint')
IQR=Q3-Q1
low=Q1-1.5*IQR
up=Q3+1.5*IQR
#Replacing outliers using median
data.loc[((data['validVotesPercentage'] > up) | (data['validVotesPercentage'] < low)),'validVotesPercentage'] =data['validVotesPercentage'].median()

#6. Votes
Q1=np.percentile(data['Votes'],25,interpolation='midpoint')
Q3=np.percentile(data['Votes'],75,interpolation='midpoint')
IQR=Q3-Q1
low=Q1-1.5*IQR
up=Q3+1.5*IQR
#Replacing outliers using median
data.loc[((data['Votes'] > up) | (data['Votes'] < low)),'Votes'] =data['Votes'].median()

#LABEL ENCODING
obj_col =[]
for i in data.columns:
    if data[i].dtypes=="O":
        obj_col.append(i)
le = LabelEncoder()
for i in obj_col:
    data[i]=pd.DataFrame(le.fit_transform(data[i]))

#STANDARDIZATION
x = data.drop(columns=['FinalMandates'])
y = data[["FinalMandates"]]    
sc = StandardScaler()
a = sc.fit_transform(x)
data_x = pd.DataFrame(a,columns=x.columns)

#FEATURE ENGINEERING
#Creating new feature invalidVotersPercentage
data_x['invalidVotesPercentage']=(data_x['blankVotesPercentage']+data_x['nullVotesPercentage'])*.5
#Droping existing features 'blankVotesPercentage','nullVotesPercentage'
data_x=data_x.drop(['blankVotesPercentage','nullVotesPercentage'],axis=1)

#MODELING
#Taking x and y values
x=data_x
y= data[["FinalMandates"]]
#Splitting data into train and test set
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.20,random_state=45)

regressor=RandomForestRegressor()
m=regressor.fit(x_train,y_train)


# save the model to disk

import pickle
pickle.dump(regressor,open('model.pkl','wb'))
from flask import Flask, request, render_template
import numpy as np
#import pandas as pd
import pickle
app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))
