import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder
data=pd.read_excel('iris (3).xls')
label_en=LabelEncoder()
a=['Classification']
for i in np.arange(len(a)):
    data[a[i]]=label_en.fit_transform(data[a[i]])
X=data.drop('Classification',axis=1)
Y=data['Classification']
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(X,Y,random_state=42,test_size=.25)
from sklearn.ensemble import RandomForestClassifier
model=RandomForestClassifier()
model.fit(x_train,y_train)
pickle.dump(model,open('model.pkl','wb'))