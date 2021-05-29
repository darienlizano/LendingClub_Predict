#import packages
import pandas as pd
import numpy as np

#load packages for models
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

#load in data
loans = pd.read_csv('loan_data.csv')

#transforming 'purpose' col into dummy variable 

cat_feats = ['purpose'] #Creates list of 1 element containing the string 'purpose'

final_data = pd.get_dummies(loans,columns=cat_feats,drop_first=True) #adding cat_feat into loans

#split data
X = final_data.drop('not.fully.paid',axis=1)
y= final_data['not.fully.paid']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

#Decision Tree
#train decision tree
dtree = DecisionTreeClassifier()
dtree.fit(X_train,y_train)
pred = dtree.predict(X_test)

#print reports
print(classification_report(y_test,pred))
print(confusion_matrix(y_test,pred))

#Random Forest
#train random forest
rfc = RandomForestClassifier(n_estimators=200)
rfc.fit(X_train,y_train)
rfc_pred = rfc.predict(X_test) 

#print reports
print(classification_report(y_test,rfc_pred))
print(confusion_matrix(y_test,rfc_pred))
