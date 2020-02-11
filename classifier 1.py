# -*- coding: utf-8 -*-
"""
Spyder Editor

Code by: Sanjay Sekar Samuel
"""


'''
Below are the modeules needed to be exported to build the CNN
Followed by the modlules needed to preprocess the image
'''
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score

names = ['Age','RestBP','Cholostrol','Thalassemia','Fasting Blood Sugar',"CP", 'Rest ECG', 'Old Peak',"sex","Thal",'Chest Pain Cass']
HeartPatientFile = pd.read_csv("heart_data.csv", names=names)

X = HeartPatientFile.iloc[:,0:10].values               # Get all the features from the csv file
y = HeartPatientFile.select_dtypes(include=[object])   # Get all the lables from the csv file



lableE = preprocessing.LabelEncoder()       # Preprocess the labels by transforming them to numerical values (0,1,2,3)

Y = y.apply(lableE.fit_transform)           # This will make the catagorical values into numbnerical values

'''
We have to now split the data set into trained and test data set to avoid any overfitting in the data set
'''
X_train, X_validation, Y_train, Y_validation = train_test_split(X, Y, test_size = 0.05)


'''
Once we split the data, we now have to do some feature scaling on the training dataset to have uniformity on the dataset
'''
scaler = StandardScaler()
scaler.fit(X_train)

X_train = scaler.transform(X_train)
X_validation = scaler.transform(X_validation)

'''
Now we use the build in scilearn function called MPL to build the ANN
By default the relu activation function is used with the adam cost optimizer
'''
multiLayerPrec = MLPClassifier(hidden_layer_sizes=(10, 10, 10), max_iter=100)    # This is 3 hidden layer with 10 nodes each and max_iter is the number of epoches
multiLayerPrec.fit(X_train, Y_train.values.ravel())

predictionANN = multiLayerPrec.predict(X_validation)

print("Accuracy score:", accuracy_score(Y_validation, predictionANN))
print(confusion_matrix(Y_validation,predictionANN))
print(classification_report(Y_validation,predictionANN))
