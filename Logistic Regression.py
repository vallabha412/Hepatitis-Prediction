#Importing Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix,f1_score
from sklearn.preprocessing import LabelEncoder,MinMaxScaler
from sklearn.metrics import classification_report, precision_score, recall_score 
from sklearn.metrics import roc_curve, roc_auc_score, average_precision_score
from sklearn.metrics import precision_recall_curve 

#Importing the Dataset
df = pd.read_csv("titanic.csv")

#Encoding the string values to integers using Label Encoder
le = LabelEncoder()
df['sex'] = le.fit_transform(df['sex'])
df['age'].fillna(df['age'].mean(),inplace=True)
df['fare'].fillna(df['fare'].mean(),inplace=True)

#Separating the Dependent and Independent Variables
x = df.iloc[:,1:-1]
y = df.iloc[:,-1]

#Scaling the data
mm = MinMaxScaler()
x = mm.fit_transform(x)

#Splitting the data into Training and Testing sets
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.3,random_state=11)

#Implementing Logisitc Regression
lr = LogisticRegression()

#Fitting the data
lr.fit(x_train,y_train)

#Predicting the values using test data
y_pred = lr.predict(x_test)

#Calculating the metrics
acc = accuracy_score(y_test,y_pred)
cm = confusion_matrix(y_test,y_pred)
f1 = f1_score(y_test,y_pred)
print("Accuracy={} Confusion_Matrix={} F1-Score={}".format(acc,cm,f1))
print("Classification Report = \n {}".format(classification_report(y_test,y_pred)))
print("Precision Score = {}".format(precision_score(y_test,y_pred)))
print("Recall Score = {}".format(recall_score(y_test,y_pred)))
print("Average Precision Score = {}".format(average_precision_score(y_test,y_pred)))
print("Area Under Curve : ",roc_auc_score(y_test, y_pred))

#Plotting
precision,recall,threshold = precision_recall_curve(y_test,y_pred)
fpr, tpr, thresholds = roc_curve(y_test, y_pred)
plt.plot(recall,precision,color="red",label="Preicison_Recall_Curve")
plt.plot(fpr,tpr,color="blue",label="ROC_Curve")
plt.ylabel("Precision , True Positive Rate")
plt.xlabel("Recall, False Positive Rate")
plt.title("Precision Recall Curve & Roc Curve")
plt.legend()
plt.show()
