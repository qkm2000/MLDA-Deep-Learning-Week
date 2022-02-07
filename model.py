import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from collections import Counter
from sklearn.compose import ColumnTransformer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, InputLayer
from sklearn.metrics import classification_report
from tensorflow.keras.utils import to_categorical
import numpy as np

data = pd.read_csv("diabetes.csv")
y = data['Outcome']
x = data[['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']]


print(data['Glucose'].median())

#Replace 0 with Mean Value
glucose_mean = data['Glucose'].median()
blood_pressure_mean = data['BloodPressure'].median()
skin_mean = data['SkinThickness'].median()
insulin_mean = data['Insulin'].median()
bmi_mean = data['BMI'].median()
def clean_glucose(x):
    return glucose_mean if x==0 else x
def clean_bloodpressure(x):
    return blood_pressure_mean if x==0 else x
def clean_skin(x):
    return skin_mean if x==0 else x
def clean_insulin(x):
    return insulin_mean if x==0 else x
def clean_bmi(x):
    return bmi_mean if x==0 else x

data['Glucose'] = data['Glucose'].apply(clean_glucose)
data['BloodPressure'] = data['BloodPressure'].apply(clean_bloodpressure)
data['SkinThickness'] = data['SkinThickness'].apply(clean_skin)
data['Insulin'] = data['Insulin'].apply(clean_insulin)
data['BMI'] = data['BMI'].apply(clean_bmi)
print(data.head())

#Convert categorical variable into dummy/indicator variables.
x = pd.get_dummies(x)
x_train,x_try,y_train,y_try = train_test_split(x,y,train_size=0.8,random_state=42)
x_test,x_val,y_test,y_val = train_test_split(x_try,y_try,train_size = 0.5, random_state=42)
#scaling and standardizing our training and test data.
ct = ColumnTransformer([("numeric", StandardScaler(),['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'])])
x_train = ct.fit_transform(x_train)
x_test = ct.fit_transform(x_test)
le = LabelEncoder()
y_train = to_categorical(le.fit_transform(y_train))
y_test = to_categorical(le.fit_transform(y_test))

#Build model
model = Sequential()
model.add(Dense(64,input_dim=8,activation='relu'))
model.add(Dense(32,activation='relu'))
model.add(Dense(2,activation='softmax'))
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
model.fit(x_train,y_train,epochs=11,batch_size=12)

y_estimate = np.argmax(model.predict(x_test),axis=1)
y_true = np.argmax(y_test,axis=1)
print(classification_report(y_true,y_estimate))

x_val = ct.transform(x_val)
y_val = to_categorical(le.transform(y_val))
y_estimate2 = np.argmax(model.predict(x_val),axis=1)
y_true2 = np.argmax(y_val,axis=1)
print(classification_report(y_true2,y_estimate2))