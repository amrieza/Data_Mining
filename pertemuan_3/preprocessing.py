import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('Data.csv')
x = dataset.iloc[:, :-1].values # Ambil semua kolom kecuali kolom terakhir
y = dataset.iloc[:, -1].values # Ambil kolom terakhir

print('x : \n', x)
print('y : \n', y)

import sklearn.impute
from sklearn import impute
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(missing_values=np.nan, strategy='mean') # Replace NaN dengan strategi Mean(Rata - Rata)
imputer.fit(x[:,1:3])
x[:,1:3] = imputer.transform(x[:,1:3])

print('Tanpa NaN : \n', x)

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder',OneHotEncoder(),[0])], remainder='passthrough') # Mengganti String dengan integer
x = np.array(ct.fit_transform(x))

print ('Enkoding : \n', x)

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder() # Mengganti Label 'YES' 'NO' dengan angka 0 atau 1.
y = le.fit_transform(y)

print ('LabelEncoder : \n', y)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 1)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train[:, 3:] = sc.fit_transform(x_train[:, 3:])
x_test[:, 3:] = sc.transform(x_test[:, 3:])

print('X_train : \n', x_train)
print('X_test : \n', x_test)