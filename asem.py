import sklearn
import streamlit as st
import pandas as pd 
import numpy as np 
import warnings
from sklearn.metrics import make_scorer, accuracy_score,precision_score
warnings.filterwarnings('ignore', category=UserWarning, append=True)

# data
data = 'https://raw.githubusercontent.com/aliefakbrp/dataset/main/wine.csv'
df = pd.read_csv(data)
df.head(10)

# pembeda data dan label
x = df.iloc[:, :-1]
y = df.loc[:, "quality"]
y = df['quality'].values

# split data
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=1)

#  normalisasi
scaler = sklearn.preprocessing.MinMaxScaler()
scaled = scaler.fit_transform(x)
features_names = x.columns.copy()
scaled_features = pd.DataFrame(scaled, columns=features_names)
scaled_features

# normalisasi inputan
minmax=[]
maxfa = max(x_train[:]["fixed acidity"])
minfa = min(x_train[:]["fixed acidity"])
maxva = max(x_train[:]["volatile acidity"])
minva = min(x_train[:]["volatile acidity"])
maxca = max(x_train[:]["citric acid"])
minca = min(x_train[:]["citric acid"])
maxrs = max(x_train[:]["residual sugar"])
minrs = min(x_train[:]["residual sugar"])
maxc = max(x_train[:]["chlorides"])
minc = min(x_train[:]["chlorides"])
maxfsd = max(x_train[:]["free sulfur dioxide"])
minfsd = min(x_train[:]["free sulfur dioxide"])
maxtsd = max(x_train[:]["total sulfur dioxide"])
mintsd = min(x_train[:]["total sulfur dioxide"])
maxd = max(x_train[:]["density"])
mind = min(x_train[:]["density"])
maxpH = max(x_train[:]["pH"])
minpH = min(x_train[:]["pH"])
maxs = max(x_train[:]["sulphates"])
mins = min(x_train[:]["sulphates"])
maxa = max(x_train[:]["alcohol"])
mina = min(x_train[:]["alcohol"])

minmax.append(maxfa)
minmax.append(minfa)
minmax.append(maxva)
minmax.append(minva)
minmax.append(maxca)
minmax.append(minca)
minmax.append(maxrs)
minmax.append(minrs)
minmax.append(maxc)
minmax.append(minc)
minmax.append(maxfsd)
minmax.append(minfsd)
minmax.append(maxtsd)
minmax.append(mintsd)
minmax.append(maxd)
minmax.append(mind)
minmax.append(maxpH)
minmax.append(minpH)
minmax.append(maxs)
minmax.append(mins)
minmax.append(maxa)
minmax.append(mina)
minmax

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(scaled_features,y,test_size=0.2,random_state=1)

# input
option = st.selectbox(
     'Pilih Jenis Model yang ingin dipakai',
     ('KNN', 'Home phone', 'Mobile phone'))
st.write('You selected:', option)


x_new = [[11.2,	0.28,	0.56,	1.9,	0.075,	17.0,	60.0,	0.99800,	3.16,	0.58,	9.8]]
maximal=0
minimal=1
for i in range(len(x_new[0])):
  x_new[0][i]=(x_new[0][i]-minmax[minimal])/(minmax[maximal]-minmax[minimal])
  maximal+=2
  minimal+=2
x_new

def KNN(x_new):
  from sklearn.neighbors import KNeighborsClassifier
  knn = KNeighborsClassifier(n_neighbors=3)
  knn.fit(x_train,y_train)
  Y_pred = knn.predict(x_test) 
  accuracy_knn=round(accuracy_score(y_test,Y_pred)* 100, 2)
  acc_knn = round(knn.score(x_train, y_train) * 100, 2)
  accuracy_knn
  acc_knn
  y_predict = knn.predict(x_new)
  return y_predict[0]

mc=option(x_new)
st.write("Hasil prediksi adalah ",mc)
