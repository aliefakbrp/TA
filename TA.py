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
# df.head(10)

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
# scaled_features

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
# minmax

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(scaled_features,y,test_size=0.2,random_state=1)

st.set_page_config(page_title="Alief Akbar Purnama")
@st.cache()
def progress():
    with st.spinner("Bentar ya....."):
        time.sleep(1)
        
st.title("UAS PENDAT")

dataframe, preporcessing, modeling, implementation = st.tabs(
    ["Wine Quality Data", "Prepocessing", "Modeling", "Implementation"])

with dataframe:
    st.write('Data Wine Quality')
    dataset, ket = st.tabs(['Dataset', 'Ket Dataset'])
    with ket:
        st.write("""
                Data : https://www.kaggle.com/datasets/nareshbhat/wine-quality-binary-classification
                * Attribute Information:
                Input variables (based on physicochemical tests):
                * 1 - fixed acidity
                * 2 - volatile acidity
                * 3 - citric acid
                * 4 - residual sugar
                * 5 - chlorides
                * 6 - free sulfur dioxide
                * 7 - total sulfur dioxide
                * 8 - density
                * 9 - pH
                * 10 - sulphates
                * 11 - alcohol
                Output variable (based on sensory data):
                * 12 - quality 
                """)
    with dataset:
        st.dataframe(df)
        
        
        
with preporcessing:
    st.write('MinMax Scaler')
    st.dataframe(scaled_features)
    
    
with modeling:
    # pisahkan fitur dan label
    knn, nb, m3 = st.tabs(
        ["K-Nearest Neighbor","Metode 2", "Metode 3"])
    
    with knn:
        from sklearn.neighbors import KNeighborsClassifier
        knn = KNeighborsClassifier(n_neighbors=3)
        knn.fit(x_train,y_train)
        y_pred_knn = knn.predict(x_test) 
        accuracy_knn=round(accuracy_score(y_test,y_pred_knn)* 100, 2)
        acc_knn = round(knn.score(x_train, y_train) * 100, 2)
        label_knn = pd.DataFrame(
        data={'Label Test': y_test, 'Label Predict': y_pred_knn}).reset_index()
        st.success(f'Tingkat akurasi = {acc_knn}')
        st.dataframe(label_knn)
        
    with nb:
        # library for Naive Bayes Gaussian
        from sklearn.naive_bayes import GaussianNB
        from sklearn.metrics import accuracy_score
        from sklearn.metrics import make_scorer, accuracy_score,precision_score
        from sklearn.metrics import classification_report
        from sklearn.metrics import confusion_matrix
        from sklearn.metrics import precision_score,recall_score,f1_score
        from sklearn.preprocessing import LabelEncoder

        #Model Select
        from sklearn.model_selection import train_test_split
        from sklearn.naive_bayes import GaussianNB


        # classifier
        gaussian = GaussianNB()
        # 
        gaussian.fit(x_train, y_train)
        Y_pred = gaussian.predict(x_test) 
        accuracy_nb=round(accuracy_score(y_test,Y_pred)* 100, 2)
        acc_gaussian = round(gaussian.score(x_train, y_train) * 100, 2)

        cm = confusion_matrix(y_test, Y_pred)
        accuracy = accuracy_score(y_test,Y_pred)
        precision =precision_score(y_test, Y_pred,average='micro')
        recall =  recall_score(y_test, Y_pred,average='micro')
        f1 = f1_score(y_test,Y_pred,average='micro')
        print('Confusion matrix for Naive Bayes\n',cm)
        print('accuracy_Naive Bayes: %.3f' %accuracy)
        print('precision_Naive Bayes: %.3f' %precision)
        print('recall_Naive Bayes: %.3f' %recall)
        print('f1-score_random_Forest : %.3f' %f1)
        st.success(f'Tingkat akurasi = {accuracy}')


with implementation:
    
    # option = st.selectbox(
    #      'Pilih Jenis Model yang ingin dipakai',
    #      ('KNN', 'Home phone', 'Mobile phone'))
    # a = options
    # st.text('asu')
    fixedacidity=0
    fixedacidity=st.text_input('fixed acidity')
    

    volatileacidity=0
    volatileacidityt=st.text_input('volatile acidity')
    

    citricacid=0
    citricacid=st.text_input('citric acid')
    

    residualsugar=0
    residualsugar=st.text_input('residual sugar')
    

    chlorides=0
    chlorides=st.text_input('chlorides')
    

    freesulfurdioxide=0
    freesulfurdioxide=st.text_input('free sulfur dioxide')
    

    totalfurdioxide=0
    totalfurdioxide=st.text_input('total sulfur dioxide')
    

    density=0
    density=st.text_input('density')
    

    pH=0
    pH=st.text_input('pH')
    

    sulphates=0
    sulphates=st.text_input('sulphates')
    

    alcohol=0
    alcohol=st.text_input('alcohol')
    


    fixedacidity=float(fixedacidity)
    volatileacidityt=float(volatileacidityt)
    citricacid=float(citricacid)
    residualsugar=float(residualsugar)
    chlorides=float(chlorides)
    freesulfurdioxide=float(freesulfurdioxide)
    totalfurdioxide=float(totalfurdioxide)
    density=float(density)
    pH=float(pH)
    sulphates=float(sulphates)
    alcohol=float(alcohol)
      
    x_new = [[fixedacidity,	volatileacidity,	citricacid,	residualsugar,	chlorides,	freesulfurdioxide,	totalfurdioxide,	density,	pH,	sulphates,	alcohol]]
    maximal=0
    minimal=1
    for i in range(len(x_new[0])):
      x_new[0][i]=(x_new[0][i]-minmax[minimal])/(minmax[maximal]-minmax[minimal])
      maximal+=2
      minimal+=2
    x_new
    from sklearn.neighbors import KNeighborsClassifier 
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(x_train,y_train)
    Y_pred = knn.predict(x_test)
    y_predict = knn.predict(x_new)
    st.write("Hasil prediksi adalah",y_predict[0]) 

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


# input
# option = st.selectbox(
#     'How would you like to be contacted?',
#     ['KNN', 'Home phone', 'Mobile phone'])
# a=option
# b=st.write(a)
# b

# mc=option(x_new)
# st.write("Hasil prediksi adalah ",b(x_new))
