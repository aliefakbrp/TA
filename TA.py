import sklearn
import joblib
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


#  normalisasi
scaler = sklearn.preprocessing.MinMaxScaler()
scaled = scaler.fit_transform(x)
features_names = x.columns.copy()
scaled_features = pd.DataFrame(scaled, columns=features_names)
# scaled_features

# splitdata
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(scaled_features,y,test_size=0.2,random_state=1)

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



st.set_page_config(page_title="Alief Akbar Purnama")
@st.cache()
def progress():
    with st.spinner("Bentar ya....."):
        time.sleep(1)
        
st.title("UAS PENDAT")
st.write("Disini kita akan memprediksi apakah suatu anggur memupunyai kualitas yang baik atau tidak dengan menggunakan 3 metode yaitu K-NN, Naive Bayes, dan Decision Tree. Dari ketiga metode Decision Tree mempunyai tingkat akurasi yang lebih tinggi daripada 2 metode lainnya yaitu sebesar 78,12%.")

dataframe, preporcessing, modeling, implementation = st.tabs(
    ["Wine Quality Data", "Prepocessing", "Modeling", "Implementation"])

with dataframe:
    st.write('Data Wine Quality ')
    dataset, ket = st.tabs(['Dataset', 'Ket Dataset'])
    with ket:
        st.write("""
                Data : https://www.kaggle.com/datasets/nareshbhat/wine-quality-binary-classification
                
                The input data is
                * fixed acidity : sebagian besar asam terlibat dengan anggur atau tetap atau tidak mudah menguap (tidak mudah menguap)
                * volatile acidity : jumlah asam asetat dalam anggur, yang pada kadar yang terlalu tinggi dapat menyebabkan rasa cuka yang tidak enak
                * citric acid : ditemukan dalam jumlah kecil, asam sitrat dapat menambah 'kesegaran' dan rasa anggur
                * residual sugar : jumlah gula yang tersisa setelah fermentasi berhenti, jarang ditemukan wine dengan berat kurang dari 1 gram/liter dan wine dengan berat lebih dari 45 gram/liter dianggap manis
                * chlorides : jumlah garam dalam anggur
                * free sulfur dioxide : bentuk bebas SO2 ada dalam kesetimbangan antara molekul SO2 (sebagai gas terlarut) dan ion bisulfit;  itu mencegah pertumbuhan mikroba dan oksidasi anggur
                * total sulfur dioxide : jumlah bentuk bebas dan terikat dari S02;  dalam konsentrasi rendah, SO2 sebagian besar tidak terdeteksi dalam anggur, tetapi pada konsentrasi SO2 bebas lebih dari 50 ppm, SO2 menjadi jelas di hidung dan rasa anggur
                * density : kerapatan air mendekati kerapatan air tergantung pada persen alkohol dan kandungan gula
                * pH : menjelaskan seberapa asam atau basa anggur dalam skala dari 0 (sangat asam) hingga 14 (sangat basa);  kebanyakan anggur antara 3-4 pada skala pH
                * sulphates : aditif anggur yang dapat berkontribusi pada kadar gas sulfur dioksida (S02), yang bertindak sebagai antimikroba dan antioksidan
                * alcohol : persen kandungan alkohol anggur
                Output variable (based on sensory data):
                * quality 
                """)
    with dataset:
        st.dataframe(df)
        
        
        
with preporcessing:
    st.write('MinMax Scaler')
    st.dataframe(scaled_features)
    
    
with modeling:
    # pisahkan fitur dan label
    knn, nb, pk = st.tabs(
        ["K-Nearest Neighbor","Naive Bayes", "Decision Tree"])
    
    with knn:        
        knn = joblib.load('knn.pkl')
        y_pred_knn = knn.predict(x_test) 
        accuracy_knn=round(accuracy_score(y_test,y_pred_knn)* 100, 2)
        acc_knn = round(knn.score(x_train, y_train) * 100, 2)
        label_knn = pd.DataFrame(
        data={'Label Test': y_test, 'Label Predict': y_pred_knn})
        st.success(f'Level of Accuracy = {acc_knn}%')
        label_knn
        
    with nb:
        # library for Naive Bayes Gaussian
        from sklearn.naive_bayes import GaussianNB
        from sklearn.metrics import accuracy_score
        from sklearn.metrics import make_scorer, accuracy_score,precision_score
        from sklearn.metrics import classification_report
        from sklearn.metrics import precision_score,recall_score,f1_score
        from sklearn.preprocessing import LabelEncoder

        #Model Select
        from sklearn.model_selection import train_test_split
        from sklearn.naive_bayes import GaussianNB

        gaussian = joblib.load('gaussian.pkl')
        y_pred_nb = gaussian.predict(x_test)
        label_nb = pd.DataFrame(
        data={'Label Test': y_test, 'Label Predict': y_pred_nb})
        accuracy_nb=round(accuracy_score(y_test,y_pred_nb)* 100, 2)
        acc_gaussian = round(gaussian.score(x_train, y_train) * 100, 2)
        accuracy = accuracy_score(y_test,y_pred_nb)
        st.success(f'Level of Accuracy = {accuracy*100}%')
        label_nb

        
    with pk:
        from sklearn.tree import DecisionTreeClassifier, export_graphviz
        d3 = joblib.load('d3.pkl')
        y_predic = d3.predict(x_test)
        data_predic = pd.DataFrame(
        data={'Label Test': y_test, 'Label Predict': y_predic})
        from sklearn.metrics import accuracy_score
        a=f'Level of Accuracy = {"{:,.2f}".format(accuracy_score(y_test, y_predic)*100)}%'
        st.success(a)
        data_predic
        
        
        
        
with implementation:
    fixedacidity=0
    fixedacidity=st.text_input('fixed acidity',value=0)
    

    volatileacidity=0
    volatileacidityt=st.text_input('volatile acidity',value=0)
    

    citricacid=0
    citricacid=st.text_input('citric acid',value=0)
    

    residualsugar=0
    residualsugar=st.text_input('residual sugar',value=0)
    

    chlorides=0
    chlorides=st.text_input('chlorides',value=0)
    

    freesulfurdioxide=0
    freesulfurdioxide=st.text_input('free sulfur dioxide',value=0)
    

    totalfurdioxide=0
    totalfurdioxide=st.text_input('total sulfur dioxide',value=0)
    

    density=0
    density=st.text_input('density',value=0)
    

    pH=0
    pH=st.text_input('pH',value=0)
    

    sulphates=0
    sulphates=st.text_input('sulphates',value=0)
    

    alcohol=0
    alcohol=st.text_input('alcohol',value=0)
    
    x_new = [[float(fixedacidity),	float(volatileacidity), float(citricacid), float(residualsugar),	float(chlorides),	float(freesulfurdioxide),	float(totalfurdioxide), float(density),	float(pH),	float(sulphates),	float(alcohol)]]

    maximal=0
    minimal=1
    for i in range(len(x_new[0])):
      x_new[0][i]=(x_new[0][i]-minmax[minimal])/(minmax[maximal]-minmax[minimal])
      maximal+=2
      minimal+=2
#     x_new
    from sklearn.neighbors import KNeighborsClassifier 
    d3 = joblib.load('d3.pkl')
    y_predict = d3.predict(x_new)
    hasil = 'The prediction result of wine quality is '+y_predict[0]
    if st.button("Predict"):
        st.success(hasil) 
