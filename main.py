import numpy as np
import pandas as pd
import streamlit as st
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# :::::::::::::::::::      Header      :::::::::::::::::::

st.set_page_config(
    page_title="ML Compi 2",
    page_icon=":dog:",
    # layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/gomzalo/OLC2P2_VJ22',
        'Report a bug': 'https://streamlit.io',
        'About': """
                    # MACHINE LEARNING
                    ## OLC 2 - Junio 2022
                    ### Gonzalo García - 201318652
                    Universidad de San Carlos de Guatemala
                """
    }
)

# :::::::::::::::::::      Intro      :::::::::::::::::::

st.title("MACHINE LEARNING")

st.write("""
    Esta app te permite analizar un set de datos, siguiendo los siguientes pasos:
    - Sube un archivo.
    - Parametriza el archivo.
    - Selecciona un algoritmo.
    """)


# :::::::::::::::::::      Functions      :::::::::::::::::::

def add_parameter_ui(clf_name, headers):
    params = dict()
    if clf_name == "KNN":
        K = st.sidebar.slider("K", 1, 15)
        params["K"] = K
    elif clf_name == "SVM":
        C = st.sidebar.slider("C", 0.01, 10.0)
        params["C"] = C
    elif clf_name == "Regresión lineal":
        # st.write(headers.iloc[1].tolist())
        params["x"] = st.selectbox("Elige la columna para el eje X", headers)
        params["y"] = st.selectbox("Elige la columna para el eje y", headers)
    else:
        max_depth = st.sidebar.slider("max_depth", 2, 15)
        n_estimators = st.sidebar.slider("n_estimators", 1, 100)
        params["max_depth"] = max_depth
        params["n_estimators"] = n_estimators
    return params


def get_classifier(clf_name, params, data):
    if clf_name == "Regresión lineal":
        x_selected = params["x"]
        y_selected = params["y"]
        linear_regresion(x_selected, y_selected, data)
    elif clf_name == "KNN":
        KNeighborsClassifier(n_neighbors=params["K"])
    elif clf_name == "SVM":
        SVC(C=params["C"])
    else:
        RandomForestClassifier(n_estimators=params["n_estimators"], max_depth=params["max_depth"], random_state=1234)


#   :::::::::::::::::::      Algoritmos      :::::::::::::::::::

# |||||||||||||     Regresión lineal    |||||||||||||

def linear_regresion(x_col, y_col, data):
    X = np.asarray(data[x_col]).reshape(-1, 1)
    Y = data[y_col]

    linear_regression = LinearRegression()
    linear_regression.fit(X, Y)
    Y_pred = linear_regression.predict(X)

    # st.write(Y_pred)
    st.write("Error medio: ", mean_squared_error(Y, Y_pred, squared=True))
    st.write("Coef: ", linear_regression.coef_)
    st.write("R2: ", r2_score(Y, Y_pred))

    Y_new = linear_regression.predict([[2050]])
    st.write("Prediccion: ", Y_new)

    plt.scatter(X, Y)
    plt.colorbar()
    plt.plot(X, Y_pred, color='red')
    st.pyplot()
    # plt.show()


#   ::::::::::::::::::::::   File upload    ::::::::::::::::::::::

st.write(""" ## Subir archivo """)

uploaded_file = st.file_uploader("Elige el archivo a analizar", type=["csv", "xls", "xlsx", "json"])

if uploaded_file is not None:
    st.write("Subiste un archivo: ", uploaded_file.type)
    st.write("*Previsualizando el archivo*")
    if uploaded_file.type == "text/csv":
        data_frame = pd.read_csv(uploaded_file)
        st.write(data_frame.head())
    elif uploaded_file.type == "application/json":
        data_frame = pd.read_json(uploaded_file)
        st.write(data_frame.head())
    elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet" or uploaded_file.type == "application/vnd.ms-excel":
        data_frame = pd.read_excel(uploaded_file)
        st.write(data_frame.head())
    else:
        st.write("Archivo no soportado")
        data_frame = None

    if data_frame is not None:
        headers = data_frame.columns
        # st.write("Cabeceras: ")
        # st.write(headers)

        # dataset_name = st.sidebar.selectbox("Select Dataset", ("Iris", "Breast Cancer", "Wine Dataset"))
        # st.write(dataset_name)

        # --------      Eligiendo el algoritmo      --------
        classifier_name = st.sidebar.selectbox("Elige el algoritmo",
                                               ("KNN", "SVM", "Random Forest", "Regresión lineal"))

        st.write(""" ## Parametrización """)
        params = add_parameter_ui(classifier_name, headers)

        get_classifier(classifier_name, params, data_frame)

        # X, y = get_dataset(data_frame)
        # st.write("Shape of dataset:", X.shape)
        # st.write("Number of classes:", len(np.unique(y)))
        #
        # # --------      Classification      --------
        # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)
        # clf.fit(X_train, y_train)
        # y_pred = clf.predict(X_test)
        #
        # acc = accuracy_score(y_test, y_pred)
        # st.write(f"classifier = {classifier_name}")
        # st.write(f"accuracy = {acc}")
        #
        # # --------      PLOT        --------
        # pca = PCA(2)
        # X_projected = pca.fit_transform(X)
        #
        # x1 = X_projected[:, 0]
        # x2 = X_projected[:, 1]
        #
        # fig = plt.figure(figsize=(8, 8))
        # plt.scatter(x1, x2, c=y, alpha=0.8, cmap="rainbow")
        # plt.xlabel("PCA 1")
        # plt.ylabel("PCA 2")
        # # plt.title(f"{dataset_name} - {classifier_name}")
        # plt.colorbar()
        # st.set_option('deprecation.showPyplotGlobalUse', False)
        # # plt.show()
        # st.pyplot()
else:
    st.write("Archivo invalido")
