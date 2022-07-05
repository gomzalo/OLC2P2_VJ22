import numpy as np
import pandas as pd
import streamlit as st
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

st.set_option('deprecation.showPyplotGlobalUse', False)

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

def get_algoritmo(clf_name, headers, data):
    if clf_name == "Regresión lineal":
        x_selected = st.selectbox("Elige la columna para el eje X", headers)
        y_selected = st.selectbox("Elige la columna para el eje y", headers)
        pred = st.text_input("Ingresa el valor a predecir")
        linear_regresion(x_selected, y_selected, int(pred), data)
    elif clf_name == "Regresión polinomial":
        x_selected = st.selectbox("Elige la columna para el eje X", headers)
        y_selected = st.selectbox("Elige la columna para el eje y", headers)
        pred = st.text_input("Ingresa el valor a predecir")
        # array from 0 to 25
        grado = st.slider("grado", 1, 20)
        polinomial_reg(x_selected, y_selected, grado, int(pred), data)
    else:
        st.write("xd?")


#   :::::::::::::::::::      Algoritmos      :::::::::::::::::::

# |||||||||||||     Regresión lineal    |||||||||||||

def linear_regresion(x_col, y_col, pred, data):
    X = np.asarray(data[x_col]).reshape(-1, 1)
    Y = data[y_col]
    # Linear
    linear_regression = LinearRegression()
    linear_regression.fit(X, Y)
    Y_pred = linear_regression.predict(X)

    # st.write(Y_pred)
    st.write("Error medio: ", mean_squared_error(Y, Y_pred, squared=True))
    st.write("Coef: ", linear_regression.coef_)
    st.write("R2: ", r2_score(Y, Y_pred))

    Y_new = linear_regression.predict([[pred]])
    st.write("Prediccion: ", Y_new)

    plt.scatter(X, Y)
    plt.colorbar()
    plt.plot(X, Y_pred, color='red')
    st.pyplot()


# |||||||||||||     Regresión polinomial    |||||||||||||

def polinomial_reg(x_col, y_col, grado, pred, data):
    X = np.asarray(data[x_col]).reshape(-1, 1)
    Y = data[y_col]
    # Polynomial
    poly = PolynomialFeatures(degree=grado)
    X_trans = poly.fit_transform(X)
    # Linear
    linear_regression = LinearRegression()
    linear_regression.fit(X_trans, Y)
    Y_pred = linear_regression.predict(X_trans)

    # st.write(Y_pred)
    st.write("Error medio: ", np.sqrt(mean_squared_error(Y, Y_pred, squared=False)))
    st.write("R2: ", r2_score(Y, Y_pred))

    Y_new = linear_regression.predict(poly.fit_transform([[int(pred)]]))
    st.write("Predicción:", Y_new)

    plt.scatter(X, Y)
    plt.plot(X, Y_pred, color='purple')
    st.pyplot()


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
        # data_frame = data_frame.replace('', np.nan, regex=True)
        # data_frame = data_frame.fillna(0)
        headers = data_frame.columns
        # --------      Eligiendo el algoritmo      --------
        classifier_name = st.sidebar.selectbox("Elige el algoritmo", ("Regresión lineal", "Regresión polinomial"))

        st.write(""" ## Parametrización """)
        get_algoritmo(classifier_name, headers, data_frame)
else:
    st.write("Archivo invalido")
