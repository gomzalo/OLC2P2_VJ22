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

st.title("MACHINE LEARNING")

st.write("""
    # Elegir archivo
    This is a simple linear regression demo.
    It shows how to use the Streamlit framework to build a simple app.
    """)

#   ::::::::::::::::::::::   File upload    ::::::::::::::::::::::

uploaded_file = st.file_uploader("Elige el archivo a analizar", type=["csv", "xls", "xlsx", "json"])

if uploaded_file is not None:
    st.write("Subiste un archivo: ", uploaded_file.type)
    if uploaded_file.type == "text/csv":
        data_frame = pd.read_csv(uploaded_file)
        st.write(data_frame)
    elif uploaded_file.type == "application/json":
        data_frame = pd.read_json(uploaded_file)
        st.write(data_frame)
    elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet" or uploaded_file.type == "application/vnd.ms-excel":
        data_frame = pd.read_excel(uploaded_file)
        st.write(data_frame)
    else:
        st.write("Archivo no soportado")
        data_frame = None

    if data_frame is not None:
        row_0 = data_frame.head(0)
        st.write("El archivo tiene: ")
        st.write(row_0)

    dataset_name = st.sidebar.selectbox("Select Dataset", ("Iris", "Breast Cancer", "Wine Dataset"))
    # st.write(dataset_name)

    classifier_name = st.sidebar.selectbox("Select Classifier", ("KNN", "SVM", "Random Forest", "Linear Regression"))


    def get_dataset(dataset_name):
        if dataset_name == "Iris":
            data = datasets.load_iris()
        elif dataset_name == "Breast Cancer":
            data = datasets.load_breast_cancer()
        elif dataset_name == "Wine Dataset":
            data = datasets.load_wine()
        else:
            data = data_frame
        X = data.data
        y = data.target
        return X, y


    X, y = get_dataset(dataset_name)
    st.write("Shape of dataset:", X.shape)
    st.write("Number of classes:", len(np.unique(y)))


    def add_parameter_ui(clf_name):
        params = dict()
        if clf_name == "KNN":
            K = st.sidebar.slider("K", 1, 15)
            params["K"] = K
        elif clf_name == "SVM":
            C = st.sidebar.slider("C", 0.01, 10.0)
            params["C"] = C
        else:
            max_depth = st.sidebar.slider("max_depth", 2, 15)
            n_estimators = st.sidebar.slider("n_estimators", 1, 100)
            params["max_depth"] = max_depth
            params["n_estimators"] = n_estimators
        return params


    params = add_parameter_ui(classifier_name)


    def get_classifier(clf_name, params):
        if clf_name == "KNN":
            clf = KNeighborsClassifier(n_neighbors=params["K"])
        elif clf_name == "SVM":
            clf = SVC(C=params["C"])
        else:
            clf = RandomForestClassifier(n_estimators=params["n_estimators"], max_depth=params["max_depth"], random_state=1234)
        return clf


    clf = get_classifier(classifier_name, params)

    # Classification
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    st.write(f"classifier = {classifier_name}")
    st.write(f"accuracy = {acc}")

    # PLOT
    pca = PCA(2)
    X_projected = pca.fit_transform(X)

    x1 = X_projected[:, 0]
    x2 = X_projected[:, 1]

    fig = plt.figure(figsize=(8, 8))
    plt.scatter(x1, x2, c=y, alpha=0.8, cmap="rainbow")
    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")
    plt.title(f"{dataset_name} - {classifier_name}")
    plt.colorbar()
    st.set_option('deprecation.showPyplotGlobalUse', False)
    # plt.show()
    st.pyplot()
