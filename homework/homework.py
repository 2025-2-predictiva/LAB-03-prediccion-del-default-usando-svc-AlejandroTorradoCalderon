# flake8: noqa: E501
#
# En este dataset se desea pronosticar el default (pago) del cliente el próximo
# mes a partir de 23 variables explicativas.
#
#   LIMIT_BAL: Monto del credito otorgado. Incluye el credito individual y el
#              credito familiar (suplementario).
#         SEX: Genero (1=male; 2=female).
#   EDUCATION: Educacion (0=N/A; 1=graduate school; 2=university; 3=high school; 4=others).
#    MARRIAGE: Estado civil (0=N/A; 1=married; 2=single; 3=others).
#         AGE: Edad (years).
#       PAY_0: Historia de pagos pasados. Estado del pago en septiembre, 2005.
#       PAY_2: Historia de pagos pasados. Estado del pago en agosto, 2005.
#       PAY_3: Historia de pagos pasados. Estado del pago en julio, 2005.
#       PAY_4: Historia de pagos pasados. Estado del pago en junio, 2005.
#       PAY_5: Historia de pagos pasados. Estado del pago en mayo, 2005.
#       PAY_6: Historia de pagos pasados. Estado del pago en abril, 2005.
#   BILL_AMT1: Historia de pagos pasados. Monto a pagar en septiembre, 2005.
#   BILL_AMT2: Historia de pagos pasados. Monto a pagar en agosto, 2005.
#   BILL_AMT3: Historia de pagos pasados. Monto a pagar en julio, 2005.
#   BILL_AMT4: Historia de pagos pasados. Monto a pagar en junio, 2005.
#   BILL_AMT5: Historia de pagos pasados. Monto a pagar en mayo, 2005.
#   BILL_AMT6: Historia de pagos pasados. Monto a pagar en abril, 2005.
#    PAY_AMT1: Historia de pagos pasados. Monto pagado en septiembre, 2005.
#    PAY_AMT2: Historia de pagos pasados. Monto pagado en agosto, 2005.
#    PAY_AMT3: Historia de pagos pasados. Monto pagado en julio, 2005.
#    PAY_AMT4: Historia de pagos pasados. Monto pagado en junio, 2005.
#    PAY_AMT5: Historia de pagos pasados. Monto pagado en mayo, 2005.
#    PAY_AMT6: Historia de pagos pasados. Monto pagado en abril, 2005.
#
# La variable "default payment next month" corresponde a la variable objetivo.
#
# El dataset ya se encuentra dividido en conjuntos de entrenamiento y prueba
# en la carpeta "files/input/".
#
# Los pasos que debe seguir para la construcción de un modelo de
# clasificación están descritos a continuación.
#
#
# Paso 1.
# Realice la limpieza de los datasets:
# - Renombre la columna "default payment next month" a "default".
# - Remueva la columna "ID".
# - Elimine los registros con informacion no disponible.
# - Para la columna EDUCATION, valores > 4 indican niveles superiores
#   de educación, agrupe estos valores en la categoría "others".
# - Renombre la columna "default payment next month" a "default"
# - Remueva la columna "ID".
#
#
# Paso 2.
# Divida los datasets en x_train, y_train, x_test, y_test.
#
#
# Paso 3.
# Cree un pipeline para el modelo de clasificación. Este pipeline debe
# contener las siguientes capas:
# - Transforma las variables categoricas usando el método
#   one-hot-encoding.
# - Descompone la matriz de entrada usando PCA. El PCA usa todas las componentes.
# - Estandariza la matriz de entrada.
# - Selecciona las K columnas mas relevantes de la matrix de entrada.
# - Ajusta una maquina de vectores de soporte (svm).
#
#
# Paso 4.
# Optimice los hiperparametros del pipeline usando validación cruzada.
# Use 10 splits para la validación cruzada. Use la función de precision
# balanceada para medir la precisión del modelo.
#
#
# Paso 5.
# Guarde el modelo (comprimido con gzip) como "files/models/model.pkl.gz".
# Recuerde que es posible guardar el modelo comprimido usanzo la libreria gzip.
#
#
# Paso 6.
# Calcule las metricas de precision, precision balanceada, recall,
# y f1-score para los conjuntos de entrenamiento y prueba.
# Guardelas en el archivo files/output/metrics.json. Cada fila
# del archivo es un diccionario con las metricas de un modelo.
# Este diccionario tiene un campo para indicar si es el conjunto
# de entrenamiento o prueba. Por ejemplo:
#
# {'dataset': 'train', 'precision': 0.8, 'balanced_accuracy': 0.7, 'recall': 0.9, 'f1_score': 0.85}
# {'dataset': 'test', 'precision': 0.7, 'balanced_accuracy': 0.6, 'recall': 0.8, 'f1_score': 0.75}
#
#
# Paso 7.
# Calcule las matrices de confusion para los conjuntos de entrenamiento y
# prueba. Guardelas en el archivo files/output/metrics.json. Cada fila
# del archivo es un diccionario con las metricas de un modelo.
# de entrenamiento o prueba. Por ejemplo:
#
# {'type': 'cm_matrix', 'dataset': 'train', 'true_0': {"predicted_0": 15562, "predicte_1": 666}, 'true_1': {"predicted_0": 3333, "predicted_1": 1444}}
# {'type': 'cm_matrix', 'dataset': 'test', 'true_0': {"predicted_0": 15562, "predicte_1": 650}, 'true_1': {"predicted_0": 2490, "predicted_1": 1420}}
#
import os
import gzip
import json
import pickle
import time
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    balanced_accuracy_score,
    confusion_matrix,
)

def load_datasets(train_path='files/input/train_data.csv.zip',
                  test_path='files/input/test_data.csv.zip'):
    """Carga los datasets de entrenamiento y prueba."""
    train_df = pd.read_csv(train_path, index_col=False)
    test_df = pd.read_csv(test_path, index_col=False)
    return train_df, test_df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Realiza limpieza y preprocesamiento básico en el dataset."""
    df = df.copy()
    df.rename(columns={"default payment next month": "default"}, inplace=True)
    df.drop(columns=["ID"], inplace=True)
    df = df[df["MARRIAGE"] != 0]
    df = df[df["EDUCATION"] != 0]
    df["EDUCATION"] = df["EDUCATION"].apply(lambda x: 4 if x > 4 else x)

    df.dropna(inplace=True)
    df.drop_duplicates(inplace=True)

    return df

def split_features_and_target(train_df, test_df, target_col="default"):
    """Separa variables independientes y objetivo en train y test."""
    x_train = train_df.drop(columns=[target_col])
    y_train = train_df[target_col]
    x_test = test_df.drop(columns=[target_col])
    y_test = test_df[target_col]
    return x_train, y_train, x_test, y_test


def create_train_test_split(x, y, test_size=0.25, random_state=42):
    """Divide el conjunto de datos en entrenamiento y validación."""
    x_train, x_val, y_train, y_val = train_test_split(
        x, y, test_size=test_size, random_state=random_state
    )

    print("Tamaños:")
    print("x_train:", x_train.shape)
    print("y_train:", y_train.shape)
    print("x_val:", x_val.shape)
    print("y_val:", y_val.shape)

    return x_train, x_val, y_train, y_val


def build_pipeline(categorical_features, numeric_features, estimator):
    """Crea un pipeline con preprocesamiento, PCA, selección de características y clasificador."""
    preprocessor = ColumnTransformer(
        transformers=[
            ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical_features),
            ("numeric", StandardScaler(), numeric_features),
        ],
        remainder='passthrough'
    )

    pipeline = Pipeline(steps=[
        ('preprocessing', preprocessor),
        ('pca', PCA()),
        ('feature_selection', SelectKBest(score_func=f_classif)),
        ('classifier', estimator)
    ])

    return pipeline


def perform_grid_search(pipeline, param_grid, cv, scoring, x_train, y_train):
    """Ejecuta una búsqueda de grilla para optimizar hiperparámetros."""
    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=cv,
        scoring=scoring,
        n_jobs=-1,
        verbose=1
    )
    grid_search.fit(x_train, y_train)
    return grid_search


def save_model(model, output_path="files/models/model.pkl.gz"):
    """Guarda el modelo entrenado en un archivo comprimido."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with gzip.open(output_path, "wb") as f:
        pickle.dump(model, f)
    print(f"Modelo guardado en: {output_path}")


def load_model(model_path):
    """Carga un modelo almacenado desde un archivo comprimido."""
    if not os.path.exists(model_path):
        return None
    with gzip.open(model_path, "rb") as f:
        return pickle.load(f)


def evaluate_model(model, x_train, y_train, x_test, y_test):
    """Calcula métricas de desempeño y matrices de confusión."""
    y_train_pred = model.predict(x_train)
    y_test_pred = model.predict(x_test)

    cm_train = confusion_matrix(y_train, y_train_pred)
    cm_test = confusion_matrix(y_test, y_test_pred)

    metrics = [
        {
            'type': 'metrics',
            'dataset': 'train',
            'precision': round(precision_score(y_train, y_train_pred, zero_division=0), 4),
            'balanced_accuracy': round(balanced_accuracy_score(y_train, y_train_pred), 4),
            'recall': round(recall_score(y_train, y_train_pred, zero_division=0), 4),
            'f1_score': round(f1_score(y_train, y_train_pred, zero_division=0), 4)
        },
        {
            'type': 'metrics',
            'dataset': 'test',
            'precision': round(precision_score(y_test, y_test_pred, zero_division=0), 4),
            'balanced_accuracy': round(balanced_accuracy_score(y_test, y_test_pred), 4),
            'recall': round(recall_score(y_test, y_test_pred, zero_division=0), 4),
            'f1_score': round(f1_score(y_test, y_test_pred, zero_division=0), 4)
        },
        {
            'type': 'cm_matrix',
            'dataset': 'train',
            'true_0': {'predicted_0': int(cm_train[0, 0]), 'predicted_1': int(cm_train[0, 1])},
            'true_1': {'predicted_0': int(cm_train[1, 0]), 'predicted_1': int(cm_train[1, 1])}
        },
        {
            'type': 'cm_matrix',
            'dataset': 'test',
            'true_0': {'predicted_0': int(cm_test[0, 0]), 'predicted_1': int(cm_test[0, 1])},
            'true_1': {'predicted_0': int(cm_test[1, 0]), 'predicted_1': int(cm_test[1, 1])}
        }
    ]
    return metrics


def save_metrics_to_json(metrics, output_path="files/output/metrics.json"):
    """Guarda las métricas calculadas en formato JSON."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for metric in metrics:
            f.write(json.dumps(metric, ensure_ascii=False))
            f.write('\n')


def main():
    start_time = time.time()

    # Cargar y limpiar datos
    train_df, test_df = load_datasets()
    train_df = clean_data(train_df)
    test_df = clean_data(test_df)

    x_train, y_train, x_test, y_test = split_features_and_target(train_df, test_df, "default")

    categorical_features = ["EDUCATION", "MARRIAGE", "SEX"]
    numeric_features = list(set(x_train.columns) - set(categorical_features))

    # Modelo base
    estimator = SVC(random_state=42)
    pipeline = build_pipeline(categorical_features, numeric_features, estimator)

    # Espacio de búsqueda
    param_grid = {
        "pca__n_components": [20, 21],
        "feature_selection__k": range(1, len(x_train.columns) + 1),
        "classifier__C": [0.8],
        "classifier__kernel": ["rbf"],
        "classifier__gamma": [0.099]
    }

    # Entrenar modelo
    model = perform_grid_search(
        pipeline,
        param_grid,
        cv=10,
        scoring="balanced_accuracy",
        x_train=x_train,
        y_train=y_train
    )

    # Guardar modelo y métricas
    save_model(model)
    metrics = evaluate_model(model, x_train, y_train, x_test, y_test)
    save_metrics_to_json(metrics)

    # Tiempo total
    duration = (time.time() - start_time) / 60
    print(f"Tiempo de ejecución: {duration:.2f} minutos")


if __name__ == "__main__":
    main()