import pandas as pd
from surprise import Dataset, Reader

def guardar_dataset(df, nombre):
    """
    Guarda el dataset pasado por argumento en un csv con el nombre pasado por argumento.
    """
    df.to_csv(nombre)

    return

def leer_dataset(path):
    """
    Se devuelve el dataset guardado en el csv con el nombre pasado por argumento.
    """
    with open(path, 'rb') as f:
        dataset = pd.read_csv(f)

    return dataset


def leer_dataset_as_folds(path):
    """
    Se devuelve el dataset guardado en el csv, con el nombre pasado por argumento, en una clase dataset.
    """
    with open(path, 'rb') as f:
        data_peq = pd.read_csv(f)

    reader = Reader(rating_scale=(1, 10))
    dataset = Dataset.load_from_df(
        data_peq[["userId", "gameId", "rating"]], reader)

    return dataset

def guardar_matriz(matriz, nombre):
    """
    Guarda la matriz pasada por argumento en un csv con el nombre pasado por argumento.
    """
    filas = [matriz[i] for i in range(len(matriz))]
    data_export = pd.DataFrame(filas)

    data_export.to_csv(nombre)

    return

def leer_matriz(nombre):
    """
    Se devuelve la matriz guardada en el csv.
    """

    with open(nombre, 'rb') as f:
        matriz = pd.read_csv(f)

    return matriz

def get_key(dictionary, val):
    """
    Se devuelve la clave del diccionario pasada por argumento que tiene el valor pasado por argumento.
    """
    for key, value in dictionary.items():
        if val == value:
            return key

    return "No existe esa clave"
