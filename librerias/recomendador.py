import surprise
from surprise import Dataset, accuracy, Reader
from surprise.model_selection import cross_validate
from sklearn.model_selection import train_test_split
import librerias.util as util


class Recomendador():
    """
    Clase que contiene los métodos necesarios para entrenar un modelo de recomendación basado en el 
    algoritmo de predicción de filtrado colaborativo NMF, y obtener las predicciones de dicho modelo.
    """
    def __init__(self):
        self.users_ids = {}
        self.games_ids= {}
        self.trainset = Dataset
        self.testset = Dataset
        self.W = surprise.prediction_algorithms.matrix_factorization.NMF
        self.porcentaje_test = float
        self.num_factores = int
        self.data = util.leer_dataset("Datasets/dataset.csv")
        pass
    
    ############################## MÉTODOS USADOS PARA EL ENTRENAMIENTO DEL MODELO ##############################
    def dividir_dataset(self, dataset, porcentaje_test):
        """ 
        Manualmente se divide el dataset usando la función train_test_split de sklearn
        """
        trainset, testset = train_test_split(dataset, test_size = porcentaje_test, random_state=7)
        self.porcentaje_test = porcentaje_test
        # se guardan en la clase para su uso posterior
        self.trainset = trainset
        self.testset = testset
        
        # diccionario para relacionar los ids de los usuarios con la fila que ocupan en la matriz p
        users = list(set(trainset['userId']))
        users_sort = sorted(users)
        self.users_ids = {users_sort[i]:i for i in range(len(users_sort))}
        
        # diccionario para relacionar los ids de los items con la columna que ocupan en la matriz q
        items = list(set(trainset['gameId']))
        items_sort = sorted(items)
        self.games_ids = {items_sort[i]:i for i in range(len(items_sort))}
        
        util.guardar_dataset(trainset, "Datasets/Trainset.csv")
        util.guardar_dataset(testset, "Datasets/Testset.csv")
        
        return trainset, testset
    
    def entrenamiento(self, num_factores):
        """
        Convertimos el dataframe trainset a la clase trainset de surprise para poder entrenar el modelo, y se entrena.
        """
        reader = Reader(line_format='user item rating', rating_scale=(1,10))
        train_data = Dataset.load_from_df(self.trainset[["userId", "gameId", "rating"]], reader).build_full_trainset()

        modelo = surprise.prediction_algorithms.matrix_factorization.NMF(n_factors=num_factores)
        W = modelo.fit(train_data)
        
        # se guardan las matrices p y q en csv
        util.guardar_matriz(W.pu, "Matrices PQ/TFG_matriz_p_{}".format(self.porcentaje_test))
        util.guardar_matriz(W.qi, "Matrices PQ/TFG_matriz_q_{}".format(self.porcentaje_test))

        self.W = W
        self.num_factores = num_factores

        return W
    
    def get_juegos_jugados(self, user):
        """ 
        Dado un usuario, devuelve los juegos del trainset, los que ha jugado dicho usuario.
        """
        return list(self.data[self.data['userId']==user].iloc[:,2])
    
    def get_rating(self, uid, iid):
        """
        Dado un id de usuario y uno de juego/item, se devuelve el rating estimado de la predicción.
        """

        rating = self.W.predict(uid, iid)

        return rating[3]
    
    def get_recomendados(self, uid, k):
        """
        Dado un usuario y un número k, se devuelven k juegos recomendados para ese usuario, que
        no ha jugado todavía.
        """
        # obtenemos los juegos que ha jugado el usuario
        games_played = self.get_juegos_jugados(uid)
        # quitamos los que ha jugado de la lista de juegos
        games = list(self.data['gameId'].unique())
        for game in games_played:
            if game in games:
                games.remove(game)
            
        # predicciones de esos juegos no jugados por el usuario
        predicciones = {}
        for game in games:
            prediccion = self.W.predict(uid, game)
            predicciones.update({game:prediccion[3]})
            
        # ordenamos las predicciones
        predicciones_ordenadas = sorted(predicciones.items(), key=lambda k:k[1], reverse=True)
        # elegimos las k primeras
        k_primeras = predicciones_ordenadas[:k]
        
        ids = []
        for prediccion in k_primeras:
            ids.append(prediccion[0])
            
        return ids
    
    def get_users(self):
        """
        Devuelve un diccionario con los ids de los usuarios y la posición que ocupan en la matriz p
        """
        return self.users_ids
    
    def get_games(self):
        """
        Devuelve un diccionario con los ids de los juegos y la posición que ocupan en la matriz q
        """
        return self.games_ids
    
    def get_porcentaje_test(self):
        """
        Devuelve el porcentaje de test que se ha usado para entrenar el modelo
        """
        return self.porcentaje_test
    
    ############################## MÉTODOS USADOS PARA LA CONFIGURACIÓN DEL MODELO ##############################
    def calcular_predicciones(self, W, df):
        """
        Dado un modelo entrenado W, y un dataframe con ratings, se calculan las predicciones 
        de dichos ratings.
        """
        # se obtiene una lista con todos los ratings del dataframe pasado
        ratings = [(df.iloc[i, 1],df.iloc[i, 2], df.iloc[i, 3]) for i in range(len(df.iloc[1:,:]))]
        predictions = []
        for (uid,iid, rui) in ratings:
            prediction = W.predict(uid, iid, rui)
            predictions.append(prediction)
            
        return predictions
    
    def calcular_mae(self, W, df):
        """
        Se calcula el error mae a partir de las predicciones pasado un dataframe con ratings.
        """
        # se obtienen las predicciones
        predictions = self.calcular_predicciones(W, df)
        
        return accuracy.mae(predictions)
    
    def calcular_mse(self, W, df):
        """
        Se calcula el error mae a partir de las predicciones pasado un dataframe con ratings.
        """
        # se obtienen las predicciones
        predictions = self.calcular_predicciones(W, df)
        
        return accuracy.mse(predictions)
    
    def calcular_rmse(self, W, df):
        """
        Se calcula el error rmse a partir de las predicciones pasado un dataframe.
        """
        # se obtienen las predicciones
        predictions = self.calcular_predicciones(W, df)
        
        return accuracy.rmse(predictions)

    def errores_train_test(self):
        """
        A partir de los conjuntos de entrenamiento y test, obtenemos los errores mae para un conjunto 
        de factores latentes, entre 1 y 40.
        """
        errores_test = []
        errores_train = []
        for i in range(1, 40):
            W = self.entrenamiento(i)
            errores_test.append(self.calcular_mae(W, self.testset))
            errores_train.append(self.calcular_mae(W, self.trainset))
        return errores_test, errores_train
    
    ############################## MÉTODOS PARA LA EVALUACIÓN DEL MODELO ##############################
    def get_validacion_cruzada(self, df):
        """
        Se obtiene la validación cruzada de 5 iteraciones para un dataframe pasado por argumento.
        """
        reader = Reader(line_format='user item rating', rating_scale=(1,10))
        data = Dataset.load_from_df(df[["userId", "gameId", "rating"]], reader)

        modelo = surprise.prediction_algorithms.matrix_factorization.NMF(n_factors=26)

        return cross_validate(modelo, data, measures=['RMSE', 'MAE', 'MSE'], cv = 5, verbose=True)
    
    def get_tabla_resultados(self):
        """
        Se obtienen los resultados de mae, rmse y mse para los conjuntos de test y train, para el porcentaje y número
        de factores del modelo.
        """
        _, test = self.dividir_dataset(self.data, self.porcentaje_test)
        W = self.entrenamiento(self.num_factores)
        rmse = self.calcular_rmse(W, test)
        mae = self.calcular_mae(W, test)
        mse = self.calcular_mse(W, test)
        return rmse, mae, mse