from sklearn.metrics import pairwise
import pandas as pd
import random as random
import librerias.util as util
import librerias.recomendador as Rec
import cv2
import matplotlib.pyplot as plt

class Explicaciones():
    """
    Clase que contiene los métodos necesarios para obtener las explicaciones del sistema de recomendación.
    """
    def __init__(self, nmf):
        self.nmf = nmf
        self.porcentaje_test = nmf.porcentaje_test
        self.users_ids = nmf.get_users()
        self.games_ids = nmf.get_games()
        self.data = util.leer_dataset('BGGrec-main/data/bgg_items.csv')
        self.designers = util.leer_dataset('BGGrec-main/data/bgg_items_designers.csv')
        self.categories = util.leer_dataset('BGGrec-main/data/bgg_items_categories.csv')
        self.types = util.leer_dataset('BGGrec-main/data/bgg_items_types.csv')
        self.mechanisms = util.leer_dataset('BGGrec-main/data/bgg_items_mechanisms.csv')
        pass
    
    def get_qu(self, user_id):
        """ 
        Se calcula la matriz Q_u para un usuario dado y se guarda en un fichero.
        """
        # comprobamos que el usuario está en el trainset, sino no se puede obtener su matriz qu
        if user_id not in self.users_ids.keys():
            print("Usuario no valido ")
            return
        
        # se obtienen las matrices p y q
        matriz_p = util.leer_matriz("Matrices PQ/TFG_matriz_p_{}".format(self.porcentaje_test))
        matriz_q = util.leer_matriz("Matrices PQ/TFG_matriz_q_{}".format(self.porcentaje_test))
        
        # se obtiene la fila del usuario en la matriz p
        user = self.users_ids[user_id]
        
        # se obtiene el vector del usuario
        vector_u = matriz_p.iloc[user, 1:]
        # inicializamos la matriz qu a 0
        Q_u = [[0 for i in range(len(matriz_q))] for i in range(len(vector_u))]

        # se calcula la matriz qu
        for columna in range(len(vector_u)):
            for fila in range(len(matriz_q)):
                Q_u[columna][fila] = vector_u[columna]*matriz_q.iloc[fila, columna+1]

        util.guardar_matriz(Q_u, "Matrices explicaciones/Q_{}".format(user_id))
        return Q_u
    
    def get_similitudes(self, juego, user):
        """
        Se obtienen las similitudes entre el item recomendado (juego, user) y los juegos jugados por el usuario,
        usando la métrica de similitud del coseno.
        """
        # se obtienen la matriz qu del usuario y los ids de juegos del trainset jugados por el usuario
        matriz_qu = util.leer_matriz("Matrices explicaciones/Q_"+str(user))
        games_played_ids = self.nmf.get_juegos_jugados(user)
        
        # para poder encontrar estos juegos en la matriz qu, se obtienen las posiciones
        games_played_values = [self.games_ids[id] for id in games_played_ids]
        
        # se obtiene la posición del juego recomendado en la matriz qu
        juego_value = self.games_ids[juego]
        similitudes = {}
        # se obtiene la fila del juego recomendado en qu
        vector = matriz_qu.iloc[:, juego_value]
        for i in range(len(matriz_qu.iloc[0, :])):
            # si el juego es el propio recomendado o no ha sido jugado por el usuario, lo saltamos
            if (i == juego_value) or (i not in games_played_values):
                continue
            # en otro caso, se obtiene su fila en qu
            vector2 = matriz_qu.iloc[:, i]
            # se calcula la similitud entre ambas filas y se guarda en un diccionario con 
            # la clave el id del juego con valor i.
            similitud = pairwise.cosine_similarity([vector, vector2])
            similitudes.update({util.get_key(self.games_ids, i): similitud[0][1]})
        return similitudes
    
    def get_k_expl(self, juego, user, k):
        """
        Se obtienen k explicaciones para el item recomendado (juego, user).
        """
        
        # se comprueba que el juego está en el trainset
        if juego not in self.games_ids.keys():
            print("Elige uno de estos juegos: ")
            print(self.games_ids.keys())
            return
        
        # se obtienen las similitudes entre el item recomendado y los juegos jugados por el usuario
        similitudes = self.get_similitudes(juego, user)
        # se ordenan de mayor a menor para elegir después los k primeros
        similitudes_ordenados = sorted(similitudes.items(), key=lambda k:k[1], reverse=True)
        return similitudes_ordenados[:k]
    
    def get_expl_limite(self, juego, user, limite):
        """
        Se obtienen las explicaciones que tienen una similitud a partir del limite pasado con el item 
        recomendado (juego, user).
        """
        
        # se comprueba que el juego está en el trainset
        if juego not in self.games_ids.keys():
            print("Elige uno de estos juegos: ")
            print(self.games_ids.keys())
            return
        
        # se obtienen las similitudes entre el item recomendado y los juegos jugados por el usuario
        similitudes = self.get_similitudes(juego, user)
        # se filtra eligiendo solo los que cuya similitud sea mayor o igual al "limite" pasado por argumento
        similitudes_filtrados = dict(filter(lambda k:k[1] >= limite, similitudes.items()))
        return sorted(similitudes_filtrados.items(), key=lambda k:k[1], reverse=True)
    
    def explain_age_best_players(self, iid, items_similares, number_show):
        """
        Se obtienen las explicaciones para el item recomendado (iid) en función de la edad y el número de jugadores.
        """
        # se obtiene la información del item recomendado
        info_recommended = self.data[self.data["id"]==iid]
        info_items = []
        i = 0

        # se obtiene la información de los items similares y se guardan los ids de los que tengan la misma edad 
        # y el mismo número de jugadores
        similar_age = []
        similar_number_players = []
        for item in items_similares:
            value = self.data[self.data["id"]==item[0]]
            info_items += [value]
            r_players = info_items[i].iloc[0, 8]
            age = info_items[i].iloc[0, 10]
            
            if age == info_recommended.iloc[0, 10]:
                similar_age.append(item[0])
            if r_players == info_recommended.iloc[0, 8]:
                similar_number_players.append(item[0])
            i += 1

        # si hay más de "number_show" items similares, se eligen los "number_show" primeros
        if similar_age:
            if len(similar_age) > number_show:
                lista = similar_age[:number_show]
                similar_age = lista

        if similar_number_players:
            if len(similar_number_players) > number_show:
                lista = similar_number_players[:number_show]
                similar_number_players = lista

        nombres_juegos_age = []
        edad = info_recommended.iloc[0,10]
        nombres_juegos_players = []
        jugadores= info_recommended.iloc[0,6]
        if similar_age:
            i = 1
            for item in similar_age:
                value = self.data[self.data["id"]==item]
                nombres_juegos_age.append(value.iloc[0,1])
                i +=1

        if similar_number_players:
            i = 1
            for item in similar_number_players:
                value = self.data[self.data["id"]==item]
                nombres_juegos_players.append(value.iloc[0,1])
                i +=1

        return nombres_juegos_age, nombres_juegos_players, edad, jugadores
    
    def explain_dificultad(self, iid, uid, items_similares, number_show):
        """
        Se obtienen las explicaciones para el item recomendado (iid) en función de la dificultad.
        """
        info_recommended = self.data[self.data["id"]==iid]
        info_items = []
        dificiles = {"dificiles":[]}
        faciles = {"faciles":[]}
        igual = {"iguales":[]}
        i = 0

        # se obtiene la información de los items similares y se guardan los ids en la lista correspondiente
        # dependiendo de si son más fáciles, más difíciles o iguales que el recomendado
        for item in items_similares:
            value = self.data[self.data["id"]==item[0]]
            info_items += [value]
            aweight = info_items[i].iloc[0, 11]
            rating = self.nmf.get_rating(uid, item[0])
            if (aweight <= info_recommended.iloc[0, 11] + 0.5) and (aweight >= info_recommended.iloc[0, 11]-0.5):
                igual["iguales"].append({"id": item[0], "aweight":aweight, "nombre":value.iloc[0,1], "rating":rating})
            elif info_recommended.iloc[0, 11] - 0.15 > aweight:
                faciles["faciles"].append({"id": item[0], "aweight":aweight, "nombre":value.iloc[0,1], "rating":rating})
            else:
                dificiles["dificiles"].append({"id": item[0], "aweight":aweight, "nombre":value.iloc[0,1], "rating":rating})
            i += 1

        # si hay más de "number_show" items similares, se eligen los "number_show" primeros
        if faciles["faciles"]:
            if len(faciles["faciles"]) > number_show:
                lista = faciles["faciles"][:number_show]
                faciles["faciles"] = lista
            
        if dificiles["dificiles"]:
            if len(dificiles["dificiles"]) > number_show:
                lista = dificiles["dificiles"][:number_show]
                dificiles["dificiles"] = lista
            
        if igual["iguales"]:
            if len(igual["iguales"]) > number_show:
                lista = igual["iguales"][:number_show]
                igual["iguales"] = lista

        return faciles, dificiles, igual, info_recommended 
    
    def explain_mecanicas(self, iid, items_similares, number_show):
        """
        Se obtienen las explicaciones para el item recomendado (iid) en función de los tipos, categorías y mecánicas.
        """
        # tipos
        info_recommended_tipo = self.types[self.types["gameId"]==iid]
        items_tipo = {}
        i = 0

        # se obtiene la información de los items similares y se guardan los tipos en una lista
        # y los nombres de los juegos en un diccionario, donde la clave es el tipo
        for item in items_similares:
            value = self.types[self.types["gameId"]==item[0]]
            name = self.data[self.data["id"]==item[0]]
            tipos = value.iloc[0, 1]
            # como pueden ser de varios tipos, se mira cada tipo por separado 
            for tipo in tipos.split("|"):
                # si el tipo está en el recomendado, se guarda el nombre del juego en el diccionario
                if tipo in info_recommended_tipo.iloc[0, 1]:
                    if tipo not in items_tipo.keys():
                        items_tipo[tipo] = [name.iloc[0,1]]
                    else:
                        items_tipo[tipo].append(name.iloc[0,1])
            i += 1
            
        # categoria
        info_recommended_categ = self.categories[self.categories["gameId"]==iid]
        items_categ = {}
        i = 0

        # se obtiene la información de los items similares y se guardan las categorías en una lista
        # y los nombres de los juegos en un diccionario, donde la clave es la categoría
        for item in items_similares:
            value = self.categories[self.categories["gameId"]==item[0]]
            name = self.data[self.data["id"]==item[0]]
            categs = value.iloc[0, 1]
            # como pueden ser de varias categorías, se mira cada categoría por separado
            for categ in categs.split("|"):
                # si la categoría está en el recomendado, se guarda el nombre del juego en el diccionario
                if categ in info_recommended_categ.iloc[0, 1]:
                    if categ not in items_categ.keys():
                        items_categ[categ] = [name.iloc[0,1]]
                    else:
                        items_categ[categ].append(name.iloc[0,1])
            i += 1
            
        # mechanisms
        info_recommended_mechanisms = self.mechanisms[self.mechanisms["gameId"]==iid]
        items_mechanism = {}
        i = 0

        # se obtiene la información de los items similares y se guardan las mecánicas en una lista
        # y los nombres de los juegos en un diccionario, donde la clave es la mecánica
        for item in items_similares:
            value = self.mechanisms[self.mechanisms["gameId"]==item[0]]
            name = self.data[self.data["id"]==item[0]]
            mechs = value.iloc[0, 1]
            # como pueden ser de varias mecánicas, se mira cada mecánica por separado
            for mechanism in mechs.split("|"):
                # si la mecánica está en el recomendado, se guarda el nombre del juego en el diccionario
                if mechanism in info_recommended_mechanisms.iloc[0, 1]:
                    if mechanism not in items_mechanism.keys():
                        items_mechanism[mechanism] = [name.iloc[0,1]]
                    else:
                        items_mechanism[mechanism].append(name.iloc[0,1])
            i += 1

        # si hay más de "number_show" items similares, se eligen los "number_show" primeros
        if items_tipo:
            tipos = info_recommended_tipo.iloc[0, 1].split("|")
            for tipo in tipos:
                if tipo in items_tipo.keys():
                    if len(items_tipo[tipo]) > number_show:
                        lista = items_tipo[tipo][:number_show]
                        items_tipo[tipo] = lista
            
        if items_categ:
            categs = info_recommended_categ.iloc[0, 1].split("|")
            for categ in categs:
                if categ in items_categ.keys():
                    if len(items_categ[categ]) > number_show:
                        lista = items_categ[categ][:number_show]
                        items_categ[categ] = lista
                
        if items_mechanism:
            mechs = info_recommended_mechanisms.iloc[0, 1].split("|")
            for mechanism in mechs:
                if mechanism in items_mechanism.keys():
                    if len(items_mechanism[mechanism]) > number_show:
                        lista = items_mechanism[mechanism][:number_show]
                        items_mechanism[mechanism] = lista

        return items_tipo, items_categ, items_mechanism, tipos, categs, mechs