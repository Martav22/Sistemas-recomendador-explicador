# Implementación de un sistema de recomendación con extracción de explicaciones
A continuación, se va a explicar cómo se han implementado los sistemas de recomendación y de explicaciones. La implementación de ambos sistemas se ha realizado en Python, y se ha creado una clase por cada uno de ellos: `class Recomendador()` y `class Explicaciones()`. También se ha creadoun fichero util.py, con funciones como leer o guardar información, como pueden ser matrices o valoraciones, en un fichero .csv.

## Sistema recomendador
En la clase Recomendador se han incluido los métodos de entrenamiento, configuración y evaluación del modelo, así como los usados para obtener los juegos de mesa recomendados dado un usuario. 

Empezamos por los métodos de entrenamiento, ya que sin ellos no se puede configurar el modelo. Por un lado, se tiene la función que divide el dataset pasado por argumento, en nuestro caso será el *dataset* mencionado del portal BGG, en los conjuntos trainset y testset, según el porcentaje de test pasado también por argumento (`dividir_dataset(dataset, porcentaje_test)`). Para ello, se usa la función `train_test_split(dataset, test_size, random_state)` del paquete sklearn. Además, en esta función se crean los diccionarios que relacionan a los usuarios y los juegos de mesa, del trainset, con su fila o columna en las matrices $P$ y $Q$, respectivamente. En estas matrices, los identificadores de los usuarios y de los juegos de mesa se ordenan de menor a mayor para asociarlos con la fila, en el caso de los usuario, o de la columna, en el caso de los juegos de mesa.

Una vez que se ha divido el *dataset*, se puede entrenar el modelo usando el conjunto de entrenamiento (trainset). Para ello, se crea una función, que recibe como argumento el número de factores latentes, y con ellos entrena el modelo usando el algoritmo NMF del paquete Surprise (`entrenamiento(num_factores)`). Esto se hace usando la función `fit(trainset)` del modelo, que recibe como argumento el trainset como la clase Trainset de Surprise. Es por ello que se necesita convertir el trainset a dicha clase usando la función `built_full_trainset()`.

Ahora, una vez que podemos dividir el *dataset* en los conjuntos de entrenamiento y de test con distintos porcentajes de entrenamiento, y entrenar el modelo usando distinto número de factores latentes, pasamos a las funciones que ayudan a configurar el modelo. Se ha implementado la función que, pasado un modelo entrenado con unos parámetros y un dataset con un conjunto de valoraciones de la forma (usuario, juego, valoración), calcula las predicciones de todas las entradas de ese dataset usando el modelo pasado (`calcular_predicciones(W, df)`). Esta función es usada por otra con los mismos argumentos que calcula el MAE de las predicciones devueltas por esta (`calcular_mae(W, df)`). Por último, se ha implementado una función que para el número de factores latentes entre 1 y 40, entrena el modelo usando la función de entrenamiento, de la que se ha hablado anteriormente, y calcula el MAE del trainset y del testset para cada número de factores latentes (`errores_train_test()`). Devuelve dos listas, una con los MAE del testset, y otra con los del trainset. 

En cuanto a la evaluación, se han creado dos funciones, una que realiza una validación cruzada de 5 iteraciones que calcula el MAE, RMSE, y MSE del *dataset* pasado por argumento (`get_validacion_cruzada(df)`), y la otra, que calcula esas mismas métricas de error con las predicciones generadas por nuestro modelo sobre el testset (`get_tabla_resultados()`).

Ahora, vamos a hablar de dos funciones que no entrenan ni evalúan ni son un medio para configurar el modelo, sino que lo utilizan. Una de ellas, recibe el identificador de un usuario y de un juego como argumentos, y devuelve la estimación de la valoración de ese usuario sobre ese juego de mesa (`get_rating(uid, iid)`). Para ello, usa la función `predict(uid, iid)` del modelo, que, pasados los identificadores del usuario y el juego, devuelve, entre otros valores, la estimación. Y la otra es una función con la que se hace una recomendación de $k$ juegos a un usuario. Esta recibe el identificador de un usuario y un número $k$, que indica la cantidad de juegos a recomendar (`get_recomendados(uid, k)`). 

## Sistema explicador
En cuanto a la clase Explicaciones, se ha incluido la función que calcula las matrices $Q_u$, que se obtienen multiplicando una fila fija $p_u \in M_{1\times F}(\mathbb{R})$ de la matriz $P$ (información sobre un usuario determinado en la fila $u$ de la matriz) por la matriz $Q$. Esta multiplicación no es la usual entre matrices, sino que consiste en multiplicar cada fila de la matriz de $Q$ por la fila fija, posición a posición También se han implementado las funciones que obtienen las similitudes entre juegos de mesa, y las que generan las explicaciones.

Empezamos con la función que, dado el identificador de un usuario, devuelve su matriz $Q_u$ correspondiente (`get_qu(user_id)`). Luego, tenemos la función que calcula las similitudes entre el juego recomendado y los juegos de mesa jugados por el usuario al que se le ha recomendado dicho juego. Esta función recibe el identificador del juego recomendado y del usuario al que se le recomienda, y devuelve un diccionario con los identificadores de los juegos de mesa jugados por el usuario, junto con su similitud con el juego recomendado (`get\_similitudes(juego, user)`). Para obtener las similitudes entre los juegos, se usa la métrica del coseno, que dado dos vectores $x, y$ en forma de fila, la similitud entre $x$ e $y$ se calcula como
$$
    cosine(x, y) = \frac{x\cdot y}{||x|| ||y||}
$$
De manera que, el vector $x$ será la fila de la matriz $Q_u$ correspondiente al juego recomendado, y el vector $y$ será una fila de la matriz $Q_u$ correspondiente a uno de los juegos jugados por el usuario. Los juegos de ejemplo, que se usarán en la generación de explicaciones, se obtienen mediante un algoritmo \acs{knn}

A partir de estas similitudes, se crean dos funciones para obtener los juegos que se usarán en la generación de explicaciones. Una obtiene los $k$ juegos de mesa más similares al recomendado, donde $k$ es un valor que se pasa por argumento (`get_k_expl(juego, user, k)`), y otra para obtener los juegos de mesa con una similitud con el juego recomendado por encima de un umbral, que se pasa por argumento (`get_expl_limite(juego, user, limite)`). Ambas funciones también reciben como argumento el identificador del juego recomendado y del usuario al que se le recomienda.

Por último, tenemos las funciones que generan las explicaciones. Para ello, se va a usar un enfoque basado en características, es decir, se analizan las características de los juegos similares, comparándolas con las del juego recomendado. Se obtienen cuatro tipos de explicaciones, una por característica:
1. Número de jugadores: aquellos juegos similares que tienen el mismo mejor número de jugadores que le juego recomendado.
2. Edad: aquellos juegos con la misma edad recomendada que el juego recomendado.
3. Dificultad: se hace una clasificación de dificultad de los juegos similares, distinguiendo los más difíciles, los más fáciles y los de dificultad similar al recomendado.
4. Tipos, categorías y mecánicas: se realiza una agrupación de juegos en distintas categorías, tipo y mecánicas.  Para ello, se obtienen los tipos, categorías y mecánicas en común entre las del juegos recomendado y las de todos los juegos similares a él, y se van guardando los juegos similares que comparten tipo, categoría o mecánica con el recomendado.

Se han implementado tres funciones, que reciben el identificador del juego recomendado, los juegos similares a este y el número máximo de estos que se quiere mostrar al usuario. Una que genera las explicaciones acerca de la edad y el número de jugadores recomendados (`explain_age_best_players(iid, items_similares, number_show)`). Una segunda, que genera las explicaciones sobre la dificultad, y que además, recibe el identificador del usuario, para poder obtener el \textit{rating} sobre cada juego similar (`explain_dificultad(iid, uid, items_similares, number_show)`). Y una última, que genera las explicaciones de los tipos, categorías y mecánicas (`explain_mecanicas(iid, items_similares, number_show)`). Estas funciones devuelven la información necesaria sobre los juegos de mesa para poder presentar las explicaciones a los usuarios.

## Tutorial
En el fichero Tutorial.ipynb se encuentra un ejemplo de cómo usar las librerías explicadas. En el ejemplo, primero se entrena el modelo, y una vez entrenado, se le hace una recomendación de un juego a un usuario en concreto. A partir de esta recomendación, se obtienen sus juegos similares y se extraen las cuatro explicaciones: edad recomendada, número de jugadores, dificultad, y tipos, categorías y mecánicas. Una vez se tienen las explicaciones, se presentan al usuario. Además, se muestra una manera de presentar las explicaciones.

# A tener en cuenta
A la hora de presentar las explicaciones, la carátula se obtiene de la carpeta "imagenes_games", por lo que si hay un juego similar que no tiene su carátula en esa carpeta, va a dar un error. Se está trabajando para obtener la imagen del portal BGG vía web (ya que las imágenes están en una ruta distinta a la del identificador del juego).