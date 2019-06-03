# Todas las liberías necesarias:
from docplex.mp.model import Model # importando la clase 'Model' de Programación Lineal
import numpy as np # álgebra lineal
import pandas as pd # DataFrames para manipular tablas de datos
import matplotlib.pyplot as plt # ploteo genérico
import networkx as nx # ploteo de conexiones en grafos
from scipy.spatial import distance_matrix # para sacar matriz de distancias


###############
# UTILITARIOS #
###############
def ObtenerMatrizViajes(dataframe_distancias, etiquetas_nodos, modelo_resuelto):
    """
    Obtenemos la matriz de viajes, el cual es el resultado del modelo TSP Tucker-Miller-Zemlin.
    """
    # Obtenemos las conexiones a modo de lista:
    conexiones = list(
        map(
            lambda x: tuple(x.split('_')),
            [
                i[2:] for i in list(modelo_resuelto.solution.as_dict().keys()) if i[0] == 'x' # magia!
            ]
        )
    )
    
    # Construimos la matriz binaria de conexiones:
    matriz_conexiones = pd.DataFrame(data=np.zeros_like(dataframe_distancias.copy().values), 
                                     index=etiquetas_nodos, 
                                     columns=etiquetas_nodos)
    for i,j in conexiones:
        matriz_conexiones.loc[i,j] = 1

    return matriz_conexiones


def PlotearGrafoViajes(dataframe_matriz_viajes):
    """
    Graficamos un grafo de conexiones que representa la ruta resultante de la optimización lineal
    """
    G = nx.from_pandas_adjacency(dataframe_matriz_viajes, 
                                 create_using=nx.MultiDiGraph()) # Grafo dirigido!!!!
    nx.draw(G, with_labels=True)
    return G

def GenerarEjemploGrafo(num_clientes, con_demandas=False):
    loc_xy = pd.DataFrame(data = np.random.random(size=(num_clientes+1,2)),
                         columns=['x','y']) # el primer indice ('0') SERÁ el depot
    if con_demandas:
        loc_xy['demanda'] = np.random.randint(1,10,size=num_clientes+1) # muestreados a partir de una dist unif discreta [1,9]
    
    # Agregamos etiquetas (en este caso '0' siempre será el depot):
    indices_generados = list(map(str, range(num_clientes + 1)))
    loc_xy.index = indices_generados
    return loc_xy

def GraficarCiudades(dataframe_localidades, tamano_punto=25):
    num_clientes = dataframe_localidades.shape[0] - 1
    plt.scatter(x = dataframe_localidades['x'],
                y = dataframe_localidades['y'], 
                s = [3.5*tamano_punto] + [tamano_punto] * num_clientes,
                c = ['r'] + ['b'] * num_clientes)
    return None

def GenerarTablaDistancias(dataframe_localidades, etiquetas_nodos):
    matriz_distancias_euclidianas = distance_matrix(dataframe_localidades[['x','y']], dataframe_localidades[['x','y']])
    tabla_distancias = pd.DataFrame(data = matriz_distancias_euclidianas, 
                                    index = etiquetas_nodos, 
                                    columns = etiquetas_nodos)
    return tabla_distancias

def AdaptarDataFrameLocalidades(lista_grafo):
    array_demandas = np.array([0] + lista_grafo[2])[:,None] # agregamos una dimensión
    array_localidades = np.vstack((np.array(lista_grafo[0]), np.array(lista_grafo[1])))
    dataframe = pd.DataFrame(data = np.hstack((array_localidades, array_demandas)), columns = ['x','y','demanda'])
    dataframe.index = list(map(str, dataframe.index))
    return dataframe

###########
# MODELOS #
###########
def ModeloTSP(dataframe_distancias, etiquetas_nodos, str_nombre_modelo):
    """
    Devuelve un objeto 'Model' de CPLEX que resuelve el TSP con la formulación propuesta por Tucker-Miller-Zemlin.
    """
    # Creamos nuestra instancia del modelo:
    m = Model(name = str_nombre_modelo)
    
    cantidad_nodos = len(etiquetas_nodos)
    
    ## Parámetros del modelo:
    # SET's:
    indices_nodos = etiquetas_nodos.copy()

    # VAR's:
    # Lo de abajo se llama comprensión de diccionario! Nótese que eliminamos todos los i==j, ahorrando variables y tiempo
    x = {(i,j): m.binary_var(name='x_{0}_{1}'.format(i,j)) for i in indices_nodos for j in indices_nodos if i!=j}
    u = {i: m.integer_var(name='u_{0}'.format(i)) for i in indices_nodos}

    # Función Objetivo:
    m.minimize(
        m.sum(
            x[i,j] * dataframe_distancias.loc[i,j] for i in indices_nodos for j in indices_nodos if i!=j
        )
    )
    
    ## RESTRICCIONES:
    # Entrada:
    for j in indices_nodos:
        m.add_constraint(m.sum(x[i,j] for i in indices_nodos if i!=j) == 1, 
                         ctname='entrada_{0}'.format(j))

    # Salida:
    for i in indices_nodos:
        m.add_constraint(m.sum(x[i,j] for j in indices_nodos if i!=j) == 1, 
                         ctname='salida_{0}'.format(i))

    # Subtours:
    for i in indices_nodos[1:]: # tomamos a partir del segundo indice (empieza en '0')
        for j in indices_nodos[1:]: # tomamos a partir del segundo indice (empieza en '0')
            if i!=j:
                m.add_constraint(u[i] - u[j] + cantidad_nodos * x[i,j] <= cantidad_nodos - 1, 
                                 ctname='subtour_{0}_{1}'.format(i, j))
    
    print(m.print_information())
    
    return m

def Modelo_M_TSP(dataframe_distancias, etiquetas_nodos, str_nombre_depot, num_rutas, str_nombre_modelo):
    """
    Devuelve un objeto 'Model' de CPLEX que resuelve el m-TSP con la formulación Tucker-Miller-Zemlin **extendida**.
    """
    # Creamos nuestra instancia del modelo:
    m = Model(name = str_nombre_modelo)
    
    cantidad_nodos = len(etiquetas_nodos)
    
    ## Parámetros del modelo:
    # SET's:
    indices_nodos = etiquetas_nodos.copy()
    indices_nodos.remove(str_nombre_depot)
    indices_nodos.insert(0,str_nombre_depot) # ponemos el depot al inicio

    # VAR's:
    # Lo de abajo se llama comprensión de diccionario! Nótese que eliminamos todos los i==j, ahorrando variables y tiempo
    x = {(i,j): m.binary_var(name='x_{0}_{1}'.format(i,j)) for i in indices_nodos for j in indices_nodos if i!=j}
    u = {i: m.integer_var(name='u_{0}'.format(i)) for i in indices_nodos}

    # Función Objetivo:
    m.minimize(
        m.sum(
            x[i,j] * dataframe_distancias.loc[i,j] for i in indices_nodos for j in indices_nodos if i!=j
        )
    )
    
    ## RESTRICCIONES:
    # Entrada:
    for j in indices_nodos:
        if j == str_nombre_depot:
            m.add_constraint(m.sum(x[i,j] for i in indices_nodos if i!=j) == num_rutas, 
                             ctname='entrada_{0}'.format(j))
        else:
            m.add_constraint(m.sum(x[i,j] for i in indices_nodos if i!=j) == 1, 
                             ctname='entrada_{0}'.format(j))

    # Salida:
    for i in indices_nodos:
        if i == str_nombre_depot:
            m.add_constraint(m.sum(x[i,j] for j in indices_nodos if i!=j) == num_rutas, 
                             ctname='salida_{0}'.format(i))
        else:
            m.add_constraint(m.sum(x[i,j] for j in indices_nodos if i!=j) == 1, 
                             ctname='salida_{0}'.format(i))

    # Subtours:
    for i in indices_nodos[1:]: # tomamos a partir del segundo indice (empieza en '0'). Ignora el depot
        for j in indices_nodos[1:]: # tomamos a partir del segundo indice (empieza en '0'). Ignora el depot
            if i!=j:
                m.add_constraint(u[i] - u[j] + cantidad_nodos * x[i,j] <= cantidad_nodos - 1, 
                                 ctname='subtour_{0}_{1}'.format(i, j))
    
    print(m.print_information())
    
    return m

def ModeloCVRP(dataframe_distancias, etiquetas_nodos, str_nombre_depot, num_rutas, capacidad_vehiculo,
               serie_demandas_nodos, str_nombre_modelo):
    """
    Devuelve un objeto 'Model' de CPLEX que resuelve el CVRP, usando una familia de restricciones
    que eliminan subtours que extiende la formulación TSP de Tucker-Miller-Zemlin, incorporando capacidades.
    
    Nótese que 'num_rutas' es igual a la cantidad de vehículos que se cuenta.
    """
    # Creamos nuestra instancia del modelo:
    m = Model(name = str_nombre_modelo)
    
    ## Parámetros del modelo:
    # SET's:
    indices_nodos = etiquetas_nodos.copy()
    indices_nodos.remove(str_nombre_depot)
    indices_nodos.insert(0,str_nombre_depot) # ponemos el depot al inicio

    # VAR's:
    # Lo de abajo se llama comprensión de diccionario! Nótese que eliminamos todos los i==j, ahorrando variables y tiempo
    x = {(i,j): m.binary_var(name='x_{0}_{1}'.format(i,j)) for i in indices_nodos for j in indices_nodos if i!=j}
    # Nótese que ahora el 'u' es una variable continua, en comparación con la formulación TSP. Agregamos cota inf y sup:
    u = {i: m.continuous_var(name='u_{0}'.format(i), 
                             lb=serie_demandas_nodos[i],
                             ub=capacidad_vehiculo) for i in indices_nodos if i!=str_nombre_depot}

    # Función Objetivo:
    m.minimize(
        m.sum(
            x[i,j] * dataframe_distancias.loc[i,j] for i in indices_nodos for j in indices_nodos if i!=j
        )
    )
    
    ## RESTRICCIONES:
    # Entrada:
    for j in indices_nodos:
        if j == str_nombre_depot:
            if num_rutas == 'auto_1.2_trivial':
                trivial_lb = int(np.ceil(serie_demandas_nodos.sum() / capacidad_vehiculo))
                calc_ub = int(np.ceil(serie_demandas_nodos.sum() / capacidad_vehiculo * 1.2))
                m.add_constraint(m.sum(x[i,j] for i in indices_nodos if i!=j) >= trivial_lb, 
                                 ctname='entrada_{0}'.format(j))
                m.add_constraint(m.sum(x[i,j] for i in indices_nodos if i!=j) <= calc_ub, 
                                 ctname='entrada_{0}'.format(j))
            else:
                m.add_constraint(m.sum(x[i,j] for i in indices_nodos if i!=j) == num_rutas, 
                                 ctname='entrada_{0}'.format(j))
        else:
            m.add_constraint(m.sum(x[i,j] for i in indices_nodos if i!=j) == 1, 
                             ctname='entrada_{0}'.format(j))

    # Salida:
    for i in indices_nodos:
        if i == str_nombre_depot:
            if num_rutas == 'auto_1.2_trivial':
                m.add_constraint(m.sum(x[i,j] for j in indices_nodos if i!=j) >= trivial_lb, 
                                 ctname='salida_{0}'.format(i))
                m.add_constraint(m.sum(x[i,j] for j in indices_nodos if i!=j) <= calc_ub, 
                                 ctname='salida_{0}'.format(i))
            else:
                m.add_constraint(m.sum(x[i,j] for j in indices_nodos if i!=j) == num_rutas, 
                                 ctname='salida_{0}'.format(i))
        else:
            m.add_constraint(m.sum(x[i,j] for j in indices_nodos if i!=j) == 1, 
                             ctname='salida_{0}'.format(i))

    # Subtours:
    for i in indices_nodos[1:]: # tomamos a partir del segundo indice (empieza en '0'). Ignora el depot
        for j in indices_nodos[1:]: # tomamos a partir del segundo indice (empieza en '0'). Ignora el depot
            # if i!=j and (serie_demandas_nodos[i] + serie_demandas_nodos[j] <= capacidad_vehiculo): # Esto no ayuda
            if i!=j:
                m.add_constraint(u[i] - u[j] + capacidad_vehiculo * x[i,j] <= capacidad_vehiculo - serie_demandas_nodos[j], 
                                 ctname='subtour_{0}_{1}'.format(i, j))
    
    print(m.print_information())
    
    return m

def ModeloCVRP_alternativo(dataframe_distancias, etiquetas_nodos, str_nombre_depot, num_rutas, capacidad_vehiculo,
               serie_demandas_nodos, str_nombre_modelo):
    """
    Devuelve un objeto 'Model' de CPLEX que resuelve el CVRP, usando una familia de restricciones
    que eliminan subtours que extiende la formulación TSP de Tucker-Miller-Zemlin, incorporando capacidades.
    
    Nótese que 'num_rutas' es igual a la cantidad de vehículos que se cuenta.
    """
    # Creamos nuestra instancia del modelo:
    m = Model(name = str_nombre_modelo)
    
    ## Parámetros del modelo:
    # SET's:
    indices_nodos = etiquetas_nodos.copy()
    indices_nodos.remove(str_nombre_depot)
    indices_nodos.insert(0,str_nombre_depot) # ponemos el depot al inicio

    # VAR's:
    # Lo de abajo se llama comprensión de diccionario! Nótese que eliminamos todos los i==j, ahorrando variables y tiempo
    x = {(i,j): m.binary_var(name='x_{0}_{1}'.format(i,j)) for i in indices_nodos for j in indices_nodos if i!=j}
    # Nótese que ahora el 'u' es una variable continua, en comparación con la formulación TSP. Agregamos cota inf y sup:
    u = {i: m.continuous_var(name='u_{0}'.format(i), 
                             lb=serie_demandas_nodos[i],
                             ub=capacidad_vehiculo) for i in indices_nodos if i!=str_nombre_depot}

    # Función Objetivo:
    m.minimize(
        m.sum(
            x[i,j] * dataframe_distancias.loc[i,j] for i in indices_nodos for j in indices_nodos if i!=j
        )
    )
    
    ## RESTRICCIONES:
    # Entrada:
    for j in indices_nodos:
        if j == str_nombre_depot:
            m.add_constraint(m.sum(x[i,j] for i in indices_nodos if i!=j) == num_rutas, 
                             ctname='entrada_{0}'.format(j))
        else:
            m.add_constraint(m.sum(x[i,j] for i in indices_nodos if i!=j) == 1, 
                             ctname='entrada_{0}'.format(j))

    # Salida:
    for i in indices_nodos:
        if i == str_nombre_depot:
            m.add_constraint(m.sum(x[i,j] for j in indices_nodos if i!=j) == num_rutas, 
                             ctname='salida_{0}'.format(i))
        else:
            m.add_constraint(m.sum(x[i,j] for j in indices_nodos if i!=j) == 1, 
                             ctname='salida_{0}'.format(i))
    
    # Subtours:
    m.add_indicator_constraints(
        m.indicator_constraint(x[i,j], u[i] + serie_demandas_nodos[j] == u[j], name='subtour_{0}_{1}'.format(i, j))
        for i in indices_nodos[1:]
        for j in indices_nodos[1:]
        if i!=j
    )
    
    print(m.print_information())
    
    return m

##########################
# FUNCIONES INTEGRADORAS #
##########################
def CrearTSPOptimizarObtenerViajesPlotearGrafo(dataframe_distancias, etiquetas_nodos, str_nombre_modelo):
    """
    Realiza el procesamiento completo!
    """
    print("========================= CREANDO MODELO =========================")
    modelo_tsp = ModeloTSP(dataframe_distancias=dataframe_distancias,
                           etiquetas_nodos=etiquetas_nodos, 
                           str_nombre_modelo=str_nombre_modelo)
    
    print("======================= OPTIMIZANDO MODELO =======================")
    modelo_tsp.solve(log_output=True)
    
    print("============================ SOLUCIÓN ============================")
    modelo_tsp.print_solution()
    
    print("=================== OBTENIENDO MATRIZ DE VIAJES ==================")
    viajes_modelo = ObtenerMatrizViajes(dataframe_distancias=dataframe_distancias,
                                        etiquetas_nodos=etiquetas_nodos, 
                                        modelo_resuelto=modelo_tsp)
    print(viajes_modelo)
    
    print("==================== CREANDO GRAFO DE VIAJES =====================")
    grafo_viajes = PlotearGrafoViajes(dataframe_matriz_viajes=viajes_modelo)
    
    return modelo_tsp, viajes_modelo, grafo_viajes

def Crear_M_TSPOptimizarObtenerViajesPlotearGrafo(dataframe_distancias, etiquetas_nodos, str_nombre_depot, num_rutas,
                                                  str_nombre_modelo):
    """
    Realiza el procesamiento completo!
    """
    print("========================= CREANDO MODELO =========================")
    modelo_m_tsp = Modelo_M_TSP(dataframe_distancias=dataframe_distancias,
                              etiquetas_nodos=etiquetas_nodos, 
                              str_nombre_depot=str_nombre_depot, 
                              num_rutas=num_rutas,
                              str_nombre_modelo=str_nombre_modelo)
    
    print("======================= OPTIMIZANDO MODELO =======================")
    modelo_m_tsp.solve(log_output=True)
    
    print("============================ SOLUCIÓN ============================")
    modelo_m_tsp.print_solution()
    
    print("=================== OBTENIENDO MATRIZ DE VIAJES ==================")
    viajes_modelo = ObtenerMatrizViajes(dataframe_distancias=dataframe_distancias,
                                        etiquetas_nodos=etiquetas_nodos, 
                                        modelo_resuelto=modelo_m_tsp)
    print(viajes_modelo)
    
    print("==================== CREANDO GRAFO DE VIAJES =====================")
    grafo_viajes = PlotearGrafoViajes(dataframe_matriz_viajes=viajes_modelo)
    
    return modelo_m_tsp, viajes_modelo, grafo_viajes

def CrearCVRPOptimizarObtenerViajesPlotearGrafo(dataframe_distancias, etiquetas_nodos, str_nombre_depot, num_rutas,
                                                capacidad_vehiculo, serie_demandas_nodos, str_nombre_modelo, limite_mins=9999999):
    """
    Realiza el procesamiento completo! Nótese que 'num_rutas' es igual a la cantidad de vehículos que se cuenta.
    """
    print("========================= CREANDO MODELO =========================")
    modelo_cvrp = ModeloCVRP(dataframe_distancias=dataframe_distancias,
                              etiquetas_nodos=etiquetas_nodos, 
                              str_nombre_depot=str_nombre_depot, 
                              num_rutas=num_rutas,
                              capacidad_vehiculo=capacidad_vehiculo,
                              serie_demandas_nodos=serie_demandas_nodos,
                              str_nombre_modelo=str_nombre_modelo)
    modelo_cvrp.parameters.timelimit = limite_mins*60 # tiempo máximo de resolución en segundos
    
    print("======================= OPTIMIZANDO MODELO =======================")
    modelo_cvrp.solve(log_output=True)
    
    print("============================ SOLUCIÓN ============================")
    modelo_cvrp.print_solution()
    
    print("=================== OBTENIENDO MATRIZ DE VIAJES ==================")
    viajes_modelo = ObtenerMatrizViajes(dataframe_distancias=dataframe_distancias,
                                        etiquetas_nodos=etiquetas_nodos, 
                                        modelo_resuelto=modelo_cvrp)
    print(viajes_modelo)
    
    print("==================== CREANDO GRAFO DE VIAJES =====================")
    grafo_viajes = PlotearGrafoViajes(dataframe_matriz_viajes=viajes_modelo)
    
    return modelo_cvrp, viajes_modelo, grafo_viajes