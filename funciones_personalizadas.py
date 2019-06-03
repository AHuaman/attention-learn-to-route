import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D

from utils.data_utils import save_dataset
from utils.data_utils import load_dataset
from sklearn.datasets.samples_generator import make_blobs # generar puntos clusterizados

# La primera es una función creada por el autor del paper:
def discrete_cmap(N, base_cmap=None):
    """
    Create an N-bin discrete colormap from the specified input map
    """
    # Note that if base_cmap is a string or None, you can simply do
    #    return plt.cm.get_cmap(base_cmap, N)
    # The following works for string, None, or a colormap instance:
    base = plt.cm.get_cmap(base_cmap)
    color_list = base(np.linspace(0, 1, N))
    cmap_name = base.name + str(N)
    return base.from_list(cmap_name, color_list, N)

def generar_un_grafo_personalizable(vrp_size, n_clusters=None, perc_clustered_points=None):
    n_clustered_datapoints=int(np.round(vrp_size * perc_clustered_points))
    n_random_datapoints=int(vrp_size - n_clustered_datapoints)
    
    # Definimos el ratio de desviación estándar para mantener una distancia razonable entre centros de clúster:
    ratio_std = 0.025/3 if n_clusters<=3 else 0.015/3
    # Generamos clústeres:
    points, _ = make_blobs(n_samples=n_clustered_datapoints, n_features=2, centers=n_clusters, 
                           cluster_std=ratio_std*n_clusters, center_box=(0.2,0.8))
    
    # Generamos el resto de los puntos:
    if perc_clustered_points != 1:
        points = np.vstack([
            points,
            np.random.uniform(size=(n_random_datapoints, 2))
        ])
        np.random.shuffle(points) # shuffle para evitar que el orden de las muestras afecte a la red
    
    return points

def generar_vrp_personalizable(dataset_size, vrp_size, capacity, n_clusters=None, 
                               perc_clustered_points=None, demands_distribution='integer_uniform', **kwargs):
    """
    Crear instancias de VRP personalizables para experimentación.
    Argumentos:
    - dataset_size: Cantidad de instancias a crear
    - vrp_size: Cantidad de clientes por grafo
    - capacity: Capacidad de los vehículos a utilizarse
    - n_clusters: Si no es 'None', indica cuántas ubicaciones de puntos clusterizados se
                  generarán
    - perc_clustered_points: Si no es 'None', indica cuál es la proporción de puntos que se encontrarán
                           dentro de algún clúster geográfico. El resto se generará aleatoriamente dentro
                           del espacio restante
    - demands_distribution: La distribución a partir de la cual se meustrean los valores de las demandas
    
    Modificado a partir de la función 'generate_vrp_data()'
    """
    # Generar ubicaciones:
    if n_clusters is None:
        # Generar varios grafos normales
        full_locations = np.random.uniform(size=(dataset_size, vrp_size, 2))
    else:
        # Generar varios grafos personalizados:
        full_locations = [] # acumulador
        for i in range(dataset_size):
            full_locations.append(
                generar_un_grafo_personalizable(vrp_size, n_clusters, perc_clustered_points)
            )
        full_locations = np.vstack([full_locations])
    
    # Generar demandas a partir de una distribución determinada:
    if demands_distribution == 'integer_uniform':
        demands = np.random.randint(low=kwargs['low'], high=kwargs['high'], size=(dataset_size, vrp_size))
    elif demands_distribution == 'float_normal':
        demands = np.random.normal(loc=kwargs['loc'], scale=kwargs['scale'], size=(dataset_size, vrp_size))
    elif demands_distribution == 'integer_normal':
        demands = np.round(np.random.normal(loc=kwargs['loc'], scale=kwargs['scale'], size=(dataset_size, vrp_size)))
    else:
        raise ValueError("El argumento 'demands_distribution' debe ser 'integer_uniform', 'float_normal' o 'integer_normal'")
    
    return list(zip(
        np.random.uniform(size=(dataset_size, 2)).tolist(),  # Depot location
        full_locations.tolist(),  # Node locations
        demands.tolist(),  # Demands
        np.full(dataset_size, capacity).tolist()  # Capacity, same for whole dataset
    ))

def plot_muestra_varios_grafos(all_datasets_list, n_samples, seed, subplot_size, indiv_figsize=24):
    np.random.seed(seed)
    indices = np.random.choice(np.arange(len(all_datasets_list)), replace=False, size=n_samples)
    num_filas = subplot_size[0]
    num_cols = subplot_size[1]
    
    # Separamos la ventana de gráficas:
    fig, axes = plt.subplots(num_filas, num_cols, figsize=(indiv_figsize, indiv_figsize))
    
    for i in range(n_samples):
        plt.subplot(num_filas, num_cols, i+1)
        sns.scatterplot(
            x = np.array(all_datasets_list[indices[i]][1])[:,0], 
            y = np.array(all_datasets_list[indices[i]][1])[:,1]
        )
        sns.scatterplot(
            x = all_datasets_list[indices[i]][0][0],
            y = all_datasets_list[indices[i]][0][1] ,
            s=100, style=['Depot'], color=['red'], markers=['s']
        )
    plt.show()
    return None

def crear_y_guardar_datasets(dataset_size, vrp_size, capacity, n_clusters, perc_clustered_points, 
                             demands_distribution, seed, datadir, name, **kwargs):
    filename = os.path.join(datadir, "VRP_{}_{}_seed{}.pkl".format(vrp_size, name, seed))
    np.random.seed(seed) # reproducibilidad
    
    dataset = generar_vrp_personalizable(dataset_size, vrp_size, capacity, n_clusters, perc_clustered_points,
                                         demands_distribution, **kwargs)
    
    print('Grabando archivo {} con {} grafos'.format(filename, len(dataset)))
    save_dataset(dataset, filename)
    print('Listo!')

# Creamos una nueva función para generar el gráfico de rutas para los casos resueltos mediante código
def modified_plot_vehicle_routes(data, route, ax1, markersize=5, visualize_demands=False, demand_scale=1, round_demand=False):
    """
    Plot the vehicle routes on matplotlib axis ax1.
    """
    # route is one sequence, separating different routes with 0 (depot)
    route = np.array(route)
    routes = [r[r!=0] for r in np.split(route, np.where(route==0)[0]) if (r != 0).any()]
    depot = np.array(data[0])
    locs = np.array(data[1])
    demands = np.array(data[2]) * demand_scale
    capacity = data[3] # Capacity is always 1
    
    x_dep, y_dep = depot
    ax1.plot(x_dep, y_dep, 'sk', markersize=markersize*4)
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    
    legend = ax1.legend(loc='upper center')
    
    cmap = discrete_cmap(len(routes) + 2, 'nipy_spectral')
    dem_rects = []
    used_rects = []
    cap_rects = []
    qvs = []
    total_dist = 0
    for veh_number, r in enumerate(routes):
        color = cmap(len(routes) - veh_number) # Invert to have in rainbow order
        
        route_demands = demands[r - 1]
        coords = locs[r - 1, :]
        xs, ys = coords.transpose()

        total_route_demand = sum(route_demands)
        assert total_route_demand <= capacity
        if not visualize_demands:
            ax1.plot(xs, ys, 'o', mfc=color, markersize=markersize, markeredgewidth=0.0)
        
        dist = 0
        x_prev, y_prev = x_dep, y_dep
        cum_demand = 0
        for (x, y), d in zip(coords, route_demands):
            dist += np.sqrt((x - x_prev) ** 2 + (y - y_prev) ** 2)
            
            cap_rects.append(Rectangle((x, y), 0.01, 0.1))
            used_rects.append(Rectangle((x, y), 0.01, 0.1 * total_route_demand / capacity))
            dem_rects.append(Rectangle((x, y + 0.1 * cum_demand / capacity), 0.01, 0.1 * d / capacity))
            
            x_prev, y_prev = x, y
            cum_demand += d
            
        dist += np.sqrt((x_dep - x_prev) ** 2 + (y_dep - y_prev) ** 2)
        total_dist += dist
        qv = ax1.quiver(
            xs[:-1],
            ys[:-1],
            xs[1:] - xs[:-1],
            ys[1:] - ys[:-1],
            scale_units='xy',
            angles='xy',
            scale=1,
            color=color,
            label='R{}, # {}, c {} / {}, d {:.2f}'.format(
                veh_number, 
                len(r), 
                int(total_route_demand) if round_demand else total_route_demand, 
                int(capacity) if round_demand else capacity,
                dist
            )
        )
        
        qvs.append(qv)
        
    ax1.set_title('{} routes, total distance {:.2f}'.format(len(routes), total_dist))
    ax1.legend(handles=qvs)
    
    pc_cap = PatchCollection(cap_rects, facecolor='whitesmoke', alpha=1.0, edgecolor='lightgray')
    pc_used = PatchCollection(used_rects, facecolor='lightgray', alpha=1.0, edgecolor='lightgray')
    pc_dem = PatchCollection(dem_rects, facecolor='black', alpha=1.0, edgecolor='black')
    
    if visualize_demands:
        ax1.add_collection(pc_cap)
        ax1.add_collection(pc_used)
        ax1.add_collection(pc_dem)