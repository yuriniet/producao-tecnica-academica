import time
import random
import osmnx as ox
import numpy as np
import pandas as pd
import networkx as nx
import osmnx.distance as distance
import igraph as ig



network, name = 'snfile/sf_original', 'SF'
data = pd.read_csv('routes/sfroutes_rede_original.csv')
data = data.values.tolist()


def shuffle_routes():
    random.shuffle(data)
    return data


g = ox.load_graphml(network, edge_dtypes={'capacity':float})

g = ox.project_graph(g)
g = nx.DiGraph(g)
n = list(nx.nodes(g))

x = nx.get_node_attributes(g, 'x')
y = nx.get_node_attributes(g, 'y')

alfa = 0.15
beta = 4.00

o_paths = dict(nx.all_pairs_dijkstra_path(g, weight='travel_time'))

highway = nx.get_node_attributes(g, 'highway')
bc = nx.get_node_attributes(g, 'bc')
'''bc = nx.betweenness_centrality(g, weight='travel_time')'''

print("centralidades e caminhos mínimos calculados.")

bctf = {node: bc[node] for node in highway.keys() if highway[node] == 'traffic_signals'}
degreetf = {node: dict(nx.degree(g, weight='length'))[node] for node in highway.keys() if highway[node] == 'traffic_signals'}

capacity = nx.get_edge_attributes(g,'capacity')
print("capacidade média: " , np.average(list(capacity.values())))


def disruption(type, fr):
    global arestas
    arestas = 0
    if type == 'A':
        p = 0
        tf = [node for node in highway if highway[node] == 'traffic_signals']
        sizetf = len(tf)
        while p < fr*sizetf:
            vs = random.choice(tf)
            for e in g.in_edges(vs):
                g.edges[e]['speed'] = g.edges[e]['speed']/2
                arestas += 1
            p += 1
            tf.remove(vs)
    elif type == 'B':
        p = 0
        node = {k: v for k, v in sorted(bctf.items(), key=lambda item: item[1], reverse=True)}
        while p < fr*len(bctf.keys()):
            for e in g.in_edges(list(node.keys())[p]):
                g.edges[e]['speed'] = g.edges[e]['speed']/2
                arestas += 1
            p += 1
    elif type == 'C':
        p = 0
        node = {k: v for k, v in sorted(degreetf.items(), key=lambda item: item[1], reverse=True)}
        while p < fr*len(degreetf.keys()):
            for e in g.in_edges(list(node.keys())[p]):
                g.edges[e]['speed'] = g.edges[e]['speed']/2
                arestas += 1
            p += 1


def convert_to_igraph(g):
    osmids = list(g.nodes)
    osmid_values = {k: v for k, v in zip(g.nodes, osmids)}
    nx.set_node_attributes(g, osmid_values, "osmid")
    G_ig = ig.Graph(directed=True)
    G_ig.add_vertices(g.nodes)
    G_ig.add_edges(g.edges())
    G_ig.vs["osmid"] = osmids
    for i in ['time','travel_time','flow','length','speed','speed_kph','capacity']:
        G_ig.es[i] = list(nx.get_edge_attributes(g, i).values())
    assert len(g.nodes()) == G_ig.vcount()
    assert len(g.edges()) == G_ig.ecount()
    return G_ig


def initialize():
    for e in g.edges():
        g.edges[e]['flow'] = 0
        g.edges[e]['speed'] = g.edges[e]['speed_kph']
        g.edges[e]['time'] = g.edges[e]['length'] / g.edges[e]['speed'] * (
                    1 + alfa * (g.edges[e]['flow'] / g.edges[e]['capacity']) ** beta)


def sp(a, b):
    return distance.euclidean_dist_vec(g.nodes[a]['y'], g.nodes[a]['x'], g.nodes[b]['y'], g.nodes[b]['x'])*0.001


def angle(path):
    ang = 0
    if len(path) > 2:
        for i in range(len(path) - 2):
            a = np.array([x[path[i]], y[path[i]]])
            b = np.array([x[path[i + 1]], y[path[i + 1]]])
            c = np.array([x[path[i + 2]], y[path[i + 2]]])
            ba = a - b
            bc = c - b
            cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
            ang += np.pi - np.arccos(cosine_angle)
        return ang
    else:
        return 0


def agent():
    global N0
    c = data[N0%len(data)]
    path = ig_g.get_shortest_paths(v=c[1],to=c[2], weights='time')[0]
    actual_path.append(path)
    N0 += 1
    for _ in range(len(actual_path[-1]) - 1):
        i = actual_path[-1][_]
        j = actual_path[-1][_ + 1]
        g.edges[(i, j)]['flow'] += 1


def update():
    for edge in g.edges():
        g.edges[edge]['time'] = g.edges[edge]['length'] / g.edges[edge]['speed'] * (
                1 + alfa * (g.edges[edge]['flow'] / g.edges[edge]['capacity']) ** beta)


def result():
    global same, alternative, actual_path
    for edge in g.edges():
        g.edges[edge]['time'] = g.edges[edge]['length'] / g.edges[edge]['speed'] * (
                1 + alfa * (g.edges[edge]['flow'] / g.edges[edge]['capacity']) ** beta)
    ac = []
    tr = []
    atta = []
    ala = []
    for k in actual_path:
        l, t, t0 = 0, 0, 0
        if k == o_paths[k[0]][k[-1]]:
            same += 1
        else:
            alternative += 1
        for _ in range(len(k)-1):
            l += g.edges[(k[_], k[_ + 1])]['length']
            t += g.edges[(k[_], k[_ + 1])]['time']
            t0 += g.edges[(k[_], k[_ + 1])]['travel_time']
        ala.append(l)
        ac.append(l / sp(k[0],k[-1]))
        atta.append(t)
        tr.append(t / t0)
    return np.average(ac), np.average(tr), np.average(atta), np.average(ala)


scenarios = ['A', 'B', 'C']
z = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
N = 50000
gamma = 5000

for num_iter in range(5):
    data = shuffle_routes()
    print("caminhos embaralhados")
    for scenario in scenarios:
        for i in z:
            actual_path = []
            same, alternative = 0, 0
            initialize()
            ig_g = convert_to_igraph(g)
            disruption(type=scenario, fr=i)
            N0 = 0
            t0 = time.time()
            while N0 < 1*N:
                agent()
                if N0 % gamma == 0:
                    update()
                    ig_g = convert_to_igraph(g)
            avgac, avgtr, avgatta, avgala = result()
            sigma = np.round(i,1)
            congestion = np.average([g.edges[e]['flow']/g.edges[e]['capacity'] for e in g.edges()])
            delay = np.sum([(g.edges[e]['time'] - g.edges[e]['travel_time']) for e in g.edges()])
            tempo = np.round(time.time()-t0,1)
            outputs = [(name,
                        str(scenario),
                        sigma,
                        gamma,
                        float(N0),
                        np.round(avgac, 4),
                        np.round(avgtr, 4),
                        np.round(avgatta, 4),
                        np.round(avgala, 4),
                        np.round(congestion, 4),
                        np.round(same / (alternative + same), 4),
                        np.round(alternative / (same + alternative), 4),
                        np.round(delay, 4),
                        arestas,
                        tempo)]
            row = pd.DataFrame(outputs, columns=['rede',
                                                'cenário',
                                                'sigma',
                                                'gamma',
                                                'y',
                                                'CM',
                                                'RTV',
                                                'TMV',
                                                'DMC',
                                                'C',
                                                'igual',
                                                'alternativo',
                                                'delay',
                                                'arestas',
                                                'tempo']
                                   )
            row.to_csv('test_astar3.csv', mode='a', header=False, sep=';')
            print(scenario, np.round(i, 1), tempo, end=" ")
            print()