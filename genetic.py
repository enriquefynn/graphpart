import operator
import scipy.sparse as sp
from tqdm import tqdm
from copy import copy
import numpy as np
import networkx as nx
import nn
from test import test_with_metis

class TestIndividual(object):
    """Test individual.

    Arguments:
    number_of_nodes -- number of nodes
    density -- density parameter to the Holme-Kim graph from [0,1]
    n_partitions -- number of partitions
    edges -- number of edges to form in each iteration (default 2)
    """

    def __init__(self, number_of_nodes, density, n_partitions, edges=2):
        self.indices = []
        self.n_partitions = n_partitions
        self.n_nodes = number_of_nodes

        graph = nx.powerlaw_cluster_graph(number_of_nodes, edges, density)
        self.graph = max(nx.connected_component_subgraphs(graph), key=len)

        for edge in graph.edges_iter():
            self.indices.append(list(edge))

        fr_l = []
        to_l = [] 
        vals = []
        for (fr, to) in self.indices:
                fr_l.append(fr)
                to_l.append(to)
                vals.append(.5 / graph.degree(fr))
        for node in graph:
            fr_l.append(node)
            to_l.append(node)
            vals.append(.5)
        
        self.matrix = sp.coo_matrix((vals, (fr_l, to_l)),
            shape = (graph.number_of_nodes(), graph.number_of_nodes()))

    def get_edge_cut_balance(self, partition):
        """Return the edge-cut and balance"""
        edge_cut = 0
        balance = [0 for _ in xrange(self.n_partitions)]
        for node in partition:
            balance[node] += 1
        balance = max(balance) / (self.n_nodes/float(self.n_partitions))

        #TODO: Maybe can do this with matrix operations
        for edge in self.indices:
            if partition[edge[0]] != partition[edge[1]]:
                edge_cut += 1
        edge_cut = edge_cut / float(len(self.indices))

        return edge_cut, balance

class Individual(object):
    
    def __init__(self, id, layer_size):
        self.id = id
        self.neural_net = nn.Network(layer_size)
        self.fitness = []
        self.fitness_acc = 0.
    
    def __str__(self):
        return str(self.fitness_acc)

    def forward(self, n_times, test_graph, features):
        for _ in xrange(n_times):
            result = self.neural_net.forward(test_graph, features)
        return np.argmax(result, axis = 0)

    #Select a point in the NN for each layer and mix
    def crossover(self, fathers):
        for layer in range(len(self.neural_net.weights)):
            crossover_point = np.random.choice(len(self.neural_net.weights[layer]), 1)[0]
            self.neural_net.weights[layer] = np.concatenate((fathers[0].neural_net.weights[layer][:crossover_point], 
                                             fathers[1].neural_net.weights[layer][crossover_point:]))
            self.fitness = 0

    #mutate every layer
    def mutate(self):
        #print 'Mutating', self.id
        for layer in range(len(self.neural_net.weights)):
            mutation_point = np.random.choice(len(self.neural_net.weights[layer]), 1)[0]
            self.neural_net.weights[layer][mutation_point] = np.random.random()
    
    def calculate_fitness(self):
        average_edge_cut = [f[0] for f in self.fitness]
        average_balance = [f[1] for f in self.fitness]
        self.fitness_acc = np.average(average_edge_cut) + np.average(average_balance)

class GA(object):
    def __init__(self, partitions, initial_population, test_cases, mutation_rate=0.1, elitism=0.2):
        self.population = initial_population
        self.test_cases = test_cases
        self.partitions = partitions
        self.elitism = elitism
        self.mutation_rate = mutation_rate

    def evolve(self, convolutions):
        print 'Evolving...'
        for individual in tqdm(self.population):
            individual.fitness = []
            for test in self.test_cases:
                n_nodes = test.matrix.get_shape()[0]
                init = np.random.uniform(0, 1, (self.partitions, n_nodes))
                init = init / np.sum(init, axis=0, keepdims=True)
                #Forward NN
                partitions = individual.forward(convolutions, test.matrix, init)
                #Calculate Fitness
                individual.fitness.append(test.get_edge_cut_balance(partitions))
            individual.calculate_fitness()

        #sort
        self.population.sort(key=operator.attrgetter('fitness_acc'))
        for i in self.population:
            print i,
        print
        #Pick n_elite for crossover
        n_elite = int(len(self.population)*self.elitism)
        print 'Crossover...'
        for i in range(n_elite, len(self.population)):
            fathers = np.random.choice(n_elite, 2)
            self.population[i].crossover([self.population[fathers[0]], self.population[fathers[1]]])
            #Mutate
            if np.random.uniform() <= self.mutation_rate:
                self.population[i].mutate()
        
        #print test_with_metis(self.population[0], 50000, 1, self.partitions)

if __name__ == '__main__':
    a = TestIndividual(10, 0.5, 2)
    print a.get_edge_cut_balance([0,1,0,1,0,1,0,1,1,1])