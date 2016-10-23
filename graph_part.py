#!/usr/bin/env python
import networkx as nx
import random, string
import tempfile, getopt, sys, os
import numpy as np
import scipy.sparse as sp

def get_metis_instancen(partitions, n, m, seed):
    graph = nx.Graph(nx.barabasi_albert_graph(n, m, seed = seed))
    tmp_filename = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(5))
    with open(tmp_filename, 'w') as tmp_graph:
        tmp_graph.write('{} {}\n'.format(graph.number_of_nodes(), graph.number_of_edges()))
        for node in range(n):
            for edg in graph[node]:
                tmp_graph.write('{} '.format(edg + 1))
            tmp_graph.write('\n')
        
    os.system('{} {} {} &> /dev/null'.format('gpmetis', tmp_graph.name, partitions))
    partition_filename = '{}.part.{}'.format(tmp_filename, partitions)
    partition_data = []
    with open(partition_filename, 'r') as p:
        for l in p:
            partition_data.append(int(l))

    os.system('{} {}'.format('rm', tmp_filename))
    os.system('{} {}'.format('rm', partition_filename))
    fr = []
    to = [] 
    vals = []
    for node in graph:
        for edge in graph[node]:
            fr.append(node)
            to.append(edge)
            vals.append(.5 / len(graph[node]))
    for node in graph:
        fr.append(node)
        to.append(node)
        vals.append(.5)

    m = sp.coo_matrix((vals, (fr, to)), shape = (graph.number_of_nodes(), graph.number_of_nodes()))
    return m, partition_data




class Network:
    def __init__(self, layer_sizes):
        self.layer_sizes = layer_sizes

        self.weights = []
        for i in range(0, len(self.layer_sizes) - 1):
            self.weights.append(np.random.normal(0.0, .01, (layer_sizes[i+1], layer_sizes[i])))
        self.buffers = [None] * len(self.layer_sizes)
            
    def forward(self, A, x):
        n_nodes = x.shape[1]
        n_features = x.shape[0]
        self.buffers[0] = x.copy()

        for i in range(len(self.weights)):
            xx = self.buffers[i]
            step = np.dot(self.weights[i], xx)
            xxh = A.dot(step.T).T
            
            if i < len(self.weights) - 1:
                self.buffers[i+1] = np.tanh(xxh)
            else:
                self.buffers[i+1] = np.exp(xxh) / np.sum(np.exp(xxh), axis=0, keepdims=True)
        return self.buffers[-1]

    
    def backward(self, A, delta, lr = .0001):
        d = delta.copy()
        for i in range(len(self.weights) - 1, -1, -1):
            if i < len(self.weights) - 1:
                before_f = d * (1.0 - self.buffers[i+1] ** 2)

            d = A.dot(d.T).T
            dw = np.dot(self.buffers[i], d.T).T
            self.weights[i] += dw * lr  ##learning

            d = np.dot(self.weights[i].T, d)
        
                 
if __name__ == '__main__':
    try:
        opts, args = getopt.getopt(sys.argv[1:], 's:n:m:')
    except getopt.GetoptError:
        sys.exit(1)
    n_parts = 2
    seed = 0
    n = 20
    m = 3

    for opt, arg in opts:
        if opt == '-s':
            seed = int(arg)
        if opt == '-n':
            n = int(arg)
        if opt == '-m':
            m = int(arg)

    net = Network([n_parts, 10, 10, n_parts])


    n_epoch = 10000
    epoch = 0
    while epoch < n_epoch:
        #print "epoch: ", epoch
        #Create Random Graph with Partitions
        seed = np.random.rand()
        A, part = get_metis_instancen(n_parts, n, m, seed = seed)

        n_times = 1000
        y = None
        for i in range(n_times):
            #Create Input
            x = np.random.uniform(1.0 / n_parts, 0., size=(n_parts, n))
            x = x / np.sum(x, axis=0, keepdims=True)
            
            #Run Network
            #y = net.forward(A, x)
            y = net.forward(A, x)
            #y = net.forward(A, y)
                

            #Find optimal assignment
            guess = np.argmax(y, axis=0)
            match = np.zeros((n_parts, n_parts))
            for p, g in zip(part, guess):
                match[p, g] += 1

            mapping = np.zeros(n_parts, dtype=np.int)    
            for i in range(n_parts):
                v = match[i]
                assign = np.argmax(v)
                mapping[i] = assign
                match[:,assign] = -1

            mapped_part = [mapping[v] for v in part]
            
            #Create Targets
            t = np.zeros((n_parts, n))
            for i in range(n):
                t[mapped_part[i], i] = 1

            #Run Backward pass (Learn)
            print "T"
            print t
            print "Y"
            print y
            delta = t - y
            net.backward(A, delta, .001)
                        
        if epoch%100 == 0:
            print y
            print t
            print guess
            print net.weights[0]
        epoch += 1

