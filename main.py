#!/usr/bin/env python
import genetic
import nn
from tqdm import tqdm

if __name__ == '__main__':
    n_population = 100
    test_n_nodes = 100
    test_n_partition = 2
    n_convolutions = 3
    mutation_rate = 0.1
    graph_density = 1

    n_layers = [test_n_partition, 32, 32, 32, 32, test_n_partition]

    generations = 10

    adversaries = []
    individuals = []
    print 'Loading tests'
    for i in tqdm(range(n_population)):
        adversaries.append(genetic.TestIndividual(test_n_nodes, graph_density, test_n_partition))
        individuals.append(genetic.Individual(i, n_layers))

    ga = genetic.GA(test_n_partition, individuals, adversaries, mutation_rate=mutation_rate)
    for g in range(generations):
        ga.evolve(n_convolutions)
