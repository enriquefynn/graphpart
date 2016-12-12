import networkx as nx
import numpy as np
import helpers
import genetic

def test_with_metis(individual, number_of_nodes, density, n_partitions, edges=2, convolutions=10):
    test_individual = genetic.TestIndividual(number_of_nodes, density, n_partitions, edges=edges)
    init = np.random.uniform(0, 1, (n_partitions, number_of_nodes))
    init = init / np.sum(init, axis=0, keepdims=True)
    partitions = individual.forward(convolutions, test_individual.matrix, init)
    individual_fitness = test_individual.get_edge_cut_balance(partitions)
    metis_result = helpers.get_metis_edge_cut(test_individual.graph, n_partitions)

    return {"METIS:" : metis_result, "Individual": individual_fitness}