import genetic
import nn
from tqdm import tqdm

if __name__ == '__main__':
    n_population = 100
    test_n_nodes = 10
    test_n_partition = 2
    n_layers = [test_n_partition, 32, 32, 32, test_n_partition]

    generations = 10

    adversaries = []
    individuals = []
    print 'Loading tests'
    for i in tqdm(range(n_population)):
        adversaries.append(genetic.TestIndividual(test_n_nodes, 1, test_n_partition))
        individuals.append(genetic.Individual(n_layers))

    ga = genetic.GA(test_n_partition, individuals, adversaries)
    ga.evolve(1)
