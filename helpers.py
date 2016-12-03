import random
import string
import subprocess
import os

def get_metis_edge_cut(graph, partitions):
    tmp_file_name = 'tmp.' + ''.join(random.choice(
        string.ascii_uppercase + string.digits) for _ in range(10))
    metis_file = tmp_file_name + '.part.' + str(partitions)
    try:
        n_users = graph.number_of_nodes()
        edges = graph.number_of_edges()
        with open(tmp_file_name, 'w') as metis_f:
            metis_f.write('{} {}\n'.format(n_users, edges))
            for node in graph:
                for edge in graph.edges_iter(node):
                    metis_f.write('{} '.format(int(edge[1]) + 1))
                metis_f.write('\n')
        p = subprocess.Popen(["gpmetis", tmp_file_name, str(partitions)],
                stdout = subprocess.PIPE, stderr=subprocess.PIPE)
        out, err = p.communicate()
        if err != '':
            print('Error running metis')

        metis_partition = []
        with open(metis_file, 'r') as metis_f:
            for line in metis_f:
                metis_partition.append(int(line) + 1)
        edge_cut = 0.
        for node in graph.edges_iter():
            if metis_partition[node[0]] != metis_partition[node[1]]:
                edge_cut += 1
        edge_cut /= graph.number_of_edges()
        return edge_cut, metis_partition

    finally:
        os.remove(tmp_file_name)
        os.remove(metis_file)
