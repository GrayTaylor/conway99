import networkx as nx
import numpy as np
import oapackage
from datetime import datetime as dt

# Template building


def get_supermatrix_template(adj, forced_edge=None):
    # Given a graph, return template for a graph with an additional vertex
    # This vertex is forced to neighbour the first unsaturated vertex of input
    order = len(adj) + 1
    supermatrix = np.empty((order, order), dtype='int')

    for i in range(order - 1):
        for j in range(order - 1):
            supermatrix[i, j] = adj[i, j]

    first_unsat = 0
    while is_saturated_vertex(first_unsat, adj):
        first_unsat += 1

    for i in range(first_unsat):
        supermatrix[order - 1, i] = 0
        supermatrix[i, order - 1] = 0

    supermatrix[order - 1, first_unsat] = 1
    supermatrix[first_unsat, order - 1] = 1

    for i in range(first_unsat + 1, order - 1):
        supermatrix[order - 1, i] = 2
        supermatrix[i, order - 1] = 2

    supermatrix[order - 1, order - 1] = 0

    if forced_edge is not None:
        supermatrix[forced_edge[0], forced_edge[1]] = 1
        supermatrix[forced_edge[1], forced_edge[0]] = 1

    return supermatrix


# Functions on templates


def num_known_zeros(adj):
    return sum([len([x for x in r if x == 0]) for r in adj])


def num_known_ones(adj):
    return sum([len([x for x in r if x == 1]) for r in adj])


def num_unknowns(adj):
    return sum([len([x for x in r if x == 2]) for r in adj])


def has_unknown_values(adj):
    return max([len([x for x in r if x == 2]) for r in adj]) > 0


def plot_given_edges(adj):
    order = len(adj)
    A = np.empty((order, order), dtype='int')
    for i in range(order):
        for j in range(order):
            if adj[i, j] == 1:
                A[i, j] = 1
            else:
                A[i, j] = 0
    G = nx.from_numpy_matrix(A)
    nx.draw(G, with_labels=True)


# Saturated growing (defaults to degree 14 as in Conway 99-graph)


def vertex_degrees(adj):
    return [sum([x for x in r if x == 1]) for r in adj]


def vertex_saturation(adj, max_degree=14):
    return [sum([x for x in r if x == 1]) == max_degree for r in adj]


def has_saturated_vertex(adj, max_degree=14):
    for i in range(len(adj)):
        if sum([x for x in adj[i] if x == 1]) == max_degree:
            return True
    return False


def is_saturated_vertex(i, adj, max_degree=14):
    return sum([x for x in adj[i] if x == 1]) == max_degree


def first_saturated_vertex(adj, max_degree=14):
    for i in range(len(adj)):
        if sum([x for x in adj[i] if x == 1]) == max_degree:
            return i
    return None


# SRG conditions (defaults to SRG(99,14,1,2) case)


def mutual_neighbours(i, j, adj):
    return [k for k in range(len(adj)) if adj[i, k] == 1 and adj[j, k] == 1]


def lambda_compatible(adj, lmbda=1):
    # Compatibility is for a subgraph, so bound rather than equality
    for i in range(len(adj)):
        for j in range(i+1, len(adj)):
            if adj[i, j] == 1 and len(mutual_neighbours(i, j, adj)) > lmbda:
                return False
    return True


def mu_compatible(adj, mu=2):
    # Compatibility is for a subgraph, so bound rather than equality
    for i in range(len(adj)):
        for j in range(i+1, len(adj)):
            if adj[i, j] == 0 and len(mutual_neighbours(i, j, adj)) > mu:
                return False
    return True


def meets_adjacency_requirements(adj, lmbda=1, mu=2, debug=False):
    # where vertices have full degree,
    # mutual (non)-neighbour conditions can be checked
    for i in range(len(adj)):
        if is_saturated_vertex(i, adj):
            for j in range(len(adj)):
                m = mutual_neighbours(i, j, adj)
                if adj[i, j] == 1 and len(m) != lmbda:
                    if debug:
                        print('Error: Neighbour {} of {} has {} \
                              mutual neighbours {}'.format(j, i, len(m), m))
                    return False
                if i != j and adj[i, j] == 0 and len(m) != mu:
                    if debug:
                        print('Error: Non-Neighbour {} of {} has {} \
                               mutual neighbours {}'.format(j, i, len(m), m))
                    return False
    return True


def graph_is_valid(adj):
    # for proper subgraphs, not templates!
    return (lambda_compatible(adj)
            and mu_compatible(adj)
            and meets_adjacency_requirements(adj))


# Branch and bound on templates

def first_unspecified(adj):
    for i in range(len(adj)):
        for j in range(len(adj)):
            if adj[i, j] == 2:
                return i, j


def branches(adj):
    b1 = adj.copy()
    b0 = adj.copy()
    i, j = first_unspecified(adj)
    b1[i, j] = 1
    b1[j, i] = 1
    b0[i, j] = 0
    b0[j, i] = 0
    return b0, b1


def templates_to_valid_graphs(seed_templates, verbose=0):
    # For each template, populate unknowns with 0/1 values,
    # Subject to lambda and mu compatibility throughout
    # Then eliminate graphs that don't meet known adjacency requirements
    # (this check cannot be performed on templates,
    # as mutual neighbours not necessarily yet determined)

    candidates = [a for a in seed_templates]
    solutions = []

    while len(candidates) > 0:
        if verbose > 0:
            print('Currently {} graphs, {} candidates'.format(len(solutions),
                                                              len(candidates)))
        current_candidate = candidates.pop()
        adj0, adj1 = branches(current_candidate)

        if lambda_compatible(adj0) and mu_compatible(adj0):
            if has_unknown_values(adj0):
                if verbose > 1:
                    print('Adding branch 0 candidate')
                candidates.append(adj0)
            else:
                if verbose > 1:
                    print('Branch 0 yielded compatible graph')
                solutions.append(adj0)
        else:
            if verbose > 1:
                print('Branch 0 invalid')

        if lambda_compatible(adj1) and mu_compatible(adj1):
            if has_unknown_values(adj1):
                if verbose > 1:
                    print('Adding branch 1 candidate')
                candidates.append(adj1)
            else:
                if verbose > 1:
                    print('Branch 1 yielded compatible graph')
                solutions.append(adj1)
        else:
            if verbose > 1:
                print('Branch 1 invalid')

    valid_soln = [s for s in solutions if graph_is_valid(s)]
    if verbose > 0:
        print('Reduces to {} valid graphs'.format(len(valid_soln)))
    return valid_soln


def find_valid_supergraphs(seed_matrices, forced_edge=None):

    templates = [get_supermatrix_template(adj, forced_edge)
                 for adj in seed_matrices]
    print('{}: {} seed templates generated'.format(dt.now(), len(templates)))

    valid_supergraphs = templates_to_valid_graphs(templates)
    print('{}: {} valid graphs from templates'.format(dt.now(),
                                                      len(valid_supergraphs)))

    supergraph_reps = reduce_mod_equivalence(valid_supergraphs, verbose=True)
    print('{}: Reduced to {} representatives'.format(dt.now(),
                                                     len(supergraph_reps)))
    return supergraph_reps


# Isomorphism testing

def inverse_permutation(perm):
    inverse = [0] * len(perm)
    for i, p in enumerate(perm):
        inverse[p] = i
    return inverse


def are_equivalent(mat1, mat2):
    tr1 = oapackage.reduceGraphNauty(mat1, verbose=0)
    tri1 = inverse_permutation(tr1)
    mat1_reduced = oapackage.transformGraphMatrix(mat1, tri1)

    tr2 = oapackage.reduceGraphNauty(mat2, verbose=0)
    tri2 = inverse_permutation(tr2)
    mat2_reduced = oapackage.transformGraphMatrix(mat2, tri2)

    return np.all(mat1_reduced == mat2_reduced)


def reduce_mod_equivalence_old(candidates, verbose=False):
    worklist = [m for m in candidates]
    replist = []
    while len(worklist) > 0:
        if verbose:
            print('\t{} candidates, {} representatives'.format(len(worklist),
                                                               len(replist)))
        rep = worklist.pop()
        replist.append(rep)
        worklist = [m for m in worklist if not are_equivalent(rep, m)]
    if verbose:
        print('\t{} candidates, {} representatives'.format(len(worklist),
                                                           len(replist)))
    return replist


def reduce_mod_equivalence(candidates, verbose=False):
    reps = []
    reduced_reps = []
    for k in range(len(candidates)):
        cand = candidates[k]
        tr = oapackage.reduceGraphNauty(cand, verbose=0)
        tri = inverse_permutation(tr)
        cand_reduced = oapackage.transformGraphMatrix(cand, tri)
        if not any(np.array_equal(cand_reduced, c) for c in reduced_reps):
            reduced_reps.append(cand_reduced)
            reps.append(cand)
            if verbose:
                print('\t{} reps for {} candidates'.format(len(reps), k + 1))
    return reps
