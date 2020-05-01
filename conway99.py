import networkx as nx
import numpy as np
import oapackage
from datetime import datetime as dt

# Template building


def get_supermatrix_template(adj, forced_edges=None, forced_non_edges=None):
    # Given a graph, return template for a graph with an additional vertex
    # Recognise saturated vertices can't have new neighbours
    # Optionally, also force some edges
    order = len(adj) + 1
    supermatrix = np.empty((order, order), dtype='int')

    supermatrix[0:order-1, 0:order-1] = adj

    # cannot neighbour an already saturated vertex
    for i in range(order - 1):
        if is_saturated_vertex(i, adj):
            nhbr_i = 0
        else:
            nhbr_i = 2
        supermatrix[order - 1, i] = nhbr_i
        supermatrix[i, order - 1] = nhbr_i

    # cannot self-neighbour!
    supermatrix[order - 1, order - 1] = 0

    # apply specified edges
    if forced_edges is not None:
        for e in forced_edges:
            supermatrix[e[0], e[1]] = 1
            supermatrix[e[1], e[0]] = 1

    # apply specified non-edges
    if forced_non_edges is not None:
        for e in forced_non_edges:
            supermatrix[e[0], e[1]] = 0
            supermatrix[e[1], e[0]] = 0

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


def has_unknown_values_supermatrix(adj):
    # special case of a template that is a supermatrix of a graph
    # if unknowns exist, they relate to the new vertex only
    # so suffices to check final row
    # More likely to find at end of array, so search in reverse
    for k in range(len(adj[-1]) - 1, -1, -1):
        if adj[-1][k] == 2:
            return True
    return False


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
            if has_unknown_values_supermatrix(adj0):
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
            if has_unknown_values_supermatrix(adj1):
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


def find_valid_supergraphs(seed_matrices,
                           forced_edges=None,
                           forced_non_edges=None,
                           verbose=True):

    templates = [get_supermatrix_template(adj, forced_edges, forced_non_edges)
                 for adj in seed_matrices]
    print('{}: {} seed templates generated'.format(dt.now(), len(templates)))

    valid_supergraphs = templates_to_valid_graphs(templates)
    print('{}: {} valid graphs from templates'.format(dt.now(),
                                                      len(valid_supergraphs)))

    supergraph_reps = reduce_mod_equivalence(valid_supergraphs, verbose)
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


def reduce_mod_equivalence_short(candidates, verbose=False):
    # May offer advantages over standard proc if few candidates
    # As then cost of array list comparison cheaper than
    # conversion to tuple for hash lookup
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


def reduce_mod_equivalence(candidates, verbose=False):
    reps = []
    reduced_reps = {}
    for k in range(len(candidates)):
        cand = candidates[k]
        tr = oapackage.reduceGraphNauty(cand, verbose=0)
        tri = inverse_permutation(tr)
        cand_reduced = oapackage.transformGraphMatrix(cand, tri)
        cand_reduced_list = tuple(map(tuple, cand_reduced))
        if cand_reduced_list not in reduced_reps:
            reduced_reps[cand_reduced_list] = 1
            reps.append(cand)
            if verbose:
                print('\t{} reps for {} candidates'.format(len(reps), k + 1))
    return reps


# Refine to special case of a supergraph of a known subgraph
# Here most adjacency is already known, so can save on recompute

def mutual_neighbours_supergraph(i, j, adj, subgraph_mutuals):
    new_v = len(adj) - 1
    if i == new_v or j == new_v:
        return mutual_neighbours(i, j, adj)
    else:
        mutuals = [v for v in subgraph_mutuals[i, j]]
        if adj[i, new_v] == 1 and adj[j, new_v] == 1:
            mutuals.append(new_v)
        return mutuals


def lambda_compatible_supergraph(adj, subgraph_mutuals, lmbda=1):
    # Compatibility is for a subgraph, so bound rather than equality
    for i in range(len(adj)):
        for j in range(i+1, len(adj)):
            if adj[i, j] == 1 and len(mutual_neighbours_supergraph(i, j, adj, subgraph_mutuals)) > lmbda:
                return False
    return True


def mu_compatible_supergraph(adj, subgraph_mutuals, mu=2):
    # Compatibility is for a subgraph, so bound rather than equality
    for i in range(len(adj)):
        for j in range(i+1, len(adj)):
            if adj[i, j] == 0 and len(mutual_neighbours_supergraph(i, j, adj, subgraph_mutuals)) > mu:
                return False
    return True


def templates_to_valid_graphs_supergraph(seed_templates, verbose=0):
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

        sub_order = len(current_candidate) - 1
        subgraph_mutuals = np.empty((sub_order, sub_order), dtype='object')
        for i in range(sub_order):
            for j in range(sub_order):
                subgraph_mutuals[i, j] = mutual_neighbours(i, j, current_candidate[:sub_order, :sub_order])

        if lambda_compatible_supergraph(adj0, subgraph_mutuals) and mu_compatible_supergraph(adj0, subgraph_mutuals):
            if has_unknown_values_supermatrix(adj0):
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

        if lambda_compatible_supergraph(adj1, subgraph_mutuals) and mu_compatible_supergraph(adj1, subgraph_mutuals):
            if has_unknown_values_supermatrix(adj1):
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


def find_valid_supergraphs_v2(seed_matrices,
                              forced_edges=None,
                              forced_non_edges=None,
                              verbose=True):

    templates = [get_supermatrix_template(adj, forced_edges, forced_non_edges)
                 for adj in seed_matrices]
    print('{}: {} seed templates generated'.format(dt.now(), len(templates)))

    valid_supergraphs = templates_to_valid_graphs_supergraph(templates)
    print('{}: {} valid graphs from templates'.format(dt.now(),
                                                      len(valid_supergraphs)))

    supergraph_reps = reduce_mod_equivalence(valid_supergraphs, verbose)
    print('{}: Reduced to {} representatives'.format(dt.now(),
                                                     len(supergraph_reps)))
    return supergraph_reps
