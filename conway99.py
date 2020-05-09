import networkx as nx
import numpy as np
import oapackage
from datetime import datetime as dt

# Template building


def get_supermatrix_template(adj, forced_edges=None, forced_non_edges=None,
                             max_degree=14):
    # Given a graph, return template for a graph with an additional vertex
    # Recognise saturated vertices can't have new neighbours
    # Optionally, also force some edges
    order = len(adj) + 1
    supermatrix = np.empty((order, order), dtype='int')

    supermatrix[0:order-1, 0:order-1] = adj

    vd = vertex_degrees(adj)

    # if vertex already saturated, new vertex can't be a nhbr
    for i in range(order - 1):
        if vd[i] >= max_degree:
            nhbr_i = 0
        else:
            nhbr_i = 2
        supermatrix[order - 1, i] = nhbr_i
        supermatrix[i, order - 1] = nhbr_i

    # can't self-neighbour
    supermatrix[order - 1, order - 1] = 0

    if forced_edges is not None:
        for e in forced_edges:
            supermatrix[e[0], e[1]] = 1
            supermatrix[e[1], e[0]] = 1

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


def graph_is_valid(adj, lmbda=1, mu=2):
    # for proper subgraphs, not templates!
    return (lambda_compatible(adj, lmbda)
            and mu_compatible(adj, mu)
            and meets_adjacency_requirements(adj, lmbda, mu))


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
    for k, cand in enumerate(candidates):
        tr = oapackage.reduceGraphNauty(cand, verbose=0)
        tri = inverse_permutation(tr)
        cand_reduced = oapackage.transformGraphMatrix(cand, tri)
        cand_reduced_list = tuple(map(tuple, cand_reduced))
        if cand_reduced_list not in reduced_reps:
            reduced_reps[cand_reduced_list] = 1
            reps.append(cand)
            if verbose:
                print('\t{} reps from {} candidates'.format(len(reps), k + 1))
    return reps


# Exploit known subgraph during growing

def known_mutuals(adj):
    order = len(adj)
    mutuals = np.empty((order, order), dtype='object')
    for i in range(order):
        for j in range(i, order):
            mn = mutual_neighbours(i, j, adj)
            mutuals[i, j] = mn
            mutuals[j, i] = mn
    return mutuals


def mutual_nhbrs_given_subgraph_mutuals(i, j, adj, subgraph_mutuals):
    new_v = len(adj) - 1
    if i == new_v or j == new_v:
        return mutual_neighbours(i, j, adj)
    else:
        mutuals = [v for v in subgraph_mutuals[i, j]]
        if adj[i, new_v] == 1 and adj[j, new_v] == 1:
            mutuals.append(new_v)
        return mutuals


def lambda_compatible_from_subgraph_mutuals(adj, subgraph_mutuals, lmbda=1):
    # Compatibility is for a subgraph, so bound rather than equality
    for i in range(len(adj)):
        for j in range(i+1, len(adj)):
            if adj[i, j] == 1:
                nhbrs = mutual_nhbrs_given_subgraph_mutuals(i, j, adj,
                                                            subgraph_mutuals)
                if len(nhbrs) > lmbda:
                    return False
    return True


def mu_compatible_from_subgraph_mutuals(adj, subgraph_mutuals, mu=2):
    # Compatibility is for a subgraph, so bound rather than equality
    for i in range(len(adj)):
        for j in range(i+1, len(adj)):
            if adj[i, j] == 0:
                nhbrs = mutual_nhbrs_given_subgraph_mutuals(i, j, adj,
                                                            subgraph_mutuals)
                if len(nhbrs) > mu:
                    return False
    return True


def meets_adj_reqs_from_subgraph_mutuals(adj, subgraph_mutuals,
                                         lmbda=1, mu=2, debug=False):
    # where vertices have full degree,
    # mutual (non)-neighbour conditions can be checked
    for i in range(len(adj)):
        if is_saturated_vertex(i, adj):
            for j in range(len(adj)):
                m = mutual_nhbrs_given_subgraph_mutuals(i, j, adj,
                                                        subgraph_mutuals)
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


def graph_is_valid_from_subgraph_mutuals(adj, subgraph_mutuals,
                                         lmbda=1, mu=2, max_degree=14):
    # for proper subgraphs, not templates!
    return (lambda_compatible_from_subgraph_mutuals(adj, subgraph_mutuals,
                                                    lmbda)
            and mu_compatible_from_subgraph_mutuals(adj, subgraph_mutuals, mu)
            and meets_adj_reqs_from_subgraph_mutuals(adj, subgraph_mutuals,
                                                     lmbda, mu, max_degree))


def template_to_valid_graphs(seed_template, subgraph_mutuals, verbose=0,
                             lmbda=1, mu=2, max_degree=14):
    # For each template, populate unknowns with 0/1 values,
    # Subject to lambda and mu compatibility throughout
    # Then eliminate graphs that don't meet known adjacency requirements
    # (this check cannot be performed on templates,
    # as mutual neighbours not necessarily yet determined)

    candidates = [seed_template]
    solutions = []

    while len(candidates) > 0:
        if verbose > 0:
            print('Currently {} graphs, {} candidates'.format(len(solutions),
                                                              len(candidates)))
        current_candidate = candidates.pop()
        adj0, adj1 = branches(current_candidate)

        # Branch with 0
        lmbda_comp0 = lambda_compatible_from_subgraph_mutuals(adj0,
                                                              subgraph_mutuals,
                                                              lmbda)
        mu_comp0 = mu_compatible_from_subgraph_mutuals(adj0,
                                                       subgraph_mutuals,
                                                       mu)

        if lmbda_comp0 and mu_comp0:
            if has_unknown_values_supermatrix(adj0):
                if verbose > 1:
                    print('Adding branch 0 candidate')
                candidates.append(adj0)
            else:
                x = graph_is_valid_from_subgraph_mutuals(adj0,
                                                         subgraph_mutuals,
                                                         lmbda, mu,
                                                         max_degree)
                if x:
                    solutions.append(adj0)
                    if verbose > 1:
                        print('Branch 0 yielded valid graph')
                else:
                    if verbose > 1:
                        print('Branch 0 yielded compatible but invalid graph')
        else:
            if verbose > 1:
                print('Branch 0 invalid')

        # Branch with 1
        lmbda_comp1 = lambda_compatible_from_subgraph_mutuals(adj1,
                                                              subgraph_mutuals,
                                                              lmbda)
        mu_comp1 = mu_compatible_from_subgraph_mutuals(adj1,
                                                       subgraph_mutuals,
                                                       mu)

        if lmbda_comp1 and mu_comp1:
            if has_unknown_values_supermatrix(adj1):
                if verbose > 1:
                    print('Adding branch 1 candidate')
                candidates.append(adj1)
            else:
                x = graph_is_valid_from_subgraph_mutuals(adj1,
                                                         subgraph_mutuals,
                                                         lmbda, mu,
                                                         max_degree)
                if x:
                    solutions.append(adj1)
                    if verbose > 1:
                        print('Branch 1 yielded valid graph')
                else:
                    if verbose > 1:
                        print('Branch 1 yielded compatible but invalid graph')
        else:
            if verbose > 1:
                print('Branch 1 invalid')

    return solutions


def find_valid_supergraphs(seed_matrices,
                           forced_edges=None,
                           forced_non_edges=None,
                           verbose=True,
                           lmbda=1, mu=2, max_degree=14):

    valid_supergraphs = []
    print('{}: Starting with {} seeds'.format(dt.now(), len(seed_matrices)))

    for s in seed_matrices:
        template = get_supermatrix_template(s, forced_edges, forced_non_edges,
                                            max_degree)
        subgraph_mutuals = known_mutuals(s)
        valid_supergraphs_of_s = template_to_valid_graphs(template,
                                                          subgraph_mutuals,
                                                          lmbda, mu,
                                                          max_degree)
        valid_supergraphs.extend(valid_supergraphs_of_s)

    print('{}: {} valid graphs from templates'.format(dt.now(),
                                                      len(valid_supergraphs)))

    supergraph_reps = reduce_mod_equivalence(valid_supergraphs, verbose)
    print('{}: Reduced to {} representatives'.format(dt.now(),
                                                     len(supergraph_reps)))
    return supergraph_reps


def templates_to_valid_graphs(seed_templates, verbose=0,
                              lmbda=1, mu=2, max_degree=14):
    valid_graphs = []
    for s in seed_templates:
        subgraph = s[:-1, :-1]
        subgraph_mutuals = known_mutuals(subgraph)
        valid_graphs.extend(template_to_valid_graphs(s, subgraph_mutuals,
                                                     verbose,
                                                     lmbda, mu, max_degree))
    return valid_graphs


# 'Greedy' saturation - always look to introduce edges on the
# highest degree unsaturated vertex, and to construct missing mutuals

def get_supermatrix_template_greedy(adj, max_degree=14):
    # Given a graph, return template for a graph with an additional vertex
    # Recognise saturated vertices can't have new neighbours
    # identify an unsaturated vertex of highest degree,
    # and a vertex that it requires a mutual nhbr with
    # set the new vertex to neighbour both

    order = len(adj) + 1
    supermatrix = np.empty((order, order), dtype='int')

    supermatrix[0:order-1, 0:order-1] = adj

    vd = vertex_degrees(adj)
    max_unsat = max([d for d in vd if d < max_degree])
    first_max_unsat = 0
    while vd[first_max_unsat] != max_unsat:
        first_max_unsat += 1

    # if vertex already saturated, new vertex can't be a nhbr
    # our first maximal unsaturated should be forced as a nhbr
    for i in range(order - 1):
        if vd[i] >= max_degree:
            nhbr_i = 0
        elif i == first_max_unsat:
            nhbr_i = 1
        else:
            nhbr_i = 2
        supermatrix[order - 1, i] = nhbr_i
        supermatrix[i, order - 1] = nhbr_i

    # can't self-neighbour
    supermatrix[order - 1, order - 1] = 0

    # look for a required mutual nhbr
    req_mutual = 0
    while (adj[first_max_unsat, req_mutual]
           + len(mutual_neighbours(first_max_unsat, req_mutual, adj))) >= 2:
        req_mutual += 1
        if req_mutual > order - 1:
            break

    if req_mutual <= order - 1:
        supermatrix[req_mutual, order - 1] = 1
        supermatrix[order - 1, req_mutual] = 1

    return supermatrix


def find_valid_supergraphs_greedy(seed_matrices, verbose=True):
    valid_supergraphs = []
    print('{}: Starting with {} seeds'.format(dt.now(), len(seed_matrices)))

    for s in seed_matrices:
        template = get_supermatrix_template_greedy(s)
        subgraph_mutuals = known_mutuals(s)
        valid_supergraphs_of_s = template_to_valid_graphs(template,
                                                          subgraph_mutuals)
        valid_supergraphs.extend(valid_supergraphs_of_s)

    print('{}: {} valid graphs from templates'.format(dt.now(),
                                                      len(valid_supergraphs)))

    supergraph_reps = reduce_mod_equivalence(valid_supergraphs, verbose)
    print('{}: Reduced to {} representatives'.format(dt.now(),
                                                     len(supergraph_reps)))
    return supergraph_reps
