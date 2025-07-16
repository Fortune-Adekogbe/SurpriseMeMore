import networkx as nx
import numpy as np
import scipy
from numba import jit
from numba.typed import List
from scipy.sparse import isspmatrix
from scipy.special import comb

from . import cp_functions as cp


@jit(nopython=True)
def compute_neighbours_alt(adj):
    neig_dict = {}
    for i in range(adj.shape[0]):
        neigh1 = np.where(adj[i, :])[0]
        neig_dict[i] = neigh1
    return neig_dict


@jit(nopython=True)
def compute_max_neighbor(node_idx, neighbors_dict, num_nodes): 
    max_cn = -1 
    index_max_cn = node_idx 

    # Check if node_idx is in neighbors_dict (it should be if dict is built for 0..num_nodes-1)
    if node_idx not in neighbors_dict: return index_max_cn # Should not happen
    node_ng = neighbors_dict[node_idx]

    if not node_ng.size: 
        return index_max_cn 

    for ii in range(num_nodes): 
        if ii == node_idx:
            continue
        
        if ii not in neighbors_dict: continue # Should not happen
        ng_ii = neighbors_dict[ii]
        if not ng_ii.size: 
            continue
            
        aux_cn = intersection(node_ng, ng_ii)
        if aux_cn > max_cn:
            max_cn = aux_cn
            index_max_cn = ii
        elif aux_cn == max_cn: 
             # Tie-breaking: prefer smaller index if that was the original intent.
            if ii < index_max_cn : index_max_cn = ii
            
    return index_max_cn


@jit(nopython=True)
def intersection(arr1, arr2):
    intersect = 0
    if arr1.shape[0] > arr2.shape[0]:
        arr1, arr2 = arr2, arr1 
    
    for i_val in arr1:
        # Numba's 'in' operator on arrays can be slow for large arrays.
        # For typical neighbor lists, this might be okay.
        # If performance is an issue here, and arrays are sorted, a merge-like scan is better.
        # If unsorted and large, set intersection logic (not directly available in nopython mode without workarounds).
        is_present = False
        for j_val in arr2:
            if i_val == j_val:
                is_present = True
                break
        if is_present:
            intersect += 1
            
    return intersect


def compute_neighbours(adj):
    lista_neigh = []
    for ii in np.arange(adj.shape[0]):
        if hasattr(adj, "tobsr"): 
            lista_neigh.append(adj[ii, :].nonzero()[1])
        else: 
            lista_neigh.append(np.where(adj[ii, :])[0])
    return lista_neigh


@jit(nopython=True)
def compute_cn(adjacency_binary):
    num_nodes = adjacency_binary.shape[0]
    cn_table = np.zeros((num_nodes, num_nodes), dtype=adjacency_binary.dtype) 
    
    # Precompute neighbor representations for all nodes if graph is undirected-like logic
    # For common successors (directed): use adj[i,:] and adj[j,:]
    # For common predecessors: use adj[:,i] and adj[:,j]
    # Original code's (adj[i,:] + adj[:,i]).astype(bool) implies undirected interpretation of "neighbor"
    
    node_influence_sets = [ (adjacency_binary[i, :] + adjacency_binary[:, i]).astype(np.bool_) for i in range(num_nodes) ]

    for i in np.arange(num_nodes):
        neighbour_i_infl = node_influence_sets[i]
        for j in np.arange(i + 1, num_nodes):
            neighbour_j_infl = node_influence_sets[j]
            # Common neighbors based on this "influence set"
            cn_val = 0
            for k_node in range(num_nodes): # Iterate over all possible common neighbors
                if neighbour_i_infl[k_node] and neighbour_j_infl[k_node]:
                    cn_val +=1
            cn_table[i, j] = cn_table[j, i] = cn_val
            # Original: np.sum(np.logical_and(neighbour_i, neighbour_j))
            # This might be faster if Numba optimizes array ops well.
            # cn_table[i, j] = cn_table[j, i] = np.sum(np.logical_and(neighbour_i_infl, neighbour_j_infl))

    return cn_table

@jit(nopython=True)
def common_neigh_init_guess_strong_old(adjacency):
    """Generates a preprocessed initial guess based on the common neighbours
     of nodes. It makes a stronger aggregation of nodes based on
      the common neighbours similarity.

    :param adjacency: Adjacency matrix.
    :type adjacency: numpy.ndarray
    :return: Initial guess for nodes memberships.
    :rtype: np.array
    """
    cn_table = compute_cn(adjacency)
    memberships = np.array(
        [k for k in np.arange(adjacency.shape[0], dtype=np.int32)])
    argsorted = np.argsort(adjacency.astype(np.bool_).sum(axis=1))[::-1]
    for aux_node1 in argsorted:
        aux_tmp = memberships == aux_node1
        memberships[aux_tmp] = memberships[np.argmax(cn_table[aux_node1])]
    return memberships


@jit(nopython=True)
def common_neigh_init_guess_strong(adjacency_binary):
    num_nodes = adjacency_binary.shape[0]
    # Make sure neighbors_dict keys are compatible with NumbaDict if it were used explicitly
    # For untyped dict passed to JIT func, Numba tries to infer.
    # Explicit Numba typed.Dict could be more robust if issues arise.
    neighbors = compute_neighbours_alt(adjacency_binary) 
    memberships = np.arange(num_nodes, dtype=np.int32)
    
    degrees = np.zeros(num_nodes, dtype=np.int32)
    is_symmetric_adj = np.all(adjacency_binary == adjacency_binary.T) # Correct symmetry check

    for i in range(num_nodes):
        degrees[i] = np.sum(adjacency_binary[i, :])
        if not is_symmetric_adj: 
             degrees[i] += np.sum(adjacency_binary[:, i])

    argsorted_degrees = np.argsort(degrees)[::-1]

    for aux_node1 in argsorted_degrees:
        current_membership_val = memberships[aux_node1]
        idx_max_cn_node = compute_max_neighbor(aux_node1, neighbors, num_nodes)
        target_membership_val = memberships[idx_max_cn_node]
        
        if current_membership_val != target_membership_val: 
            for i in range(num_nodes):
                if memberships[i] == current_membership_val:
                    memberships[i] = target_membership_val
    return memberships


@jit(nopython=True)
def common_neigh_init_guess_weak_old(adjacency):
    """Generates a preprocessed initial guess based on the common neighbours
     of nodes. It makes a weaker aggregation of nodes based on
      the common neighbours similarity.

    :param adjacency: Adjacency matrix.
    :type adjacency: numpy.ndarray
    :return: Initial guess for nodes memberships.
    :rtype: np.array
    """
    cn_table = compute_cn(adjacency)
    memberships = np.array(
        [k for k in np.arange(adjacency.shape[0], dtype=np.int32)])
    degree = (adjacency.astype(np.bool_).sum(axis=1)
              + adjacency.astype(np.bool_).sum(axis=0))
    avg_degree = np.mean(degree)
    argsorted = np.argsort(degree)[::-1]
    for aux_node1 in argsorted:
        if degree[aux_node1] >= avg_degree:
            aux_tmp = memberships == aux_node1
            memberships[aux_tmp] = memberships[np.argmax(cn_table[aux_node1])]
    return memberships


@jit(nopython=True)
def common_neigh_init_guess_weak(adjacency_binary):
    num_nodes = adjacency_binary.shape[0]
    neighbors = compute_neighbours_alt(adjacency_binary)
    memberships = np.arange(num_nodes, dtype=np.int32)

    degrees = np.zeros(num_nodes, dtype=np.int32)
    is_symmetric_adj = np.all(adjacency_binary == adjacency_binary.T) # Correct symmetry check

    for i in range(num_nodes):
        degrees[i] = np.sum(adjacency_binary[i, :])
        if not is_symmetric_adj:
             degrees[i] += np.sum(adjacency_binary[:, i])
    
    avg_degree = 0.0
    if num_nodes > 0 : 
        sum_deg = 0
        for d_val in degrees: sum_deg += d_val
        avg_degree = float(sum_deg) / num_nodes
    
    argsorted_degrees = np.argsort(degrees)[::-1]

    for aux_node1 in argsorted_degrees:
        if degrees[aux_node1] >= avg_degree:
            current_membership_val = memberships[aux_node1]
            idx_max_cn_node = compute_max_neighbor(aux_node1, neighbors, num_nodes)
            target_membership_val = memberships[idx_max_cn_node]
            
            if current_membership_val != target_membership_val:
                for i in range(num_nodes):
                    if memberships[i] == current_membership_val:
                        memberships[i] = target_membership_val
    return memberships


def eigenvector_init_guess(adjacency, is_directed):
    """Generates an initial guess for core periphery detection method: nodes
    with higher eigenvector centrality are in the core.

    :param adjacency: Adjacency matrix.
    :type adjacency: np.ndarray
    :param is_directed: True if the network is directed.
    :type is_directed: bool
    :return: Initial guess.
    :rtype: np.ndarray
    """
    # TODO: Vedere come funziona la parte pesata

    n_nodes = adjacency.shape[0]
    aux_nodes = int(np.ceil((n_nodes * 5) / 100))
    if is_directed:
        graph = nx.from_numpy_array(adjacency, create_using=nx.DiGraph)
        centra = nx.eigenvector_centrality_numpy(graph)
        centra1 = np.array([centra[key] for key in centra])
        membership = np.ones_like(centra1, dtype=np.int32)
        membership[np.argsort(centra1)[::-1][:aux_nodes]] = 0

    else:
        graph = nx.from_numpy_array(adjacency, create_using=nx.Graph)
        centra = nx.eigenvector_centrality_numpy(graph)
        centra1 = np.array([centra[key] for key in centra])
        membership = np.ones_like(centra1, dtype=np.int32)
        membership[np.argsort(centra1)[::-1][:aux_nodes]] = 0

    return membership


def fixed_clusters_init_guess_cn(adjacency_binary, n_clust):
    num_nodes = adjacency_binary.shape[0]
    
    if n_clust <= 0 : # Handle n_clust=0 or negative
        if num_nodes > 0: return np.zeros(num_nodes, dtype=np.int32)
        else: return np.array([], dtype=np.int32)

    aux_memb = np.zeros(num_nodes, dtype=np.int32) # Default to cluster 0

    cn = compute_cn(adjacency_binary)
    
    degrees = np.zeros(num_nodes, dtype=np.int32)
    is_symmetric_adj = np.all(adjacency_binary == adjacency_binary.T) # Correct symmetry check

    for i in range(num_nodes):
        degrees[i] = np.sum(adjacency_binary[i, :]) 
        if not is_symmetric_adj:
            degrees[i] += np.sum(adjacency_binary[:, i]) 

    sorted_degree_indices = np.argsort(degrees)[::-1]
    
    if n_clust > num_nodes: # More clusters than nodes: assign each node to its own cluster
        # And pad with empty clusters if strictly n_clust labels are needed (not typical)
        # This case usually means n_clust should be num_nodes.
        # For now, assume n_clust <= num_nodes.
        # If this error is desired: raise ValueError("Number of clusters cannot exceed number of nodes.")
        # Or, cap n_clust:
        n_clust = num_nodes

    seeds = np.zeros(n_clust, dtype=np.int32)
    
    # Seed selection: take top n_clust degree nodes if available and distinct enough
    # Simplified: top n_clust distinct degree nodes.
    # If fewer than n_clust nodes have edges, some seeds might be 0-degree.
    
    # Take top `n_clust` nodes by degree as seeds
    num_actual_seeds = 0
    if len(sorted_degree_indices) > 0:
        for i in range(min(n_clust, len(sorted_degree_indices))):
            seeds[i] = sorted_degree_indices[i]
            aux_memb[seeds[i]] = i # Assign seed `i` to cluster `i`
            num_actual_seeds += 1
    
    # If n_clust > num_actual_seeds (e.g. n_clust > num nodes with edges), some clusters are empty.
    # This logic assumes seeds are assigned labels 0 to n_clust-1.

    # Assign remaining non-seed nodes to the cluster of the seed they share most CNs with
    if num_actual_seeds > 0: # Only if there are seeds
        for i_node in range(num_nodes):
            is_seed = False # Check if i_node is already a seed
            for s_idx in range(num_actual_seeds):
                if i_node == seeds[s_idx]:
                    is_seed = True
                    break
            if is_seed: continue # Skip seeds, they are already assigned

            # Assign i_node to a seed's cluster
            max_cn_to_seed_val = -1.0
            best_seed_cluster_label = 0 # Default to cluster 0 if no CNs
            
            for s_idx in range(num_actual_seeds):
                seed_node_val = seeds[s_idx]
                current_cn_val = float(cn[i_node, seed_node_val]) # Ensure float for comparison
                
                if current_cn_val > max_cn_to_seed_val:
                    max_cn_to_seed_val = current_cn_val
                    best_seed_cluster_label = aux_memb[seed_node_val] # Label of this seed's cluster
                elif current_cn_val == max_cn_to_seed_val:
                    # Tie-breaking: prefer smaller cluster label
                    if aux_memb[seed_node_val] < best_seed_cluster_label:
                        best_seed_cluster_label = aux_memb[seed_node_val]
            
            aux_memb[i_node] = best_seed_cluster_label
    else: # No seeds (e.g. n_clust = 0 or empty graph with n_clust > 0)
        # All nodes go to cluster 0 by default init of aux_memb if n_clust=0 or 1.
        # If n_clust > 1 and no seeds, randomly assign or assign to 0.
        # Current aux_memb init to zeros covers this.
        # If n_clust > 1, and graph is empty, this results in all nodes in cluster 0.
        # If random assignment for empty graph nodes:
        # for i_node in range(num_nodes): aux_memb[i_node] = np.random.randint(n_clust)
        pass


    return aux_memb


def compute_degree(a, is_directed):
    """Returns matrix *a* degree sequence.

    :param a: Matrix.
    :type a:  numpy.ndarray
    :param is_directed: True if the matrix is directed.
    :type is_directed: bool
    :return: Degree sequence.
    :rtype: numpy.ndarray.
    """
    # if the matrix is a numpy array
    if is_directed:
        if type(a) == np.ndarray:
            return np.sum(a > 0, 0), np.sum(a > 0, 1)
        # if the matrix is a scipy sparse matrix
        elif isspmatrix(a):
            return np.sum(a > 0, 0).A1, np.sum(a > 0, 1).A1
    else:
        if type(a) == np.ndarray:
            return np.sum(a > 0, 1)
        # if the matrix is a scipy sparse matrix
        elif isspmatrix(a):
            return np.sum(a > 0, 1).A1


def compute_strength(a, is_directed):
    """Returns matrix *a* strength sequence.

    :param a: Matrix.
    :type a: numpy.ndarray
    :param is_directed: True if the matrix is directed.
    :type is_directed: bool
    :return: Strength sequence.
    :rtype: numpy.ndarray
    """
    if is_directed:
        # if the matrix is a numpy array
        if type(a) == np.ndarray:
            return np.sum(a, 0), np.sum(a, 1)
        # if the matrix is a scipy sparse matrix
        elif isspmatrix(a):
            return np.sum(a, 0).A1, np.sum(a, 1).A1
    else:
        # if the matrix is a numpy array
        if type(a) == np.ndarray:
            return np.sum(a, 1)
        # if the matrix is a scipy sparse matrix
        elif isspmatrix(a):
            return np.sum(a, 1).A1


def from_edgelist(edgelist, is_sparse, is_directed):
    """Returns np.ndarray or scipy.sparse matrix from edgelist.

    :param edgelist: List of edges, eache edge must be given as a 2-tuples
     (u,v).
    :type edgelist: list or numpy.ndarray
    :param is_sparse: If true the returned matrix is sparse.
    :type is_sparse: bool
    :param is_directed: If true the graph is directed.
    :type is_directed: bool
    :return: Adjacency matrix.
    :rtype: numpy.ndarray or scipy.sparse
    """
    # TODO: vedere che tipo di sparse e'
    if is_directed:
        g = nx.DiGraph()
    else:
        g = nx.Graph()
    g.add_edges_from(edgelist)
    if is_sparse:
        return nx.to_scipy_sparse_matrix(g)
    else:
        return nx.to_numpy_array(g)


def from_weighted_edgelist(edgelist, is_sparse, is_directed):
    """Returns np.ndarray or scipy.sparse matrix from edgelist.

    :param edgelist: List of weighted edges, eache edge must be given as a
     3-tuples (u,v,w).
    :type edgelist: [type]
    :param is_sparse: If true the returned matrix is sparse.
    :type is_sparse: bool
    :param is_directed: If true the graph is directed.
    :type is_directed: bool
    :return: Weighted adjacency matrix.
    :rtype: numpy.ndarray or scipy.sparse
    """
    if is_directed:
        g = nx.DiGraph()
    else:
        g = nx.Graph()
    g.add_weighted_edges_from(edgelist)
    if is_sparse:
        return nx.to_scipy_sparse_matrix(g)
    else:
        return nx.to_numpy_array(g)


def check_symmetric(a, is_sparse, rtol=1e-05, atol=1e-08):
    """Checks if the matrix is symmetric.

    :param a: Matrix.
    :type a: numpy.ndarray or scipy.sparse
    :param is_sparse: If true the matrix is sparse.
    :type is_sparse: bool
    :param rtol: Tuning parameter, defaults to 1e-05.
    :type rtol: float, optional
    :param atol: Tuning parameter, defaults to 1e-08.
    :type atol: float, optional
    :return: True if the matrix is symmetric.
    :rtype: bool
    """
    if is_sparse:
        return np.all(np.abs(a - a.T) < atol)
    else:
        return np.allclose(a, a.T, rtol=rtol, atol=atol)


def check_adjacency(adjacency, is_sparse, is_directed):
    """Functions checking the _validty_ of the adjacency matrix.

    :param adjacency: Adjacency matrix.
    :type adjacency: numpy.ndarray or scipy.sparse
    :param is_sparse: If true the matrix is sparse.
    :type is_sparse: bool
    :param is_directed: True if the graph is directed.
    :type is_directed: bool
    :raises TypeError: Matrix not square.
    :raises ValueError: Negative entries.
    :raises TypeError: Matrix not symmetric.
    """
    if adjacency.shape[0] != adjacency.shape[1]:
        raise TypeError(
            "Adjacency matrix must be square. If you are passing an edgelist"
            " use the positional argument 'edgelist='.")
    if np.sum(adjacency < 0):
        raise ValueError(
            "The adjacency matrix entries must be positive."
        )
    if (not check_symmetric(adjacency, is_sparse)) and (not is_directed):
        raise TypeError(
            "The adjacency matrix seems to be not symmetric, we suggest to use"
            " 'DirectedGraphClass'.")


# Updated
@jit(nopython=True, fastmath=True)
def sumLogProbabilities(nextlogp, logp):
    if nextlogp == -np.inf: # Probability 0
        stop = True # No change to logp
    elif logp == -np.inf: # Current sum is 0, new term is not 0
        logp = nextlogp
        stop = False
    else:
        stop = False
        if nextlogp > logp:
            common = nextlogp
            diffexponent = logp - common
        else:
            common = logp
            diffexponent = nextlogp - common
        
        logp = common + np.log10(1 + 10**diffexponent)

        if 10**diffexponent < 1e-7: 
             stop = True
    return logp, stop

# update
@jit(nopython=True, fastmath=True)
def logc(n, k):
    if k < 0 or k > n: 
        return -np.inf 
    if k == 0 or k == n:
        return 0.0 
    if k > n / 2: 
        k = n - k
    
    res = 0.0
    for i in range(int(k)): 
        res += np.log10(n - i) - np.log10(i + 1)
    return res

# update
@jit(nopython=True, fastmath=True)
def logStirFac(n):
    if n <= 1: 
        return 0.0 
    else:
        # Using a common log10 version of Stirling's approximation:
        # log10(n!) approx n*log10(n) - n*log10(e) + 0.5*log10(2*pi*n)
        LN10 = np.log(10) # log_e(10)
        return (n + 0.5) * np.log10(n) - n / LN10 + 0.5 * np.log10(2 * np.pi)


@jit(nopython=True, fastmath=True)
def sumRange(xmin, xmax):
    """[summary]

    :param xmin: [description]
    :type xmin: [type]
    :param xmax: [description]
    :type xmax: [type]
    :return: [description]
    :rtype: [type]
    """
    csum = 0
    for i in np.arange(xmin, xmax + 1):
        csum += np.log10(i)
    return csum


@jit(nopython=True, fastmath=True)
def sumFactorial(n):
    csum = 0
    if n > 1:
        for i in np.arange(2, n + 1):
            csum += np.log10(i)
    return csum


def shuffled_edges(adjacency_matrix, is_directed):
    # Ensure input is numpy array for .T to work as expected
    adj_np = np.array(adjacency_matrix)
    adj_bool = adj_np.astype(bool) # Use boolean for structure
    
    if not is_directed:
        # For undirected, only take upper triangle to avoid duplicate edges if adj is symmetric
        adj_proc = np.triu(adj_bool, k=1) # k=1 to exclude diagonal
    else:
        adj_proc = adj_bool

    edges = np.stack(adj_proc.nonzero(), axis=-1)
    np.random.shuffle(edges)
    shuff_edges = edges.astype(np.int32)
    return shuff_edges


def jaccard_sorted_edges(adjacency_matrix):
    """Returns edges ordered based on jaccard index.

    :param adjacency_matrix: Matrix.
    :type adjacency_matrix: numpy.ndarray
    :return: Ordered edgelist.
    :rtype: numpy.ndarray
    """
    G = nx.from_numpy_array(adjacency_matrix)
    jacc = nx.jaccard_coefficient(G, ebunch=G.edges())
    jacc_array = []
    for u, v, p in jacc:
        jacc_array += [[u, v, p]]
    jacc_array = np.array(jacc_array)
    jacc_array = jacc_array[jacc_array[:, 2].argsort()][::-1]
    sorted_edges = jacc_array[:, 0:2]
    sorted_edges = sorted_edges.astype(np.int32)
    return sorted_edges


def surprise_negative_hypergeometric(Vi, w, Ve, W, V):
    """Computes the negative hypergeometric distribution.
    """
    surprise = 0
    for w_loop in range(w, W):
        surprise += ((comb(Vi + w_loop - 1, w_loop, exact=True) * comb(
            Ve + W - w_loop, W - w_loop, exact=True)) /
                     comb(V + W, W, exact=True))
    return surprise


def evaluate_surprise_cp_bin(adjacency_matrix,
                             cluster_assignment,
                             is_directed):
    """Computes core-periphery binary log-surprise given a certain nodes'
     partitioning.

    :param adjacency_matrix: Binary adjacency matrix.
    :type adjacency_matrix: numpy.ndarray
    :param cluster_assignment: Core periphery assigments.
    :type cluster_assignment: numpy.ndarray
    :param is_directed: True if the graph is directed.
    :type is_directed: bool
    :return: Log-surprise
    :rtype: float
    """
    core_nodes = np.unique(np.where(cluster_assignment == 0)[0])
    periphery_nodes = np.unique(np.where(cluster_assignment == 1)[0])

    if is_directed:
        n_c = core_nodes.shape[0]
        n_x = periphery_nodes.shape[0]
        p_c = n_c * (n_c - 1)
        p_x = n_c * n_x * 2

        l_c = cp.compute_sum(adjacency_matrix, core_nodes, core_nodes)
        l_x = cp.compute_sum(adjacency_matrix, core_nodes,
                             periphery_nodes) + cp.compute_sum(
                                                         adjacency_matrix,
                                                         periphery_nodes,
                                                         core_nodes)

        l_t = np.sum(adjacency_matrix)
        n = n_c + n_x
        p = n * (n - 1)

    else:
        n_c = core_nodes.shape[0]
        n_x = periphery_nodes.shape[0]
        p_c = (n_c * (n_c - 1)) / 2
        p_x = n_c * n_x

        l_c = cp.compute_sum(adjacency_matrix, core_nodes, core_nodes) / 2
        l_x = (cp.compute_sum(adjacency_matrix, core_nodes,
                              periphery_nodes) + cp.compute_sum(
                                                          adjacency_matrix,
                                                          periphery_nodes,
                                                          core_nodes)) / 2

        l_t = np.sum(adjacency_matrix) / 2
        n = n_c + n_x
        p = (n * (n - 1)) / 2

    if (p_c + p_x) < (l_c + l_x):
        return 0

    surprise = surprise_bipartite_cp_bin(p, p_c, p_x, l_t, l_c, l_x)
    return surprise


@jit(forceobj=True)
def surprise_bipartite_cp_bin(p, p_c, p_x, l, l_c, l_x):
    surprise = 0
    aux_first = 0
    for l_c_loop in range(l_c, p_c + 1):
        aux_first_temp = aux_first
        for l_x_loop in range(l_x, p_x + 1):
            if l_c_loop + l_x_loop > l:
                continue
            aux = multihyperprobability(p, p_c, p_x,
                                        l, l_c_loop,
                                        l_x_loop)
            surprise += aux
            if surprise == 0:
                break
            if aux/surprise < 1e-3:
                break

        aux_first = surprise
        if aux_first - aux_first_temp:
            if ((aux_first - aux_first_temp) / aux_first) < 1e-3:
                # pass
                break

    return surprise


# @jit(nopython=True)
def multihyperprobability(p, p_c, p_x, l, l_c, l_x):
    """Computes the logarithm of the Multinomial Hypergeometric
     distribution."""
    logh = comb(p_c, l_c, True) * comb(p_x, l_x) + comb(
        p - p_c - p_x,
        l - l_c - l_x) - comb(p, l)
    return logh


def evaluate_surprise_cp_enh(adjacency_matrix,
                             cluster_assignment,
                             is_directed):
    """Computes core-periphery weighted surprise given a
     certain nodes' partitioning.

    :param adjacency_matrix: Weighted adjacency matrix.
    :type adjacency_matrix: numpy.ndarray
    :param cluster_assignment: Core periphery assigments.
    :type cluster_assignment: numpy.ndarray
    :param is_directed: True if the graph is directed.
    :type is_directed: bool
    :return: Log-surprise
    :rtype: float
    """
    core_nodes = np.unique(np.where(cluster_assignment == 0)[0])
    periphery_nodes = np.unique(np.where(cluster_assignment == 1)[0])

    if is_directed:
        n_o = core_nodes.shape[0]
        n_p = periphery_nodes.shape[0]
        V_o = n_o * (n_o - 1)
        V_c = n_o * n_p * 2

        l_o, w_o = cp.compute_sum_enh(adjacency_matrix, core_nodes, core_nodes)
        l_c, w_c = cp.compute_sum_enh(adjacency_matrix,
                                      core_nodes,
                                      periphery_nodes) + cp.compute_sum_enh(
            adjacency_matrix,
            periphery_nodes,
            core_nodes)
        L = np.sum(adjacency_matrix.astype(bool))
        W = np.sum(adjacency_matrix)
        # w_p = W - w_o - w_c
        n = n_o + n_p
        V = n * (n - 1)

    else:
        n_o = core_nodes.shape[0]
        n_p = periphery_nodes.shape[0]
        V_o = n_o * (n_o - 1) / 2
        V_c = n_o * n_p

        l_o, w_o = (cp.compute_sum_enh(adjacency_matrix, core_nodes, core_nodes))
        l_c1, w_c1 = cp.compute_sum_enh(adjacency_matrix,
                                        core_nodes,
                                        periphery_nodes)
        l_c2, w_c2 = cp.compute_sum_enh(adjacency_matrix,
                                        periphery_nodes,
                                        core_nodes)

        l_o = l_o / 2
        w_o = w_o / 2
        l_c = (l_c1 + l_c2) / 2
        w_c = (w_c1 + w_c2) / 2

        L = np.sum(adjacency_matrix.astype(bool)) / 2
        W = np.sum(adjacency_matrix) / 2
        # w_p = (W - w_o - w_c) / 2
        n = n_o + n_p
        V = n * (n - 1) / 2

    # print("V_o", V_o, "l_o", l_o, "V_c", V_c, "l_c", l_c,
    #      "w_o", w_o, "w_c", w_c, "V", V, "L", L, "W", W)

    surprise = surprise_bipartite_cp_enh(V_o, l_o, V_c, l_c,
                                         w_o, w_c, V, L, W)
    return surprise


@jit(forceobj=True)
def surprise_bipartite_cp_enh(V_o, l_o, V_c, l_c, w_o, w_c, V, L, W):
    surprise = 0

    min_l_o = min(L, V_o + V_c)
    aux_first = 0
    aux_second = 0
    aux_third = 0
    for l_o_loop in np.arange(l_o, min_l_o + 1):
        aux_first_temp = aux_first
        for l_c_loop in np.arange(l_c, min_l_o + 1 - l_o_loop):
            aux_second_temp = aux_second
            for w_o_loop in np.arange(w_o, W + 1):
                aux_third_temp = aux_third
                for w_c_loop in np.arange(w_c, W + 1 - w_o_loop):
                    aux = logmulti_hyperprobability_weightenh(
                                                            V_o, l_o_loop,
                                                            V_c, l_c_loop,
                                                            w_o_loop, w_c_loop,
                                                            V, L, W)
                    surprise += aux
                    # print(surprise)
                    if surprise == 0:
                        break
                    if aux / surprise < 1e-3:
                        break

                aux_third = surprise
                if aux_third - aux_third_temp:
                    if ((aux_third - aux_third_temp) / aux_third) < 1e-3:
                        # pass
                        break
                else:
                    break

            aux_second = aux_third
            if aux_second - aux_second_temp:
                if ((aux_second - aux_second_temp) / aux_second) < 1e-3:
                    # pass
                    break
            else:
                break

        aux_first = aux_second
        if aux_first - aux_first_temp:
            if ((aux_first - aux_first_temp) / aux_first) < 1e-4:
                # pass
                break
        else:
            break

    return surprise


@jit(forceobj=True)
def logmulti_hyperprobability_weightenh(V_o, l_o, V_c, l_c, w_o, w_c, V, L, W):
    """Computes the of the Negative Multinomial Hypergeometric
     distribution."""
    aux1 = (comb(V_o, l_o, exact=True) * comb(V_c, l_c, exact=True) * comb(
        V - (V_o + V_c), L - (l_o + l_c), exact=True)) / comb(V, L, exact=True)
    aux2 = (comb(w_o - 1, l_o - 1, exact=True) * comb(w_c - 1, l_c - 1,
                                                      exact=True) * comb(
        W - (w_o + w_c) - 1, L - (l_o + l_c) - 1, exact=True)) / comb(W - 1,
                                                                      W - L,
                                                                      exact=True)
    return aux1 * aux2


@jit(forceobj=True)
def surprise_clust_enh(V_o, l_o, w_o, V, L, W):
    min_l_loop = min(L, V_o)

    surprise = 0.0
    aux_first = 0.0
    # print("l_o", l_o, "min l", min_l_loop,"w_0", w_o, "W-L", W)
    for l_loop in range(l_o, min_l_loop + 1):
        aux_first_temp = aux_first
        for w_loop in range(w_o - l_loop + l_o, W - L + l_o + 1):
            if w_loop <= 0:
                continue
            # print(l_loop,  w_loop)
            aux = logenhancedhypergeometric(V_o, l_loop, w_loop, V, L, W)

            if np.isnan(aux):
                break
            surprise += aux
            # print(aux, surprise)
            if surprise == 0:
                break
            if aux / surprise <= 1e-3:
                break
        aux_first = surprise
        if aux_first - aux_first_temp:
            if ((aux_first - aux_first_temp) / aux_first) < 1e-4:
                # pass
                break
        else:
            break

    return surprise


@jit(forceobj=True)
def logenhancedhypergeometric(V_o, l_o, w_o, V, L, W):
    if l_o < L:
        aux1 = (comb(V_o, l_o, True) * comb(V - V_o, L - l_o, True)) / comb(V, L, True)
        aux2 = (comb(w_o - 1, w_o - l_o, True) * comb(W - w_o - 1, (W - L) - (w_o - l_o), True)) / comb(W - 1, W - L, True)
    else:
        aux1 = (comb(V_o, l_o, True) / comb(V, L, True))
        aux2 = comb(w_o - 1, w_o - L, True)
    return aux1 * aux2

