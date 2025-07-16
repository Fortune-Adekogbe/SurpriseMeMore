from . import auxiliary_function as ax

import numpy as np
from mpmath import mp, ncdf, log10, quad, beta, mpf, sqrt, pi
from numba import jit

mp.dps = 100
mp.pretty = True

@jit(nopython=True)
def pertumbuhan_array_int32(): # Growable array helper for Numba
    # Initial capacity, can be tuned
    return (np.empty(10, dtype=np.int32), 10, 0) # data, capacity, current_size

@jit(nopython=True)
def pertumbuhan_array_append_int32(arr_tuple, value):
    data, capacity, size = arr_tuple
    if size == capacity:
        new_capacity = capacity * 2
        if new_capacity == 0 : new_capacity = 10 # Handle initial zero capacity if it were possible
        new_data = np.empty(new_capacity, dtype=np.int32)
        if size > 0: # Only copy if there's data
             new_data[:size] = data[:size]
        data = new_data
        capacity = new_capacity
    data[size] = value
    size += 1
    return (data, capacity, size)

@jit(nopython=True)
def pertumbuhan_array_to_numpy_int32(arr_tuple):
    data, _, size = arr_tuple
    # Return a copy of the used part of the array
    # If size is 0, data[:0] is an empty array of correct type.
    return data[:size].copy() 

def calculate_possible_intracluster_links(partitions, is_directed):
    """Computes the number of possible links, given nodes memberships.

    :param partitions: Nodes memberships.
    :type partitions: numpy.ndarray
    :param is_directed: True if the graph is directed.
    :type is_directed: bool
    :return: Total number of possible edges.
    :rtype: float
    """
    _, counts = np.unique(partitions, return_counts=True)
    nr_poss_intr_clust_links = np.sum(counts * (counts - 1))
    if is_directed:
        return nr_poss_intr_clust_links
    else:
        return nr_poss_intr_clust_links / 2


@jit(nopython=True)
def calculate_possible_intracluster_links_new(partitions, is_directed):
    """Computes the number of possible links, given nodes memberships.
    Faster implementation compiled in "nopython" mode.

    :param partitions: Nodes memberships.
    :type partitions: numpy.ndarray
    :param is_directed: True if the graph is directed.
    :type is_directed: bool
    :return: Total number of possible edges.
    :rtype: float
    """
    if partitions.size == 0: return 0.0
    # Ensure partitions are non-negative for bincount
    min_label = 0
    if partitions.size > 0:
        has_negative = False
        for p_val in partitions:
            if p_val < 0:
                has_negative = True
                break
        if has_negative: # Should not happen with correct labeling
             # Remap to non-negative if necessary, or raise error.
             # For now, assume non-negative.
             pass 
    
    counts = np.bincount(partitions) # Requires non-negative input
    nr_poss_intr_clust_links = 0.0
    for c_val in counts: # c_val is count of nodes in a cluster
        if c_val > 1: # Need at least 2 nodes for a possible link
            nr_poss_intr_clust_links += float(c_val) * (float(c_val) - 1.0)
    
    if is_directed:
        return nr_poss_intr_clust_links
    else:
        return nr_poss_intr_clust_links / 2.0

@jit(nopython=True)
def intracluster_links_new(adj, clust_labels, partitions):
    sum_weights_in_clust = np.zeros(clust_labels.shape[0], dtype=np.float64)
    for i, lab in enumerate(clust_labels):
        # Build list of indices for nodes in cluster 'lab'
        current_cluster_indices_tuple = pertumbuhan_array_int32() 
        for node_idx in range(partitions.shape[0]):
            if partitions[node_idx] == lab:
                # This append operation re-assigns the tuple, which is fine in Numba loop
                current_cluster_indices_tuple = pertumbuhan_array_append_int32(current_cluster_indices_tuple, node_idx)
        
        indices_np = pertumbuhan_array_to_numpy_int32(current_cluster_indices_tuple)

        if indices_np.size > 0:
            sum_weights_in_clust[i] = intracluster_links_aux(adj, indices_np)
        # else, sum_weights_in_clust[i] remains 0.0 (correct for empty cluster)
            
    return sum_weights_in_clust


@jit(nopython=True)
def intracluster_links_aux(adj, indices):
    """Computes intra-cluster links or weights given nodes indices.

    :param adj: [description]
    :type adj: [type]
    :param indices: [description]
    :type indices: [type]
    :return: [description]
    :rtype: [type]
    """
    n_links = 0.0 
    for ii_idx in range(indices.shape[0]):
        for jj_idx in range(indices.shape[0]):
            ii = indices[ii_idx]
            jj = indices[jj_idx]
            n_links += adj[ii, jj]
    return n_links


def calculate_surprise_logsum_clust_bin_new(
        adjacency_matrix,
        cluster_assignment,
        mem_intr_link,
        clust_labels,
        args,
        approx,
        is_directed):
    """Calculates the logarithm of the surprise given the current partitions
     for a binary network. New faster implementation reducing the number of
      redundant computations.

    :param adjacency_matrix: Binary adjacency matrix.
    :type adjacency_matrix: numpy.ndarray
    :param cluster_assignment: Nodes memberships.                                                       
    :type cluster_assignment: numpy.ndarray
    :param mem_intr_link: Intracluster links per cluster
    :type mem_intr_link: np.array
    :param clust_labels: Labels of changed clusters.
    :type clust_labels: np.array
    :param args: (Observed links, nodes number, possible links)
    :type args: (int, int)
    :param approx:
    :type approx:
    :param is_directed: True if the graph is directed.
    :type is_directed: bool
    :return: Log-surprise.                                                                              
    :rtype: float                                                                                       
    """
    if is_directed:
        # intracluster links
        if len(clust_labels):
            int_links = intracluster_links_new(
                adj=adjacency_matrix,
                clust_labels=clust_labels,
                partitions=cluster_assignment)

            for node_label, nr_links in zip(clust_labels, int_links):
                mem_intr_link[0][node_label] = nr_links

        p = np.sum(mem_intr_link[0])
        int_links = int(p)
        # All the possible intracluster links                                                           
        poss_int_links = int(calculate_possible_intracluster_links_new(
            cluster_assignment,
            is_directed))
        # Observed links                                                                                
        obs_links = int(args[0])
        # Possible link
        poss_links = int(args[2])
    else:
        # intracluster links
        if len(clust_labels):
            int_links = intracluster_links_new(
                adj=adjacency_matrix,
                clust_labels=clust_labels,
                partitions=cluster_assignment)

            for node_label, nr_links in zip(clust_labels, int_links):
                mem_intr_link[0][node_label] = nr_links

        p = np.sum(mem_intr_link[0])
        int_links = int(p / 2)
        # All the possible intracluster links                                                           
        poss_int_links = int(calculate_possible_intracluster_links_new(
            cluster_assignment,
            is_directed))
        # Observed links                                                                                
        obs_links = int(args[0] / 2)
        # Possible links
        poss_links = int(args[2] / 2)

    if int_links == 0:
        return 0, mem_intr_link

    if approx == "gaussian":
        surprise = binary_surp_gauss_approx(poss_links,
                                            poss_int_links,
                                            obs_links,
                                            int_links)
        if surprise > 0:
            surprise = np.float64(-log10(surprise))
        else:
            surprise = 0
    elif approx == "asymptotic":
        surprise = asymptot_surp_cd_bin_sum(l_o=int_links,
                                            V=poss_links,
                                            L=obs_links,
                                            V_o=poss_int_links)
        if surprise > 0:
            surprise = np.float64(-log10(surprise))
        else:
            surprise = 0
    else:
        surprise = surprise_logsum_clust_bin(
            poss_links,
            int_links,
            poss_int_links,
            obs_links)

    return surprise, mem_intr_link


def asymptot_surp_cd_bin(l_o, V, L, V_o):
    f = lambda x: integrand_asympt_cd_b(x, V, L, V_o)
    aux = quad(f, [l_o, L], method="gauss-legendre")
    return aux


def asymptot_surp_cd_bin_sum(l_o, V, L, V_o):
    f = lambda x: integrand_asympt_cd_b(x, V, L, V_o)
    aux = 0
    for l_o_loop in range(l_o, L):
        aux += f(l_o_loop)
    return aux


def integrand_asympt_cd_b(l_o, V, L, V_o):
    p = mpf(L / V)
    p_d = mpf(l_o / V_o)
    p_c = mpf((L - l_o) / (V - V_o))
    a = a_l_o(l_o, V, L, V_o, p, p_d, p_c)
    aux = (a * bernoulli(V, L, p)) / (
                bernoulli(V_o, l_o, p_d) * bernoulli(V - V_o, L - l_o, p_c))
    return aux


def bernoulli(x, y, z):
    aux = (mpf(z) ** y) * (mpf(1 - z) ** (x - y))
    return aux


def a_l_o(l_o, V, L, V_o, p, p_d, p_c):
    aux1 = mpf(V*p*(1-p))**2
    aux2 = mpf(2) * pi * (mpf(V_o * p_d * (1-p_d))**2) * (mpf((V-V_o) * p_c * (1-p_c))**2)
    if aux2:
        aux = sqrt(aux1/aux2)
    else:
        aux = 1
    return aux


def binary_surp_gauss_approx(V, V0, L, l0):
    pi = V0 / V
    P = ncdf(-(l0 - 1 - L * pi) / (np.sqrt(L * pi * (1 - pi))))
    return P


@jit(nopython=True)
def surprise_logsum_clust_bin(F, p, M, m):
    """[summary]

    :param F: [description]
    :type F: [type]
    :param p: [description]
    :type p: [type]
    :param M: [description]
    :type M: [type]
    :param m: [description]
    :type m: [type]
    :return: [description]
    :rtype: [type]
    """
    # stop = False
    min_p = min(M, m)

    logP = loghyperprobability(F, p, M, m)
    for p_loop in range(p, min_p + 1):
        if p_loop == p:
            continue
        nextLogP = loghyperprobability(F, p_loop, M, m)
        [logP, stop] = ax.sumLogProbabilities(nextLogP, logP)
        if stop:
            break

    return -logP


@jit(nopython=True)
def loghyperprobability(F, p, M, m):
    """Evaluates logarithmic hypergeometric distribution

    :param F:
    :type F:
    :param p:
    :type p:
    :param M:
    :type M:
    :param m:
    :type m:
    :return:
    :rtype:
    """
    logH = ax.logc(M, p) + ax.logc(F - M, m - p) - ax.logc(F, m)
    return logH


def calculate_surprise_logsum_clust_weigh_new(
        adjacency_matrix_weighted, 
        cluster_assignment,
        mem_intr_link, 
        clust_labels,  
        args,          
        approx,        
        is_directed):

    # args unpacking needs to be explicit for Numba if it struggles with tuple indexing inside JIT
    obs_links_total_binary_arg = args[0] # Not used but part of tuple
    obs_weights_total_arg = args[1]
    poss_links_total_arg = args[2]

    if len(clust_labels) > 0:
        updated_weights_for_changed_clusters = intracluster_links_new(
            adj=adjacency_matrix_weighted, 
            clust_labels=clust_labels,
            partitions=cluster_assignment
        )
        for i in range(clust_labels.shape[0]): # Iterate with index for Numba
            label_val = clust_labels[i]
            if label_val >= 0 and label_val < mem_intr_link.shape[1]:
                 mem_intr_link[1][label_val] = updated_weights_for_changed_clusters[i]

            # else: error or resize, for now assume it's sized for max possible cluster label.
            # This implies `correct_partition_labeling` ensures dense 0..K-1 labels for mem_intr_link size.
            # Or mem_intr_link needs to be a dictionary or dynamically sized.

    w_obs_total_intracluster_sum = 0.0 # Numba compatible sum
    for val_w in mem_intr_link[1]: w_obs_total_intracluster_sum += val_w
    w_obs_total_intracluster = w_obs_total_intracluster_sum
    
    poss_intracluster_links_total = calculate_possible_intracluster_links_new(
        cluster_assignment,
        is_directed
    )

    # Make copies for local modification if undirected
    w_intra_eff = w_obs_total_intracluster
    W_total_eff = obs_weights_total_arg
    V_total_poss_eff = poss_links_total_arg
    Vi_eff = poss_intracluster_links_total # Already adjusted by its func if undirected

    if not is_directed:
        w_intra_eff /= 2.0
        W_total_eff /= 2.0
        V_total_poss_eff /= 2.0
        # Vi_eff is already using calculate_possible_intracluster_links_new which handles directedness.

    if w_intra_eff < 0 : w_intra_eff = 0 # Cannot have negative summed weight. Should not happen.
    if W_total_eff < 0 : W_total_eff = 0

    if w_intra_eff == 0 and W_total_eff == 0: # No weights anywhere
         return 0.0, mem_intr_link # S=1, logS=0. No surprise.
    if w_intra_eff == 0 and Vi_eff > 0 : # No intra-weights but possible links exist.
         # This means w_obs = 0. surprise_logsum_clust_weigh will sum from P(w_in=0)...
         # If S=1 (e.g. P(w_in=0)=1, all other P(w_in>0)=0), then -logS = 0.
         # Original returns 0 for log_surprise if no links/weights. Let function calculate.
         pass

    Ve_eff = V_total_poss_eff - Vi_eff
    if Ve_eff < 0 : Ve_eff = 0 # Cannot have negative external possible links.

    log_surprise_value = surprise_logsum_clust_weigh(
        Vi_eff, w_intra_eff, Ve_eff, W_total_eff, V_total_poss_eff
    )
    
    return log_surprise_value, mem_intr_link


def asymptot_surp_cd_wei(V, W, V_o, w_o):
    f = lambda x: integrand_asympt_cd_w(x, V, W, V_o)
    aux = quad(f, [w_o, W], method="gauss-legendre")
    return aux


@jit(forceobj=True)
def asymptot_surp_cd_wei_sum(V, W, V_o, w_o):
    aux_surp = 0
    for w_o_loop in range(int(w_o), int(W)):
        aux = integrand_asympt_cd_w(w_o_loop, V, W, V_o)
        aux_surp += aux
        if aux_surp == 0:
            break
        if aux / aux_surp < 1e-3:
            break
    return aux_surp


def integrand_asympt_cd_w(w_o, V, W, V_o):
    q = mpf(W/(V + W -1))
    q_d = mpf(w_o/(V_o + w_o - 1))
    q_c = mpf((W - w_o)/(V - V_o + W - w_o - 1))
    C = C_w_o(V, W, V_o, w_o, q, q_d, q_c)
    aux = (C * geometric(V, W, q))/(geometric(V_o, w_o, q_d)*geometric(V-V_o, W-w_o, q_c))
    return aux


def geometric(x, y, z):
    aux = (mpf(z)**y) * (mpf(1 - z)**x)
    return aux


def C_w_o(V, W, V_o, w_o, q, q_d, q_c):
    aux = sqrt(mpf(V*q)**2 / (mpf(2)*pi*(mpf(V_o*q_d)**2)*(mpf((V-V_o)*q_c)**2)))
    return aux


def weighted_suprise_approx(V, W, w_o, V_o):
    """Gaussian approximation of surprise.
    """
    rho = W / V
    aux = ncdf(-(w_o - 1 - V_o * rho) / (np.sqrt(V_o * rho * (1 + rho))))
    return aux


@jit(nopython=True)
def surprise_logsum_clust_weigh(Vi, w_obs, Ve, W_total, V_total_possible):
    logP_sum = -np.inf 
    
    # Iterate w_loop from observed w_obs up to W_total (max possible in Vi)
    # or until probability becomes too small.
    for w_loop_float in np.arange(float(w_obs), float(W_total) + 1.0):
        w_loop = int(round(w_loop_float)) # Ensure integer for combinatorics

        current_logP = lognegativehyperprobability(Vi, w_loop, Ve, W_total, V_total_possible)
        
        if current_logP == -np.inf: # This P(config) is 0
            if logP_sum == -np.inf and w_loop == int(round(float(w_obs))): 
                pass # Sum remains log(0)
            else: # Adding 0 to sum, no change unless sum was also 0.
                  # If current_logP is -inf, sumLogProbabilities should handle it.
                pass # No change to logP_sum from this term. Continue to check stopping.

        # Use sumLogProbabilities to add current_logP to logP_sum
        # Ensure it handles -np.inf correctly.
        # sumLogProbabilities(nextlogp, logp)
        # nextlogp = current_logP, logp = logP_sum
        
        new_logP_sum, stop_early = ax.sumLogProbabilities(current_logP, logP_sum)
        logP_sum = new_logP_sum
        
        if stop_early and w_loop > int(round(float(w_obs))): # Stop if adding negligible prob, but not on first term
            break

    # If logP_sum is still -np.inf (e.g. all terms were 0 prob, or w_obs > W_total), then S=0. -logS = inf.
    # If S=1 (max surprise, e.g. w_obs is the only possible outcome), -logS=0.
    # If S=0 (e.g. impossible observation), -logS=inf.
    # The function should return -log(S). If S=0, -log(0) = inf.
    if logP_sum == -np.inf: return np.inf 
    return -logP_sum 

@jit(nopython=True)
def lognegativehyperprobability(Vi, w, Ve, W_total, V_total_possible):
    # Parameter validation
    if Vi < 0 or Ve < 0 or w < 0 or (W_total - w) < 0: return -np.inf
    if V_total_possible <= 0 and W_total > 0 : return -np.inf # Cannot place weights if no possible links
    if V_total_possible < Vi + Ve : return -np.inf # Inconsistent possible links

    # Vi: possible links in group of interest
    # w: sum of weights in group of interest
    # Ve: possible links outside group of interest (V_total_possible - Vi)
    # W_total: total sum of weights in the network
    # V_total_possible: total possible links in the network

    # log P(w_in=w | Vi, Ve, W_total) = log C(Vi+w-1, w) + log C(Ve+(W_total-w)-1, W_total-w) - log C(V_total_possible+W_total-1, W_total)
    # This is the standard formula for bivariate negative hypergeometric probability.
    # (Distributing W_total items into V_total_possible bins, what's prob of w items in Vi bins, W_total-w in Ve bins)

    # Numerator terms for C(N,K) form: (num_bins + num_items - 1, num_items) or (num_bins + num_items - 1, num_bins - 1)
    # Term 1 (group of interest): Vi bins, w items. C(Vi+w-1, w) or C(Vi+w-1, Vi-1)
    # Term 2 (outside group): Ve bins, (W_total-w) items. C(Ve+(W_total-w)-1, W_total-w) or C(Ve+(W_total-w)-1, Ve-1)
    
    # Denominator (total ways): V_total_possible bins, W_total items.
    # C(V_total_possible + W_total - 1, W_total) or C(V_total_possible + W_total - 1, V_total_possible - 1)

    # Original code used logc(V+W, W) for denominator, which is different.
    # Let's use the standard neg. hyper. denominator C(V_total+W_total-1, W_total).
    
    # Need to handle Vi=0 or Ve=0 cases for C(num_bins-1+k, k). If num_bins=0, C(-1+k,k) is not standard.
    # If Vi=0, then w must be 0. logc(w-1,w) -> logc(-1,0) if w=0. logc(X,0)=0.
    # If Vi=0 and w>0, prob should be 0 (-inf log).
    if Vi == 0 and w > 0: return -np.inf
    if Ve == 0 and (W_total - w) > 0: return -np.inf # All weight must be in Vi if Ve=0

    # Handle Vi-1 < 0 if Vi=0 for C(N, K_bins-1) form
    # Using C(N, K_items) form is safer:
    term1_logc_n = Vi + w - 1
    term1_logc_k = w 
    # If Vi=0, w=0: n=-1, k=0. logc(-1,0) should be 0. Our logc handles k=0.
    # If Vi=1, w=0: n=0, k=0. logc(0,0)=0.
    if Vi == 0 : term1_logc_n = w -1 # Avoid Vi=0 making n negative if w=0

    term1 = ax.logc(term1_logc_n, term1_logc_k)

    term2_logc_n = Ve + (W_total - w) - 1
    term2_logc_k = (W_total - w)
    if Ve == 0 : term2_logc_n = (W_total-w) - 1
        
    term2 = ax.logc(term2_logc_n, term2_logc_k)
    
    # Denominator:
    # If V_total_possible = 0 (e.g. single node graph), and W_total=0, should be log(1)=0.
    # If V_total_possible = 0, and W_total > 0, prob is 0. Denom C(-1+W,W) is problematic.
    # This situation should be caught earlier if possible.
    if V_total_possible == 0 and W_total > 0 : return -np.inf # Cannot place weight
    if V_total_possible == 0 and W_total == 0: return 0.0 # Prob 1 of 0 weight in 0 links

    denom_logc_n = V_total_possible + W_total - 1
    denom_logc_k = W_total
    if V_total_possible == 0: denom_logc_n = W_total -1

    denominator = ax.logc(denom_logc_n, denom_logc_k)

    # Check for -inf terms indicating zero probability component
    if term1 == -np.inf or term2 == -np.inf or denominator == -np.inf:
        # If numerator is 0 (-inf log) and denominator non-zero, result is -inf.
        # If numerator non-zero and denom is 0 (-inf log), result is +inf (bad, P > 1).
        # This indicates an issue with formula or parameters if P > 1.
        # For now, assume parameters are such that probabilities are valid.
        if denominator == -np.inf and (term1 != -np.inf or term2 != -np.inf):
             # This case (P_total_configs = 0 but P_specific_config > 0) is an error state.
             # Or implies the specific config is impossible if its terms are also -inf.
             # For safety, if denom is 0, overall P should be considered undefined or error.
             # However, if term1 or term2 is also -inf, then result is -inf (0/0 needs care).
             # If term1+term2 is finite and denom is -inf: log(finite_P / zero_P_total) -> problem.
             # Assume valid scenarios where denom_P > 0 unless W_total=0.
             if W_total == 0: # If W_total=0, then w=0. All terms should be logC(X-1,0)=0. Result 0.
                 return 0.0 # log(1)
             else: # Denom P=0 but num P>0 implies issue.
                 return -np.inf # Default to 0 probability for safety.

    log_prob = term1 + term2 - denominator
    return log_prob


@jit(nopython=True)
def intracluster_links_enh_new(adj, clust_labels, partitions):
    """Computes intracluster links and weights for enhanced community
     detection method.

    :param adj: Adjacency matrix.
    :type adj: numpy.array
    :param clust_labels:
    :type clust_labels:
    :param partitions: Nodes memberships.
    :type partitions: numpy.array
    :return: Number of intra-cluster links/weights.
    :rtype: float
    """
    nr_intr_clust_links = np.zeros(clust_labels.shape[0])
    intr_weight = np.zeros(clust_labels.shape[0])
    for ii, lab in enumerate(clust_labels):
        indices = np.where(partitions == lab)[0]
        aux_l, aux_w = intracluster_links_aux_enh(adj, indices)
        nr_intr_clust_links[ii] = aux_l
        intr_weight[ii] = aux_w
    return nr_intr_clust_links, intr_weight


@jit(nopython=True)
def intracluster_links_aux_enh(adj, indices): 
    weight_sum = 0.0
    n_binary_links = 0.0 
    
    for ii_outer_idx in range(indices.shape[0]):
        for jj_inner_idx in range(indices.shape[0]):
            ii_node_true_idx = indices[ii_outer_idx]
            jj_node_true_idx = indices[jj_inner_idx]
            
            current_weight_val = adj[ii_node_true_idx, jj_node_true_idx]
            if current_weight_val > 0: # Assuming positive weights contribute
                weight_sum += current_weight_val
                n_binary_links += 1.0 # Count as one binary link if weight > 0
                                     # Using float for consistency if it were ever non-integer counts
                
    return n_binary_links, weight_sum

def calculate_surprise_logsum_clust_enhanced_new(
        adjacency_matrix,
        cluster_assignment,
        mem_intr_link,
        clust_labels,
        args,
        approx,
        is_directed):
    """Calculates, for a weighted network, the logarithm of the enhanced
     surprise given the current partitioning.

    :param adjacency_matrix: Weighted adjacency matrix.
    :type adjacency_matrix: numpy.ndarray
    :param cluster_assignment: Nodes memberships.
    :type cluster_assignment: numpy.ndarray
    :param mem_intr_link:
    :type mem_intr_link:
    :param clust_labels:
    :type clust_labels:
    :param args:
    :type args:
    :param approx:
    :type approx:
    :param is_directed: True if the graph is directed.
    :type is_directed: bool
    :return: Log-surprise.
    :rtype: float
    """
    if is_directed:
        # intracluster weights
        if len(clust_labels):
            l_aux, w_aux = intracluster_links_enh_new(
                adj=adjacency_matrix,
                clust_labels=clust_labels,
                partitions=cluster_assignment)

            for node_label, nr_links, w_int in zip(clust_labels, l_aux, w_aux):
                mem_intr_link[0][node_label] = nr_links
                mem_intr_link[1][node_label] = w_int

        l_o = int(mem_intr_link[0].sum())
        w_o = int(mem_intr_link[1].sum())

        # intracluster possible links
        V_o = int(calculate_possible_intracluster_links_new(
            cluster_assignment,
            is_directed))
        # Total Weight
        W = int(args[1])
        L = int(args[0])
        # Possible links
        # n = adjacency_matrix.shape[0]
        V = int(args[2])
        # extracluster links
        # inter_links = V - V_o
    else:
        # intracluster weights
        if len(clust_labels):
            l_aux, w_aux = intracluster_links_enh_new(
                adj=adjacency_matrix,
                clust_labels=clust_labels,
                partitions=cluster_assignment)

            for node_label, nr_links, w_int in zip(clust_labels, l_aux, w_aux):
                mem_intr_link[0][node_label] = nr_links
                mem_intr_link[1][node_label] = w_int

        l_o = int(mem_intr_link[0].sum() / 2)
        w_o = int(mem_intr_link[1].sum() / 2)

        # intracluster possible links
        V_o = int(calculate_possible_intracluster_links_new(
            cluster_assignment,
            is_directed))
        # Total Weight
        W = int(args[1] / 2)
        L = int(args[0] / 2)
        # Possible links
        # n = adjacency_matrix.shape[0]
        V = int(args[2] / 2)
        # extracluster links
        # inter_links = V - V_o

    if l_o == 0:
        return 0, mem_intr_link

    # print("V_0", V_o, "l_0", l_o, "w_0", w_o, "V", V, "L", L, "W", W)

    surprise = surprise_logsum_clust_enh(V_o, l_o, w_o, V, L, W)
    return surprise, mem_intr_link


@jit(nopython=True)
def surprise_logsum_clust_enh(V_o, l_o, w_o, V, L, W):
    # stop = False
    # stop1 = False
    min_l_loop = min(L, V_o)

    logP = logenhancedhypergeometric(V_o, l_o, w_o, V, L, W)
    logP1 = logP
    w_loop = w_o

    for l_loop in range(l_o, min_l_loop + 1):
        if l_loop == 0:
            continue
        for w_loop in range(w_o - l_loop + l_o, W - L + l_o + 1):
            if w_loop <= 0:
                continue
            if (w_loop == w_o) and (l_loop == l_o):
                continue
            nextLogP = logenhancedhypergeometric(V_o, l_loop, w_loop, V, L, W)
            [logP, stop] = ax.sumLogProbabilities(nextLogP, logP)
            if stop:
                break
        nextLogP1 = logenhancedhypergeometric(V_o, l_loop, w_loop, V, L, W)
        [logP1, stop1] = ax.sumLogProbabilities(nextLogP1, logP1)
        if stop1:
            break

    return -logP1


@jit(nopython=True)
def logenhancedhypergeometric(V_o, l_o, w_o, V, L, W):
    if l_o < L:
        aux1 = (ax.logc(V_o, l_o) + ax.logc(V - V_o, L - l_o)) - ax.logc(V, L)
        aux2 = (ax.logc(w_o - 1, w_o - l_o) + ax.logc(W - w_o - 1, (W - L) - (
                    w_o - l_o))) - ax.logc(W - 1, W - L)
    else:
        aux1 = (ax.logc(V_o, L) - ax.logc(V, L))
        aux2 = (ax.logc(w_o, w_o - L) - ax.logc(W, L))
    return aux1 + aux2


def calculate_surprise_logsum_clust_weigh_continuos(
        adjacency_matrix,
        cluster_assignment,
        mem_intr_link,
        clust_labels,
        args,
        approx,
        is_directed):
    """Calculates the logarithm of the continuos surprise given the current
     partitions for a weighted network. New faster implementation.


    :param adjacency_matrix: Weighted adjacency matrix.
    :type adjacency_matrix: numpy.ndarray
    :param cluster_assignment: Nodes memberships.
    :type cluster_assignment: numpy.ndarray
    :param mem_intr_link:
    :type mem_intr_link:
    :param clust_labels:
    :type clust_labels:
    :param args:
    :type args:
    :param approx:
    :type approx:
    :param is_directed: True if the graph is directed.
    :type is_directed: bool
    :return: Log-surprise.
    :rtype: float
    """
    if is_directed:
        # intracluster weights
        if len(clust_labels):
            w = intracluster_links_new(
                adj=adjacency_matrix,
                clust_labels=clust_labels,
                partitions=cluster_assignment)

            for node_label, nr_links in zip(clust_labels, w):
                mem_intr_link[1][node_label] = nr_links

        p = np.sum(mem_intr_link[1])
        intr_weights = p
        # intracluster possible links
        poss_intr_links = calculate_possible_intracluster_links_new(
            cluster_assignment,
            is_directed)
        # Total Weight
        tot_weights = args[1]
        # Possible links
        poss_links = args[2]
        # extracluster links
        # inter_links = poss_links - poss_intr_links
    else:
        # intracluster weights
        if len(clust_labels):
            w = intracluster_links_new(
                adj=adjacency_matrix,
                clust_labels=clust_labels,
                partitions=cluster_assignment)

            for node_label, nr_links in zip(clust_labels, w):
                mem_intr_link[1][node_label] = nr_links

        p = np.sum(mem_intr_link[1])
        intr_weights = p / 2
        # intracluster possible links
        poss_intr_links = calculate_possible_intracluster_links_new(
            cluster_assignment,
            is_directed)
        # Total Weight
        tot_weights = args[1] / 2
        # Possible links
        poss_links = args[2] / 2
        # extracluster links
        # inter_links = poss_links - poss_intr_links

    if intr_weights == 0:
        return 0, mem_intr_link

    if approx == "gaussian":
        surprise = weighted_suprise_approx(
            poss_links,
            tot_weights,
            intr_weights,
            poss_intr_links)
        if surprise > 0:
            surprise = np.float64(-log10(surprise))
        else:
            surprise = 0
    elif approx == "asymptotic":
        surprise = asymptot_surp_cd_wei(
            poss_links,
            tot_weights,
            poss_intr_links,
            intr_weights)
        if surprise > 0:
            surprise = np.float64(-log10(surprise))
        else:
            surprise = 0
    else:
        surprise = continuous_surprise_clust(
            poss_links,
            tot_weights,
            intr_weights,
            poss_intr_links)

        if surprise > 0:
            surprise = np.float64(-log10(surprise))
        else:
            surprise = 0

    return surprise, mem_intr_link


def continuous_surprise_clust(V, W, w_o, V_o):
    f = lambda x: integrand_clust(x, V, W, V_o)
    aux_surp = quad(f, [w_o, W], method="gauss-legendre")
    return aux_surp


def integrand_clust(w_o, V, W, V_o):
    aux = W * beta(V, W) / (w_o*beta(V_o, w_o) * (W-w_o) * beta(V-V_o, W-w_o))
    return aux


def labeling_communities(partitions):
    """Gives labels to communities from 0 to number of communities minus one.

    :param partitions: Nodes memberships.
    :type partitions: numpy.ndarray
    :return: Re-labeled nodes memberships.
    :rtype: numpy.ndarray
    """
    if partitions.size == 0: return np.array([], dtype=np.int32)
    
    unique_labels, counts = np.unique(partitions, return_counts=True)
    if unique_labels.size == 0 : return np.array([], dtype=np.int32)

    sorted_indices_of_unique_labels = np.argsort(counts)[::-1]
    ordered_old_labels = unique_labels[sorted_indices_of_unique_labels]

    new_partitioning = np.zeros_like(partitions, dtype=np.int32)
    
    # Robust mapping for potentially non-contiguous or large labels
    map_old_to_new = {old_label: new_label for new_label, old_label in enumerate(ordered_old_labels)}
    
    for i in range(partitions.shape[0]):
        new_partitioning[i] = map_old_to_new[partitions[i]]
        
    return new_partitioning


#@jit(nopython=True)
def flipping_function_comdet_agl_new( 
        calculate_surprise_func, 
        adj_matrix,
        current_membership,
        current_mem_intr_link, 
        graph_args,        
        current_surprise,
        approx_method, 
        is_directed_graph,
        list_of_neighbors 
    ):
    
    n_nodes = current_membership.shape[0]
    if n_nodes == 0: # Handle empty graph explicitly
        return current_membership, current_surprise, current_mem_intr_link

    best_membership = current_membership.copy()
    best_surprise = current_surprise
    # Ensure current_mem_intr_link is copied correctly for modification state
    best_mem_intr_link_state = current_mem_intr_link.copy()


    for node_idx in range(n_nodes):
        original_node_cluster = best_membership[node_idx] # Use current best state for decisions
        
        node_neighbors_indices = list_of_neighbors[node_idx]
        
        for neighbor_idx in node_neighbors_indices:
            neighbor_actual_cluster = best_membership[neighbor_idx] # Cluster of neighbor in current best state
            
            if original_node_cluster != neighbor_actual_cluster: 
                temp_membership_try = best_membership.copy()
                temp_membership_try[node_idx] = neighbor_actual_cluster 
                
                # Determine affected labels for recalculation
                # These are original_node_cluster (lost a node) and neighbor_actual_cluster (gained a node)
                # Numba compatible way to list unique changed labels:
                changed_labels_np_arr = np.empty(2, dtype=np.int32)
                changed_labels_np_arr[0] = original_node_cluster
                changed_labels_np_arr[1] = neighbor_actual_cluster
                # Sort them for consistent order if calculate_surprise relies on it (it shouldn't matter for sum)
                if changed_labels_np_arr[0] > changed_labels_np_arr[1]:
                    changed_labels_np_arr[0], changed_labels_np_arr[1] = changed_labels_np_arr[1], changed_labels_np_arr[0]
                # If they were same (not possible here due to outer 'if'), make unique.
                # Here, they are distinct.
                
                # Pass a *copy* of best_mem_intr_link_state for trial, calculate_surprise_func will update it.
                temp_surprise_val, temp_updated_mem_link = calculate_surprise_func(
                    adj_matrix,
                    temp_membership_try,
                    best_mem_intr_link_state.copy(), 
                    changed_labels_np_arr, # Pass the two distinct, sorted labels
                    graph_args,
                    approx_method,
                    is_directed_graph
                )
                
                if temp_surprise_val > best_surprise:
                    best_surprise = temp_surprise_val
                    best_membership = temp_membership_try # Commit: already a copy
                    best_mem_intr_link_state = temp_updated_mem_link # Commit: this is the new state

    return best_membership, best_surprise, best_mem_intr_link_state


def flipping_function_comdet_div_new( 
        calculate_surprise_func,
        adj_matrix,
        current_membership, # Current best assignment
        current_mem_intr_link, # Its corresponding state
        graph_args,
        current_surprise, # Its surprise value
        approx_method,
        is_directed_graph,
        list_of_neighbors
    ):

    n_nodes = current_membership.shape[0]
    if n_nodes == 0: return current_membership, current_surprise, current_mem_intr_link

    best_membership = current_membership.copy()
    best_surprise = current_surprise
    best_mem_intr_link_state = current_mem_intr_link.copy()

    # Calculate initial cluster sizes based on `best_membership` (which is `current_membership` at start)
    max_label_val = 0
    if n_nodes > 0: 
        for label_idx in range(best_membership.shape[0]):
            if best_membership[label_idx] > max_label_val:
                max_label_val = best_membership[label_idx]
    
    # Size cluster_sizes array appropriately. Add 1 because labels are 0-indexed.
    # If max_label_val is 0 (e.g. all nodes in cluster 0), size should be 1.
    # If graph is empty (n_nodes=0), max_label_val remains 0, size 1. Handle.
    if n_nodes == 0: cluster_sizes_arr_size = 0
    else: cluster_sizes_arr_size = max_label_val + 1

    current_cluster_sizes = np.zeros(cluster_sizes_arr_size, dtype=np.int32)
    if n_nodes > 0 : # Only populate if there are nodes
        for i_node_cs in range(n_nodes):
            # Ensure label is within bounds of current_cluster_sizes, may need resizing if labels sparse/large
            # This assumes labels in best_membership are already dense 0..K-1.
            # If not, this part can fail or need more robust max_label finding and array sizing.
            # For now, trust that max_label_val correctly found the max index needed.
            label_for_size = best_membership[i_node_cs]
            if label_for_size >=0 and label_for_size < current_cluster_sizes.shape[0]:
                 current_cluster_sizes[label_for_size] += 1
            # else: label out of bounds, error or dynamic resize of current_cluster_sizes.

    for node_idx in range(n_nodes):
        original_node_cluster = best_membership[node_idx]
        
        # Preserve K fixed clusters: if node is sole member, don't move it.
        # Check bounds for original_node_cluster before indexing current_cluster_sizes
        if original_node_cluster >=0 and original_node_cluster < current_cluster_sizes.shape[0]:
            if current_cluster_sizes[original_node_cluster] == 1:
                continue
        else: # Label out of bounds - indicates an issue with label consistency or array sizing
            continue # Skip this node to be safe

        node_neighbors_indices = list_of_neighbors[node_idx]
        for neighbor_idx in node_neighbors_indices:
            neighbor_actual_cluster = best_membership[neighbor_idx] 
            
            if original_node_cluster != neighbor_actual_cluster:
                temp_membership_try = best_membership.copy()
                temp_membership_try[node_idx] = neighbor_actual_cluster
                
                changed_labels_np_arr = np.empty(2, dtype=np.int32)
                changed_labels_np_arr[0] = original_node_cluster
                changed_labels_np_arr[1] = neighbor_actual_cluster
                if changed_labels_np_arr[0] > changed_labels_np_arr[1]:
                    changed_labels_np_arr[0], changed_labels_np_arr[1] = changed_labels_np_arr[1], changed_labels_np_arr[0]

                temp_surprise_val, temp_updated_mem_link = calculate_surprise_func(
                    adj_matrix, temp_membership_try, best_mem_intr_link_state.copy(),
                    changed_labels_np_arr, graph_args, approx_method, is_directed_graph
                )
                
                if temp_surprise_val > best_surprise:
                    # Accept the move
                    best_surprise = temp_surprise_val
                    best_membership = temp_membership_try 
                    best_mem_intr_link_state = temp_updated_mem_link
                    
                    # Update current_cluster_sizes to reflect the accepted move
                    # Ensure labels are within bounds before decrementing/incrementing
                    if original_node_cluster >=0 and original_node_cluster < current_cluster_sizes.shape[0]:
                        current_cluster_sizes[original_node_cluster] -= 1
                    if neighbor_actual_cluster >=0 and neighbor_actual_cluster < current_cluster_sizes.shape[0]:
                        current_cluster_sizes[neighbor_actual_cluster] += 1
                    # If a label became out of bounds due to re-labeling or sparse labels, this needs robust handling.
                    # For now, assume labels remain within the initial max_label_val range or are re-mapped.

    return best_membership, best_surprise, best_mem_intr_link_state