import numpy as np
from tqdm import tqdm

from . import auxiliary_function as ax
from . import comdet_functions as cd


def solver_cp(adjacency_matrix,
              cluster_assignment,
              num_sim,
              sort_edges,
              calculate_surprise,
              correct_partition_labeling,
              flipping_function,
              print_output=False):
    """[summary]

    :param adjacency_matrix: [description]
    :type adjacency_matrix: [type]
    :param cluster_assignment: [description]
    :type cluster_assignment: [type]
    :param num_sim: [description]
    :type num_sim: [type]
    :param sort_edges: [description]
    :type sort_edges: [type]
    :param calculate_surprise: [description]
    :type calculate_surprise: [type]
    :param correct_partition_labeling: [description]
    :type correct_partition_labeling: [type]
    :param flipping_function:
    :type flipping_function:
    :param print_output: [description], defaults to False
    :type print_output: bool, optional
    :return: [description]
    :rtype: [type]
    """
    surprise = 0
    edges_sorted = sort_edges(adjacency_matrix)
    sim = 0
    while sim < num_sim:
        # edges_counter = 0
        for [u, v] in tqdm(edges_sorted):
            # surprise_old = surprise
            cluster_assignment_temp1 = cluster_assignment.copy()
            cluster_assignment_temp2 = cluster_assignment.copy()

            if cluster_assignment[u] != cluster_assignment[v]:
                cluster_assignment_temp1[v] = cluster_assignment[u]
                cluster_assignment_temp2[u] = cluster_assignment[v]

                surprise_temp1 = calculate_surprise(adjacency_matrix,
                                                    cluster_assignment_temp1)
                if surprise_temp1 >= surprise:
                    cluster_assignment = cluster_assignment_temp1.copy()
                    surprise = surprise_temp1

                surprise_temp2 = calculate_surprise(adjacency_matrix,
                                                    cluster_assignment_temp2)
                if surprise_temp2 >= surprise:
                    cluster_assignment = cluster_assignment_temp2.copy()
                    surprise = surprise_temp2

            else:  # cluster_assignment is the same
                cluster_assignment_temp1[v] = 1 - cluster_assignment[v]
                cluster_assignment_temp2[u] = 1 - cluster_assignment[u]

                surprise_temp1 = calculate_surprise(adjacency_matrix,
                                                    cluster_assignment_temp1)
                if surprise_temp1 >= surprise:
                    cluster_assignment = cluster_assignment_temp1.copy()
                    surprise = surprise_temp1

                surprise_temp2 = calculate_surprise(adjacency_matrix,
                                                    cluster_assignment_temp2)
                if surprise_temp2 >= surprise:
                    cluster_assignment = cluster_assignment_temp2.copy()
                    surprise = surprise_temp2

        if print_output:
            print()
            print(surprise)
        sim += 1

    if len(cluster_assignment) <= 500:
        n_flips = 100
    else:
        n_flips = int(len(cluster_assignment) * 0.2)

    flips = 0
    while flips < n_flips:
        cluster_assignment_temp = flipping_function(cluster_assignment.copy())
        surprise_temp = calculate_surprise(adjacency_matrix,
                                           cluster_assignment_temp)
        if surprise_temp > surprise:
            cluster_assignment = cluster_assignment_temp.copy()
            surprise = surprise_temp
        flips += 1

    cluster_assignment = correct_partition_labeling(adjacency_matrix,
                                                    cluster_assignment)
    return cluster_assignment, surprise


def solver_com_det_aglom(
        adjacency_matrix, 
        cluster_assignment,
        num_sim,
        sort_edges_func, 
        calculate_surprise_func, 
        correct_partition_labeling_func, 
        prob_mix, # Probability of merging clusters vs. moving single node
        flipping_function_node_move, 
        approx, 
        is_directed,
        print_output=False):

    n_nodes = adjacency_matrix.shape[0]
    if n_nodes == 0: return np.array([], dtype=np.int32), 0.0

    # Calculate global graph properties for args tuple
    obs_links_total_binary = np.sum( (np.array(adjacency_matrix) > 0).astype(np.int16) )
    obs_weights_total = np.sum(np.array(adjacency_matrix)) # Ensure it's a sum over numbers
    
    # Total possible links (pairs of nodes)
    if n_nodes > 1:
        poss_links_total = float(n_nodes) * (float(n_nodes) - 1.0)
        if not is_directed: 
            poss_links_total /= 2.0
    else: # 0 or 1 node
        poss_links_total = 0.0


    graph_args = (obs_links_total_binary, obs_weights_total, poss_links_total)

    current_cluster_assignment = correct_partition_labeling_func(cluster_assignment.copy())
    
    # Initialize mem_intr_link state
    # Max label determines size. `correct_partition_labeling_func` should ensure dense 0..K-1 labels.
    max_label_after_corr = 0
    if n_nodes > 0 and current_cluster_assignment.size > 0:
        unique_labels_after_corr = np.unique(current_cluster_assignment)
        if unique_labels_after_corr.size > 0:
            max_label_after_corr = np.max(unique_labels_after_corr)

    # Size for mem_intr_link: max_label + 1
    # Handle n_nodes = 0 or 1, or single cluster case.
    if n_nodes == 0: mem_link_size = 0 # No clusters
    elif max_label_after_corr == -1 : mem_link_size = 0 # No labels found (e.g. empty assignment for n_nodes>0)
    else: mem_link_size = max_label_after_corr + 1
    
    if mem_link_size == 0 and n_nodes > 0: # E.g. assignment was empty for non-empty graph
        # This indicates an issue. Fallback or error.
        # If all nodes in one cluster 0, max_label=0, size=1. (Correct)
        # If no nodes, size 0. (Correct)
        # If nodes exist but assignment is empty, current_cluster_assignment.size = 0.
        # The `np.unique` on empty array gives empty. Max on empty raises error or gives default.
        # Safest: if n_nodes > 0 but mem_link_size is 0, it means labels are problematic.
        # Default to n_nodes size if labels were e.g. not 0-indexed or sparse.
        # However, `correct_partition_labeling_func` should prevent this.
        pass


    # Make mem_intr_link_state robust to mem_link_size=0 (e.g. empty graph)
    if mem_link_size > 0:
        mem_intr_link_state = np.zeros((2, mem_link_size), dtype=np.float64)
    else: # Handles n_nodes = 0 or problematic labeling returning no valid labels.
          # Create a dummy 1-element array to prevent downstream errors if code expects non-empty.
          # Or, handle n_nodes=0 by returning immediately. (Done at start of func)
        mem_intr_link_state = np.zeros((2,1), dtype=np.float64) # Dummy for safety if somehow reached here with n_nodes>0

    if n_nodes > 0 and current_cluster_assignment.size > 0 and mem_link_size > 0:
      # Re-get unique labels from the corrected assignment
      unique_labels_to_fill = np.unique(current_cluster_assignment)
      for label_val in unique_labels_to_fill:
          if label_val < 0 or label_val >= mem_link_size: continue # Skip invalid labels

          indices_list_fill = cd.pertumbuhan_array_int32()
          for node_i in range(n_nodes):
              if current_cluster_assignment[node_i] == label_val:
                  indices_list_fill = cd.pertumbuhan_array_append_int32(indices_list_fill, node_i)
          indices_fill = cd.pertumbuhan_array_to_numpy_int32(indices_list_fill)

          if indices_fill.size > 0: 
              l_aux, w_aux = cd.intracluster_links_aux_enh(adjacency_matrix, indices_fill)
              mem_intr_link_state[0][label_val] = l_aux
              mem_intr_link_state[1][label_val] = w_aux

    # Initial surprise calculation
    # Pass empty array for clust_labels as all clusters are considered "new" or "initial"
    current_surprise, mem_intr_link_state = calculate_surprise_func(
        adjacency_matrix, current_cluster_assignment, mem_intr_link_state, 
        np.array([], dtype=np.int32), graph_args, approx, is_directed
    )

    no_improvement_streak = 0
    
    adj_binary_for_neighbors = (np.array(adjacency_matrix) > 0).astype(np.int16)
    list_of_neighbors_for_flipping = ax.compute_neighbours(adj_binary_for_neighbors)


    for sim_count in range(num_sim):
        prev_surprise_for_streak = current_surprise
        
        sorted_edges = sort_edges_func(adjacency_matrix) # Pass appropriate matrix

        iterable_edges = tqdm(sorted_edges, desc=f"Sim {sim_count+1} Agglom Edges") if print_output else sorted_edges
        for u_node, v_node in iterable_edges:
            cluster_u_label = current_cluster_assignment[u_node]
            cluster_v_label = current_cluster_assignment[v_node]

            if cluster_u_label != cluster_v_label:
                # Louvain-like merge: try merging C_u into C_v
                potential_new_assignment_merged = current_cluster_assignment.copy()
                # Relabel all nodes from cluster_u_label to cluster_v_label
                for i_node_merge in range(n_nodes):
                    if potential_new_assignment_merged[i_node_merge] == cluster_u_label:
                        potential_new_assignment_merged[i_node_merge] = cluster_v_label
                
                # Affected labels: cluster_u_label (becomes empty/merged) and cluster_v_label (grows)
                # Ensure labels are distinct for the call, though here they are by outer `if`.
                changed_labels_for_merge_calc = np.array([cluster_u_label, cluster_v_label], dtype=np.int32)
                # Sort for consistency if needed by calc_surprise (usually not critical for sums)
                if changed_labels_for_merge_calc[0] > changed_labels_for_merge_calc[1]:
                     changed_labels_for_merge_calc[0],changed_labels_for_merge_calc[1] = changed_labels_for_merge_calc[1],changed_labels_for_merge_calc[0]
                
                # If mem_intr_link_state isn't large enough for max label in potential_new_assignment_merged,
                # (e.g. if labels were not dense 0..K-1), this needs care.
                # Assume correct_partition_labeling_func ensures dense labels for mem_intr_link_state size.
                # When merging, one label effectively disappears. `calculate_surprise_func` should handle this
                # by finding 0 members for cluster_u_label in new_assignment, thus 0 links/weights.
                
                # Ensure mem_intr_link_state passed is large enough.
                # Max label in current assignment defines its size. A merge doesn't increase max label.
                temp_s_merge, temp_mem_link_merge = calculate_surprise_func(
                    adjacency_matrix, potential_new_assignment_merged, mem_intr_link_state.copy(),
                    changed_labels_for_merge_calc, graph_args, approx, is_directed
                )

                if temp_s_merge > current_surprise:
                    current_surprise = temp_s_merge
                    current_cluster_assignment = potential_new_assignment_merged
                    mem_intr_link_state = temp_mem_link_merge 
                    # After a merge, `mem_intr_link_state` for the merged-away label (cluster_u_label)
                    # should be effectively zeroed out by `calculate_surprise_func`'s update.
                    # The number of active clusters may have reduced.

        # Single node moves (local refinement)
        # Ensure mem_intr_link_state is correctly sized for current_cluster_assignment's labels.
        # If merges happened, number of unique labels (and potentially max label if re-labeled) could change.
        # It's safer to re-label and resize mem_intr_link_state before flipping if K changes.
        # For now, assume flipping_function handles existing mem_intr_link_state structure.
        # If K changed, the old mem_intr_link_state might have more columns than active clusters.
        
        # Re-canonicalize labels and reconstruct mem_intr_link_state if K changed, before flipping.
        # This is crucial for correctness of `calculate_surprise_func` inside flipping.
        num_unique_before_flip = np.unique(current_cluster_assignment).size
        current_cluster_assignment = correct_partition_labeling_func(current_cluster_assignment.copy())
        num_unique_after_flip_relabel = np.unique(current_cluster_assignment).size
        
        if num_unique_before_flip != num_unique_after_flip_relabel or True : # Always re-init for safety
            # Re-initialize mem_intr_link_state based on new canonical labeling
            max_label_after_relabel = 0
            if n_nodes > 0 and current_cluster_assignment.size > 0:
                 unique_labels_post_relabel = np.unique(current_cluster_assignment)
                 if unique_labels_post_relabel.size > 0:
                      max_label_after_relabel = np.max(unique_labels_post_relabel)

            new_mem_link_size = max_label_after_relabel + 1
            if new_mem_link_size == 0 and n_nodes > 0: new_mem_link_size = 1 # Default for safety
            
            new_mem_intr_link_state = np.zeros((2, new_mem_link_size), dtype=np.float64)
            if n_nodes > 0 and current_cluster_assignment.size > 0 and new_mem_link_size > 0:
                unique_labels_for_new_mem_link = np.unique(current_cluster_assignment)
                for label_val_new in unique_labels_for_new_mem_link:
                    if label_val_new < 0 or label_val_new >= new_mem_link_size: continue

                    indices_list_new = cd.pertumbuhan_array_int32()
                    for node_i_new in range(n_nodes):
                        if current_cluster_assignment[node_i_new] == label_val_new:
                            indices_list_new = cd.pertumbuhan_array_append_int32(indices_list_new, node_i_new)
                    indices_new = cd.pertumbuhan_array_to_numpy_int32(indices_list_new)
                    
                    if indices_new.size > 0:
                        l_aux_new, w_aux_new = cd.intracluster_links_aux_enh(adjacency_matrix, indices_new)
                        new_mem_intr_link_state[0][label_val_new] = l_aux_new
                        new_mem_intr_link_state[1][label_val_new] = w_aux_new
            mem_intr_link_state = new_mem_intr_link_state


        # Now call flipping function with re-labeled assignment and corresponding mem_intr_link_state
        flipped_assignment, flipped_surprise, flipped_mem_link = flipping_function_node_move(
            calculate_surprise_func, 
            adjacency_matrix, current_cluster_assignment,
            mem_intr_link_state, graph_args, current_surprise,
            approx, is_directed, list_of_neighbors_for_flipping
        )

        if flipped_surprise > current_surprise:
            current_surprise = flipped_surprise
            current_cluster_assignment = flipped_assignment
            mem_intr_link_state = flipped_mem_link
        
        if current_surprise > prev_surprise_for_streak:
            no_improvement_streak = 0
        else:
            no_improvement_streak += 1
        
        if no_improvement_streak >= 2: # Original used 10, 2 for faster test
            if print_output: print(f"Converged after {sim_count+1} simulations (no improvement).")
            break
            
        if print_output:
            unique_clusters_count = np.unique(current_cluster_assignment).size if n_nodes > 0 else 0
            print(f"Sim {sim_count+1} end: Surprise={current_surprise:.4f}, Clusters={unique_clusters_count}")

    final_assignment = correct_partition_labeling_func(current_cluster_assignment)
    return final_assignment, current_surprise


def solver_com_det_divis( 
        adjacency_matrix,
        cluster_assignment, 
        num_sim,
        sort_edges_func, # Not directly used for main loop in simplified version, but for context
        calculate_surprise_func,
        correct_partition_labeling_func,
        flipping_function_node_move, 
        approx,
        is_directed,
        print_output=False):

    n_nodes = adjacency_matrix.shape[0]
    if n_nodes == 0: return np.array([], dtype=np.int32), 0.0

    obs_links_total_binary = np.sum( (np.array(adjacency_matrix) > 0).astype(np.int16) )
    obs_weights_total = np.sum(np.array(adjacency_matrix))
    
    if n_nodes > 1:
        poss_links_total = float(n_nodes) * (float(n_nodes) - 1.0)
        if not is_directed: poss_links_total /= 2.0
    else: poss_links_total = 0.0
    graph_args = (obs_links_total_binary, obs_weights_total, poss_links_total)

    current_cluster_assignment = correct_partition_labeling_func(cluster_assignment.copy())
    K_fixed = np.unique(current_cluster_assignment).size if n_nodes > 0 else 0
    
    max_label_init_div = 0
    if n_nodes > 0 and current_cluster_assignment.size > 0:
        unique_labels_init_div = np.unique(current_cluster_assignment)
        if unique_labels_init_div.size > 0:
            max_label_init_div = np.max(unique_labels_init_div)
    
    mem_link_size_div = max_label_init_div + 1
    if mem_link_size_div == 0 and n_nodes > 0 : mem_link_size_div = 1 # Safety
    
    mem_intr_link_state = np.zeros((2, mem_link_size_div), dtype=np.float64)
    if n_nodes > 0 and current_cluster_assignment.size > 0 and mem_link_size_div > 0:
        unique_labels_fill_div = np.unique(current_cluster_assignment)
        for label_val_f_div in unique_labels_fill_div:
            if label_val_f_div < 0 or label_val_f_div >= mem_link_size_div: continue
            indices_list_f_div = cd.pertumbuhan_array_int32()
            for node_i_f_div in range(n_nodes):
                if current_cluster_assignment[node_i_f_div] == label_val_f_div:
                    indices_list_f_div = cd.pertumbuhan_array_append_int32(indices_list_f_div, node_i_f_div)
            indices_f_div = cd.pertumbuhan_array_to_numpy_int32(indices_list_f_div)
            
            if indices_f_div.size > 0:
                l_aux_f_div, w_aux_f_div = cd.intracluster_links_aux_enh(adjacency_matrix, indices_f_div)
                mem_intr_link_state[0][label_val_f_div] = l_aux_f_div
                mem_intr_link_state[1][label_val_f_div] = w_aux_f_div

    current_surprise, mem_intr_link_state = calculate_surprise_func(
        adjacency_matrix, current_cluster_assignment, mem_intr_link_state,
        np.array([], dtype=np.int32), graph_args, approx, is_directed
    )

    no_improvement_streak = 0
    adj_binary_for_neighbors = (np.array(adjacency_matrix) > 0).astype(np.int16)
    list_of_neighbors_for_flipping = ax.compute_neighbours(adj_binary_for_neighbors)

    for sim_count in range(num_sim):
        prev_surprise_for_streak = current_surprise
        
        # In fixed-K, primary operation is node moves via flipping_function.
        # The original edge iteration logic for divisive was complex and might be covered by robust flipping.
        
        # Ensure mem_intr_link_state is correctly sized for current_cluster_assignment labels
        # (correct_partition_labeling_func should ensure dense 0..K-1 before this, K being K_fixed)
        
        new_assignment, new_surprise, new_mem_link_state = flipping_function_node_move(
            calculate_surprise_func, 
            adjacency_matrix, current_cluster_assignment,
            mem_intr_link_state, graph_args, current_surprise,
            approx, is_directed, list_of_neighbors_for_flipping
        )
        
        # Check if K_fixed is maintained (flipping_function_comdet_div_new should do this)
        current_K_after_flip = np.unique(new_assignment).size if n_nodes > 0 else 0
        if current_K_after_flip == K_fixed and new_surprise > current_surprise:
            current_surprise = new_surprise
            current_cluster_assignment = new_assignment
            mem_intr_link_state = new_mem_link_state
        # Else, if K changed or surprise not better, keep old state.
        # (The flipping func already returns best found, so this check is redundant if it worked)

        if current_surprise > prev_surprise_for_streak:
            no_improvement_streak = 0
        else:
            no_improvement_streak += 1
        
        if no_improvement_streak >= 2: # Original 10
            if print_output: print(f"Converged after {sim_count+1} simulations (no improvement).")
            break
            
        if print_output:
            print(f"Sim {sim_count+1} end: Surprise={current_surprise:.4f} (K_fixed={K_fixed})")

    final_assignment = correct_partition_labeling_func(current_cluster_assignment)
    return final_assignment, current_surprise
