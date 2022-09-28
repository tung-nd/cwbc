import numpy as np

def reweight_bin(orig_hist, edges, sorted_returns, percentile, lamb):
    orig_hist = orig_hist / np.sum(orig_hist)
    rtg_max = sorted_returns[-1]
    rtg_percentile = sorted_returns[int(percentile * len(sorted_returns))]
    tau = rtg_max - rtg_percentile
    if tau == 0:
        tau = 0.1
    bin_avg = np.array([(edges[i] + edges[i-1])/2 for i in range(1, len(edges))])
    if lamb != 0:
        reweighted_hist = orig_hist / (orig_hist + lamb) * np.exp(-np.abs(bin_avg - rtg_max) / tau)
    else:
        reweighted_hist = np.exp(-np.abs(bin_avg - rtg_max) / tau)
    reweighted_hist = reweighted_hist / np.sum(reweighted_hist)
    return reweighted_hist

def bin_dict_from_ids(ids):
    bin_dict = {}
    for i, bin_id in enumerate(ids):
        if bin_id not in bin_dict.keys():
            bin_dict[bin_id] = [i]
        else:
            bin_dict[bin_id].append(i)
    return bin_dict

def sample_from_bins(orig_bin_dist, target_bin_dist, bin_edges, bin_dict, returns, noise_scale, noise_constant):
    ### sample a bin
    bin_id = np.random.choice(np.arange(len(target_bin_dist)), 1, p=target_bin_dist)[0]

    best_r, worst_r = np.max(returns), np.min(returns)

    # if the bin is not empty, sample a random data point from this bin
    if orig_bin_dist[bin_id] != 0:
        data_id = np.random.choice(bin_dict[bin_id], 1)[0]
        r = returns[data_id]
        std = noise_scale if noise_constant else noise_scale * (r - worst_r) / (best_r - worst_r)
        noise = np.random.normal(loc=0.0, scale=std)
    # if the bin is empty, sample a random data point from the closet bin, and add an offset
    else:
        if np.sum(orig_bin_dist[bin_id:]) > 0: # if a bin on the right has weights
            bin_id_alter = np.arange(bin_id, len(orig_bin_dist))[orig_bin_dist[bin_id:] > 0][0]
            data_id = np.random.choice(bin_dict[bin_id_alter], 1)[0]
            r = returns[data_id]
            low, high = bin_edges[bin_id], bin_edges[bin_id+1]
            offset = (high - r) + np.random.uniform(low-high, 0.0)
        else:
            bin_id_alter = np.arange(0, bin_id)[orig_bin_dist[:bin_id] > 0][-1]
            data_id = np.random.choice(bin_dict[bin_id_alter], 1)[0]
            r = returns[data_id]
            low, high = bin_edges[bin_id], bin_edges[bin_id+1]
            offset = (low - r) + np.random.uniform(0.0, high - low)
        r = r + offset
        std = noise_scale if noise_constant else noise_scale * (r - worst_r) / (best_r - worst_r)
        noise = np.random.normal(loc=0.0, scale=std)
        noise = offset + noise
        
    return data_id, r, noise

def sample_batch_from_bins(batch_size, orig_bin_dist, target_bin_dist, bin_edges, bin_dict, returns, noise_scale, noise_constant):
    ids, rs, noises = [], [], []
    for i in range(batch_size):
        i, r, n = sample_from_bins(orig_bin_dist, target_bin_dist, bin_edges, bin_dict, returns, noise_scale, noise_constant)
        ids.append(i)
        rs.append(r)
        noises.append(n)
    return ids, rs, noises