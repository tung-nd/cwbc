import numpy as np
import logging
import matplotlib.pyplot as plt

EXPERT_RETURN = {"Breakout": 30, "Seaquest": 42055, "Qbert": 13455, "Pong": 15}
MAX_CONDITION_RETURN = {"Breakout": 90, "Seaquest": 1450, "Qbert": 2500, "Pong": 20}
RANDOM_RETURN = {"Breakout": 2, "Seaquest": 68, "Qbert": 164, "Pong": -21}

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

def draw_hist(returns, game, nbins, path):
    _, _, _ = plt.hist(returns, bins=nbins, density=True, facecolor='g', alpha=0.75)
    plt.xlabel('Returns')
    plt.ylabel('Probability')
    plt.title(f'{game}-{nbins}bins')
    plt.tight_layout()
    plt.savefig(path)
    plt.clf()
    plt.close()

def get_logger(filename, mode='a'):
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    logger = logging.getLogger()
    for hdlr in logger.handlers:
        logger.removeHandler(hdlr)
    logger.addHandler(logging.FileHandler(filename, mode=mode))
    logger.addHandler(logging.StreamHandler())
    return logger