import torch
import time
import logging
from collections import OrderedDict
import re
import matplotlib
from matplotlib import pyplot as plt
from os.path import split, splitext
import matplotlib.pyplot as plt

def get_logger(filename, mode='a'):
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    logger = logging.getLogger()
    for hdlr in logger.handlers:
        logger.removeHandler(hdlr)
    logger.addHandler(logging.FileHandler(filename, mode=mode))
    logger.addHandler(logging.StreamHandler())
    return logger

def draw_hist(returns, env_name, dataset, nbins, path):
    _, _, _ = plt.hist(returns, bins=nbins, density=True, facecolor='g', alpha=0.75)
    plt.xlabel('Returns')
    plt.ylabel('Probability')
    plt.title(f'{env_name}-{dataset}-{nbins}bins')
    plt.tight_layout()
    plt.savefig(path)
    plt.clf()
    plt.close()