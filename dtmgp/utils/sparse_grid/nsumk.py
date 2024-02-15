import itertools
import torch


def n_sum_k(n, k):
    """
    ------------------------
    Parameters:
    n: # of positive integer
    k: sum of the integers = k

    ------------------------
    Returns:
    a list of all possible combinations of n positive integers adding up to a given number k 
    """
    if n == 1:
        return torch.tensor( [[k]] )
    else:
        res = []
        for i in itertools.product(range(1, k - n + 2), repeat=n):
            if sum(i) == k:
                res.append(i)#yield i
        return torch.tensor( res )