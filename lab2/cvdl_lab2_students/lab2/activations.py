import numpy as np

def softmax(x, t=1):
    """"
    Applies the softmax temperature on the input x, using the temperature t
    """
    # TODO your code here
    temperateX = x / t
    e_j = np.exp(temperateX - np.max(temperateX))
    sum_j = np.sum(e_j)
    return e_j / sum_j
    # end TODO your code here