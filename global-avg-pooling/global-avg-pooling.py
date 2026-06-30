import numpy as np

def sum_(x):
    sum=0
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            sum+=x[i][j]
    return  sum 

def average_func(x):
    result = []
    for c in range(x.shape[0]):
        ch = x[c]
        avg = sum_(ch)/ch.size
        result.append(avg)
        
    return result

def global_avg_pool(x):
    """
    Compute global average pooling over spatial dims.
    Supports (C,H,W) => (C,) and (N,C,H,W) => (N,C).
    """

    if x.ndim not in (3, 4):
        raise ValueError(f"Expected 3D (C,H,W) or 4D (N,C,H,W) input, got {x.ndim}D array with shape {x.shape}")
    if len(x.shape)==3:

        return np.array(average_func(x))
    else:
        arr=[]
        for i in range(x.shape[0]):
            arr.append(average_func(x[i]))
        return np.array(arr)
        