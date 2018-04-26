import numpy as np

def softmax(x, scale = 1):
    x = np.array(x)/scale
    max_x = np.max(x)
    e_x = np.exp(x - max_x)
    p = e_x/e_x.sum()
    p = p/p.sum()

    return p

def logsumexp(x, scale = 1):
    x = np.array(x)/scale
    max_x = np.max(x)
    lse_x = max_x + np.log(np.exp(x-max_x).sum())
    lse_x = scale*lse_x
    return lse_x

def sparsedist(z, scale=1.):
    z = np.array(z/scale)
    if len(z.shape) == 1:
        z = np.reshape(z,(1,-1))
    z = z - np.mean(z, axis=1)[:, np.newaxis]

    # sort z
    z_sorted = np.sort(z, axis=1)[:, ::-1]

    # calculate k(z)
    z_cumsum = np.cumsum(z_sorted, axis=1)
    k = np.arange(1, z.shape[1] + 1)
    z_check = 1 + k * z_sorted > z_cumsum
    k_z = z.shape[1] - np.argmax(z_check[:, ::-1], axis=1)

    # calculate tau(z)
    tau_sum = z_cumsum[np.arange(0, z.shape[0]), k_z - 1]
    tau_z = ((tau_sum - 1) / k_z).reshape(-1, 1)

    # calculate p
    p = np.maximum(0, z - tau_z)
    return p 

def sparsemax(z, scale=1.):
    z = np.array(z/scale)    
    z = z - np.mean(z, axis=1)[:, np.newaxis]

    # calculate sum over S(z)
    p = sparsedist(z)
    s = p > 0
    # z_i^2 - tau(z)^2 = p_i (2 * z_i - p_i) for i \in S(z)
    S_sum = np.sum(s * p * (2 * z - p), axis=1)

    return 0.5 * S_sum + 0.5

if __name__ == '__main__':
    print("Main Started")
    x = np.random.rand(5)
    print(x[range(0,len(x))])
    print(np.sort(x))
    print(np.max(x))
    print(logsumexp(x))
    print(sparsemax(x))
    print(softmax(x))
    print(sparsedist(x))