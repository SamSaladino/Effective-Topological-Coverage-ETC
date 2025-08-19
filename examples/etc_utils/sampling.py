import random, numpy as np

def random_subsets(nodes, k, iters, seed=1):
    rng = random.Random(seed); nodes = list(nodes)
    for _ in range(iters):
        yield [nodes[i] for i in rng.sample(range(len(nodes)), k)]

def sweep_gamma(H_fn, A, Dinv2, k, nodes, mu, gammas, iters=300, seed=1):
    vals = []
    for g in gammas:
        Hs = []
        for S in random_subsets(nodes, k, iters, seed=seed):
            S_idx = [int(x) for x in S]
            Hs.append(H_fn(A, Dinv2, S_idx, mu=mu, gamma=g)[0])
        vals.append((g, np.array(Hs)))
    return vals
