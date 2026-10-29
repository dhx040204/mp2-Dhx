import numpy as np
from pathlib import Path

def cos_sim(a, b):
    dot = np.dot(a, b)
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < 1e-8 or nb < 1e-8:
        return 0.0
    return dot / (na * nb)

def predict(bank, arrays, checkpoint: Path | None):
    emb = arrays["embeddings"]
    N = len(bank)
    assert emb.shape[0] == N
    w = 0.3 
    pred = []
    for i in range(N):
        vecs = emb[i]
        sim_mat = np.zeros((4,4))
        for p in range(4):
            for q in range(4):
                sim_mat[p,q] = cos_sim(vecs[p], vecs[q])
        sim_var = np.var(sim_mat, axis=1)
        aff = np.zeros(4)
        for c in range(4):
            aff[c] = np.sum([sim_mat[c,o]**2 for o in range(4) if c != o])
        aff_norm = aff / (np.max(aff)+1e-8)
        var_norm = 1 - sim_var/(np.max(sim_var)+1e-8)
        total = w * aff_norm + (1-w)*var_norm
        best_idx = int(np.argmax(total))
        pred.append(best_idx)
    return pred
