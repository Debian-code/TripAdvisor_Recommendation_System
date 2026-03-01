import numpy as np 

def cos_sim(vec1, vec2):
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0 

    return np.dot(vec1, vec2) / (norm1 * norm2)