import os, sys, time
import numpy as np
module_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(module_dir, "../..")))
from phaser.similarities import *
from scipy.spatial.distance import hamming
from scipy.ndimage import convolve

def generate_hashes(num_values, length):
    vals = []
    z = np.zeros(length)
    for n in range(num_values):
        # generate non zero array pairs as a tuple, add to vals
        a = z
        while np.array_equal(a, z):
            a = np.random.randint(2, size=length)
        b = z
        while np.array_equal(b, z):
            b = np.random.randint(2, size=length)
            
        vals.append((a, b))
    
    return np.array(vals)

num_runs = 1
num_comparisons = 100_000
hash_size = 64

times = {}

# get max value for convolution
ones_m = np.ones(64,).reshape(8,8)
max_conv = np.sum(convolve(ones_m, np.ones((4,4)), mode='constant', cval=0))

# cache fast alg
hatched_matrix_fast(u=np.random.randint(2, size=64), v=np.random.randint(2, size=64))


for r in range(num_runs):
    print(f"run: {r}")
    hash_pairs = generate_hashes(num_comparisons, hash_size)
    
    
    # hatched metric
    t0 = time.time()
    for i in hash_pairs:
        hatched_matrix(u=i[0], v=i[1], distance_fun=hamming)
    t1 = time.time()
    elapsed = t1-t0
    times.setdefault(hatched_matrix.__name__, []).append(elapsed)
    
    # convolve 4_4
    t0 = time.time()
    for i in hash_pairs:
        convolution_distance(u=i[0], v=i[1], max_value=max_conv, filter=np.ones((4,4)))
    t1 = time.time()
    elapsed = t1-t0
    times.setdefault(convolution_distance.__name__, []).append(elapsed)
    
    # 2gram
    t0 = time.time()
    for i in hash_pairs:
        ngram_cosine_distance(u=i[0], v=i[1], ngram_size=2)
    t1 = time.time()
    elapsed = t1-t0
    times.setdefault(ngram_cosine_distance.__name__, []).append(elapsed)
    
    # fast hatched - note: currently using  Hamming distance in the calculation as 
    # fastdist.cosine frequently throws zero division errors with its implementaiton.
    t0 = time.time()
    for i in hash_pairs:
        hatched_matrix_fast(u=i[0], v=i[1])
    t1 = time.time()
    elapsed = t1-t0
    times.setdefault(hatched_matrix_fast.__name__, []).append(elapsed)
    
    # hamming
    t0 = time.time()
    for i in hash_pairs:
        hamming(u=i[0], v=i[1])
    t1 = time.time()
    elapsed = t1-t0
    times.setdefault(hamming.__name__, []).append(elapsed)
    
for k,v in times.items():
    print(f"{k}: {round(np.mean(v), 2)}")
    
