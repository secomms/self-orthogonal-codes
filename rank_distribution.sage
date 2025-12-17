import time
import os
import random
import numpy as np
from multiprocessing import Pool

def get_random_permutation(field,n):
    """
    Generate a random permutation matrix of size n over the specified field.
    """
    perm_list = Permutations([i+1 for i in range(n)]).random_element()
    perm = Permutation(perm_list)
    P = perm.to_matrix()
    return P

def swap_cols(A,i,j):
    """
    Return a copy of A with columns i and j swapped. 
    """
    B = copy(A)
    tmp = B[:,i]
    B[:,i] = B[:,j]
    B[:,j] = tmp
    return B

def swap_rows(A,i,j):
    """
    Return a copy of A with rows i and j swapped. 
    """
    B = copy(A)
    tmp = B[i,:]
    B[i,:] = B[j,:]
    B[j,:] = tmp
    return B

def compute_roots(q):
    """
    Compute all roots and (consequent squares) in Fq. 
    """
    roots = {}
    if q == 2:
        roots[0] = 0
        roots[1] = 1
    else:
        for i in range(1,(q-1)/2+1):
            roots[i^2 % q] = i
    return roots 

def generate_antiorthogonal_matrix(Fq,k,n_k):
    """
    Generate antiorthogonal matrix of size k times n_k over Fq. 
    """
    q = Fq.characteristic()
    num_attempts=0;num_attempts_max = 300; #counter for abortion

    kp = 1
    while kp < k:
        num_attempts=0;
        roots = compute_roots(q) 
        c = random_vector(Fq,n_k)
        while c*c != Fq(-1):
            c = random_vector(Fq,n_k)
            if q != 2 and c*c != 0 and -(c*c)^-1 in roots:
                b = roots[-(c*c)^-1]
                b = random.choice([b,q-b])
                c = b*c
                
        # First element of the vector non null
        i = 0
        while c[i]==0:
            i+=1;
        tmp = c[0];
        c[0] = c[i];
        c[i] = tmp;

        A = matrix(Fq,c) # antiorthogonal matrix
        G = matrix(Fq,c) # vector space basis
        G = G[0,0]^-1*G # normalizing vector space basis

        kp = 1; #this is the number of found linearly independent codewords

        queue = []

        ctr = 0
        
        while kp < k:

            P = G[:,kp:];
            H = (-P.T).augment(identity_matrix(Fq,n_k-kp))
            
            u = random_vector(Fq,n_k-kp)
            c = u*H
            while c*c != Fq(-1) and num_attempts < num_attempts_max:
                u = random_vector(Fq,n_k-kp)
                c = u*H
                if q != 2 and c*c != 0 and -(c*c)^-1 in roots:
                    b = roots[-(c*c)^-1]
                    b = random.choice([b,q-b])
                    c = b*c
                num_attempts += 1

            if num_attempts >= num_attempts_max:
                break

            Gp = G.stack(c)

            #Elimination for lower part
            for i in range(kp):
                Gp[kp,:] += (-Gp[kp,i]*Gp[i,:]);

            if Gp[kp,kp] == 0:
                for i in range(kp+1,2*k):
                    flag = False
                    if Gp[kp,i] !=0:
                        Gp = swap_cols(Gp,i,kp)
                        A = swap_cols(A,i,kp)
                        tmp = c[i]
                        c[i] = c[kp]
                        c[kp] = tmp
                        flag = True
                        break

            Gp[kp,:] = Gp[kp,kp]^-1*Gp[kp,:];
            #Do elimination for upper part
            for i in range(kp):
                Gp[i,:] += (-Gp[i,kp]*Gp[kp,:]);
            G = Gp;
            A = A.stack(c)
            kp += 1;

    LP = get_random_permutation(Fq,k)
    RP = get_random_permutation(Fq,n_k)
    A = LP*A*RP

    return A, ctr

def gaussian_binomial(n,k,q):
    """
    Compute gaussian binomial (n,k)_q.
    """
    coeff = 1
    for i in range(k):
        coeff = coeff * (1 - q^(n-i))/(1 - q^(i+1)) *1.0
    return coeff

def rank_probability(m,n,r,q):
    """
    Compute probability that a random (m x n) matrix in Fq has rank r.
    """
    prob = gaussian_binomial(n,r,q)
    for i in range(0,r):
        prob = prob * (q^m - q^i) * 1.0
    prob = prob / (q^(m*n))
    return prob

def antiorthogonal_rank_simulation(k,n_k,subk,q,no):
    """
    Generate 'no' antiorthogonal matrices and compute (subk x subk+1) sub rank.
    """
    Fq = GF(q)
    sub_ranks = []
    for i in range(no):
        A, ctr = generate_antiorthogonal_matrix(Fq,k,n_k) 
        LP = get_random_permutation(Fq,k)
        RP = get_random_permutation(Fq,n_k)
        A = LP*A*RP
        subA = A[:subk,:subk+1]
        sub_ranks.append(subA.rank())

    return sub_ranks 

def seed_initializer():
    """
    Init seeds for random generation.
    """
    seed = int.from_bytes(os.urandom(8), 'big')
    set_random_seed(seed)
    seed = seed % 2^31
    random.seed(int(seed))
    np.random.seed(int(seed))

def run_test(q,k,n_k,sub_dim,samples_per_proc,proc_num):
    """
    Generate proc_num processes that compute rank for samples_per_proc matrices.
    """
    pool = Pool(processes=proc_num, initializer=seed_initializer)
    results = []
    print(f"Generating {samples_per_proc*proc_num} antiorthogonal matrices ({proc_num} processes)")
    for _ in range(proc_num):
        results.append(pool.apply_async(antiorthogonal_rank_simulation, [k,n_k,sub_dim,q,samples_per_proc]))
    pool.close()
    time.sleep(float(1))
    pool.join()
    print("Done!")
    ranks = []
    for result in results:
        ranks = ranks + result.get()
    print("Antiorthogonal AVG rank",N(np.mean(ranks)))

    points = []
    for i in range(sub_dim+1):
        points.append((i,N(rank_probability(sub_dim,sub_dim+1,sub_dim-i,q))))
    print(ranks)

    rank_defs = []
    for rank in ranks:
        rank_defs.append(sub_dim - rank)

    rank_defs_count = [0 for i in range(5)]
    for rank_def in rank_defs:
        rank_defs_count[rank_def] = rank_defs_count[rank_def] + 1

    with open(f"q{q}k{sub_dim}sim.dat",'w') as file:
        for i in range(len(rank_defs_count)):
            file.write(f"{i} {N(rank_defs_count[i]/(proc_num*samples_per_proc))}\n")

    with open(f"q{q}k{sub_dim}real.dat",'w') as file:
        for i in range(len(rank_defs_count)):
            file.write(f"{points[i][0]} {N(points[i][1])}\n")

"""
k = 100
n_k = k + 10
sub_dim = 50
samples_per_proc = 625
proc_num = 16

for q in [2,3,11,127,1021]:
    run_test(q,k,n_k,sub_dim,samples_per_proc,proc_num)
"""

A,ctr = generate_antiorthogonal_matrix(GF(127),126,126)
B,ctr = generate_antiorthogonal_matrix(GF(127),126,126)
C = A.stack(B)
G = C.augment(identity_matrix(GF(127),252))
C = LinearCode(G)
print(C.dual_code())
#G2 = C.dual().generator_matrix()
#u = random_vector(GF(127),126)
