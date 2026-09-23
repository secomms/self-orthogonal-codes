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
    num_attempts=0;num_attempts_max = 50; #counter for abortion

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
                print("PORCODIOOOOOOO")
                break

            Gp = G.stack(c)

            #Elimination for lower part
            for i in range(kp):
                Gp[kp,:] += (-Gp[kp,i]*Gp[i,:]);

            if Gp[kp,kp] == 0:
                for i in range(kp+1,n_k):
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


def compress_antiorthogonal(A):
    M = copy(A)
    n_k = A.ncols()
    k = A.nrows()

    roots = compute_roots(127)
    roots[0] = 0

    extra_vars = []
    bitstring = [0,0]

    # First bitstring
    if A[0,0] > 63:
        bitstring[0] = 1

    pivoted = [0 for _ in range(n_k)]
    rows = [0 for _ in range(k)]

    for i in range(1,k):

        # Delete values on new row using pivoted columns
        for col in range(i+1):
            if pivoted[col]:
                f = M[i-1,col]                      # save factor before the row changes
                for j in range(n_k):                # all columns
                    M[i-1,j] = M[i-1,j] - M[rows[col],j]*f


        # See what column can be pivoted using the new row
        for col in range(i+1):
            if (pivoted[col] == 0) and (M[i-1,col] != 0):

                inv = M[i-1,col]^-1                 # save factor before the row changes
                for j in range(n_k):
                    M[i-1,j] = M[i-1,j]*inv

                for j in range(i-1):
                    g = M[j,col]                    # save factor before the row changes
                    for l in range(n_k):            # renamed from k
                        M[j,l] = M[j,l] - M[i-1,l]*g

                pivoted[col] = 1
                rows[col] = i-1
                break

        # Now compute if extra_vars are needed

        affine_var = -1

        for j in range(i+1):
            if pivoted[j] == 0 and affine_var == -1:
                affine_var = j
            elif pivoted[j] == 0:
                extra_vars.append(A[i,j]) 

        # Compute bitstring

        v = [0 for _ in range(i+1)]
        #print(affine_var)

        for j in range(i+1):
            if pivoted[j] == 1:

                # Left component of v
                for l in range(i+1):                # renamed from k
                    if pivoted[l] == 0 and l != affine_var:
                        v[j] -= M[rows[j],l]*M[i,l]

                # Right component of v
                for l in range(i+1,n_k):            # renamed from k
                    v[j] -= M[rows[j],l]*M[i,l]

        Fq = M.base_ring()

        a = Fq(1)
        b = Fq(0)
        c = Fq(1)

        # TODO: decide what to do when every column in 0..i is pivoted (no affine var)
        for j in range(i+1):
            if pivoted[j] == 1:
                a += M[rows[j],affine_var]^2
                b += (-M[rows[j],affine_var])*v[j]
                c += v[j]^2

            elif pivoted[j] == 0 and affine_var != j:
                print("yo",M[i,j])
                c += M[i,j]^2

        for j in range(i+1,n_k):
            c += M[i,j]^2

        b = 2*b

        if a == 0:
            x = -c*(b^-1)
            print(x,M[i,affine_var])
        else:
            discriminant = (b^2) - (4*a*c)            
            x1 = (2*a)^-1*(-b + roots[discriminant])
            if x1 != M[i,affine_var]:
                bitstring[(i >> 6)] += (1 << i)

        # Compute other values back to check
        for j in range(i+1):
            if pivoted[j] == 1:
                M[i,j] = (-M[rows[j],affine_var]*M[i,affine_var]) + v[j]

    # Triangularize A and give it back

    B = copy(A)
    for i in range(k):
        for j in range(i+1):
            B[i,j] = 0

    #print(B)
    #print(extra_vars)
    #print(bitstring)
    return B, bitstring, extra_vars

def decompress_antiorthogonal(B,bitstring,extra_vars):

    M = copy(B)
    A = copy(B)

    n_k = B.ncols()
    k = B.nrows()

    extra_vars_used = 0

    roots = compute_roots(127)
    roots[0] = 0

    pivoted = [0 for _ in range(n_k)]
    rows = [0 for _ in range(k)]

    #Recover first element

    b = -1
    for i in range(n_k-1):
        b -= M[0,i+1]^2

    
    if (bitstring[0] & 1) == 1:
        A[0,0] = 127 - roots[b]
        M[0,0] = 127 - roots[b]
    else:
        A[0,0] = roots[b]
        M[0,0] = roots[b]
    

    for i in range(1,k):

        # Delete values on new row using pivoted columns
        for col in range(i+1):
            if pivoted[col]:
                f = M[i-1,col]                      # save factor before the row changes
                for j in range(n_k):                # all columns
                    M[i-1,j] = M[i-1,j] - M[rows[col],j]*f


        # See what column can be pivoted using the new row
        for col in range(i+1):
            if (pivoted[col] == 0) and (M[i-1,col] != 0):

                inv = M[i-1,col]^-1                 # save factor before the row changes
                for j in range(n_k):
                    M[i-1,j] = M[i-1,j]*inv

                for j in range(i-1):
                    g = M[j,col]                    # save factor before the row changes
                    for l in range(n_k):            # renamed from k
                        M[j,l] = M[j,l] - M[i-1,l]*g

                pivoted[col] = 1
                rows[col] = i-1
                break

        # Now compute if extra_vars are needed

        affine_var = -1

        for j in range(i+1):
            if pivoted[j] == 0 and affine_var == -1:
                affine_var = j
            elif pivoted[j] == 0:
                M[i,j] = extra_vars[extra_vars_used] 
                A[i,j] = M[i,j]
                extra_vars_used +=1


        v = [0 for _ in range(i+1)]
        #print(affine_var)

        for j in range(i+1):
            if pivoted[j] == 1:

                # Left component of v
                for l in range(i+1):                
                    if pivoted[l] == 0 and l != affine_var:
                        v[j] -= M[rows[j],l]*M[i,l]

                # Right component of v
                for l in range(i+1,n_k):            
                    v[j] -= M[rows[j],l]*M[i,l]

        Fq = M.base_ring()

        a = Fq(1)
        b = Fq(0)
        c = Fq(1)
        
        # TODO: decide what to do when every column in 0..i is pivoted (no affine var)
        for j in range(i+1):
            if pivoted[j] == 1:
                a += M[rows[j],affine_var]^2
                b += (-M[rows[j],affine_var])*v[j]
                c += v[j]^2

            elif pivoted[j] == 0 and affine_var != j:
                # The variable got passed
                #print("got passedd")
                c += M[i,j]^2

        for j in range(i+1,n_k):
            c += M[i,j]^2

        b = 2*b

        # Compute bitstring

        if a == 0:
            x = -c*(b^-1)
            M[i,affine_var] = x
            A[i,affine_var] = x
        else:
            discriminant = (b^2) - (4*a*c)            
            x1 = (2*a)^-1*(-b + roots[discriminant])
            #print(A)
            if (bitstring[i >> 6] & (1<<i)) == (1 << i):
                #print("x1", x1)
                M[i,affine_var] = (2*a)^-1*(-b - roots[discriminant])
                A[i,affine_var] = (2*a)^-1*(-b - roots[discriminant])
            else: 
                M[i,affine_var] = x1
                A[i,affine_var] = x1

        # Recompute the rest of the variables!

        for j in range(i+1):
            if pivoted[j] == 1:
                M[i,j] = (-M[rows[j],affine_var]*M[i,affine_var]) + v[j]
                A[i,j] = (-M[rows[j],affine_var]*M[i,affine_var]) + v[j]

    print(len(extra_vars))
    print(extra_vars_used)
    return A
        

"""
k = 100
n_k = k + 10
sub_dim = 50
samples_per_proc = 625
proc_num = 16

for q in [2,3,11,127,1021]:
    run_test(q,k,n_k,sub_dim,samples_per_proc,proc_num)
"""

#A,ctr = generate_antiorthogonal_matrix(GF(127),126,126)
#B,ctr = generate_antiorthogonal_matrix(GF(127),126,126)
#C = A.stack(B)
#G = C.augment(identity_matrix(GF(127),252))
#C = LinearCode(G)
#print(C.dual_code())
#G2 = C.dual().generator_matrix()
#u = random_vector(GF(127),126)

"""
A,ctr = generate_antiorthogonal_matrix(GF(127),4,4)
#print("ANTIORTHOGONAL?")
#print(A*A.T)
print(A)

x = A[0,0]^-1
for i in range(4):
    A[0,i] = A[0,i]*x

for i in range(4):
    A[1,i] = A[1,i] - A[0,i]*A[1,0]

x = A[1,1]^-1
for i in range(4):
    A[1,i] = A[1,i]*x

for i in range(1,4):
    A[0,i] = A[0,i] - A[1,i]*A[0,1]

print(A)

#print(A)


vec1 = [0 for _ in range(3)]
vec2 = [0 for _ in range(3)]

sqrt_table = [0 for i in range(127)]

for i in range(64):
    sqrt_table[i^2 % 127] = i


# x0 = - a2*x2 - a3*c3
# x1 = - b2*x2 - b3*c3

# x0^2 = (-a2x2 - a3c3)^2 = a2^2x2^2 + 2a2a3c3x2 + a3c3^2
# x1^2 = (-b2x2 - b3c3)^2 = b2^2x2^2 + 2b2b3c3x2 + b3c3^2
# => coeff_0 = 1 + a2^2 + b2^2
# => coeff_1 = -2a2a3c3 -2b2b3c3
# => coeff_2 = a3c3^2 + b3c3^2 + 1

a = 1 + A[0,2]^2 + A[1,2]^2
b = (2*A[0,2]*A[0,3]*A[2,3]) + (2*A[1,2]*A[1,3]*A[2,3])
c = (A[0,3]*A[2,3])^2 + (A[1,3]*A[2,3])^2 + A[2,3]^2 + 1
discriminant = b^2 - 4*a*c

#print(discriminant, sqrt_table[discriminant])
solution0 = (-b + sqrt_table[discriminant])/(2*a)
solution1 = (-b - sqrt_table[discriminant])/(2*a)
print("Solutions:",solution0,solution1)

# Testing solutions

vec1 = vector(GF(127),4)
vec1[0] = -A[0,2]*solution0 - A[0,3]*A[2,3]
vec1[1] = -A[1,2]*solution0 - A[1,3]*A[2,3]
vec1[2] = solution0 
vec1[3] = A[2,3]
print(vec1*A[0])
print(vec1*A[1])
print(vec1 == A[2])
print(vec1*vec1)

vec1 = vector(GF(127),4)
vec1[0] = -A[0,2]*solution1 - A[0,3]*A[2,3]
vec1[1] = -A[1,2]*solution1 - A[1,3]*A[2,3]
vec1[2] = solution1
vec1[3] = A[2,3]
print(vec1)
print(vec1*A[0])
print(vec1*A[1])
print(vec1 == A[2])
print(vec1*vec1)
"""

#vec1 = vector(GF(127),4)
#vec1[0] = -A[0,2]*solution0 - A[0,3]*A[2,3]
#vec1[1] = -A[1,2]*solution1 - A[1,3]*A[2,3]
#vec1[2] = solution0 
#vec1[3] = A[2,3]
#print(vec1)
    

#B,ctr = generate_antiorthogonal_matrix(GF(127),6,6)
#print(B)


#for i in range(10):
#    A,ctr = generate_antiorthogonal_matrix(GF(127),126,126)
#    #print(A)
#    #print("=======")
#    B,bitstring,extra_vars = compress_antiorthogonal(A)
#    recA = decompress_antiorthogonal(B,bitstring,extra_vars)
#    print(A==recA)
#    #print("=====")
#    #print(recA)

#print(A)
#print("===================")
#print(recA)
#print(rank_probability(5,6,5,5))

#def count_self_dual(q,n,k):
#    count = (q ^ (n/2 - k) + 1)/(q ^ (n/2) + 1)
#
#    for i in range(1,k+1):
#        count *= (q ^ (n - 2*i + 2) - 1)/(q ^ (i) - 1)
#
#    return count

def num_antiorthogonal(q, k, n_k):
    """Number of k x n_k matrices A over GF(q), q odd, with A*A.T == -I."""
    F = GF(q)
    eta = lambda x: 1 if F(x).is_square() else -1   # quadratic character
    c = F(-1)
    total = 1
    for i in range(k):
        d = n_k - i                  # dimension of the remaining complement
        disc = F(-1)^i               # its discriminant, up to squares
        if d <= 0:
            return 0
        if d % 2 == 1:
            N = q^(d-1) + q^((d-1)//2) * eta((-1)^((d-1)//2) * c * disc)
        else:
            N = q^(d-1) - q^(d//2 - 1) * eta((-1)^(d//2) * disc)
        total *= N
    return total

#print(num_antiorthogonal(5, 4, 4))   # 28800

from collections import Counter
from scipy.stats import chisquare

q = 2
n = 8
k = 4

counts = Counter()
for i in range(100000):
    A,ctr = generate_antiorthogonal_matrix(GF(q),k,n-k)
    A.set_immutable()
    counts[A] += 1
    print(i)
    #print(i)

K = int(num_antiorthogonal(q,k,n-k))
print(K)

print("Number of matrices:", K)
observed = list(counts.values())
observed += [0] * (K - len(observed))   # unseen matrices

observed = [int(c) for c in observed]   # plain Python ints for scipy
stat, p = chisquare(observed)           # default expectation is uniform: N/K each
print(stat, p)











