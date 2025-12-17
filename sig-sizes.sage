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

def exp_com_cost(q,k):
    """
    Compute expected compression algorithm cost.
    """
    com_cost = 1 
    for i in range(2,k):
        for rank_def in range(1,i):
            prob = rank_probability(i-1,i,i-1-rank_def,q)
            com_cost += 1 + prob*rank_def*ceil(log(q,2)) + 1.0
            if rank_def*prob < 10^(-10):
                break
    return com_cost

def less_speck_pk_size(q,n,k,s,lamb, alg=0):
    """
    Compute LESS/SPECK pk size for given parameters.
    Alg:
        - 0: no compression size 
        - 1: worst case output size
        - 2: expected output size
    """
    pk_size = lamb # Seed for G resampling
    pk_size += (s-1)*log(binomial(n,k),2) # Transformation for IS on left
    if alg == 0:
        pk_size += (s - 1)*(k*(n-k)*log(q,2))
    elif alg == 1:
        pk_size += (s - 1)*((k*(n-k) - k*(k-1)/2)*log(q,2) + log(factorial(k)*binomial(n, k)*binomial(n-k,k),2))
    elif alg == 2:
        pk_size += (s - 1)*((k*(n-k) - k*(k+1)/2)*log(q,2) + exp_com_cost(q,k))

    return pk_size

def abl_pk_size(q,n,k,alg=0):
    """
    Compute ABL pk size for given parameters.
    Alg:
        - 0: no compression size 
        - 1: worst case output size
        - 2: expected output size
    """
    pk_size = log(binomial(n,k),2) # Transformation for IS on left
    pk_size += log(q,2)*k # u in Fq^n
    if alg == 0:
        pk_size += (k*(n-k)*log(q,2))
    elif alg == 1:
        pk_size += ((k*(n-k) - k*(k-1)/2)*log(q,2) + log(factorial(k)*binomial(n, k)*binomial(n-k,k),2))
    elif alg == 2:
        pk_size += ((k*(n-k) - k*(k+1)/2)*log(q,2) + exp_com_cost(q,k))

    return pk_size

def print_less_speck_sizes(q,k,n,s,lamb):
    print(f"{q:<10}{k:<10}{n:<10}{s:<5}{lamb:<6}{'':<15}{N(less_speck_pk_size(q,n,k,s,lamb))/(8*1024):<20.2f}{N(less_speck_pk_size(q,n,k,s,lamb,1))/(8*1024):<20.2f}{N(less_speck_pk_size(q,n,k,s,lamb,2))/(8*1024):<20.2f}")

def print_abl_sizes(q,k,n,s,lamb):
    print(f"{q:<10}{k:<10}{n:<10}{s:<5}{lamb:<6}{'':<15}{N(abl_pk_size(q,n,k))/(8*1024):<20.2f}{N(abl_pk_size(q,n,k,1))/(8*1024):<20.2f}{N(abl_pk_size(q,n,k,2))/(8*1024):<20.2f}")

def print_param_line(scheme):
    print("")
    print('='*60 + f"{scheme:^5}" + '='*60)
    print("")
    print(f"{'Parameters':^41}{'':<15}{'Sizes':^60}")
    print(f"{'q':<10}{'k':<10}{'n':<10}{'s':<5}{'lambda':<6}{'':<15}{'Full pk (KB)':<20}{'Worst case (KB)':<20}{'Expected (KB)':<20}")

# SPECK SIZES =====================

print_param_line('SPECK')

k = 126;n = 2*k;s = 2;lamb = 128
for q in [127,8861]:
    print_less_speck_sizes(q,k,n,s,lamb)

# LESS SIZES =====================

print_param_line('LESS')

q = 127;k = 126;n = 2*k;lamb = 128
for s in [2,4,8]:
    print_less_speck_sizes(q,k,n,s,lamb)

k = 200;n = 2*k;lamb = 192
for s in [2,4]:
    print_less_speck_sizes(q,k,n,s,lamb)

k = 274;n = k*2;lamb = 256
for s in [2,4]:
    print_less_speck_sizes(q,k,n,s,lamb)

# ABL SIZES =====================

print_param_line('ABL')

s = 2
k = 450;n = 7313;q = 2^13;lamb=128
print_abl_sizes(q,k,n,s,lamb)

k = 550;n = 11000;q = 2^16;lamb=128
print_abl_sizes(q,k,n,s,lamb)

k = 900;n = 20250;q = 2^18;lamb=192
print_abl_sizes(q,k,n,s,lamb)

k = 1250;n = 29688;q = 2^19;lamb=256
print_abl_sizes(q,k,n,s,lamb)
