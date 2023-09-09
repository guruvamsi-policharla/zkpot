# Script to estimate the offline proof size
N_PARTIES=512
L=N_PARTIES/8
RANGE_m=16
RANGE_M=64
RANGE_k=48
FRI_OPENINGS = 95
FIELD_SIZE = 128 #bits

# format bits
def format_bits(size):
    # convert to bits
    size = size/8

    power = 1024
    n = 0
    power_labels = {0 : '', 1: 'kilo', 2: 'mega', 3: 'giga', 4: 'tera'}
    while size > power:
        size /= power
        n += 1
    return str(size)+power_labels[n]+'bytes'

# we count the number of polynomials and multiply by openings to compute approximate FRI opening sizes
def ProdSumCheck(k):
    # k is number of poly in the linear combination
    return (k+3)

def pss_check():
    size=0

    size += 2*L + N_PARTIES
    size += ProdSumCheck(N_PARTIES)
    return size

def range_check(range):
    size=0
    
    size+=2*L*range
    size+=L
    return size

def data_check(D, B):
    size=0
    
    # Size of X-Xpack consistency
    size += D/L*pss_check() #N
        
    # Sizr of Xt-Xtpack consistency
    size += B/L*pss_check() #ND/B
    
    # A ProdSumCheck over D+B polynomials where degree is max(N,ND/B)
    size += ProdSumCheck(D+B)

    return size

def rand_check():
    size = 0
    # alpha
    size += 2*pss_check() # for t and 2t
    # gamma
    size += 2*pss_check() # for t and 2t
    # theta
    size += 2*pss_check() # for t and 2t
    # mu
    size += 2*pss_check() # for t and 2t
    # tau
    size += pss_check() # for 2t    
    size += pss_check() # for 2t    
    # nu
    size += pss_check() # for 2t 
    size += pss_check() # for 2t 

    # rand consistency check
    size += ProdSumCheck(2*L)
    size += ProdSumCheck(2*L)

    return size

def mask_check():
    size = 0
    
    # PSS consistency and range proof   
    #maskd
    size += pss_check() 
    size += range_check(RANGE_m)
    #masku
    size += pss_check()
    size += range_check(RANGE_M+RANGE_k)
    #ltd
    size += pss_check()
    size += range_check(RANGE_M)
    #ltu
    size += pss_check()
    size += range_check(RANGE_k+1)
    # ltbd
    size += pss_check()
    size += range_check(RANGE_M)
    # ltbu
    size += pss_check()
    size += range_check(RANGE_k+1)
    # maskbd
    size += pss_check()
    size += range_check(RANGE_m)
    # maskbu
    size += pss_check()
    size += range_check(RANGE_M+RANGE_k)

    return size

# N*D/(B*L), N/L, N, 
data_size = 1<<20
dim = 256 
batches = 256

# rand and mask checks don't change with dimensions and batch size
rand_poly = rand_check()
mask_poly = mask_check()
data_poly = data_check(dim, batches)

total_poly = mask_poly + rand_poly

print("rand_poly:"+str(rand_poly))
print("mask_poly:"+str(mask_poly))
print("data_poly:"+str(data_poly))

print("mask+rand opening size:"+str(format_bits(total_poly*FRI_OPENINGS*FIELD_SIZE)))
print("==================================")