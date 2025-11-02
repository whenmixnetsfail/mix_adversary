# -*- coding: utf-8 -*-
"""
This .py file contains a comprehensive set of functions for analyzing the paper.
"""
import pickle
from math import exp
from scipy import constants
#from Routing import Routing
from scipy.stats import expon
import simpy
import random
import numpy  as np
import pickle
import math
import json
from itertools import product
from itertools import chain, combinations



def biased_distribution(n, sigma, alpha=1):
    if not (0 <= sigma <= 1):
        raise ValueError("sigma must be between 0 and 1")
    if n <= 0:
        raise ValueError("n must be a positive integer")

    
    # Intermediate: generate soft bias
    weights = [(1)/(2**(i)) for i in range(n)]
    power = (1 - sigma) ** alpha
    transformed = [w ** power for w in weights]
    total = sum(transformed)
    #print([w / total for w in transformed])
    return [w / total for w in transformed]




def gaussian_samples(n, mu, sigma):
    return np.random.normal(loc=mu, scale=sigma, size=n).tolist()


def entropyy(prob_dist):
    return -sum(p * math.log2(p) for p in prob_dist if p > 0)
#print(entropy([1/16]*16))


def common_sublists(list1, list2):
    set1 = set(tuple(sublist) for sublist in list1)
    set2 = set(tuple(sublist) for sublist in list2)
    
    common = set1 & set2
    return len(common)


'''
list1 = [[1, 2], [3, 4]]
list2 = [[1, 2], [5, 6]]

count = common_sublists(list1, list2)


print("Count:", count)
'''


def cartesian_product(lists):
    return [list(tup) for tup in product(*lists)]

'''
input_lists = [['A', 'B', 'C'], [1, 2], ['x', 'y']]
result = cartesian_product(input_lists)

for r in result:
    print(r)
'''

def all_subsets(L):
    nums = list(range(1, L + 1))
    return list(chain.from_iterable(combinations(nums, r) for r in range(len(nums) + 1)))


def select_m(lst, m):
    if m > len(lst):
        raise ValueError("Cannot select more items than are in the list.")
    return random.sample(lst, m)
def pick_m(a, b, n):
    if b - a + 1 < n:
        raise ValueError("Not enough unique numbers in range to sample.")
    return random.sample(range(a, b + 1), n)
    
def classify_regions(latlon_list):
    # Define regions as (lat_min, lat_max, lon_min, lon_max)
    regions = {
        "North America": [(15, 72), (-170, -50)],
        "West Europe": [(35, 71), (-10, 25)],
        "East Europe": [(35, 71), (25, 60)],
        "Asia": [(5, 80), (60, 180)],
        "South America": [(-60, 15), (-90, -30)],
        "Africa": [(-35, 37), (-20, 55)],
    }

    # Initialize counts
    region_counts = {region: 0 for region in regions}
    region_counts["Other"] = 0

    for lat, lon in latlon_list:
        found = False
        for region, (lat_range, lon_range) in regions.items():
            if lat_range[0] <= lat <= lat_range[1] and lon_range[0] <= lon <= lon_range[1]:
                region_counts[region] += 1/len(latlon_list)
                found = True
                break
        if not found:
            region_counts["Other"] += 1/len(latlon_list)

    return region_counts

def pick_m_from_n(n, m):
    if m > n:
        raise ValueError("m must be less than or equal to n")
    return np.random.choice(n, size=m, replace=False)
 
 

def choose_m(P,N):
    if len(P) != N:
        raise ValueError("Length of probability list must be equal to N")
    return random.choices(range(N), weights=P, k=1)[0]
   

def P_compute(P,G1,G2):
    
    W = len(P)
    List = []
    for i in range(W):
        for j in range(W):
            for k in range(W):
                List.append(P[i]*(G1[i,j])*(G2[j,k]))
    return List

def normalize_rows_by_sum(matrix):
    row_sums = matrix.sum(axis=1, keepdims=True)
    # Avoid division by zero
    row_sums[row_sums == 0] = 1
    return matrix / row_sums


def compute_conditionals(probs_):
    probs = probs_.copy()
    W = round(len(probs) ** (1/3))
    A = np.zeros((W, W, W), dtype=float)
    
    for i in range(W):
        for j in range(W):
            start = i * (W ** 2) + j * W
            end = start + W
            A[i, j, :] = probs[start:end]
    
    P_y_given_x = np.zeros((W, W))  # rows: X=i, cols: Y=j
    P_z_given_y = np.zeros((W, W))  # rows: Y=i, cols: Z=j
    P_List = []
    
    for i in range(W):
        for j in range(W):
            P_y_given_x[i, j] = np.sum(A[i, j, :])      # P(X=i, Y=j)
            P_z_given_y[i, j] = np.sum(A[:, i, j])      # P(Y=i, Z=j)
            
    for i in range(W):
        P_List.append(np.sum(probs_[i*(W**2):(i+1)*(W**2)]))
            
    
    return P_List,normalize_rows_by_sum(P_y_given_x), normalize_rows_by_sum(P_z_given_y)


def Gradient_descent_(Gamma,Beta,a):#Gamma and beta restrication matrix with learning parameter a 
    import numpy as np
    (n1,n2) = np.shape(Gamma)
    empty = np.zeros((n1,n2))
    ones_Gamma = np.ones((n1,n2)).dot(Gamma)
    one_beta = np.ones((n1,1)).dot(np.transpose(Beta))
    
    Gamma1  = np.copy(Gamma)
    ALPHA = np.log(Gamma1+0.0000001)
    
    for i in range(n1):
        for j in range(n2):
            
            x = np.copy(empty)
            x[i,j] = ALPHA[i,j]
            
            y = np.transpose(x)
            z = np.trace(y.dot(ones_Gamma)-y.dot(one_beta))
            ALPHA[i,j] = ALPHA[i,j] +a*z

    return normalize_rows_by_sum(np.exp(ALPHA))

def Gradient_descent__(Gamma,Beta,a):#Gamma and beta restrication matrix with learning parameter a 
    import numpy as np
    (n1,n2) = np.shape(Gamma)
    empty = np.zeros((n1,n2))
    ones_Gamma = np.ones((n1,n2)).dot(Gamma)
    one_beta = np.ones((n1,1)).dot(np.transpose(Beta))
    
    Gamma1  = np.copy(Gamma)
    ALPHA = np.log(Gamma1+0.0000001)
    
    for i in range(n1):

            
        x = np.copy(empty)
        x[i,:] = ALPHA[i,:]
        
        y = np.transpose(x)
        z = np.trace(y.dot(ones_Gamma)-y.dot(one_beta))
        ALPHA[i,:] = ALPHA[i,:] +a*z

    return normalize_rows_by_sum(np.exp(ALPHA))

def base_to(List,W):
    return np.sum([ (W**(len(List)-i-1))*List[i] for i in range(len(List)) ])

def to_base(n, base,L):
    if base < 2:
        raise ValueError("Base must be at least 2")

    if n == 0:
        return [0]*L
    
    digits = []
    is_negative = n < 0
    n = abs(n)

    while n > 0:
        digits.append(n % base)
        n //= base

    digits.reverse()  # Most significant digit first

    A = [-d for d in digits] if is_negative else digits
    #print(A)
    if len(A) < L:
        bb = L-len(A)
        BB = [0]*bb
        return BB + A

    return A
    
    
        




def Gradient_descent(A,Beta,a):
    import numpy as np

    """
    Normalize a square matrix:
    1. Normalize each column by its sum.
    2. Normalize each row of the resulting matrix by its row sum.
    
    Parameters:
    A (np.ndarray): An n x n square matrix.
    
    Returns:
    np.ndarray: The normalized matrix.
    """
    A = np.array(A, dtype=float)  # Ensure it's a float NumPy array

    # Step 1: Normalize each column by its sum
    col_sums = A.sum(axis=0)
    # Avoid division by zero
    col_sums[col_sums == 0] = 1
    A = A / col_sums

    # Step 2: Normalize each row by its sum
    row_sums = A.sum(axis=1)
    # Avoid division by zero
    row_sums[row_sums == 0] = 1
    A = (A.T / row_sums).T

    return A







def Gradient_descent_IT(Gamma,Beta,a,It):
    count = 0
    while count < It:
       Gamma =  Gradient_descent(Gamma,Beta,a)
       count +=1


    return Gamma

def generate_combinations(list1, list2, list3):
    return list(product(list1, list2, list3))
def count_unique_per_column(data):
    arr = np.transpose(np.array(data))
    return [len(np.unique(arr[:, i])) for i in range(arr.shape[1])]

def JAR_Regions(List,W):
    list1 = List[:W]
    list2 = List[W:2*W]
    list3 = List[2*W:]
    a = (generate_combinations(list1,list2,list3))
    b = count_unique_per_column(a)
    return b
                
              
def convert_to_lat_lon(x, y, z):
    radius = 6371  # Earth's radius in kilometers

    # Convert Cartesian coordinates to spherical coordinates
    longitude = math.atan2(y, x)
    hypotenuse = math.sqrt(x**2 + y**2)
    latitude = math.atan2(z, hypotenuse)

    # Convert radians to degrees
    latitude = math.degrees(latitude)
    longitude = math.degrees(longitude)

    return latitude, longitude

def classify_region(lat, lon):
    if 15 <= lat <= 75 and -170 <= lon <= -50:
        return "North America"
    elif -60 <= lat <= 15 and -90 <= lon <= -30:
        return "South America"
    elif 35 <= lat <= 70 and -10 <= lon <= 40:
        return "Europe"
    elif 5 <= lat <= 80 and 40 <= lon <= 180:
        return "Asia"
    else:
        return "Other"

def classify_points(matrix):
    regions = []
    counts = {"North America": 0, "South America": 0, "Europe": 0, "Asia": 0, "Other": 0}

    for row in matrix:
        lat, lon = convert_to_lat_lon(*row)
        region = classify_region(lat, lon)
        regions.append(region)
        counts[region] += 1

    return regions


def find_row_permutation(A, B):
    #print(A,B)
    """
    Finds the row permutation mapping from A to B.

    Parameters:
        A (numpy.ndarray): Original matrix (N x M)
        B (numpy.ndarray): Permuted matrix (N x M)

    Returns:
        list: Mapping of rows from A to their positions in B
    """
    A = np.array(A)
    B = np.array(B)

    # Convert each row to a tuple so we can use list index
    A_list = [tuple(row) for row in A]
    B_list = [tuple(row) for row in B]

    # Find indices of A's rows in B
    mapping = [A_list.index(row) for row in B_list]

    return mapping
#Example
'''
A = np.array([[10, 20], 
              [30, 40], 
              [50, 60]])

B = np.array([[50, 60],  # Row from index 2 in A
              [10, 20],  # Row from index 0 in A
              [30, 40]]) # Row from index 1 in A

L = find_row_permutation(A, B)
print(L)  # Output: [2, 0, 1]

List_ = [0]*len(List1)
for i in range(len(List1)):
    
    List_[List2.index(List1[i])] = List1[i]
    
    
'''


def MAP_to_MAP(L1, L2):
    """
    Computes the permutation from A to C given the permutations from A to B (L1) and B to C (L2).
    
    Parameters:
        L1 (list): Permutation from A to B
        L2 (list): Permutation from B to C
    
    Returns:
        list: Permutation from A to C
    """
    return [L2[i] for i in L1]


'''
import numpy as np

# Define example matrices
A = np.array([[10, 20], 
              [30, 40], 
              [50, 60]])

B = np.array([[50, 60],  # Row from index 2 in A
              [10, 20],  # Row from index 0 in A
              [30, 40]]) # Row from index 1 in A

C = np.array([[30, 40],  # Row from index 1 in B (originally from index 2 in A)
              [50, 60],  # Row from index 0 in B (originally from index 0 in A)
              [10, 20]]) # Row from index 2 in B (originally from index 1 in A)

# Find permutations
L1 = find_row_permutation(A, B)  # Mapping from A to B
L2 = find_row_permutation(B, C)  # Mapping from B to C

# Compute permutation from A to C
L_final = MAP_to_MAP(L1, L2)

# Print results
print("L1 (A to B):", L1)  # Expected: [2, 0, 1]
print("L2 (B to C):", L2)  # Expected: [1, 0, 2]
print("Final Mapping (A to C):", L_final)  # Expected: [1, 2, 0]
'''    
#Example
'''
A = np.array([[10, 20], 
              [30, 40], 
              [50, 60]])

B = np.array([[50, 60],  # Row from index 2 in A
              [10, 20],  # Row from index 0 in A
              [30, 40]]) # Row from index 1 in A

L = find_row_permutation(C,A)
print(L)  # Output: [2, 0, 1]    
    
C = np.array([[50, 60],
              [30, 40],# Row from index 2 in A
              [10, 20]]) # Row from index 1 in A    
L2 = find_row_permutation(A,B)
print(L2)

print(MAP_to_MAP(L,L2))
'''
def remove_elements_by_index(values, indices):
    """
    Removes elements from 'values' based on the indices given in 'indices'.
    
    Parameters:
        values (list): A list of values.
        indices (list): A list of indices referring to elements to be removed.

    Returns:
        list: The modified 'values' list with specified indices removed.
    """
    # Convert indices to a set to avoid duplicate processing
    index_set = set(indices)
    
    # Create a new list excluding elements at specified indices
    filtered_values = [val for i, val in enumerate(values) if i not in index_set]
    
    return filtered_values


def add_elements_by_index(values, indices):
    """
    Inserts `0` at the specified indices in `values`, maintaining order.

    Parameters:
        values (list): The original list of values.
        indices (list): The list of indices where `0` should be inserted.

    Returns:
        list: A new list with `0` inserted at the specified indices.
    """
    result = []  # Store the final modified list
    value_index = 0  # Track the index in the original values list
    
    for i in range(len(values) + len(indices)):  # Iterate through new length
        if i in indices:  
            result.append(0)  # Insert `0` at specified index
        else:
            result.append(values[value_index])  # Insert original element
            value_index += 1  # Move to the next element in values

    return result

def Corruption_c(List,N):
    Corrupted_List ={}
    
    for i in range(N):

        Corrupted_List['PM'+str(i+1)] = False
        
    for j in List:
        Corrupted_List['PM'+str(j+1)] = True
        
    return Corrupted_List
            

def permutation_matrix(AA, BB):
    
    A = [(item*10000)/10000 for item in AA]
    B = [(item*10000)/10000 for item in BB]    
    """
    Computes the permutation matrix that maps list A to list B.

    Args:
        A (list): The original list.
        B (list): The target list (a permutation of A).

    Returns:
        numpy.ndarray: The permutation matrix P such that P @ A_sorted = B.
    """
    if sorted(A) != sorted(B):
        print(A,B)
        raise ValueError("Lists must be permutations of each other.")
    
    n = len(A)
    P = np.zeros((n, n))

    # Create index mapping from A to B
    index_map = {value: i for i, value in enumerate(B)}

    for i, value in enumerate(A):
        P[index_map[value], i] = 1  # Place a 1 at the corresponding position

    return P

'''
# Example usage:
A = [3, 1, 2, 4]
B = [1, 3, 4, 2]

P = permutation_matrix(A, B)
print("Permutation Matrix:\n", P)
'''










def Latency_extraction(data0,Positions,L):
    List = []
    n1,n2 = np.shape(data0)
    for i in range(L-1):
        List_ = []
        for j in range(int(n1/L)):
            List__ = []
            for k in range(int(n1/L)):
                List__.append(data0[Positions[i][j],Positions[i+1][k]])
            List_.append(List__)
        List.append(List_)
    return List
                
def Norm_List(List,term):
    S = np.sum(List)
    return [List[i]*(term/S)for i in range(len(List))]
def To_list(List):
    if type(List) == list:
        return List
    import numpy as np
    List_ = List.tolist()
    if len(List_)==1:
        output = List_[0]
    else:
        output = List_
    
    return output
def dist_List(List):
    Sum = np.sum(List)
    
    return [List[i]/Sum for i in range(len(List))]

#print(dist_List([14,17,9]))
    
'''
#Ex:
A = np.matrix([[2,4,7,5,6,7],[5,6,7,1,1,1],[6,7,8,5,3,2],[2,4,7,5,6,7],[5,6,7,1,1,1],[6,7,8,5,3,2]])
P = [[2,5],[0,4],[1,3]]
List = Latency_extraction(A, P, 3)

print(List)
print(Norm_List([1,2,3,4],4))
'''



def find_median_from_cdf(cdf):
    """
    Finds the median of a discrete distribution given its CDF.

    Args:
        cdf (list): A list representing the cumulative probabilities of a discrete distribution.

    Returns:
        int: The index of the median value in the distribution.
    """
    for i, value in enumerate(cdf):
        if value >= 0.5:
            return i  # The first index where CDF reaches or exceeds 0.5 is the median index.
    
    raise ValueError("Invalid CDF: It should reach at least 0.5 somewhere.")



def I_key_finder(x,y,z,matrix,data):
    
    List = [x,y,z]
    Index1 = np.sum(np.abs(matrix - List),axis = 1)
    index = Index1.tolist()
    Index2 = min(index)
    Index = index.index(Index2)

    
    return data[Index]
            
def Medd(List):
    N = len(List)

    List_ = []
    import statistics
    for i in range(N):

        List_.append( statistics.median(List[i]))
        
    return List_


#print(Medd([[1,2,7,9,10],[1,9,10]]))

def Loc_finder(I_key,data):
    for i in range(len(data)):
        if data[i]['i_key'] == I_key:
            return i
    
def Ent(List):
    L =[]
    for item in List:
       
        if item!=0:
            L.append(item)
    l = sum(L)
    for i in range(len(L)):
        L[i]=L[i]/l
    ent = 0
    for item in L:
        ent = ent - item*(np.log(item)/np.log(2))
    return ent

def Med(List):
    N = len(List)

    List_ = []
    import statistics
    for i in range(N):

        List_.append( statistics.median(List[i]))
        
    return List_

def SC_Latency(Matrix,Positions,L):
    N = len(Matrix)
    A = np.zeros((N,N))
    W = int(N/L)
    for i in range(N):
        n1 = Positions[i//W][i%W]
        for j in range(N):
            n2 = Positions[j//W][j%W]
            A[i,j] = Matrix[n1,n2]
  
    return A
    



from itertools import combinations
from math import radians, sin, cos, sqrt, atan2

from collections import Counter
import json
from geopy.geocoders import Nominatim
from collections import Counter
import time

def classify_countries(latlon_list, delay=1.0, limit=None):
    """
    Reverse geocodes lat/lon to countries, counts nodes per country,
    and returns a sorted list of (country, count, percentage).
    
    Args:
        latlon_list (list): List of (lat, lon) tuples
        delay (float): Time delay between queries (to avoid rate limits)
        limit (int): Max number of points to process (optional)

    Returns:
        List of tuples: (country_name, count, percentage)
    """
    geolocator = Nominatim(user_agent="node_locator")
    country_counts = Counter()

    total = len(latlon_list) if not limit else min(limit, len(latlon_list))
    
    for i, (lat, lon) in enumerate(latlon_list[:total]):
        try:
            location = geolocator.reverse((lat, lon), language='en', exactly_one=True, timeout=10)
            country = location.raw['address'].get('country', 'Unknown')
        except Exception:
            country = 'Unknown'

        country_counts[country] += 1
        time.sleep(delay)  # polite usage for Nominatim

    # Convert counts to percentages
    results = []
    for country, count in country_counts.items():
        percentage = 100 * count / total
        results.append((country, count, round(percentage, 2)))

    # Sort by number of nodes descending
    results.sort(key=lambda x: x[1], reverse=True)

    return results






class CirMixNet(object):
    
    def __init__(self,Targets,Iteration,Capacity,run,delay1,delay2,base,Initial = False):
        self.Iterations = Iteration
        #print( self.Iterations )
        self.ML_a = 0.2
        self.ML_It = 1
        self.CAP = Capacity
        self.delay1 = delay1
        self.delay2 = delay2
        self.Targets = Targets
        self.b = base
        self.run = run
        #self.Method = ['L_C']        
        self.Tau = [0.085,0.2,0.4,0.6,0.8,1]
        self.T = [2,12,25,38,50,80]  
        self.alpha = [0.1*i  for i in range(10)]
        self.alpha += [0.9+0.01*(i+1) for i in range(10)]
        self.EPS = [ 0,3,5,7,8]
        self.nn = 20
        self.CF = 0.3
        self.Initial = Initial
        self.RST_tau = 0.6
        self.RST_T = 12
        self.CDF = [i/10 for i in range(51)]
        #self.Data_Set_General = self.data_generator(Iteration)

        

    def Basic_check(self,n,L,C,title):
        from PLOTTER import Plotter
        
        Y = []
        
        for CC in C:
            #W = NN/L
            fcp0 = CC
            FCP = [1-(1-(fcp0)**(L))**(item) for item in n]
            Y.append(FCP)
            
        
            
        
        
        
        X = []
        for nn in n:
            if nn<1024:
                X.append(str(int(nn))+"KB")
            elif nn<1024**2:
                X.append(str(int(nn/1024))+"MB")
            else:
                X.append(str(int(nn/1024**2))+"GB")
        #X[0] = "2KB"
        #print(X)
        Descriptions = [r'$\beta =$ '+str(round(100*nnn)) +"%" for nnn in C]
        X_label = 'Data size'
        Y_label = r'$\mathbb{P}\left(\mathsf{B}\right)$' # Likiliehood of Compromising Anonymity
        name =  title
        Plot_class = Plotter(X, Y, Descriptions, X_label, Y_label, name)
     
        Plot_class.simple_plot(1.02,False,1)  
        
        
        
        
    def Percentage_latlon(self):
        
        

        import json
        
        with open('Nym_dataset_nodes_description.json','r') as j_file:
            
            data0 = json.load(j_file)
            
            
        List = [ (float(item['latitude']), float(item['longitude'])) for item in data0]
            
            
        
        counts = classify_regions(List)
        
        return counts
    
    def Con_latlon(self):
        
        

        import json
        
        with open('Nym_dataset_nodes_description.json','r') as j_file:
            
            data0 = json.load(j_file)
            
            
        List = [ (float(item['latitude']), float(item['longitude'])) for item in data0]
            
            
        
        counts = classify_countries(List)
        
        return counts

  
    
    
    def w_strategy(self,alpha,N,L,n):
        W = int(N/L)
        alpha_W = int(W*alpha)
        Set = []
        
        for i in range(L):
            Set.append(pick_m(0,W-1,alpha_W))
        #print(Set)   
        Path_List = []
        
        for j in range(n):
            List_ = []
            for k in range(L):
                List_.append(select_m(Set[k],1)[0])
            Path_List.append(List_)
      
        return Path_List
                
        
    def alpha_strategy(self,alpha0,N,L,n):
        W = int(N/L)
        P_N = W**(L)
        Path_0 = []
        initial_path = pick_m(0,P_N-1,1)[0]
        Path_0.append(initial_path)
        PATHS = []
        for alpha in alpha0:
            Path_ = Path_0.copy()
            for i in range(n-1):
                #print(P_N-(i+1))
                P = [(1-alpha)/(P_N-(i+1))]*(P_N)
                for j in Path_:
                    P[j] = alpha/(i+1)
                Path_.append(choose_m(P,P_N))
            PATHS.append([to_base(item,W,L) for item in Path_])
                

      
        return PATHS




    def fix_strategy(self,list_1,N,L,n):
        list_ = [item-1 for item in list_1]
        W = int(N/L)
        
        Pre_selected = {}
        
        for i in list_:
            Pre_selected[i] = pick_m(0,W-1,1)
            
        
        Path_ = []
        
        for i in range(n):
            LIST = []
            for k in range(L):
                if k in list_:
                    LIST.append(Pre_selected[k][0])
                else:
                    LIST.append(pick_m(0,W-1,1)[0])
            Path_.append(LIST)
                    

      
        return Path_





    def Corruption(self,N,L,beta):
        List = []
        W = int(N/L)
        CC = int(beta*W)
        
        for i in range(L):
            List.append(pick_m(0,W-1,CC))
            
        return cartesian_product(List)
    
    
    
    
    def ACC_measurement(self,N,L,beta,n,It):
        count = []
        
        for I in range(It):
            count_ = []
            c_path = self.Corruption(N,L,beta)
            #print(self.alpha)
            paths = self.alpha_strategy(self.alpha,N,L,n)
            for i in range(len(self.alpha)):

                
                #print(paths)
                common = common_sublists(c_path,paths[i])
                if common <1:
                    x=0
                else:
                    x=1
                #print(common)
                count_.append(x)
            count.append(count_)
        
        return np.mean(np.matrix(count),axis=0)


    def ACC_measurement2(self,N,L,beta,n,It):
        Alpha = [0.01*(i+1) for i in range(10)]
        Alpha += [0.1*(i+2) for i in range(9)]
        #Alpha[0] = 0.02
        count = []
        
        for I in range(It):
            count_ = []

            c_path = self.Corruption(N,L,beta)
            
            for alpha in Alpha:
                paths = self.w_strategy(alpha,N,L,n)

                
                #print(paths)
                common = common_sublists(c_path,paths)
                #print(common)
                if common <1:
                    x=0
                else:
                    x=1
                count_.append(x)
            count.append(count_)
        count0 = np.mean(np.matrix(count),axis=0)[0]   
        #print(count0)
        return count0
    
    
    

    def ACC_measurement3(self,N,L,beta,n,It):
        Alpha = all_subsets(L)
        #Alpha[0] = 0.02
        count = []
        
        for I in range(It):
            count_ = []

            c_path = self.Corruption(N,L,beta)
            
            for alpha in Alpha:
                paths = self.fix_strategy(list(alpha),N,L,n)

                
                #print(paths)
                common = common_sublists(c_path,paths)
                #print(common)
                if common <1:
                    x=0
                else:
                    x=1
                count_.append(x)
            count.append(count_)
        count0 = np.mean(np.matrix(count),axis=0)[0]   
        #print(count0)
        return count0   
    
    
    
    
    
    def ENT_Loss(self,N,L,It,sig,m):
        nn = [1000]
        #print('ok')
        Traffic_rate = biased_distribution(It,sig)
        self.alpha = [0.1*i+0.1  for i in range(10)]
        #self.alpha += [0.9+0.01*(i+1) for i in range(10)]        
        
        pp = []
        count = []
        for I in range(m):
            n = nn[I%len(nn)]
            #print(n)
            count_ = []
            paths = self.alpha_strategy(self.alpha,N,L,n)
            pp.append(paths)
            
        ppp = []
        
        for j in range(len(self.alpha)):
            pppp = []
            for i in range(m):
                pppp.append(pp[i][j])
            ppp.append(pppp)
                
        for item in ppp:
            count.append(self.path_ent(item, Traffic_rate, int(N/L), L))
        
        return count   
    
    
    
    def ENT_Loss2(self,N,L,It,sig,m):
        Alpha = []
        Alpha += [0.1*(i)+0.1 for i in range(10)]        
        nn = [1000]
        #print('ok')
        Traffic_rate = biased_distribution(It,sig)
        #print(Traffic_rate)
        #print('yes')
        pp = []
        count = []
        for I in range(m):
            n = nn[I%len(nn)]
            #print(n)
            count_ = []
            paths = []
            for alpha in Alpha:
                paths.append( self.w_strategy(alpha,N,L,n))
            pp.append(paths)
            
        ppp = []
        
        for j in range(len(Alpha)):
            pppp = []
            for i in range(It):
                pppp.append(pp[i][j])
            ppp.append(pppp)
                
        for item in ppp:
            count.append(self.path_ent(item, Traffic_rate, int(N/L), L))
        
        return count   
    
    
    
    def ENT_Loss3(self,N,L,It,sig,m):
        Alpha = all_subsets(L)
        nn = [1000]
        #print('ok')
        Traffic_rate = biased_distribution(It,sig)
        pp = []
        count = []
        for I in range(m):
            n = nn[I%len(nn)]
            #print(n)
            count_ = []
            paths = []
            for alpha in Alpha:
                paths.append( self.fix_strategy(list(alpha),N,L,n))
            pp.append(paths)
            
        ppp = []
        
        for j in range(len(Alpha)):
            pppp = []
            for i in range(It):
                pppp.append(pp[i][j])
            ppp.append(pppp)
                
        for item in ppp:
            count.append(self.path_ent(item, Traffic_rate, int(N/L), L))
        
        return count      
    
    
    def path_ent(self,Paths_List,Traffic_rate,W,L):
        data0 = {}
        for i in range(W**L):
            data0[i] = 0
        
        for j, path in enumerate(Paths_List):
            
            for item in path:
                index = base_to(item,W)
                data0[index] += Traffic_rate[j]
                
        List = []
        
        for i in range(W**L):
            List.append(data0[i])
            
        List0 = To_list(np.array(List)/np.sum(List))
        
        
        ent = entropyy(List0)
                
        return ent
    
    def Corruption_to_dic(self,N,L,beta):
        List = []
        W = int(N/L)
        CC = int(beta*W)
        data0 = {}
        for k in range(N):
            data0['PM'+str(k+1)] = False
        
        for i in range(L):
            List0 = pick_m(0,W-1,CC)
            for item in List0:
                data0['PM'+str(item+1+W*i)] = True
        return data0
            
            

    
    def Simulators(self,Fun_name,nn,W,L,Corrupted_Mix,num_packets,param):
        from Sim import Simulation
        

        Sim_ = Simulation(self.Targets,self.run,self.delay1,self.delay2,W*L,L )
        
        Entropy_Sim = Sim_.Simulator(Corrupted_Mix,Fun_name,nn,num_packets,param)
        
        
        return Entropy_Sim            
    

    
    def ACC_n(self,N,L,beta,alpha,It, state=0):
        if state==0:
            nn = [1,10,100,1000,10000]
            count = []
            for I in range(It):
                count_ = []
                c_path = self.Corruption(N,L,beta)
                for n in nn:
                    paths = self.alpha_strategy([alpha],N,L,n)[0]
    
    
                    common = common_sublists(c_path,paths)
                    if common <1:
                        x=0
                    else:
                        x=1
                    count_.append(x)
                count.append(count_)
        elif state ==1:
            return [0.00823808000881776,
         0.010140664229024487,
         0.03129487844811618,
         0.2195004295266595,
         0.9100133191820654]
        elif state ==2:
            return [0.00347590828498745,
         0.004282806845088749,
         0.013315520120472657,
         0.0992564039858248,
         0.6379005480450417]
        
        elif state ==3:
            return [0.0010299700011099366,
         0.0012696976458517195,
         0.003962672964175518,
         0.030496280331981818,
         0.2599049664498072]       
        elif state ==4:
            return [0.00012874953138886092,
         0.00015874527646753123,
         0.0004961354273054219,
         0.0038637833591782522,
         0.036922639677527846]     
      
        return np.mean(np.matrix(count),axis=0)


    def ACC_n2(self,N,L,beta,alpha,It,state= 0):
        nn = [1,10,100,1000,10000]

        count = []
        
        for I in range(It):
            count_ = []

            c_path = self.Corruption(N,L,beta)
            
            for n in nn:
                paths = self.w_strategy(alpha,N,L,n)

                
                #print(paths)
                common = common_sublists(c_path,paths)
                #print(common)
                if common <1:
                    x=0
                else:
                    x=1
                count_.append(x)
            count.append(count_)
        count0 = np.mean(np.matrix(count),axis=0)[0]   
        if state ==1:
            return [0.0075,0.071,0.2535,0.309,0.319]
        elif state ==2:
            return [0.0035,0.028,0.1455,0.1805,0.181]
        
        elif state ==3:
            return [0.0005,0.0105,0.0495,0.0815,0.082]       
        elif state ==4:
            return [0,0.001,0.0085,0.018,0.0095]      
        return count0
    
    
    

    def ACC_n3(self,N,L,beta,alpha,It):
        nn = [1,10,100,1000,10000]
        #Alpha[0] = 0.02
        count = []
        
        for I in range(It):
            count_ = []

            c_path = self.Corruption(N,L,beta)
            
            for n in nn:
                paths = self.fix_strategy(list(alpha),N,L,n)

                
                #print(paths)
                common = common_sublists(c_path,paths)
                #print(common)
                if common <1:
                    x=0
                else:
                    x=1
                count_.append(x)
            count.append(count_)
        count0 = np.mean(np.matrix(count),axis=0)[0]   
        #print(count0)
        return count0      
    
    
    






    def ACC_L(self,beta,It,state=0):
        if state==0:
            N = 300
            LL = [2,3,4]
            alpha = 0.95
            n = 500
            count = []
            for I in range(It):
                count_ = []
                
                for L in LL:
                    c_path = self.Corruption(N,L,beta)
                    paths = self.alpha_strategy([alpha],N,L,n)[0]
    
                    
                    #print(paths)
                    common = common_sublists(c_path,paths)
                    if common <1:
                        x=0
                    else:
                        x=1
                    #print(common)
                    count_.append(x)
                count.append(count_)
            
            return np.mean(np.matrix(count),axis=0)
        elif state ==1:
            return [0.7107763096920346,
         0.2195004295266595,
         0.048346643998458894,
         0.009861550097691185,
         0.0019801208583396512]
        elif state ==2:
            return [0.5021365507352609,
         0.0992564039858248,
         0.015556533319465382,
         0.0023490265596110005,
         0.0003527055641778798]
        
        elif state ==3:
            return [0.2664419241007453,
         0.030496280331981818,
         0.0030922187148058056,
         0.0003096521427838894,
         3.0969521423562796e-05]     
        elif state ==4:
            return [0.07452137154024085,
         0.0038637833591782522,
         0.00019354380544267524,
         9.678078259467426e-06,
         4.839061717998305e-07]  



    def ACC_L2(self,beta,It,state=0):
        N = 300
        LL = [2,3,4,5,6]
        alpha = 0.02
        n = 1000
        count = []
        for I in range(It):
            count_ = []

            
            
            for L in LL:
                c_path = self.Corruption(N,L,beta)
                paths = self.w_strategy(alpha,N,L,n)

                
                #print(paths)
                common = common_sublists(c_path,paths)
                #print(common)
                if common <1:
                    x=0
                else:
                    x=1
                count_.append(x)
            count.append(count_)
        count0 = np.mean(np.matrix(count),axis=0)[0]   
        
        if state ==1:
            return [0.259, 0.058, 0.001, 0.]
        elif state ==2:
            return [0.148, 0.019, 0. , 0.  ]
        
        elif state ==3:
            return [0.069, 0.005, 0.,  0.  ]      
        elif state ==4:
            return [0.023, 0.004, 0.,  0.  ] 
        return count0
    
    
    

    def ACC_L3(self,beta,It):
        N = 300
        LL = [2,3,4,5,6]
        alpha = [1]
        n = 1000
        count = []
        for I in range(It):
            count_ = []

            
            
            for L in LL:
                c_path = self.Corruption(N,L,beta)
                paths = self.fix_strategy(list(alpha),N,L,n)

                
                #print(paths)
                common = common_sublists(c_path,paths)
                #print(common)
                if common <1:
                    x=0
                else:
                    x=1
                count_.append(x)
            count.append(count_)
        count0 = np.mean(np.matrix(count),axis=0)[0]   
        #print(count0)
        return count0      
    
    

    def KHF(self,L,m):
        beta = 0.15
        gamma = 0.003
        sets = all_subsets(L)
        x = []
        Y_P = []
        Y_H = []
        Y_G = []
        for item in sets:
            x.append(str(set(list(item))))
            h_f = len(item)
            term1 = beta**(h_f)
            term2 =1- (1-beta**(L-h_f))**(m)
            term3 = (1-gamma)**(h_f)
            term4 = (1-gamma)**((L-h_f)*m) 
            Y_P.append(term1*term2)
            Y_H.append((L-h_f)*(np.log(300)/np.log(2)))
            Y_G.append(term3*term4)
        x[0] = "{}"   
        return x,Y_P,Y_H, Y_G
            





    def KW(self,L,m):
        beta = 0.15
        gamma = 0.003
        K = [1,2,3,4,5,6,8,16,32,64,128,300]
        K[-1] = 300
        x = []
        Y_P = []
        Y_H = []
        Y_G = []
        for k in K:
            x.append(str(k))
            #h_f = len(item)
            #term1 = beta**(h_f)
            term2 =1- (1-beta**(L))**(min(m,k**L))
            Y_P.append(term2)
            Y_H.append((L)*(np.log(k)/np.log(2)))
            Y_G.append((1-gamma)**(L*(min(m,k**L))))
        #print(x)   
        return x,Y_P,Y_H,Y_G




    def aSS(self,L,m):
        beta = 0.15
        gamma = 0.003
        alpha0 = (m-2)/(300**L+m-1)
        Alpha = [i/10 for i in range(11)]
        Alpha[0] = int(10**4*alpha0)/10000
        x = []
        Y_P = []
        Y_H = []
        Y_G = []
        for alpha in Alpha:
            x.append(str(alpha))
            term0 = (1-beta**L)
            term1 = (300**L)*beta**L
            xx = 1
            for j in range(m-1):
                i = j+2
                term2 = term1/(300**L-1-(i-2)*(1-alpha)) 
                xx *= alpha+(1-alpha)*(1-term2)
            Y_P.append(1-term0*xx)
            item0 = alpha*((np.log(1+(i-2)*(1-alpha)))/(np.log(2)))
            item1 = (1-alpha)*((np.log(300**L-1-(i-2)*(1-alpha)))/(np.log(2)))            
            Y_H.append(item0+item1)
            
            term0 = (1-gamma)**L
            xx = 1
            for j in range(m-1):
                i = j+2
                term11 = (300**L)*(1-gamma)**L-1-(i-2)*(1-alpha)
                term1 = term11/((300**L)-1-(i-2)*(1-alpha)) 
                xx *= alpha+(1-alpha)*(term1)
            Y_G.append(term0*xx)            
            
            
            #Y_G.append((1-gamma)**(L*(min(m,k**L))))
            
        return x,Y_P,Y_H,Y_G




    def aSS_L(self,LL):
        m = 1000
        beta = 0.05
        gamma = 0.003
        #alpha0 = (m-2)/(300**L+m-1)
        #Alpha = [i/10 for i in range(11)]
        #Alpha[0] = int(10**4*alpha0)/10000
        alpha = 0.97
        x = []
        Y_P = []
        Y_H = []
        Y_G = []
        for L in LL:
            x.append(L)
            term0 = (1-beta**L)
            term1 = (300**L)*beta**L
            xx = 1
            for j in range(m-1):
                i = j+2
                term2 = term1/(300**L-1-(i-2)*(1-alpha)) 
                xx *= alpha+(1-alpha)*(1-term2)
            Y_P.append(1-term0*xx)
            item0 = alpha*((np.log(1+(i-2)*(1-alpha)))/(np.log(2)))
            item1 = (1-alpha)*((np.log(300**L-1-(i-2)*(1-alpha)))/(np.log(2)))            
            Y_H.append(item0+item1)
            
            term0 = (1-gamma)**L
            xx = 1
            for j in range(m-1):
                i = j+2
                term11 = (300**L)*(1-gamma)**L-1-(i-2)*(1-alpha)
                term1 = term11/((300**L)-1-(i-2)*(1-alpha)) 
                xx *= alpha+(1-alpha)*(term1)
            Y_G.append(term0*xx)            
            
            
            #Y_G.append((1-gamma)**(L*(min(m,k**L))))
            
        return x,Y_P








    def aSS_n(self,mm):
        L = 3
        beta = 0.2
        gamma = 0.003
        #alpha0 = (m-2)/(300**L+m-1)
        #Alpha = [i/10 for i in range(11)]
        #Alpha[0] = int(10**4*alpha0)/10000
        alpha = 0.97
        x = []
        Y_P = []
        Y_H = []
        Y_G = []
        for m in mm:
            x.append(m)
            term0 = (1-beta**L)
            term1 = (300**L)*beta**L
            xx = 1
            for j in range(m-1):
                i = j+2
                term2 = term1/(300**L-1-(i-2)*(1-alpha)) 
                xx *= alpha+(1-alpha)*(1-term2)
            Y_P.append(1-term0*xx)
            item0 = alpha*((np.log(1+(i-2)*(1-alpha)))/(np.log(2)))
            item1 = (1-alpha)*((np.log(300**L-1-(i-2)*(1-alpha)))/(np.log(2)))            
            Y_H.append(item0+item1)
            
            term0 = (1-gamma)**L
            xx = 1
            for j in range(m-1):
                i = j+2
                term11 = (300**L)*(1-gamma)**L-1-(i-2)*(1-alpha)
                term1 = term11/((300**L)-1-(i-2)*(1-alpha)) 
                xx *= alpha+(1-alpha)*(term1)
            Y_G.append(term0*xx)            
            
            
            #Y_G.append((1-gamma)**(L*(min(m,k**L))))
            
        return x,Y_P



    def KW_L(self,LL):
        beta = 0.15
        gamma = 0.003
        k = 8
        m = 1000
        x = []
        Y_P = []
        Y_H = []
        Y_G = []
        for L in LL:
            x.append(str(k))
            #h_f = len(item)
            #term1 = beta**(h_f)
            term2 =1- (1-beta**(L))**(min(m,k**L))
            Y_P.append(term2)
            Y_H.append((L)*(np.log(k)/np.log(2)))
            Y_G.append((1-gamma)**(L*(min(m,k**L))))
        #print(x)   
        return x,Y_P
    
    
    
    def ENT_Loss_(self,N,L,It,sig,m,nnn):
        nn = [nnn]

        #print('ok')
        Traffic_rate = biased_distribution(It,sig)
        self.alpha = [0.95]
        #self.alpha += [0.9+0.01*(i+1) for i in range(10)]        
        
        pp = []
        count = []
        for I in range(m):
            n = nn[I%len(nn)]
            #print(n)
            count_ = []
            paths = self.alpha_strategy(self.alpha,N,L,n)
            pp.append(paths)
            
        ppp = []
        
        for j in range(len(self.alpha)):
            pppp = []
            for i in range(m):
                pppp.append(pp[i][j])
            ppp.append(pppp)
                
        for item in ppp:
            count.append(self.path_ent(item, Traffic_rate, int(N/L), L))
        
        return count       
    
    
    
    
    
    def ENT_Loss_n(self,N,L,It,sig,m):
        

        x = []
        nn = [1,10,100,1000]
        
        for n in nn:
            
            x += self.ENT_Loss_(N, L,It,beta,m,n)


        
        return x  
    
    
    
    def ENT_Loss2_n(self,N,L,It,sig,m):
        Alpha = [0.2]
        nn = [1,10,100,1000,10000]
        #print('ok')
        Traffic_rate = biased_distribution(It,sig)
        #print(Traffic_rate)
        #print('yes')
        pp = []
        count = []
        for I in range(m):
            #n = nn[I%len(nn)]
            #print(n)
            count_ = []
            paths = []
            for n in nn:
                alpha = Alpha[0]
                paths.append( self.w_strategy(alpha,N,L,n))
            pp.append(paths)
            
        ppp = []
        
        for j in range(len(nn)):
            pppp = []
            for i in range(It):
                pppp.append(pp[i][j])
            ppp.append(pppp)
                
        for item in ppp:
            count.append(self.path_ent(item, Traffic_rate, int(N/L), L))
        
        return count   
    
    
    
    def ENT_Loss3_n(self,N,L,It,sig,m):
        Alpha = [1]
        nn = [1,10,100,1000,10000]
        #print('ok')
        Traffic_rate = biased_distribution(It,sig)
        pp = []
        count = []
        for I in range(m):
            n = nn[I%len(nn)]
            #print(n)
            count_ = []
            paths = []
            for n in nn:
                alpha = Alpha
                paths.append( self.fix_strategy(list(alpha),N,L,n))
            pp.append(paths)
            
        ppp = []
        
        for j in range(len(nn)):
            pppp = []
            for i in range(It):
                pppp.append(pp[i][j])
            ppp.append(pppp)
                
        for item in ppp:
            count.append(self.path_ent(item, Traffic_rate, int(N/L), L))
        
        return count         
    
    def ENT_Loss_L(self,N,L,It,sig,m):
        

        x = []
        n = 1000
        LL = [2,3,4,5,6]
        
        for L in LL:
            if L==2:
                n = 100
            
            x += self.ENT_Loss_(N, L,It,beta,m,n)


        
        return x  
    
    
    
    def ENT_Loss2_L(self,N,L,It,sig,m):
        Alpha = [0.4]
        n = 1000
        LL = [2,3,4,5,6]
        #print('ok')
        Traffic_rate = biased_distribution(It,sig)
        #print(Traffic_rate)
        #print('yes')
        pp = []
        count = []
        for I in range(m):
            #n = nn[I%len(nn)]
            #print(n)
            count_ = []
            paths = []
            for L in LL:
                alpha = Alpha[0]
                paths.append( self.w_strategy(alpha,N,L,n))
            pp.append(paths)
            
        ppp = []
        
        for j in range(len(LL)):
            pppp = []
            for i in range(It):
                pppp.append(pp[i][j])
            ppp.append(pppp)
                
        for item in ppp:
            count.append(self.path_ent(item, Traffic_rate, int(N/L), L))
        
        return count   
    
    
    
    def ENT_Loss3_L(self,N,L,It,sig,m):
        Alpha = [1]
        n = 1000
        LL = [2,3,4,5,6]
        #print('ok')
        Traffic_rate = biased_distribution(It,sig)
        pp = []
        count = []
        for I in range(m):
            #n = nn[I%len(LL)]
            #print(n)
            count_ = []
            paths = []
            for L in LL:
                alpha = Alpha
                paths.append( self.fix_strategy(list(alpha),N,L,n))
            pp.append(paths)
            
        ppp = []
        
        for j in range(len(LL)):
            pppp = []
            for i in range(It):
                pppp.append(pp[i][j])
            ppp.append(pppp)
                
        for item in ppp:
            count.append(self.path_ent(item, Traffic_rate, int(N/L), L))
        
        return count             
    
    
    
    
  