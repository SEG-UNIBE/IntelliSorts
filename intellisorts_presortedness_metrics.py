"""
Presortedness metrics (comparison benchmark)
"""

"""
Number of Runs
"""

def runs(arr):
    """
    The number of runs, is the number of increasing sequences in an array minus one.
    """
    count = 0

    for key in range(1,len(arr)):
        if arr[key] < arr[key-1]:
            count += 1

    return count

def runs_comp(arr):
    """
    Number of comparisons needed for Runs computation
    """
    count = 0
    comparisons = 0
    for key in range(1,len(arr)):
        comparisons += 1
        if arr[key] < arr[key-1]:
            count += 1

    return comparisons

"""
Number of Deletions
"""

def deletions(arr):
    """
    Minimum number of elements that need to be removed from array to obtain a sorted sequence.
    """
    def ceil_index(sub, val):
        l, r = 0, len(sub)-1
        while l <= r:
            mid = (l + r) // 2
            if sub[mid] >= val:
                r = mid - 1
            else:
                l = mid + 1
        return l
 
    sub = [arr[0]]
    for i in range(1, len(arr)):
        if arr[i] >= sub[-1]:
            sub.append(arr[i])
        else:
            sub[ceil_index(sub, arr[i])] = arr[i]
 
    return len(arr) - len(sub)

def deletions_comp(arr):
    """
    Number of comparisons needed for Deletions computation.
    """
    global comparisons
    comparisons = 0
    def ceil_index(sub, val):
        global comparisons
        l, r = 0, len(sub)-1
        while l <= r:
            mid = (l + r) // 2
            comparisons += 1
            if sub[mid] >= val:
                r = mid - 1
            else:
                l = mid + 1
        return l
 
    sub = [arr[0]]
    for i in range(1, len(arr)):
        comparisons += 1
        if arr[i] >= sub[-1]:
            sub.append(arr[i])
        else:
            sub[ceil_index(sub, arr[i])] = arr[i]
 
    return comparisons

"""
Number of Inversions
"""

def inversions(arr):
    """
    The number of inversion in an array, is the number of pairs j < key such that arr[j] > key.
    """
    count = 0

    for key in range(len(arr)):
        for j in range(key):
            if arr[key] < arr[j]:
                count += 1

    return count

def inversions_comp(arr):
    """
    Number of comparisons needed for inversions computation.
    """
    count = 0
    comparisons = 0
    for key in range(len(arr)):
        for j in range(key):
            comparisons += 1
            if arr[key] < arr[j]:
                count += 1

    return comparisons

"""
Max Distance by inversion
"""

def max_dist_inversion(arr):
    """
    Computes the longest distance between two elements that have to be inverted.
    """
    c_max_dist = 0

    for key in range(len(arr)):
        for j in range(key):
            if arr[key] < arr[j]:
                c_max_dist = max(key-j,c_max_dist)

    return c_max_dist

def max_dist_inversion_comp(arr):
    """
    Number of comparisons needed for Max Distance computation
    """
    c_max_dist = 0
    count = 0

    for key in range(len(arr)):
        for j in range(key):
            count += 1
            if arr[key] < arr[j]:
                c_max_dist = max(key-j,c_max_dist)

    return count

"""
Inv(arr) and Dis(arr) combination
"""

def inv_dis(arr):
    """
    Note that the amount of comparisons needed to perform inversion(arr) and max_dist_inversion(arr) are exactly the same. 
    We can combine the two algorithms in to one without having to do any more comparison than just doing one of the two.
    """
    c_max_dist = 0
    inv = 0
    
    for key in range(len(arr)):
        for j in range(key):
            if arr[key] < arr[j]:
                c_max_dist = max(key-j,c_max_dist)
                inv += 1

    return inv, c_max_dist

def inv_dis_comp(arr):
    """
    Number of comparisons needed for Inv/Dis computation 
    """
    c_max_dist = 0
    inv = 0
    comparisons = 0
    
    for key in range(len(arr)):
        for j in range(key):
            comparisons += 1
            if arr[key] < arr[j]:
                c_max_dist = max(key-j,c_max_dist)
                inv += 1

    return comparisons
    