"""
Sorting algorithm (comparison benchmark)
"""
import math

# Increase call stack size...
# import sys
# sys.setrecursionlimit(11000)

"""
Insertion sort
"""

def insertion_sort(A):
    comparisons = 0
    for i in range(1, len(A)):
        key = A[i]
        j = i - 1
        while j >= 0:
            comparisons += 1
            if A[j] > key:
                A[j + 1] = A[j]
                j -= 1
            else:
                break
        A[j + 1] = key
    return comparisons

"""
Selection sort
"""

def selection_sort(A):
    comparisons = 0
    n = len(A)
    for i in range(n - 1):
        min_index = i
        for j in range(i + 1, n):
            comparisons += 1
            if A[j] < A[min_index]:
                min_index = j
        A[i], A[min_index] = A[min_index], A[i]
    return comparisons

"""
Merge sort
"""

def merge_sort(A):
    if len(A) <= 1:
        return 0
    mid = len(A) // 2
    left = A[:mid]
    right = A[mid:]

    comparecount = merge_sort(left) + merge_sort(right)

    i = 0
    j = 0
    k = 0
    
    while i < len(left) and j < len(right):
        comparecount += 1
        comparecount += 1
        comparecount += 1
        if left[i] <= right[j]:
            A[k] = left[i]
            i += 1
        else:
            A[k] = right[j]
            j += 1
        k += 1

    while i < len(left):
        comparecount += 1
        A[k] = left[i]
        i += 1
        k += 1

    while j < len(right):
        comparecount += 1
        A[k]=right[j]
        j += 1
        k += 1

    return comparecount
    
"""
Quick sort
"""

def quicksort(A):
    comparisons = [0]
    quickSorter(A, 0, len(A) - 1, comparisons)
    return comparisons[0]

def partition(A, low, high, comparisons):
    pivot = A[high]
    i = low - 1
    for j in range(low, high):
        comparisons[0] += 1
        if A[j] <= pivot:
            i += 1
            A[i], A[j] = A[j], A[i]
    A[i + 1], A[high] = A[high], A[i + 1]
    return i + 1

def quickSorter(A, low, high, comparisons):
    if low < high:
        pi = partition(A, low, high, comparisons)
        quickSorter(A, low, pi - 1, comparisons)
        quickSorter(A, pi + 1, high, comparisons)

"""
Timsort
"""

MINIMUM = 32

def find_minrun(n): 
    r = 0
    while n >= MINIMUM: 
        r |= n & 1
        n >>= 1
    return n + r 

def tim_insertion_sort(array, left, right): 
    comparisons = 0
    for i in range(left + 1, right + 1):
        key = array[i]
        j = i - 1
        comparisons += 1
        while j >= left and key < array[j]:
            array[j + 1] = array[j]
            j -= 1
            comparisons += 1
        array[j + 1] = key
    return comparisons
              
def tim_merge(array, l, m, r): 
    comparisons = 0
    array_length1 = m - l + 1
    array_length2 = r - m 
    left = []
    right = []
    for i in range(array_length1): 
        left.append(array[l + i]) 
    for i in range(array_length2): 
        right.append(array[m + 1 + i]) 
  
    i = 0
    j = 0
    k = l
   
    while j < array_length2 and i < array_length1: 
        if left[i] <= right[j]: 
            array[k] = left[i] 
            i += 1
        else: 
            array[k] = right[j] 
            j += 1
        k += 1
        comparisons += 1
  
    while i < array_length1: 
        array[k] = left[i] 
        k += 1
        i += 1
        comparisons += 1
  
    while j < array_length2: 
        array[k] = right[j] 
        k += 1
        j += 1
        comparisons += 1

    return comparisons
  
def timsort(array):
    comparisons = 0
    n = len(array) 
    minrun = find_minrun(n) 
  
    for start in range(0, n, minrun): 
        end = min(start + minrun - 1, n - 1) 
        comparisons += tim_insertion_sort(array, start, end) 
   
    size = minrun 
    while size < n: 
        for left in range(0, n, 2 * size): 
            mid = min(n - 1, left + size - 1) 
            right = min((left + 2 * size - 1), (n - 1)) 
            comparisons += tim_merge(array, left, mid, right) 
        size = 2 * size

    return comparisons

"""
Introsort
"""

def introsort(arr):

    def partition(arr, low, high):
        pivot = arr[high]
        i = low - 1
        for j in range(low, high):
            comparisons[0] += 1
            if arr[j] <= pivot:
                i += 1
                arr[i], arr[j] = arr[j], arr[i]
        arr[i + 1], arr[high] = arr[high], arr[i + 1]
        return i + 1

    def insertion_sort(arr, low, high):
        for i in range(low + 1, high + 1):
            key = arr[i]
            j = i - 1
            comparisons[0] += 1
            while j >= low and arr[j] > key:
                comparisons[0] += 1
                arr[j + 1] = arr[j]
                j -= 1
            arr[j + 1] = key

    def heap_sort(arr):
        def heapify(arr, n, i):
            largest = i
            l = 2 * i + 1
            r = 2 * i + 2

            if l < n and arr[i] < arr[l]:
                largest = l

            if r < n and arr[largest] < arr[r]:
                largest = r

            if largest != i:
                arr[i], arr[largest] = arr[largest], arr[i]
                heapify(arr, n, largest)

        n = len(arr)

        for i in range(n // 2 - 1, -1, -1):
            heapify(arr, n, i)

        for i in range(n - 1, 0, -1):
            arr[i], arr[0] = arr[0], arr[i]
            heapify(arr, i, 0)

    def introsort_util(arr, low, high, depth_limit):
        size = high - low + 1

        if size < 16:
            insertion_sort(arr, low, high)
            return

        if depth_limit == 0:
            heap_sort(arr)
            return

        pivot = partition(arr, low, high)

        introsort_util(arr, low, pivot - 1, depth_limit - 1)
        introsort_util(arr, pivot + 1, high, depth_limit - 1)

    comparisons = [0]
    introsort_util(arr, 0, len(arr) - 1, 2 * math.log(len(arr)))
    return comparisons[0]
    