from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed

# https://docs.python.org/3/library/concurrent.futures.html 
# the sample task is multiplication of two arrays and we will implement it using 
# 1. ThreadPoolExecutor
# 2. ProcessPoolExecutor
# while performing these computations, we also measure the overall time taken by the two methods

import time
from tqdm import tqdm
import numpy as np

def multiply(arr1, arr2):
    return np.dot(arr1, arr2)


def main(size, n_workers):
    arr1 = np.random.rand(size)
    arr2 = np.random.rand(size)
    
    size_per_worker = size//n_workers # ignoring tail effect
    # print(f"Size per worker: {size_per_worker}")

    thread_out = np.zeros(size_per_worker*n_workers)
    process_out = np.zeros(size_per_worker*n_workers)
    # ThreadPoolExecutor
    start = time.time()
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        thread_futures = []
        for i in range(8):
            sub_arr1 = arr1[i*size_per_worker:(i+1)*size_per_worker]
            sub_arr2 = arr2[i*size_per_worker:(i+1)*size_per_worker]
            future = executor.submit(multiply, sub_arr1, sub_arr2)
            thread_futures.append(future)
        for thread_future in as_completed(thread_futures):
            i = thread_futures.index(thread_future)
            try:
                result = thread_future.result()
                thread_out[i*size_per_worker:(i+1)*size_per_worker] = result
            except Exception as e:
                print(f"{i} Thread -- Exception {e}")

    thread_time = time.time()-start
    # print(f"ThreadPoolExecutor: {thread_time}")
    
    # ProcessPoolExecutor
    start = time.time()
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        process_futures = []
        for i in range(8):
            sub_arr1 = arr1[i*size_per_worker:(i+1)*size_per_worker]
            sub_arr2 = arr2[i*size_per_worker:(i+1)*size_per_worker]
            future = executor.submit(multiply, sub_arr1, sub_arr2)
            process_futures.append(future)
        for future in as_completed(process_futures):
            i = process_futures.index(future)
            try:
                result = future.result()
                process_out[i*size_per_worker:(i+1)*size_per_worker] = result
            except Exception as e:
                print(f"{i} Process -- Exception {e}")
    
    process_time = time.time()-start
    # print(f"ProcessPoolExecutor: {process_time}")

    return thread_time, process_time

if __name__ == "__main__":
    t_times = []
    p_times = []
    for i in tqdm(range(10)):
        t_time, p_time = main(10000000, 8)
        t_times.append(t_time)
        p_times.append(p_time)

    print(f"Average Speedup with ThreadPoolExecutor vs ProcessPoolExecutor: {np.mean(p_times)/np.mean(t_times)}")