# CS 133 Lab 1 Report: High-Performance GEMM
**Name:** Alexander Neary  
**UID:** 805935345  
**Date:** April 2026  

---

## 1. Optimization Description

### 1.1 GemmParallel
The sequential implementation from `gemm.cpp` used the standard $i, j, k$ loop order. Since we know that a CPU will access memory in chunk sizes, I drew out the process order for matrix multiplication. Since matrices are stored as 2d arrays, each element of the array is a row (with each element in that row being the column). 
Due to k being the innermost loop, matrix B is iterated by row first, then column. Matrix A on the other hand is iterated by column first and then row, meaning it can grab a row from cache and run operations on that singular row for multiple loops of k. We want to do the same thing for B. 

**Optimization Performed:**
- **Loop Order:** Changed from $i, j, k$ to $i, k, j$.
- **Parallelization:** Added `#pragma omp parallel for` to the outermost loop $i$. 
- **Reasoning:** In the $i, k, j$ order, both matrix A and matrix B are iterated by column first, so the cache locality is maximized during matrix multiplication. Parallelizing the $i$ loop allowed for each row of the output matrix C to be computed in parallel.
- **Performance Increase:** (On my local machine, 2048 x 2048 x 2048)
        Sequential GEMM: 0.383 GFlops 
        vs
        Parallel GEMM: ~170 GFlops

### 1.2 GemmParallelBlocked
The problem with the non-blocked parallel version from `omp.cpp` is that for each row of matrix C that is computed, the processor has to read the entirety of matrix B. Once the outer loop $i$ is iterated, all of matrix B will need to be read again. This means that each thread is reading the entirety of matrix B within its loops.
Instead of using 1 row of A and all of B per thread, we can use a small block of A and a small block of B to compute the partial matrix product for that block in C. This will better utilize the cache (memory locality) by avoiding fetching the entire matrix B per thread.

- **Block Size chosen:** $B = [64]$
- **Reasoning:** Intel Xeon Platinum 8175M 24-core CPU (a custom model for Amazon AWS), each thread has its own L1d 32KiB Cache. 32KiB = 32768 bytes --> since a block of A and a block of B is needed, calculate half of the space 32768 bytes / 2 = 16384 bytes --> 16384 bytes of space fits how many words (a word is 32 bits or 4 bytes): 16384 bytes / 4 bytes = 4096 words --> 4096 words is a 64 x 64 block of words.
- **Implementation:** Block size (and number of blocks) is stored for calculations before any loops or parallelization. The first two loops are for iterating through result matrix C, and it's parallelized using `#pragma omp parallel for collapse(2)` so each thread can work on its own block of C. The following loop is used to iterate over A and B (by block), giving the block offset. The final 3 inner loops are the $i, k, j$ ordered matrix multiplication from `omp.cpp`, adjusted for the tiling format.
- **Performance Increase:** (On my local machine, 2048 x 2048 x 2048)
        Parallel GEMM: ~170 GFlops 
        vs
        Parallel-Blocked GEMM: ~210 GFlops

---

## 2. Performance Evaluation

The following results were obtained on the **m5.2xlarge** AWS instance with a problem size of $4096^3$.

| Version | GFlops | Time (s) | Execution Mode |
| :--- | :---: | :---: | :--- |
| Sequential (Baseline) | [Value] | [Value] | Single Thread |
| GemmParallel | [Value] | [Value] | OpenMP Multi-thread |
| GemmParallelBlocked | [Value] | [Value] | OpenMP + Tiling |

**Performance Range achieved:** 

---

## 3. Analysis and Observations
- **Scaling:** Adding parallelism to the sequential version of GEMM allowed for multiple cores to calculate the result at the same time, combining the overall result to receive matrix C. This change provided a major performance improvement. 
- **Cache Impact:** Through implementing blocking, the GFlops even on just my local machine jumped up by over 100. Splitting up C by blocks, having each thread calculate the result for that block of C, and grabbing blocks of A and B from memory instead of full rows/columns at a time all increased the performance significantly. These changes maximized cache locality between cores, providing a worthwile speedup.
- **Challenges:** Trying to wrap my head around the full process for blocked parallel matrix multiplication was difficult. I ended up making a couple different drawings to visually grasp the order of operations/loops, as well as where each variable ends up after a calculation. Once I broke the problem down, I was able to work through it sequentially to best optimize performance. 