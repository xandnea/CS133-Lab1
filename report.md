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

### 1.2 GemmParallelBlocked
The problem with the non-blocked parallel version from `omp.cpp` is that for each row of matrix C that is computed, the processor has to read the entirety of matrix B. Once the outer loop $i$ is iterated, all of matrix B will need to be read again. This means that each thread is reading the entirety of matrix B within its loops.
Instead of using 1 row of A and all of B per thread, we can use a small block of A and a small block of B to compute the partial matrix product for that block in C. This will better utilize the cache (memory locality) by avoiding fetching the entire matrix B per thread.

- **Block Size chosen:** $B = [64]$
- **Reasoning:** Intel Xeon Platinum 8175M 24-core CPU (a custom model for Amazon AWS), each thread has its own L1d 32KiB Cache. 32KiB = 32768 bytes --> since a block of A and a block of B is needed, calculate half of the space 32768 bytes / 2 = 16384 bytes --> 16384 bytes of space fits how many words (a word is 32 bits or 4 bytes): 16384 bytes / 4 bytes = 4096 words --> 4096 words is a 64 x 64 block of words.
- **Implementation:** 

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
- **Scaling:** [Discuss how the performance improved when moving from sequential to parallel].
- **Cache Impact:** [Discuss the jump in GFlops after implementing blocking].
- **Challenges:** [e.g., Mention the initial Makefile path issues or choosing the optimal block size].