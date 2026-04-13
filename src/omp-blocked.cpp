// Header inclusions, if any...

#include "lib/gemm.h"
#include <omp.h>

// Using declarations, if any...

void GemmParallelBlocked(const float a[kI][kK], const float b[kK][kJ], float c[kI][kJ]) {
  const int blockSize = 64; // sqrt((32KiB cache size) / (2 matrices * 4 bytes per word))
  for (int c_i = 0; c_i < kI; c_i += blockSize) {
    for (int c_j = 0; c_j < kJ; c_j += blockSize) {
    #pragma omp parallel for
      // iterate through blocks of A and B to get corresponding block of C 
      // if thread 1 is working on block (0, 0) of C, it will iterate through blocks (0, 0), (0, 1), ..., of A and (0, 0), (1, 0), ..., of B
      // each thread is utilizing the cache by working on a block of C and corresponding blocks of A and B
    }
  }
}
