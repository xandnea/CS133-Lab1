// Header inclusions, if any...

#include "lib/gemm.h"
#include <omp.h>

// Using declarations, if any...

void GemmParallelBlocked(const float a[kI][kK], const float b[kK][kJ], float c[kI][kJ]) {
  // sqrt((32KiB cache size) / (2 matrices * 4 bytes per word))
  const int block_size = 128; // was 64, testing higher for Xeon L2 cache
  const int num_blocks = kK / block_size;
  
  #pragma omp parallel for collapse(2)// each thread works on a block of C
  for (int c_i = 0; c_i < kI; c_i += block_size) {
    for (int c_j = 0; c_j < kJ; c_j += block_size) {
      // in a thread: working on block (c_i, c_j) of C

      // iterate horizontally through A blocks ==> vertically through B blocks
      for (int block_iter = 0; block_iter < num_blocks; block_iter++) {
        int block_offset = block_iter * block_size;

        // iterate through the block of A and B, and update the block of C
        for (int i = 0; i < block_size; i++) {
          // pointer for row of C and row of A
          float* row_c = &c[c_i + i][c_j];
          const float* row_a = &a[c_i + i][block_offset];

          for (int k = 0; k < block_size; k++) {
            // same as above, hoisting a_ik and row_b
            float a_ik = row_a[k];
            const float* row_b = &b[block_offset + k][c_j];

            #pragma omp simd // vectorize the innermost loop across j
            for (int j = 0; j < block_size; j++) {
              row_c[j] += a_ik * row_b[j];
            }
          }
        }
      }
    }
  }
}
