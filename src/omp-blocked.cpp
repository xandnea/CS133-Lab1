#include "lib/gemm.h"
#include <omp.h>

void GemmParallelBlocked(const float a[kI][kK], const float b[kK][kJ], float c[kI][kJ]) {
  const int block_size = 128; // 128 seems to work better for Xeon L2 cache than 64
  const int num_blocks = kK / block_size;
  
  #pragma omp parallel for collapse(2) schedule(static) num_threads(8) // each thread works on a block of C
  for (int c_i = 0; c_i < kI; c_i += block_size) {
    for (int c_j = 0; c_j < kJ; c_j += block_size) {
      // in a thread: working on block (c_i, c_j) of C

      // iterate horizontally through A blocks & vertically through B blocks 
      // block offset is calculated to add to the k dimensions of A and B
      for (int block_iter = 0; block_iter < num_blocks; block_iter++) {
        int block_offset = block_iter * block_size;

        // iterate through the block of A and B, and update the block of C
        for (int i = 0; i < block_size; i += 2) { // unroll the i loop by 2
          for (int k = 0; k < block_size; k++) {

            // hoist two values of A out of the innermost loop since it doesn't change across j
            const float a_i0_k = a[c_i + i][k + block_offset];
            const float a_i1_k = a[c_i + i + 1][k + block_offset];

            // utilize row pointers for each c row being calculated at column j
            float* c_row0 = &c[c_i + i][c_j];
            float* c_row1 = &c[c_i + i + 1][c_j];
            const float* b_row = &b[k + block_offset][c_j];

            #pragma omp simd // vectorize the innermost loop across j
            for (int j = 0; j < block_size; j++) {
              // one load of b_row[j] feeds two updates
              float b_val = b_row[j];
              c_row0[j] += a_i0_k * b_val;
              c_row1[j] += a_i1_k * b_val;
            }
          }
        }
      }
    }
  }
}