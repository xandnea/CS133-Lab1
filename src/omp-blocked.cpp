// Header inclusions, if any...

#include "lib/gemm.h"
#include <omp.h>

// Using declarations, if any...

void GemmParallelBlocked(const float a[kI][kK], const float b[kK][kJ], float c[kI][kJ]) {
  const int block_size = 64; // sqrt((32KiB cache size) / (2 matrices * 4 bytes per word))
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
          for (int k = 0; k < block_size; k++) {
            for (int j = 0; j < block_size; j++) {
              c[c_i + i][c_j + j] += a[c_i + i][k + block_offset] * b[k + block_offset][c_j + j];
            }
          }
        }
      }
    }
  }
}
