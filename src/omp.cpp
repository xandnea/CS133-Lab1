// Header inclusions, if any...

#include "lib/gemm.h"
#include <omp.h>

// Using declarations, if any...

void GemmParallel(const float a[kI][kK], const float b[kK][kJ], float c[kI][kJ]) {
  #pragma omp parallel for
  for (int i = 0; i < kI; ++i) {
    for (int k = 0; k < kK; ++k) {
      // hoist a[i][k] out of the innermost loop since it doesn't change across j
      const float a_ik = a[i][k];
      for (int j = 0; j < kJ; ++j) {
        c[i][j] += a_ik * b[k][j];
      }
    }
  }
}
