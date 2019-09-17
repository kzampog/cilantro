#pragma once

#include <Eigen/Dense>

namespace cilantro {
    namespace internal {
        template <typename ScalarT, ptrdiff_t NRows, ptrdiff_t NCols>
        struct MatrixReductions {
#pragma omp declare reduction (+: Eigen::Matrix<ScalarT,NRows,NCols>: omp_out = omp_out + omp_in) initializer(omp_priv = Eigen::Matrix<ScalarT,NRows,NCols>::Zero(omp_orig.rows(), omp_orig.cols()))
        };
    }
}
