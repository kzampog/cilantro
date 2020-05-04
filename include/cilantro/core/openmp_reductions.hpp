#pragma once

#if defined(__clang__)
#define STRINGIFY(x) #x
#define DECLARE_MATRIX_SUM_REDUCTION(Scalar,Rows,Cols) _Pragma(STRINGIFY(omp declare reduction (+: Eigen::Matrix<Scalar,Rows,Cols>: omp_out = omp_out + omp_in) initializer(omp_priv = Eigen::Matrix<Scalar,Rows,Cols>::Zero(omp_orig.rows(), omp_orig.cols()))))
#define MATRIX_SUM_REDUCTION(Scalar,Rows,Cols,var) reduction (+: var)
#elif defined(_MSC_VER)
// Never lose hope... :)
#define DECLARE_MATRIX_SUM_REDUCTION(Scalar,Rows,Cols) __pragma(omp declare reduction (+: Eigen::Matrix<Scalar,Rows,Cols>: omp_out = omp_out + omp_in) initializer(omp_priv = Eigen::Matrix<Scalar,Rows,Cols>::Zero(omp_orig.rows(), omp_orig.cols())))
#define MATRIX_SUM_REDUCTION(Scalar,Rows,Cols,var) reduction (+: var)
#else
#include <Eigen/Dense>

namespace cilantro {
    namespace internal {
        template <typename ScalarT, ptrdiff_t NRows, ptrdiff_t NCols>
        struct MatrixReductions {
#pragma omp declare reduction (+: Eigen::Matrix<ScalarT,NRows,NCols>: omp_out = omp_out + omp_in) initializer(omp_priv = Eigen::Matrix<ScalarT,NRows,NCols>::Zero(omp_orig.rows(), omp_orig.cols()))
        };
    } // namespace internal
}

#define DECLARE_MATRIX_SUM_REDUCTION(Scalar,Rows,Cols)
#define MATRIX_SUM_REDUCTION(Scalar,Rows,Cols,var) reduction (cilantro::internal::MatrixReductions<Scalar,Rows,Cols>::operator+: var)
#endif
