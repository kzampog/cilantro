#pragma once

#ifdef _MSC_VER
#define DEFINE_MATRIX_SUM_REDUCTION(Scalar,Rows,Cols) __pragma(omp declare reduction (+: Eigen::Matrix<Scalar,Rows,Cols>: omp_out = omp_out + omp_in) initializer(omp_priv = Eigen::Matrix<Scalar,Rows,Cols>::Zero(omp_orig.rows(), omp_orig.cols())))
#else
#define STRINGIFY(x) #x
#define DEFINE_MATRIX_SUM_REDUCTION(Scalar,Rows,Cols) _Pragma(STRINGIFY(omp declare reduction (+: Eigen::Matrix<Scalar,Rows,Cols>: omp_out = omp_out + omp_in) initializer(omp_priv = Eigen::Matrix<Scalar,Rows,Cols>::Zero(omp_orig.rows(), omp_orig.cols()))))
#endif

//#include <Eigen/Dense>
//
//namespace cilantro {
//    namespace internal {
//        template <typename ScalarT, ptrdiff_t NRows, ptrdiff_t NCols>
//        struct MatrixReductions {
//#pragma omp declare reduction (+: Eigen::Matrix<ScalarT,NRows,NCols>: omp_out = omp_out + omp_in) initializer(omp_priv = Eigen::Matrix<ScalarT,NRows,NCols>::Zero(omp_orig.rows(), omp_orig.cols()))
//        };
//    }
//}
