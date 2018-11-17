#pragma once

#include <cilantro/data_containers.hpp>

namespace cilantro {
    template <typename ScalarT, ptrdiff_t EigenDim>
    class PrincipalComponentAnalysis {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        typedef ScalarT Scalar;

        enum { Dimension = EigenDim };

        PrincipalComponentAnalysis(const ConstVectorSetMatrixMap<ScalarT,EigenDim> &data) {
            mean_ = data.rowwise().mean();
//#pragma omp declare reduction (+: Eigen::Matrix<ScalarT,EigenDim,EigenDim>: omp_out = omp_out + omp_in) initializer(omp_priv = Eigen::Matrix<ScalarT,EigenDim,EigenDim>::Zero(omp_orig.rows(), omp_orig.cols()))
            Eigen::Matrix<ScalarT,EigenDim,EigenDim> cov(Eigen::Matrix<ScalarT,EigenDim,EigenDim>::Zero(data.rows(), data.rows()));
//#pragma omp parallel for reduction (+: cov)
            for (size_t i = 0; i < data.cols(); i++) {
                Eigen::Matrix<ScalarT,EigenDim,1> ptc = data.col(i) - mean_;
                cov += ptc*ptc.transpose();
            }
            cov *= (ScalarT)(1.0)/(data.cols()-1);

            Eigen::SelfAdjointEigenSolver<Eigen::Matrix<ScalarT,EigenDim,EigenDim>> eig(cov);

            eigenvectors_ = eig.eigenvectors().rowwise().reverse();
            if (eigenvectors_.determinant() < (ScalarT)0.0) {
                auto last_col = eigenvectors_.col(data.rows() - 1);
                last_col = -last_col;
            }

            eigenvalues_ = eig.eigenvalues().reverse();

//            mean_ = data.rowwise().mean();
//            Eigen::Matrix<ScalarT,EigenDim,Eigen::Dynamic> centered = data.colwise() - mean_;
//
//            Eigen::Matrix<ScalarT,EigenDim,EigenDim> cov = centered*centered.transpose();
//            cov *= (ScalarT)(1.0)/(data.cols()-1);
//            Eigen::SelfAdjointEigenSolver<Eigen::Matrix<ScalarT,EigenDim,EigenDim>> eig(cov);
//
//            eigenvectors_ = eig.eigenvectors().rowwise().reverse();
//            if (eigenvectors_.determinant() < (ScalarT)0.0) {
//                auto last_col = eigenvectors_.col(data.rows() - 1);
//                last_col = -last_col;
//            }
//
//            eigenvalues_ = eig.eigenvalues().reverse();
        }

        ~PrincipalComponentAnalysis() {}

        inline const Vector<ScalarT,EigenDim>& getDataMean() const { return mean_; }

        inline const Vector<ScalarT,EigenDim>& getEigenValues() const { return eigenvalues_; }

        inline const Eigen::Matrix<ScalarT,EigenDim,EigenDim>& getEigenVectors() const { return eigenvectors_; }

        inline Eigen::Matrix<ScalarT,Eigen::Dynamic,Eigen::Dynamic> project(const ConstVectorSetMatrixMap<ScalarT,Eigen::Dynamic> &points,
                                                                            size_t target_dim) const
        {
            return (eigenvectors_.leftCols(target_dim).transpose()*(points.colwise() - mean_));
        }

        template <ptrdiff_t EigenDimOut>
        inline typename std::enable_if<EigenDimOut != Eigen::Dynamic, VectorSet<ScalarT,EigenDimOut>>::type project(const ConstVectorSetMatrixMap<ScalarT,EigenDim> &points) const {
            return (eigenvectors_.leftCols(EigenDimOut).transpose()*(points.colwise() - mean_));
        }

        inline Eigen::Matrix<ScalarT,Eigen::Dynamic,Eigen::Dynamic> reconstruct(const ConstVectorSetMatrixMap<ScalarT,Eigen::Dynamic> &points) const {
            return (eigenvectors_.leftCols(points.rows())*points).colwise() + mean_;
        }

        template <ptrdiff_t EigenDimIn>
        inline typename std::enable_if<EigenDimIn != Eigen::Dynamic, VectorSet<ScalarT,EigenDim>>::type reconstruct(const ConstVectorSetMatrixMap<ScalarT,EigenDimIn> &points) const {
            return (eigenvectors_.leftCols(EigenDimIn)*points).colwise() + mean_;
        }

    protected:
        Vector<ScalarT,EigenDim> mean_;
        Vector<ScalarT,EigenDim> eigenvalues_;
        Eigen::Matrix<ScalarT,EigenDim,EigenDim> eigenvectors_;
    };

    typedef PrincipalComponentAnalysis<float,2> PrincipalComponentAnalysis2f;
    typedef PrincipalComponentAnalysis<double,2> PrincipalComponentAnalysis2d;
    typedef PrincipalComponentAnalysis<float,3> PrincipalComponentAnalysis3f;
    typedef PrincipalComponentAnalysis<double,3> PrincipalComponentAnalysis3d;
    typedef PrincipalComponentAnalysis<float,Eigen::Dynamic> PrincipalComponentAnalysisXf;
    typedef PrincipalComponentAnalysis<double,Eigen::Dynamic> PrincipalComponentAnalysisXd;
}
