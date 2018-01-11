#pragma once

#include <cilantro/data_containers.hpp>

namespace cilantro {
    template <typename ScalarT, ptrdiff_t EigenDim>
    class PrincipalComponentAnalysis {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        PrincipalComponentAnalysis(const ConstVectorSetMatrixMap<ScalarT,EigenDim> &data) {
            mean_ = data.rowwise().mean();
            Eigen::Matrix<ScalarT,EigenDim,Eigen::Dynamic> centered = data.colwise() - mean_;

            Eigen::SelfAdjointEigenSolver<Eigen::Matrix<ScalarT,EigenDim,EigenDim>> eig((centered*centered.transpose())/(data.cols()-1));

            eigenvectors_ = eig.eigenvectors().rowwise().reverse();
            if (eigenvectors_.determinant() < 0.0) {
                ptrdiff_t last_col_ind = data.rows() - 1;
                eigenvectors_.col(last_col_ind) = -eigenvectors_.col(last_col_ind);
            }

            eigenvalues_ = eig.eigenvalues().reverse();
        }

        ~PrincipalComponentAnalysis() {}

        inline const Vector<ScalarT,EigenDim>& getDataMean() const { return mean_; }

        inline const Vector<ScalarT,EigenDim>& getEigenValues() const { return eigenvalues_; }

        inline const Eigen::Matrix<ScalarT,EigenDim,EigenDim>& getEigenVectors() const { return eigenvectors_; }

        Eigen::Matrix<ScalarT,Eigen::Dynamic,Eigen::Dynamic> project(const ConstVectorSetMatrixMap<ScalarT,Eigen::Dynamic> &points, size_t target_dim) const {
            return (eigenvectors_.leftCols(target_dim).transpose()*(points.colwise() - mean_));
        }

        template <ptrdiff_t EigenDimOut>
        typename std::enable_if<EigenDimOut != Eigen::Dynamic, VectorSet<ScalarT,EigenDimOut>>::type project(const ConstVectorSetMatrixMap<ScalarT,EigenDim> &points) const {
            return (eigenvectors_.leftCols(EigenDimOut).transpose()*(points.colwise() - mean_));
        }

        Eigen::Matrix<ScalarT,Eigen::Dynamic,Eigen::Dynamic> reconstruct(const ConstVectorSetMatrixMap<ScalarT,Eigen::Dynamic> &points) const {
            return (eigenvectors_.leftCols(points.rows())*points).colwise() + mean_;
        }

        template <ptrdiff_t EigenDimIn>
        typename std::enable_if<EigenDimIn != Eigen::Dynamic, VectorSet<ScalarT,EigenDim>>::type reconstruct(const ConstVectorSetMatrixMap<ScalarT,EigenDimIn> &points) const {
            return (eigenvectors_.leftCols(EigenDimIn)*points).colwise() + mean_;
        }

    protected:
        Vector<ScalarT,EigenDim> mean_;
        Vector<ScalarT,EigenDim> eigenvalues_;
        Eigen::Matrix<ScalarT,EigenDim,EigenDim> eigenvectors_;
    };

    typedef PrincipalComponentAnalysis<float,2> PrincipalComponentAnalysis2D;
    typedef PrincipalComponentAnalysis<float,3> PrincipalComponentAnalysis3D;
}
