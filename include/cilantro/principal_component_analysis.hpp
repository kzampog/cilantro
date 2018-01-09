#pragma once

#include <cilantro/data_containers.hpp>

namespace cilantro {
    template <typename ScalarT, ptrdiff_t EigenDim>
    class PrincipalComponentAnalysis {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        PrincipalComponentAnalysis(const ConstPointSetMatrixMap<ScalarT,EigenDim> &data) {
            mean_ = data.rowwise().mean();
            Eigen::Matrix<ScalarT,EigenDim,Eigen::Dynamic> centered = data.colwise() - mean_;

            Eigen::JacobiSVD<Eigen::Matrix<ScalarT,EigenDim,Eigen::Dynamic>> svd(centered, Eigen::ComputeFullU | Eigen::ComputeThinV);
            eigenvectors_ = svd.matrixU();
            if (eigenvectors_.determinant() < 0.0) {
                ptrdiff_t last_col_ind = data.rows() - 1;
                eigenvectors_.col(last_col_ind) = -eigenvectors_.col(last_col_ind);
            }

            eigenvalues_ = svd.singularValues().array().square();
        }

        ~PrincipalComponentAnalysis() {}

        inline const Point<ScalarT,EigenDim>& getDataMean() const { return mean_; }

        inline const Point<ScalarT,EigenDim>& getEigenValues() const { return eigenvalues_; }

        inline const Eigen::Matrix<ScalarT,EigenDim,EigenDim>& getEigenVectors() const { return eigenvectors_; }

        Eigen::Matrix<ScalarT,Eigen::Dynamic,Eigen::Dynamic> project(const ConstPointSetMatrixMap<ScalarT,Eigen::Dynamic> &points, size_t target_dim) const {
            return (eigenvectors_.leftCols(target_dim).transpose()*(points.colwise() - mean_));
        }

        template <ptrdiff_t EigenDimOut>
        typename std::enable_if<EigenDimOut != Eigen::Dynamic, PointSet<ScalarT,EigenDimOut>>::type project(const ConstPointSetMatrixMap<ScalarT,EigenDim> &points) const {
            return (eigenvectors_.leftCols(EigenDimOut).transpose()*(points.colwise() - mean_));
        }

        Eigen::Matrix<ScalarT,Eigen::Dynamic,Eigen::Dynamic> reconstruct(const ConstPointSetMatrixMap<ScalarT,Eigen::Dynamic> &points) const {
            return (eigenvectors_.leftCols(points.rows())*points).colwise() + mean_;
        }

        template <ptrdiff_t EigenDimIn>
        typename std::enable_if<EigenDimIn != Eigen::Dynamic, PointSet<ScalarT,EigenDim>>::type reconstruct(const ConstPointSetMatrixMap<ScalarT,EigenDimIn> &points) const {
            return (eigenvectors_.leftCols(EigenDimIn)*points).colwise() + mean_;
        }

    protected:
        Point<ScalarT,EigenDim> mean_;
        Point<ScalarT,EigenDim> eigenvalues_;
        Eigen::Matrix<ScalarT,EigenDim,EigenDim> eigenvectors_;
    };

    typedef PrincipalComponentAnalysis<float,2> PrincipalComponentAnalysis2D;
    typedef PrincipalComponentAnalysis<float,3> PrincipalComponentAnalysis3D;
}
