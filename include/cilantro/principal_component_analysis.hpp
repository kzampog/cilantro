#pragma once

#include <cilantro/data_matrix_map.hpp>

namespace cilantro {
    template <typename ScalarT, ptrdiff_t EigenDim>
    class PrincipalComponentAnalysis {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        PrincipalComponentAnalysis(const ConstDataMatrixMap<ScalarT,EigenDim> &data) {
            mean_ = data.rowwise().mean();
            Eigen::Matrix<ScalarT,EigenDim,Eigen::Dynamic> centered = data.colwise() - mean_;

            Eigen::JacobiSVD<Eigen::Matrix<ScalarT,EigenDim,Eigen::Dynamic> > svd(centered, Eigen::ComputeFullU | Eigen::ComputeThinV);
            eigenvectors_ = svd.matrixU();
            if (eigenvectors_.determinant() < 0.0f) {
                eigenvectors_.col(EigenDim-1) = -eigenvectors_.col(EigenDim-1);
            }
            eigenvalues_ = svd.singularValues().array().square();
        }

        ~PrincipalComponentAnalysis() {}

        inline const Eigen::Matrix<ScalarT,EigenDim,1>& getDataMean() const { return mean_; }
        inline const Eigen::Matrix<ScalarT,EigenDim,1>& getEigenValues() const { return eigenvalues_; }
        inline const Eigen::Matrix<ScalarT,EigenDim,EigenDim>& getEigenVectorsMatrix() const { return eigenvectors_; }
        inline std::vector<Eigen::Matrix<ScalarT,EigenDim,1> > getEigenVectors() const {
            std::vector<Eigen::Matrix<ScalarT,EigenDim,1> > e_vecs(EigenDim);
            Eigen::Matrix<ScalarT,EigenDim,EigenDim>::Map((ScalarT *)e_vecs.data(),EigenDim,EigenDim) = eigenvectors_;
            return e_vecs;
        }

        Eigen::Matrix<ScalarT,Eigen::Dynamic,Eigen::Dynamic> project(const Eigen::Ref<const Eigen::Matrix<ScalarT,Eigen::Dynamic,Eigen::Dynamic> > &points, size_t target_dim) const {
            return (eigenvectors_.leftCols(target_dim).transpose()*(points.colwise() - mean_));
        }

        template <ptrdiff_t EigenDimOut>
        std::vector<Eigen::Matrix<ScalarT,EigenDimOut,1> > project(const ConstDataMatrixMap<ScalarT,EigenDim> &points) const {
            std::vector<Eigen::Matrix<ScalarT,EigenDimOut,1> > points_p(points.cols());
            Eigen::Map<Eigen::Matrix<ScalarT,EigenDimOut,Eigen::Dynamic> >((ScalarT *)points_p.data(), EigenDimOut, points_p.size()) =
                    (eigenvectors_.leftCols(EigenDimOut).transpose()*(points.colwise() - mean_));
            return points_p;
        }

        Eigen::Matrix<ScalarT,Eigen::Dynamic,Eigen::Dynamic> reconstruct(const Eigen::Ref<const Eigen::Matrix<ScalarT,Eigen::Dynamic,Eigen::Dynamic> > &points) const {
            return (eigenvectors_.leftCols(points.rows())*points).colwise() + mean_;
        }

        template <ptrdiff_t EigenDimIn>
        std::vector<Eigen::Matrix<ScalarT,EigenDim,1> > reconstruct(const ConstDataMatrixMap<ScalarT,EigenDimIn> &points) const {
            std::vector<Eigen::Matrix<ScalarT,EigenDim,1> > points_r(points.cols());
            Eigen::Map<Eigen::Matrix<ScalarT,EigenDim,Eigen::Dynamic> >((ScalarT *)points_r.data(), EigenDim, points_r.size()) =
                    (eigenvectors_.leftCols(EigenDimIn)*points).colwise() + mean_;
            return points_r;
        }

    protected:
        Eigen::Matrix<ScalarT,EigenDim,1> mean_;
        Eigen::Matrix<ScalarT,EigenDim,1> eigenvalues_;
        Eigen::Matrix<ScalarT,EigenDim,EigenDim> eigenvectors_;
    };

    typedef PrincipalComponentAnalysis<float,2> PrincipalComponentAnalysis2D;
    typedef PrincipalComponentAnalysis<float,3> PrincipalComponentAnalysis3D;
}
