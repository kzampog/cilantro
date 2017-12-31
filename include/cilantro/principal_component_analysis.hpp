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

            Eigen::JacobiSVD<Eigen::Matrix<ScalarT,EigenDim,Eigen::Dynamic>> svd(centered, Eigen::ComputeFullU | Eigen::ComputeThinV);
            eigenvectors_ = svd.matrixU();
            if (eigenvectors_.determinant() < 0.0) {
                ptrdiff_t last_col_ind = data.rows() - 1;
                eigenvectors_.col(last_col_ind) = -eigenvectors_.col(last_col_ind);
            }

            eigenvalues_ = svd.singularValues().array().square();
        }

        ~PrincipalComponentAnalysis() {}

        inline const Eigen::Matrix<ScalarT,EigenDim,1>& getDataMean() const { return mean_; }

        inline const Eigen::Matrix<ScalarT,EigenDim,1>& getEigenValues() const { return eigenvalues_; }

        inline const Eigen::Matrix<ScalarT,EigenDim,EigenDim>& getEigenVectorsMatrix() const { return eigenvectors_; }

        inline std::vector<Eigen::Matrix<ScalarT,EigenDim,1>> getEigenVectors() const {
            std::vector<Eigen::Matrix<ScalarT,EigenDim,1>> e_vecs(eigenvectors_.cols());
            for (size_t i = 0; i < eigenvectors_.cols(); i++) {
                e_vecs[i] = eigenvectors_.col(i);
            }
            return e_vecs;
        }

        Eigen::Matrix<ScalarT,Eigen::Dynamic,Eigen::Dynamic> getProjectedPointsMatrix(const ConstDataMatrixMap<ScalarT,Eigen::Dynamic> &points, size_t target_dim) const {
            return (eigenvectors_.leftCols(target_dim).transpose()*(points.colwise() - mean_));
        }

        template <ptrdiff_t EigenDimOut>
        typename std::enable_if<EigenDimOut != Eigen::Dynamic, Eigen::Matrix<ScalarT,EigenDimOut,Eigen::Dynamic>>::type getProjectedPointsMatrix(const ConstDataMatrixMap<ScalarT,EigenDim> &points) const {
            return (eigenvectors_.leftCols(EigenDimOut).transpose()*(points.colwise() - mean_));
        }

        template <ptrdiff_t EigenDimOut>
        typename std::enable_if<EigenDimOut != Eigen::Dynamic, std::vector<Eigen::Matrix<ScalarT,EigenDimOut,1>>>::type getProjectedPoints(const ConstDataMatrixMap<ScalarT,EigenDim> &points) const {
            std::vector<Eigen::Matrix<ScalarT,EigenDimOut,1>> points_p(points.cols());
            Eigen::Map<Eigen::Matrix<ScalarT,EigenDimOut,Eigen::Dynamic>>((ScalarT *)points_p.data(), EigenDimOut, points_p.size()) =
                    (eigenvectors_.leftCols(EigenDimOut).transpose()*(points.colwise() - mean_));
            return points_p;
        }

        template <ptrdiff_t EigenDimOut>
        typename std::enable_if<EigenDimOut == Eigen::Dynamic, std::vector<Eigen::Matrix<ScalarT,EigenDimOut,1>>>::type getProjectedPoints(const ConstDataMatrixMap<ScalarT,EigenDim> &points) const {
            std::vector<Eigen::Matrix<ScalarT,EigenDimOut,1>> points_p(points.cols());
            for (size_t i = 0; i < points.cols(); i++) {
                points_p[i] = (eigenvectors_.leftCols(EigenDimOut).transpose()*(points.col(i) - mean_));
            }
            return points_p;
        }

        Eigen::Matrix<ScalarT,Eigen::Dynamic,Eigen::Dynamic> getReconstructedPointsMatrix(const ConstDataMatrixMap<ScalarT,Eigen::Dynamic> &points) const {
            return (eigenvectors_.leftCols(points.rows())*points).colwise() + mean_;
        }

        template <ptrdiff_t EigenDimIn>
        typename std::enable_if<EigenDim != Eigen::Dynamic, Eigen::Matrix<ScalarT,EigenDim,Eigen::Dynamic>>::type getReconstructedPointsMatrix(const ConstDataMatrixMap<ScalarT,EigenDimIn> &points) const {
            return (eigenvectors_.leftCols(points.rows())*points).colwise() + mean_;
        }

        template <ptrdiff_t EigenDimIn, ptrdiff_t Dim = EigenDim>
        typename std::enable_if<Dim != Eigen::Dynamic, std::vector<Eigen::Matrix<ScalarT,EigenDim,1>>>::type getReconstructedPoints(const ConstDataMatrixMap<ScalarT,EigenDimIn> &points) const {
            std::vector<Eigen::Matrix<ScalarT,EigenDim,1>> points_r(points.cols());
            Eigen::Map<Eigen::Matrix<ScalarT,EigenDim,Eigen::Dynamic>>((ScalarT *)points_r.data(), EigenDim, points_r.size()) =
                    (eigenvectors_.leftCols(points.rows())*points).colwise() + mean_;
            return points_r;
        }

        template <ptrdiff_t EigenDimIn, ptrdiff_t Dim = EigenDim>
        typename std::enable_if<Dim == Eigen::Dynamic, std::vector<Eigen::Matrix<ScalarT,EigenDim,1>>>::type getReconstructedPoints(const ConstDataMatrixMap<ScalarT,EigenDimIn> &points) const {
            std::vector<Eigen::Matrix<ScalarT,EigenDim,1>> points_r(points.cols());
            for (size_t i = 0; i < points.cols(); i++) {
                points_r[i] = (eigenvectors_.leftCols(points.rows())*points.col(i)) + mean_;
            }
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
