#pragma once

#include <vector>
#include <Eigen/Dense>

namespace cilantro {
    template <typename ScalarT, ptrdiff_t EigenDim>
    class PrincipalComponentAnalysis {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        PrincipalComponentAnalysis(const Eigen::Ref<const Eigen::Matrix<ScalarT,EigenDim,Eigen::Dynamic> > &points)
                : num_points_(points.cols()),
                  data_((ScalarT *)points.data())
        {
            compute_();
        }
        PrincipalComponentAnalysis(const std::vector<Eigen::Matrix<ScalarT,EigenDim,1> > &points)
                : num_points_(points.size()),
                  data_((ScalarT *)points.data())
        {
            compute_();
        }
        PrincipalComponentAnalysis(ScalarT * data, size_t num_points)
                : num_points_(num_points),
                  data_(data)
        {
            compute_();
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

        Eigen::Matrix<ScalarT,Eigen::Dynamic,Eigen::Dynamic> project(const Eigen::Matrix<ScalarT,Eigen::Dynamic,Eigen::Dynamic> &points, size_t target_dim) const {
            //if (target_dim > EigenDim || points.rows() != EigenDim) return Eigen::Matrix<ScalarT,Eigen::Dynamic,Eigen::Dynamic>(target_dim,0);
            return (eigenvectors_.leftCols(target_dim).transpose()*(points.colwise() - mean_));
        }

        template <ptrdiff_t EigenDimOut>
        Eigen::Matrix<ScalarT,EigenDimOut,Eigen::Dynamic> project(const Eigen::Ref<const Eigen::Matrix<ScalarT,EigenDim,Eigen::Dynamic> > &points) const {
            //if (EigenDimOut > EigenDim) return Eigen::Matrix<ScalarT,EigenDimOut,Eigen::Dynamic>(EigenDimOut,0);
            return (eigenvectors_.leftCols(EigenDimOut).transpose()*(points.colwise() - mean_));
        }

        template <ptrdiff_t EigenDimOut>
        std::vector<Eigen::Matrix<ScalarT,EigenDimOut,1> > project(const std::vector<Eigen::Matrix<ScalarT,EigenDim,1> > &points) const {
            //if (EigenDimOut > EigenDim) return std::vector<Eigen::Matrix<ScalarT,EigenDimOut,1> >(0);
            std::vector<Eigen::Matrix<ScalarT,EigenDimOut,1> > points_p(points.size());
            Eigen::Matrix<ScalarT,EigenDimOut,Eigen::Dynamic>::Map((ScalarT *)points_p.data(),EigenDimOut,points_p.size()) =
                    (eigenvectors_.leftCols(EigenDimOut).transpose()*((Eigen::Matrix<ScalarT,EigenDim,Eigen::Dynamic>::Map((ScalarT *)points.data(),EigenDim,points.size())).colwise() - mean_));
            return points_p;
        }

        Eigen::Matrix<ScalarT,Eigen::Dynamic,Eigen::Dynamic> reconstruct(const Eigen::Matrix<ScalarT,Eigen::Dynamic,Eigen::Dynamic> &points) const {
            //if (points.rows() > EigenDim) return Eigen::Matrix<ScalarT,Eigen::Dynamic,Eigen::Dynamic>(EigenDim,0);
            return (eigenvectors_.leftCols(points.rows())*points).colwise() + mean_;
        }

        template <ptrdiff_t EigenDimIn>
        Eigen::Matrix<ScalarT,EigenDim,Eigen::Dynamic> reconstruct(const Eigen::Ref<const Eigen::Matrix<ScalarT,EigenDimIn,Eigen::Dynamic> > &points) const {
            //if (EigenDimIn > EigenDim) return Eigen::Matrix<ScalarT,EigenDim,Eigen::Dynamic>(EigenDim,0);
            return (eigenvectors_.leftCols(EigenDimIn)*points).colwise() + mean_;
        }

        template <ptrdiff_t EigenDimIn>
        std::vector<Eigen::Matrix<ScalarT,EigenDim,1> > reconstruct(const std::vector<Eigen::Matrix<ScalarT,EigenDimIn,1> > &points) const {
            //if (EigenDimIn > EigenDim) return std::vector<Eigen::Matrix<ScalarT,EigenDim,1> >(0);
            std::vector<Eigen::Matrix<ScalarT,EigenDim,1> > points_r(points.size());
            Eigen::Map<Eigen::Matrix<ScalarT,EigenDim,Eigen::Dynamic> >((ScalarT *)points_r.data(),EigenDim,points_r.size()) =
                    (eigenvectors_.leftCols(EigenDimIn)*Eigen::Map<Eigen::Matrix<ScalarT,EigenDimIn,Eigen::Dynamic> >((ScalarT *)points.data(),EigenDimIn,points.size())).colwise() + mean_;
            return points_r;
        }

    protected:
        size_t num_points_;
        ScalarT * data_;

        Eigen::Matrix<ScalarT,EigenDim,1> mean_;
        Eigen::Matrix<ScalarT,EigenDim,1> eigenvalues_;
        Eigen::Matrix<ScalarT,EigenDim,EigenDim> eigenvectors_;

        inline void compute_() {
            Eigen::Map<Eigen::Matrix<ScalarT,EigenDim,Eigen::Dynamic> > data_mat(data_, EigenDim, num_points_);

            mean_ = data_mat.rowwise().mean();
            Eigen::Matrix<ScalarT,EigenDim,Eigen::Dynamic> centered = data_mat.colwise() - mean_;

            Eigen::JacobiSVD<Eigen::Matrix<ScalarT,EigenDim,Eigen::Dynamic> > svd(centered, Eigen::ComputeFullU | Eigen::ComputeThinV);
            eigenvectors_ = svd.matrixU();
            if (eigenvectors_.determinant() < 0.0f) {
                eigenvectors_.col(EigenDim-1) = -eigenvectors_.col(EigenDim-1);
            }
            eigenvalues_ = svd.singularValues().array().square();
        }
    };

    typedef PrincipalComponentAnalysis<float,2> PrincipalComponentAnalysis2D;
    typedef PrincipalComponentAnalysis<float,3> PrincipalComponentAnalysis3D;
}
