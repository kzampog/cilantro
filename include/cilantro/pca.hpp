#pragma once

#include <cilantro/point_cloud.hpp>

template <typename ScalarT, ptrdiff_t EigenDim>
class PCA {
public:
    PCA(const std::vector<Eigen::Matrix<ScalarT,EigenDim,1> > &points)
            : num_points_(points.size()),
              data_((ScalarT *)points.data())
    {
        compute_();
    }
    PCA(ScalarT * data, size_t num_points)
            : num_points_(num_points),
              data_(data)
    {
        compute_();
    }
    ~PCA() {}

    inline const Eigen::Matrix<ScalarT,EigenDim,1>& getDataMean() const { return mean_; }
    inline const Eigen::Matrix<ScalarT,EigenDim,1>& getEigenValues() const { return eigenvalues_; }
    inline const Eigen::Matrix<ScalarT,EigenDim,EigenDim>& getEigenVectorsMatrix() const { return eigenvectors_; }
    inline std::vector<Eigen::Matrix<ScalarT,EigenDim,1> > getEigenVectors() const {
        std::vector<Eigen::Matrix<ScalarT,EigenDim,1> > e_vecs(EigenDim);
        Eigen::Matrix<ScalarT,EigenDim,EigenDim>::Map((ScalarT *)e_vecs.data(),EigenDim,EigenDim) = eigenvectors_;
        return e_vecs;
    }

    Eigen::Matrix<ScalarT,Eigen::Dynamic,Eigen::Dynamic> project(Eigen::Matrix<ScalarT,Eigen::Dynamic,Eigen::Dynamic> points, size_t target_dim) const {
        if (target_dim > EigenDim || points.rows() != EigenDim) return Eigen::Matrix<ScalarT,Eigen::Dynamic,Eigen::Dynamic>(target_dim,0);
        return (eigenvectors_.transpose()*(points.colwise()-mean_)).topRows(target_dim);
    }

    template <ptrdiff_t EigenDimOut>
    std::vector<Eigen::Matrix<ScalarT,EigenDimOut,1> > project(const std::vector<Eigen::Matrix<ScalarT,EigenDim,1> > &points) const {
        if (EigenDimOut > EigenDim) return std::vector<Eigen::Matrix<ScalarT,EigenDimOut,1> >(0);
        std::vector<Eigen::Matrix<ScalarT,EigenDimOut,1> > points_p(points.size());
        Eigen::Matrix<ScalarT,EigenDimOut,Eigen::Dynamic>::Map((ScalarT *)points_p.data(),EigenDimOut,points_p.size()) =
                (eigenvectors_.transpose()*((Eigen::Matrix<ScalarT,EigenDim,Eigen::Dynamic>::Map((ScalarT *)points.data(),EigenDim,points.size())).colwise()-mean_)).topRows(EigenDimOut);
        return points_p;
    }

    Eigen::Matrix<ScalarT,Eigen::Dynamic,Eigen::Dynamic> reconstruct(Eigen::Matrix<ScalarT,Eigen::Dynamic,Eigen::Dynamic> points) const {
        if (points.rows() > EigenDim) return Eigen::Matrix<ScalarT,Eigen::Dynamic,Eigen::Dynamic>(EigenDim,0);
        Eigen::Matrix<ScalarT,Eigen::Dynamic,Eigen::Dynamic> points_full_dim(EigenDim,points.cols());
        points_full_dim.topRows(points.rows()) = points;
        points_full_dim.bottomRows(EigenDim-points.rows()) = Eigen::Matrix<ScalarT,Eigen::Dynamic,Eigen::Dynamic>::Zero(EigenDim-points.rows(),points.cols());
        return (eigenvectors_*points_full_dim).colwise() + mean_;
    }

    template <ptrdiff_t EigenDimIn>
    std::vector<Eigen::Matrix<ScalarT,EigenDim,1> > reconstruct(const std::vector<Eigen::Matrix<ScalarT,EigenDimIn,1> > &points) const {
        if (EigenDimIn > EigenDim) return std::vector<Eigen::Matrix<ScalarT,EigenDim,1> >(0);
        Eigen::Matrix<ScalarT,EigenDim,Eigen::Dynamic> points_full_dim(EigenDim,points.size());
        points_full_dim.topRows(EigenDimIn) = Eigen::Matrix<ScalarT,EigenDimIn,Eigen::Dynamic>::Map((ScalarT *)points.data(),EigenDimIn,points.size());
        points_full_dim.bottomRows(EigenDim-EigenDimIn) = Eigen::Matrix<ScalarT,EigenDim-EigenDimIn,Eigen::Dynamic>::Zero(EigenDim-EigenDimIn,points.size());
        std::vector<Eigen::Matrix<ScalarT,EigenDim,1> > points_r(points.size());
        Eigen::Matrix<ScalarT,EigenDim,Eigen::Dynamic>::Map((ScalarT *)points_r.data(),EigenDim,points_r.size()) = (eigenvectors_*points_full_dim).colwise() + mean_;
        return points_r;
    }

private:
    size_t num_points_;
    ScalarT * data_;

    Eigen::Matrix<ScalarT,EigenDim,1> mean_;
    Eigen::Matrix<ScalarT,EigenDim,1> eigenvalues_;
    Eigen::Matrix<ScalarT,EigenDim,EigenDim> eigenvectors_;

    void compute_() {
        Eigen::Map<Eigen::Matrix<ScalarT,EigenDim,Eigen::Dynamic> > data_mat(data_, EigenDim, num_points_);

        mean_ = data_mat.rowwise().mean();
        Eigen::Matrix<ScalarT,EigenDim,Eigen::Dynamic> centered = data_mat.colwise() - mean_;

        Eigen::JacobiSVD<Eigen::Matrix<ScalarT,EigenDim,Eigen::Dynamic> > svd(centered, Eigen::ComputeFullU | Eigen::ComputeThinV);
        eigenvectors_ = svd.matrixU();
        if (eigenvectors_.determinant() < 0.0f) {
            eigenvectors_.col(EigenDim-1) *= -1.0f;
        }
        eigenvalues_ = svd.singularValues().array().square();
    }
};

typedef PCA<float,3> PCA3D;
