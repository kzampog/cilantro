#pragma once

#include <cilantro/point_cloud.hpp>

class PCA {
public:
    PCA(const PointCloud &pc);
    PCA(const std::vector<Eigen::Vector3f> &points);
    PCA(float * data, size_t dim, size_t num_points);

    ~PCA() {}

    inline Eigen::VectorXf getMean() const { return mean_; }
    inline Eigen::VectorXf getEigenValues() const { return eigenvalues_; }
    inline Eigen::MatrixXf getEigenVectors() const { return eigenvectors_; }

private:
    size_t dim_;
    size_t num_points_;
    float * data_;

    Eigen::VectorXf mean_;
    Eigen::VectorXf eigenvalues_;
    Eigen::MatrixXf eigenvectors_;

    void compute_();
};
