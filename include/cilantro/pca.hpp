#pragma once

#include <cilantro/point_cloud.hpp>

class PCA {
public:
    PCA(const PointCloud &pc);
    PCA(const std::vector<Eigen::Vector3f> &points);
    PCA(float * data, size_t dim, size_t num_points);

    ~PCA() {}

    inline Eigen::Vector3f getMean() const { return mean_; }
    inline Eigen::Vector3f getEigenValues() const { return eigenvalues_; }
    inline Eigen::Matrix3f getEigenVectors() const { return eigenvectors_; }

private:
    size_t dim_;
    size_t num_points_;
    float * data_;

    Eigen::Vector3f mean_;
    Eigen::Vector3f eigenvalues_;
    Eigen::Matrix3f eigenvectors_;

    void compute_();
};
