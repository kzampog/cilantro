#pragma once

#include <sisyphus/point_cloud.hpp>

class PCA {
public:
    PCA(const PointCloud &pc);
    PCA(const std::vector<Eigen::Vector3f> &points);
    PCA(float * data, size_t num_points);

    ~PCA() {}

    inline Eigen::Vector3f getMean() { return mean_; }
    inline Eigen::Vector3f getEigenValues() { return eigenvalues_; }
    inline Eigen::Matrix3f getEigenVectors() { return eigenvectors_; }

private:
    size_t num_points_;
    float * data_;

    Eigen::Vector3f mean_;
    Eigen::Vector3f eigenvalues_;
    Eigen::Matrix3f eigenvectors_;

    void compute_();
};

