#pragma once

#include <sisyphus/kd_tree.hpp>

class NormalEstimation {
public:
    NormalEstimation(const PointCloud &cloud);
    NormalEstimation(const PointCloud &cloud, const KDTree &kd_tree);
    ~NormalEstimation();

    inline Eigen::Vector3f& viewPoint() { return view_point_; };

    void computeNormalsKNN(PointCloud &cloud, size_t num_neighbors) const;
    void computeNormalsRadius(PointCloud &cloud, float radius) const;

private:
    const PointCloud &input_cloud_;
    KDTree *kd_tree_ptr_;
    bool kd_tree_owned_;
    Eigen::Vector3f view_point_;

};
