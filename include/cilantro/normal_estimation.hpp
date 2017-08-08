#pragma once

#include <cilantro/kd_tree.hpp>

class NormalEstimation {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    NormalEstimation(const std::vector<Eigen::Vector3f> &points);
    NormalEstimation(const std::vector<Eigen::Vector3f> &points, const KDTree &kd_tree);
    NormalEstimation(const PointCloud &cloud);
    NormalEstimation(const PointCloud &cloud, const KDTree &kd_tree);
    ~NormalEstimation();

    inline Eigen::Vector3f getViewPoint() { return view_point_; };
    inline NormalEstimation& setViewPoint(const Eigen::Vector3f &vp) { view_point_ = vp; return *this; }

    std::vector<Eigen::Vector3f> estimateNormalsKNN(size_t num_neighbors) const;
    void estimateNormalsInPlaceKNN(size_t num_neighbors) const;

    std::vector<Eigen::Vector3f> estimateNormalsRadius(float radius) const;
    void estimateNormalsInPlaceRadius(float radius) const;

    std::vector<Eigen::Vector3f> estimateNormalsKNNInRadius(size_t k, float radius) const;
    void estimateNormalsInPlaceKNNInRadius(size_t k, float radius) const;

    std::vector<Eigen::Vector3f> estimateNormals(const KDTree::Neighborhood &nh) const;
    void estimateNormalsInPlace(const KDTree::Neighborhood &nh) const;

private:
    PointCloud *input_cloud_;
    const std::vector<Eigen::Vector3f> *input_points_;
    KDTree *kd_tree_ptr_;
    bool kd_tree_owned_;
    Eigen::Vector3f view_point_;

};
