#pragma once

#include <vector>
#include <Eigen/Dense>

struct PointCloud {
    PointCloud();
    PointCloud(const PointCloud &cloud, const std::vector<size_t> &indices);

    std::vector<Eigen::Vector3f> points;
    std::vector<Eigen::Vector3f> normals;
    std::vector<Eigen::Vector3f> colors;

    inline size_t size() const { return points.size(); }
    inline bool hasColors() const { return size() > 0 && colors.size() == size(); }
    inline bool hasNormals() const { return size() > 0 && normals.size() == size(); }
    inline bool empty() const { return points.empty(); }
    void clear();

    inline Eigen::Map<Eigen::MatrixXf> pointsMatrixMap() { return Eigen::Map<Eigen::MatrixXf>((float *)points.data(), 3, points.size()); }
    inline Eigen::Map<Eigen::MatrixXf> normalsMatrixMap() { return Eigen::Map<Eigen::MatrixXf>((float *)normals.data(), 3, normals.size()); }
    inline Eigen::Map<Eigen::MatrixXf> colorsMatrixMap() { return Eigen::Map<Eigen::MatrixXf>((float *)colors.data(), 3, colors.size()); }
};
