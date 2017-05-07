#pragma once

#include <vector>
#include <Eigen/Dense>

struct PointCloud {
    PointCloud();
    PointCloud(const PointCloud &cloud, const std::vector<size_t> &indices);

    std::vector<Eigen::Vector3f> points;
    std::vector<Eigen::Vector3f> normals;
    std::vector<Eigen::Vector3f> colors;

    inline size_t size() { return points.size(); };
    inline bool hasColors() { return size() > 0 && colors.size() == size(); };
    inline bool hasNormals() { return size() > 0 && normals.size() == size(); };

    // TODO: add empty() and clear() methods
};

