#pragma once

#include <vector>
#include <Eigen/Dense>

struct PointCloud {
    PointCloud();
    PointCloud(const PointCloud &cloud, const std::vector<size_t> &indices);

    std::vector<Eigen::Vector3f> points;
    std::vector<Eigen::Vector3f> normals;
    std::vector<Eigen::Vector3f> colors;

    inline size_t size() const { return points.size(); };
    inline bool hasColors() const { return size() > 0 && colors.size() == size(); };
    inline bool hasNormals() const { return size() > 0 && normals.size() == size(); };

    inline bool empty() const { return points.empty(); }
    void clear();
};

