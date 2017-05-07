#pragma once

#include <vector>
#include <Eigen/Dense>

struct PointCloud {
    PointCloud();
    PointCloud(const PointCloud &cloud, const std::vector<size_t> &indices);

    std::vector<Eigen::Vector3f> points;
    std::vector<Eigen::Vector3f> normals;
    std::vector<Eigen::Vector3f> colors;
};
