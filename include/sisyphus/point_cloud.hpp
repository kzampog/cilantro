#pragma once

#include <vector>
#include <Eigen/Dense>

struct PointCloud {
    int num_points;
    std::vector<Eigen::Vector3f> points;
    std::vector<Eigen::Vector3f> normals;
    std::vector<Eigen::Vector3f> colors;
};
