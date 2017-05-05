#pragma once

#include <vector>
#include <Eigen/Dense>

struct PointCloud {
    std::vector<Eigen::Vector3f> points;
    std::vector<Eigen::Vector3f> normals;
    std::vector<Eigen::Vector3f> colors;
};
