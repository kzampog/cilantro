#ifndef POINT_CLOUD_HPP
#define POINT_CLOUD_HPP

#include <vector>
#include <Eigen/Dense>

struct PointCloud {
    std::vector<Eigen::Vector3f> points;
    std::vector<Eigen::Vector3f> normals;
    std::vector<Eigen::Vector3f> colors;
};

#endif /* POINT_CLOUD_HPP */
