#pragma once

#include <sisyphus/point_cloud.hpp>

void estimatePointCloudNormals(PointCloud &cloud, float radius, const Eigen::Vector3f &view_point = Eigen::Vector3f::Zero());
