#pragma once

#include <sisyphus/point_cloud.hpp>

void estimatePointCloudNormalsKNN(PointCloud &cloud, size_t num_neighbors, const Eigen::Vector3f &view_point = Eigen::Vector3f::Zero());

void estimatePointCloudNormalsRadius(PointCloud &cloud, float radius, const Eigen::Vector3f &view_point = Eigen::Vector3f::Zero());
