#pragma once

#include <cilantro/point_cloud.hpp>

void readPointCloudFromPLYFile(const std::string &filename, PointCloud &cloud);

void writePointCloudToPLYFile(const std::string &filename, const PointCloud &cloud, bool binary = true);
