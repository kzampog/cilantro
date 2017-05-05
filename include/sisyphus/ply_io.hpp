#pragma once

#include <sisyphus/point_cloud.hpp>

void readPointCloudFromPLYFile(const std::string &filename, PointCloud &cloud);

void writePointCloudToPLYFile(const std::string &filename, PointCloud &cloud);
