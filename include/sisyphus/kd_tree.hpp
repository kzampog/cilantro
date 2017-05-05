#pragma once

#include <nanoflann/nanoflann.hpp>
#include <sisyphus/point_cloud.hpp>

class KDTree {
public:
    KDTree(const PointCloud &cloud);
    ~KDTree() {}

private:

};
