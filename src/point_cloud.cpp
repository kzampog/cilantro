#include <cilantro/point_cloud.hpp>

PointCloud::PointCloud() {}

PointCloud::PointCloud(const std::vector<Eigen::Vector3f> &points, const std::vector<Eigen::Vector3f> &normals, const std::vector<Eigen::Vector3f> &colors)
        : points(points),
          normals(normals),
          colors(colors)
{}

PointCloud::PointCloud(const PointCloud &cloud, const std::vector<size_t> &indices) {
    points.resize(indices.size());
    for (size_t i = 0; i < indices.size(); i++) {
        points[i] = cloud.points[indices[i]];
    }

    if (cloud.hasNormals()) {
        normals.resize(indices.size());
        for (size_t i = 0; i < indices.size(); i++) {
            normals[i] = cloud.normals[indices[i]];
        }
    }

    if (cloud.hasColors()) {
        colors.resize(indices.size());
        for (size_t i = 0; i < indices.size(); i++) {
            colors[i] = cloud.colors[indices[i]];
        }
    }
}

void PointCloud::clear() {
    points.clear();
    normals.clear();
    colors.clear();
}
