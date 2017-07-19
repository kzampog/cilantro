#include <cilantro/point_cloud.hpp>
#include <set>

PointCloud::PointCloud() {}

PointCloud::PointCloud(const std::vector<Eigen::Vector3f> &points, const std::vector<Eigen::Vector3f> &normals, const std::vector<Eigen::Vector3f> &colors)
        : points(points),
          normals(normals),
          colors(colors)
{}

PointCloud::PointCloud(const PointCloud &cloud, const std::vector<size_t> &indices, bool negate) {

    std::set<size_t> indices_set;
    if (negate) {
        std::vector<size_t> full_indices(cloud.size());
        for (size_t i = 0; i < cloud.size(); i++) {
            full_indices[i] = i;
        }
        std::set<size_t> indices_to_discard(indices.begin(), indices.end());
        std::set_difference(full_indices.begin(), full_indices.end(), indices_to_discard.begin(), indices_to_discard.end(), std::inserter(indices_set, indices_set.begin()));
    } else {
        indices_set = std::set<size_t>(indices.begin(), indices.end());
    }

    size_t k = 0;
    points.resize(indices_set.size());
    for (auto it = indices_set.begin(); it != indices_set.end(); ++it) {
        points[k++] = cloud.points[*it];
    }

    if (cloud.hasNormals()) {
        k = 0;
        normals.resize(indices_set.size());
        for (auto it = indices_set.begin(); it != indices_set.end(); ++it) {
            normals[k++] = cloud.normals[*it];
        }
    }

    if (cloud.hasColors()) {
        k = 0;
        colors.resize(indices_set.size());
        for (auto it = indices_set.begin(); it != indices_set.end(); ++it) {
            colors[k++] = cloud.colors[*it];
        }
    }
}

void PointCloud::clear() {
    points.clear();
    normals.clear();
    colors.clear();
}
