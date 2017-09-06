#include <cilantro/point_cloud.hpp>
#include <set>

PointCloud::PointCloud() {}

PointCloud::PointCloud(const std::vector<Eigen::Vector3f> &points)
        : points(points)
{}

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

PointCloud& PointCloud::clear() {
    points.clear();
    normals.clear();
    colors.clear();
    return *this;
}

#include <iostream>

PointCloud& PointCloud::append(const PointCloud &cloud) {
    size_t original_size = size();
    points.insert(points.end(), cloud.points.begin(), cloud.points.end());
    if (normals.size() == original_size && cloud.hasNormals()) {
        normals.insert(normals.end(), cloud.normals.begin(), cloud.normals.end());
    }
    if (colors.size() == original_size && cloud.hasColors()) {
        colors.insert(colors.end(), cloud.colors.begin(), cloud.colors.end());
    }
    return *this;
}

PointCloud& PointCloud::remove(const std::vector<size_t> &indices) {
    if (indices.empty()) return *this;

    std::set<size_t> indices_set(indices.begin(), indices.end());
    if (indices_set.size() >= size()) {
        clear();
        return *this;
    }

    size_t valid_ind = size() - 1;
    while (indices_set.find(valid_ind) != indices_set.end()) {
        valid_ind--;
    }

    auto ind_it = indices_set.begin();
    while (ind_it != indices_set.end() && *ind_it < valid_ind) {
        std::swap(points[*ind_it], points[valid_ind]);
        if (hasNormals()) {
            std::swap(normals[*ind_it], normals[valid_ind]);
        }
        if (hasColors()) {
            std::swap(colors[*ind_it], colors[valid_ind]);
        }
        valid_ind--;
        while (*ind_it < valid_ind && indices_set.find(valid_ind) != indices_set.end()) {
            valid_ind--;
        }
        ++ind_it;
    }

    size_t original_size = size();
    size_t new_size = valid_ind + 1;
    points.resize(new_size);
    if (normals.size() == original_size) {
        normals.resize(new_size);
    }
    if (colors.size() == original_size) {
        colors.resize(new_size);
    }

    return *this;
}
