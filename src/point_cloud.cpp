#include <cilantro/point_cloud.hpp>
#include <set>

namespace cilantro {
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

        bool has_normals = hasNormals();
        bool has_colors = hasColors();

        auto ind_it = indices_set.begin();
        while (ind_it != indices_set.end() && *ind_it < valid_ind) {
            std::swap(points[*ind_it], points[valid_ind]);
            if (has_normals) {
                std::swap(normals[*ind_it], normals[valid_ind]);
            }
            if (has_colors) {
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

    PointCloud& PointCloud::removeInvalidPoints() {
        std::vector<size_t> ind_to_remove;
        ind_to_remove.reserve(points.size());
        for (size_t i = 0; i < points.size(); i++) {
            if (!points[i].allFinite()) ind_to_remove.emplace_back(i);
        }

        return remove(ind_to_remove);
    }

    PointCloud& PointCloud::removeInvalidNormals() {
        if (!hasNormals()) return *this;

        std::vector<size_t> ind_to_remove;
        ind_to_remove.reserve(normals.size());
        for (size_t i = 0; i < normals.size(); i++) {
            if (!normals[i].allFinite()) ind_to_remove.emplace_back(i);
        }

        return remove(ind_to_remove);
    }

    PointCloud& PointCloud::removeInvalidColors() {
        if (!hasColors()) return *this;

        std::vector<size_t> ind_to_remove;
        ind_to_remove.reserve(colors.size());
        for (size_t i = 0; i < colors.size(); i++) {
            if (!colors[i].allFinite()) ind_to_remove.emplace_back(i);
        }

        return remove(ind_to_remove);
    }

    PointCloud& PointCloud::removeInvalidData() {
        bool has_normals = hasNormals();
        bool has_colors = hasColors();

        std::vector<size_t> ind_to_remove;
        ind_to_remove.reserve(points.size());
        for (size_t i = 0; i < points.size(); i++) {
            if (!points[i].allFinite()) {
                ind_to_remove.emplace_back(i);
            } else if (has_normals && !normals[i].allFinite()) {
                ind_to_remove.emplace_back(i);
            } else if (has_colors && (!colors[i].allFinite())) {
                ind_to_remove.emplace_back(i);
            }
        }

        return remove(ind_to_remove);
    }

    PointCloud& PointCloud::transform(const Eigen::Ref<const Eigen::Matrix3f> &rotation, const Eigen::Ref<const Eigen::Vector3f> &translation) {
        if (!empty()) {
            Eigen::Map<Eigen::Matrix<float,3,Eigen::Dynamic> > points_map(pointsMatrixMap());
            points_map = (rotation*points_map).colwise() + translation;
            if (hasNormals()) {
                Eigen::Map<Eigen::Matrix<float,3,Eigen::Dynamic> > normals_map(normalsMatrixMap());
                normals_map = rotation*normals_map;
            }
        }
        return *this;
    }

    PointCloud& PointCloud::transform(const Eigen::Ref<const Eigen::Matrix4f> &rigid_transform) {
        return transform(rigid_transform.topLeftCorner(3,3), rigid_transform.topRightCorner(3,1));
    }

    PointCloud PointCloud::transformed(const Eigen::Ref<const Eigen::Matrix3f> &rotation, const Eigen::Ref<const Eigen::Vector3f> &translation) {
        PointCloud cloud;
        cloud.points.resize(points.size());
        cloud.normals.resize(normals.size());
        cloud.colors = colors;
        if (!cloud.empty()) {
            cloud.pointsMatrixMap() = (rotation*pointsMatrixMap()).colwise() + translation;
            if (cloud.hasNormals()) {
                cloud.normalsMatrixMap() = rotation*normalsMatrixMap();
            }
        }
        return cloud;
    }

    PointCloud PointCloud::transformed(const Eigen::Ref<const Eigen::Matrix4f> &rigid_transform) {
        return transformed(rigid_transform.topLeftCorner(3,3), rigid_transform.topRightCorner(3,1));
    }
}
