#pragma once

#include <cilantro/data_containers.hpp>
#include <iterator>
#include <set>

namespace cilantro {
    template <typename ScalarT, ptrdiff_t EigenDim>
    class PointCloud {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        VectorSet<ScalarT,EigenDim> points;
        VectorSet<ScalarT,EigenDim> normals;
        VectorSet<float,3> colors;

        PointCloud() {}

        PointCloud(const ConstVectorSetMatrixMap<ScalarT,EigenDim> &points) : points(points) {}

        PointCloud(const ConstVectorSetMatrixMap<ScalarT,EigenDim> &points,
                   const ConstVectorSetMatrixMap<ScalarT,EigenDim> &normals,
                   const ConstVectorSetMatrixMap<float,3> &colors)
                : points(points), normals(normals), colors(colors)
        {}

        PointCloud(const PointCloud<ScalarT,EigenDim> &cloud,
                   const std::vector<size_t> &indices,
                   bool negate = false)
        {
            std::set<size_t> indices_set;
            if (negate) {
                std::vector<size_t> full_indices(cloud.size());
                for (size_t i = 0; i < cloud.size(); i++) full_indices[i] = i;
                std::set<size_t> indices_to_discard(indices.begin(), indices.end());
                std::set_difference(full_indices.begin(), full_indices.end(), indices_to_discard.begin(), indices_to_discard.end(), std::inserter(indices_set, indices_set.begin()));
            } else {
                indices_set = std::set<size_t>(indices.begin(), indices.end());
            }

            size_t k = 0;
            points.resize(cloud.points.rows(), indices_set.size());
            for (auto it = indices_set.begin(); it != indices_set.end(); ++it) {
                points.col(k++) = cloud.points.col(*it);
            }
            if (cloud.hasNormals()) {
                k = 0;
                normals.resize(cloud.normals.rows(), indices_set.size());
                for (auto it = indices_set.begin(); it != indices_set.end(); ++it) {
                    normals.col(k++) = cloud.normals.col(*it);
                }
            }
            if (cloud.hasColors()) {
                k = 0;
                colors.resize(3, indices_set.size());
                for (auto it = indices_set.begin(); it != indices_set.end(); ++it) {
                    colors.col(k++) = cloud.colors.col(*it);
                }
            }
        }

        inline size_t size() const { return points.cols(); }

        inline bool hasNormals() const { return points.cols() > 0 && normals.cols() == points.cols(); }

        inline bool hasColors() const { return points.cols() > 0 && colors.cols() == points.cols(); }

        inline bool isEmpty() const { return points.cols() == 0; }

        PointCloud& clear() {
            points.resize(Eigen::NoChange, 0);
            normals.resize(Eigen::NoChange, 0);
            colors.resize(Eigen::NoChange, 0);
            return *this;
        }

        PointCloud& append(const PointCloud<ScalarT,EigenDim> &cloud) {
            size_t original_size = size();

            points.conservativeResize(cloud.points.rows(), original_size + cloud.points.cols());
            points.rightCols(cloud.points.cols()) = cloud.points;
            if (normals.cols() == original_size && cloud.hasNormals()) {
                normals.conservativeResize(cloud.normals.rows(), original_size + cloud.normals.cols());
                normals.rightCols(cloud.normals.cols()) = cloud.normals;
            }
            if (colors.cols() == original_size && cloud.hasColors()) {
                colors.conservativeResize(3, original_size + cloud.colors.cols());
                colors.rightCols(cloud.colors.cols()) = cloud.colors;
            }
            return *this;
        }

        PointCloud& remove(const std::vector<size_t> &indices) {
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
                points.col(*ind_it).swap(points.col(valid_ind));
                if (has_normals) {
                    normals.col(*ind_it).swap(normals.col(valid_ind));
                }
                if (has_colors) {
                    colors.col(*ind_it).swap(colors.col(valid_ind));
                }
                valid_ind--;
                while (*ind_it < valid_ind && indices_set.find(valid_ind) != indices_set.end()) {
                    valid_ind--;
                }
                ++ind_it;
            }

            size_t original_size = size();
            size_t new_size = valid_ind + 1;
            points.conservativeResize(Eigen::NoChange, new_size);
            if (normals.cols() == original_size) {
                normals.conservativeResize(Eigen::NoChange, new_size);
            }
            if (colors.cols() == original_size) {
                colors.conservativeResize(Eigen::NoChange, new_size);
            }

            return *this;
        }

        PointCloud& removeInvalidPoints() {
            std::vector<size_t> ind_to_remove;
            ind_to_remove.reserve(points.cols());
            for (size_t i = 0; i < points.cols(); i++) {
                if (!points.col(i).allFinite()) ind_to_remove.emplace_back(i);
            }
            return remove(ind_to_remove);
        }

        PointCloud& removeInvalidNormals() {
            if (!hasNormals()) return *this;

            std::vector<size_t> ind_to_remove;
            ind_to_remove.reserve(normals.cols());
            for (size_t i = 0; i < normals.cols(); i++) {
                if (!normals.col(i).allFinite()) ind_to_remove.emplace_back(i);
            }
            return remove(ind_to_remove);
        }

        PointCloud& removeInvalidColors() {
            if (!hasColors()) return *this;

            std::vector<size_t> ind_to_remove;
            ind_to_remove.reserve(colors.cols());
            for (size_t i = 0; i < colors.cols(); i++) {
                if (!colors.col(i).allFinite()) ind_to_remove.emplace_back(i);
            }
            return remove(ind_to_remove);
        }

        PointCloud& removeInvalidData() {
            bool has_normals = hasNormals();
            bool has_colors = hasColors();

            std::vector<size_t> ind_to_remove;
            ind_to_remove.reserve(points.cols());
            for (size_t i = 0; i < points.cols(); i++) {
                if (!points.col(i).allFinite()) {
                    ind_to_remove.emplace_back(i);
                } else if (has_normals && !normals.col(i).allFinite()) {
                    ind_to_remove.emplace_back(i);
                } else if (has_colors && !colors.col(i).allFinite()) {
                    ind_to_remove.emplace_back(i);
                }
            }
            return remove(ind_to_remove);
        }

        PointCloud& transform(const Eigen::Ref<const Eigen::Matrix3f> &rotation, const Eigen::Ref<const Eigen::Vector3f> &translation) {
            points = (rotation*points).colwise() + translation;
            if (hasNormals()) normals = rotation*normals;
            return *this;
        }

        PointCloud& transform(const Eigen::Ref<const Eigen::Matrix4f> &rigid_transform) {
            return transform(rigid_transform.topLeftCorner(3,3), rigid_transform.topRightCorner(3,1));
        }

        PointCloud transformed(const Eigen::Ref<const Eigen::Matrix3f> &rotation, const Eigen::Ref<const Eigen::Vector3f> &translation) const {
            PointCloud cloud;
            cloud.points = (rotation*points).colwise() + translation;
            if (hasNormals()) cloud.normals = rotation*normals;
            cloud.colors = colors;
            return cloud;
        }

        PointCloud transformed(const Eigen::Ref<const Eigen::Matrix4f> &rigid_transform) const {
            return transformed(rigid_transform.topLeftCorner(3,3), rigid_transform.topRightCorner(3,1));
        }
    };

    typedef PointCloud<float,2> PointCloud2D;
    typedef PointCloud<float,3> PointCloud3D;
}
