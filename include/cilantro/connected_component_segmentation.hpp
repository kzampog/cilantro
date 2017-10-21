#pragma once

#include <cilantro/kd_tree.hpp>
#include <cilantro/point_cloud.hpp>

namespace cilantro {
    class ConnectedComponentSegmentation {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        ConnectedComponentSegmentation(const std::vector<Eigen::Vector3f> &points, const std::vector<Eigen::Vector3f> &normals, const std::vector<Eigen::Vector3f> &colors);
        ConnectedComponentSegmentation(const std::vector<Eigen::Vector3f> &points, const std::vector<Eigen::Vector3f> &normals, const std::vector<Eigen::Vector3f> &colors, const KDTree3D &kd_tree);
        ConnectedComponentSegmentation(const PointCloud &cloud);
        ConnectedComponentSegmentation(const PointCloud &cloud, const KDTree3D &kd_tree);
        ~ConnectedComponentSegmentation();

        ConnectedComponentSegmentation& segment(std::vector<size_t> seeds_ind,
                                                float dist_thresh,
                                                float normal_angle_thresh,
                                                float color_diff_thresh,
                                                size_t min_segment_size = 0,
                                                size_t max_segment_size = std::numeric_limits<size_t>::max());
        ConnectedComponentSegmentation& segment(float dist_thresh,
                                                float normal_angle_thresh,
                                                float color_diff_thresh,
                                                size_t min_segment_size = 0,
                                                size_t max_segment_size = std::numeric_limits<size_t>::max());

        inline const std::vector<std::vector<size_t> >& getComponentPointIndices() const { return component_indices_; }
        inline const std::vector<size_t>& getComponentIndexMap() const { return label_map_; }
        std::vector<size_t> getUnlabeledPointIndices() const;
        inline size_t getNumberOfSegments() const { return component_indices_.size(); }

    private:
        const std::vector<Eigen::Vector3f> *points_;
        const std::vector<Eigen::Vector3f> *normals_;
        const std::vector<Eigen::Vector3f> *colors_;
        KDTree3D *kd_tree_;
        bool kd_tree_owned_;

        float normal_angle_thresh_;
        float color_diff_thresh_sq_;

        std::vector<std::vector<size_t> > component_indices_;
        std::vector<size_t> label_map_;

        inline bool is_similar_(size_t i, size_t j) {
            if (normals_ != NULL) {
                float angle = std::acos((*normals_)[i].dot((*normals_)[j]));
                if (normal_angle_thresh_ < 0.0f) {
                    if (std::min(angle,(float)M_PI-angle) > -normal_angle_thresh_) return false;
                } else {
                    if (angle > normal_angle_thresh_) return false;
                }
            }
            if (colors_ != NULL && ((*colors_)[i]-(*colors_)[j]).squaredNorm() > color_diff_thresh_sq_) return false;
            return true;
        }

        static inline bool vector_size_comparator_(const std::vector<size_t> &a, const std::vector<size_t> &b) { return a.size() > b.size(); }
    };
}
