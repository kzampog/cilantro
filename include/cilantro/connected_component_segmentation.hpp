#pragma once

#include <cilantro/kd_tree.hpp>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace cilantro {
    class ConnectedComponentSegmentation {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        ConnectedComponentSegmentation(const ConstVectorSetMatrixMap<float,3> &points,
                                       const ConstVectorSetMatrixMap<float,3> &normals,
                                       const ConstVectorSetMatrixMap<float,3> &colors);

        ConnectedComponentSegmentation(const ConstVectorSetMatrixMap<float,3> &points,
                                       const ConstVectorSetMatrixMap<float,3> &normals,
                                       const ConstVectorSetMatrixMap<float,3> &colors,
                                       const KDTree3f &kd_tree);

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

        inline const std::vector<std::vector<size_t>>& getComponentPointIndices() const { return component_indices_; }

        inline const std::vector<size_t>& getComponentIndexMap() const { return label_map_; }

        std::vector<size_t> getUnlabeledPointIndices() const;

        inline size_t getNumberOfSegments() const { return component_indices_.size(); }

    private:
        ConstVectorSetMatrixMap<float,3> points_;
        ConstVectorSetMatrixMap<float,3> normals_;
        ConstVectorSetMatrixMap<float,3> colors_;

        KDTree3f *kd_tree_;
        bool kd_tree_owned_;

        std::vector<std::vector<size_t>> component_indices_;
        std::vector<size_t> label_map_;

        float normal_angle_thresh_;
        float color_diff_thresh_sq_;

        inline bool is_similar_(size_t i, size_t j) const {
            if (normals_.cols() > 0) {
                const float angle = std::acos(normals_.col(i).dot(normals_.col(j)));
                if ((normal_angle_thresh_ >= 0.0f && angle > normal_angle_thresh_) ||
                    (normal_angle_thresh_ < 0.0f && std::min(angle,(float)M_PI-angle) > -normal_angle_thresh_)) return false;
            }
            if (colors_.cols() > 0 && (colors_.col(i) - colors_.col(j)).squaredNorm() > color_diff_thresh_sq_) return false;
            return true;
        }

        static inline bool vector_size_comparator_(const std::vector<size_t> &a, const std::vector<size_t> &b) { return a.size() > b.size(); }
    };
}
