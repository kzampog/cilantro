#pragma once

#include <cilantro/kd_tree.hpp>

class ConnectedComponentSegmentation {
public:
    ConnectedComponentSegmentation(const std::vector<Eigen::Vector3f> &points, const std::vector<Eigen::Vector3f> &normals, const std::vector<Eigen::Vector3f> &colors);
    ConnectedComponentSegmentation(const std::vector<Eigen::Vector3f> &points, const std::vector<Eigen::Vector3f> &normals, const std::vector<Eigen::Vector3f> &colors, const KDTree &kd_tree);
    ConnectedComponentSegmentation(const PointCloud &cloud);
    ConnectedComponentSegmentation(const PointCloud &cloud, const KDTree &kd_tree);
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
    inline const std::vector<size_t>& getLabelMap() const { return label_map_; }

private:
    const std::vector<Eigen::Vector3f> *points_;
    const std::vector<Eigen::Vector3f> *normals_;
    const std::vector<Eigen::Vector3f> *colors_;
    KDTree *kd_tree_;
    bool kd_tree_owned_;

    float dist_thresh_;
    float normal_angle_thresh_;
    float color_diff_thresh_;
    size_t min_segment_size_;
    size_t max_segment_size_;

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
        if (colors_ != NULL && ((*colors_)[i]-(*colors_)[j]).norm() > color_diff_thresh_) return false;
        return true;
    }

    static inline bool vector_size_comparator_(const std::vector<size_t> &a, const std::vector<size_t> &b) { return a.size() > b.size(); }
};
