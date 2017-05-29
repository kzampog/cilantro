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
                                            float min_neighbor_fraction = 0.5);
    ConnectedComponentSegmentation& segment(float dist_thresh,
                                            float normal_angle_thresh,
                                            float color_diff_thresh,
                                            float min_neighbor_fraction = 0.5);

    inline const std::vector<std::vector<size_t> >& getComponentPointIndices() const { return component_indices_; }
    inline const std::vector<size_t>& getPointLabelMap() const { return label_map_; }

    std::vector<std::vector<size_t> > getComponentInSizeRangePointIndices(size_t min_segment_size = 0, size_t max_segment_size = std::numeric_limits<size_t>::max()) const;

private:
    const std::vector<Eigen::Vector3f> *points_;
    const std::vector<Eigen::Vector3f> *normals_;
    const std::vector<Eigen::Vector3f> *colors_;
    KDTree *kd_tree_ptr_;
    bool kd_tree_owned_;

    std::vector<std::vector<size_t> > component_indices_;
    std::vector<size_t> label_map_;

};
