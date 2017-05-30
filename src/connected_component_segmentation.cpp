#include <cilantro/connected_component_segmentation.hpp>
#include <stack>

#include <iostream>

ConnectedComponentSegmentation::ConnectedComponentSegmentation(const std::vector<Eigen::Vector3f> &points, const std::vector<Eigen::Vector3f> &normals, const std::vector<Eigen::Vector3f> &colors)
        : points_(&points),
          normals_((normals.size() == points.size()) ? &normals : NULL),
          colors_((colors.size() == points.size()) ? &colors : NULL),
          kd_tree_(new KDTree(points)),
          kd_tree_owned_(true)
{}

ConnectedComponentSegmentation::ConnectedComponentSegmentation(const std::vector<Eigen::Vector3f> &points, const std::vector<Eigen::Vector3f> &normals, const std::vector<Eigen::Vector3f> &colors, const KDTree &kd_tree)
        : points_(&points),
          normals_((normals.size() == points.size()) ? &normals : NULL),
          colors_((colors.size() == points.size()) ? &colors : NULL),
          kd_tree_((KDTree*)&kd_tree),
          kd_tree_owned_(false)
{}

ConnectedComponentSegmentation::ConnectedComponentSegmentation(const PointCloud &cloud)
        : points_(&cloud.points),
          normals_((cloud.normals.size() == cloud.points.size()) ? &cloud.normals : NULL),
          colors_((cloud.colors.size() == cloud.points.size()) ? &cloud.colors : NULL),
          kd_tree_(new KDTree(cloud.points)),
          kd_tree_owned_(true)
{}

ConnectedComponentSegmentation::ConnectedComponentSegmentation(const PointCloud &cloud, const KDTree &kd_tree)
        : points_(&cloud.points),
          normals_((cloud.normals.size() == cloud.points.size()) ? &cloud.normals : NULL),
          colors_((cloud.colors.size() == cloud.points.size()) ? &cloud.colors : NULL),
          kd_tree_((KDTree*)&kd_tree),
          kd_tree_owned_(false)
{}

ConnectedComponentSegmentation::~ConnectedComponentSegmentation() {
    if (kd_tree_owned_) delete kd_tree_;
}

ConnectedComponentSegmentation& ConnectedComponentSegmentation::segment(std::vector<size_t> seeds_ind,
                                                                        float dist_thresh,
                                                                        float normal_angle_thresh,
                                                                        float color_diff_thresh,
                                                                        size_t min_segment_size,
                                                                        size_t max_segment_size)
{
    dist_thresh_ = dist_thresh;
    normal_angle_thresh_ = normal_angle_thresh;
    color_diff_thresh_ = color_diff_thresh;
    min_segment_size_ = min_segment_size;
    max_segment_size_ = max_segment_size;

    std::vector<bool> has_been_assigned(points_->size(), 0);

    std::vector<size_t> neighbors;
    std::vector<float> distances;

    component_indices_.clear();
    for (size_t i = 0; i < seeds_ind.size(); i++) {
        if (has_been_assigned[seeds_ind[i]]) continue;

        std::vector<size_t> curr_cc_ind;
        std::vector<bool> is_in_stack(points_->size(), 0);

        std::stack<size_t> seeds_stack;
        seeds_stack.push(seeds_ind[i]);
        is_in_stack[seeds_ind[i]] = 1;
        while (!seeds_stack.empty()) {
            size_t curr_seed = seeds_stack.top();
            seeds_stack.pop();

            has_been_assigned[curr_seed] = 1;
            curr_cc_ind.push_back(curr_seed);

            kd_tree_->radiusSearch((*points_)[curr_seed], dist_thresh_, neighbors, distances);

            for (size_t j = 0; j < neighbors.size(); j++) {
                if (is_similar_(curr_seed,neighbors[j]) && !is_in_stack[neighbors[j]]) {
                    seeds_stack.push(neighbors[j]);
                    is_in_stack[neighbors[j]] = 1;
                }
            }

        }

        if (curr_cc_ind.size() >= min_segment_size_ && curr_cc_ind.size() <= max_segment_size_) {
            component_indices_.push_back(curr_cc_ind);
        }
    }

    std::sort(component_indices_.begin(), component_indices_.end(), vector_size_comparator_);

    label_map_ = std::vector<size_t>(points_->size(), component_indices_.size());
    for (size_t i = 0; i < component_indices_.size(); i++) {
        for (size_t j = 0; j < component_indices_[i].size(); j++) {
            label_map_[component_indices_[i][j]] = i;
        }
    }

    return *this;
}

ConnectedComponentSegmentation& ConnectedComponentSegmentation::segment(float dist_thresh,
                                                                        float normal_angle_thresh,
                                                                        float color_diff_thresh,
                                                                        size_t min_segment_size,
                                                                        size_t max_segment_size)
{
    std::vector<size_t> seeds_ind(points_->size());
    for (size_t i = 0; i < seeds_ind.size(); i++) {
        seeds_ind[i] = i;
    }
    return segment(seeds_ind, dist_thresh, normal_angle_thresh, color_diff_thresh, min_segment_size, max_segment_size);
}
