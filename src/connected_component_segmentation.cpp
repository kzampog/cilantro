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
                                                                        size_t max_segment_size,
                                                                        float min_neighbor_fraction)
{
    dist_thresh_ = dist_thresh;
    normal_angle_thresh_ = normal_angle_thresh;
    color_diff_thresh_ = color_diff_thresh;
    min_segment_size_ = min_segment_size;
    max_segment_size_ = max_segment_size;
    min_neighbor_fraction_ = std::max(0.0f, std::min(1.0f, min_neighbor_fraction));

    component_indices_.clear();
    size_t unassigned = points_->size()+1;
    label_map_ = std::vector<size_t>(points_->size(), unassigned);

    size_t curr_cc_id = 0;
    for (size_t i = 0; i < seeds_ind.size(); i++) {
        if (label_map_[seeds_ind[i]] != unassigned) continue;

        std::vector<size_t> curr_cc_ind;
        std::vector<char> is_in_stack(points_->size(), 0);

        std::stack<size_t> seeds_stack;
        seeds_stack.push(seeds_ind[i]);
        while (!seeds_stack.empty()) {
            size_t curr_seed = seeds_stack.top();
            seeds_stack.pop();

            std::vector<size_t> neighbors;
            std::vector<float> distances;
            kd_tree_->radiusSearch((*points_)[curr_seed], dist_thresh, neighbors, distances);

            size_t neighbors_in_cc = 0;
            size_t compatible_neighbors_in_cc = 0;
            for (size_t j = 0; j < neighbors.size(); j++) {
                if (neighbors[j] != curr_seed && label_map_[neighbors[j]] == curr_cc_id) {
                    neighbors_in_cc++;
                    if (is_similar_(curr_seed,neighbors[j])) {
                        compatible_neighbors_in_cc++;
                    }
                }
            }

            float compatible_neighbor_fraction = ((float)compatible_neighbors_in_cc)/((float)neighbors_in_cc);
            if (neighbors_in_cc == 0 ||
                compatible_neighbor_fraction >= min_neighbor_fraction_ && compatible_neighbors_in_cc > 0)
            {
                label_map_[curr_seed] = curr_cc_id;
                curr_cc_ind.push_back(curr_seed);
                for (size_t j = 0; j < neighbors.size(); j++) {
                    if (neighbors[j] != curr_seed && label_map_[neighbors[j]] == unassigned &&
                        is_in_stack[neighbors[j]] == 0 && is_similar_(curr_seed,neighbors[j]))
                    {
                        seeds_stack.push(neighbors[j]);
                        is_in_stack[neighbors[j]] = 1;
                    }
                }
            }
        }

        if (curr_cc_ind.size() >= min_segment_size_ && curr_cc_ind.size() <= max_segment_size_) {
            component_indices_.push_back(curr_cc_ind);
            curr_cc_id++;
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
                                                                        size_t max_segment_size,
                                                                        float min_neighbor_fraction)
{
    std::vector<size_t> seeds_ind(points_->size());
    for (size_t i = 0; i < seeds_ind.size(); i++) {
        seeds_ind[i] = i;
    }
    return segment(seeds_ind, dist_thresh, normal_angle_thresh, color_diff_thresh, min_segment_size, max_segment_size, min_neighbor_fraction);
}
