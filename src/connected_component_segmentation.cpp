#include <cilantro/connected_component_segmentation.hpp>
#include <set>

ConnectedComponentSegmentation::ConnectedComponentSegmentation(const std::vector<Eigen::Vector3f> &points, const std::vector<Eigen::Vector3f> &normals, const std::vector<Eigen::Vector3f> &colors)
        : points_(&points),
          normals_((normals.size() == points.size()) ? &normals : NULL),
          colors_((colors.size() == points.size()) ? &colors : NULL),
          kd_tree_(new KDTree3D(points)),
          kd_tree_owned_(true)
{}

ConnectedComponentSegmentation::ConnectedComponentSegmentation(const std::vector<Eigen::Vector3f> &points, const std::vector<Eigen::Vector3f> &normals, const std::vector<Eigen::Vector3f> &colors, const KDTree3D &kd_tree)
        : points_(&points),
          normals_((normals.size() == points.size()) ? &normals : NULL),
          colors_((colors.size() == points.size()) ? &colors : NULL),
          kd_tree_((KDTree3D*)&kd_tree),
          kd_tree_owned_(false)
{}

ConnectedComponentSegmentation::ConnectedComponentSegmentation(const PointCloud &cloud)
        : points_(&cloud.points),
          normals_((cloud.normals.size() == cloud.points.size()) ? &cloud.normals : NULL),
          colors_((cloud.colors.size() == cloud.points.size()) ? &cloud.colors : NULL),
          kd_tree_(new KDTree3D(cloud.points)),
          kd_tree_owned_(true)
{}

ConnectedComponentSegmentation::ConnectedComponentSegmentation(const PointCloud &cloud, const KDTree3D &kd_tree)
        : points_(&cloud.points),
          normals_((cloud.normals.size() == cloud.points.size()) ? &cloud.normals : NULL),
          colors_((cloud.colors.size() == cloud.points.size()) ? &cloud.colors : NULL),
          kd_tree_((KDTree3D*)&kd_tree),
          kd_tree_owned_(false)
{}

ConnectedComponentSegmentation::~ConnectedComponentSegmentation() {
    if (kd_tree_owned_) delete kd_tree_;
}

std::vector<size_t> ConnectedComponentSegmentation::getUnlabeledPointIndices() const {
    std::vector<size_t> res;
    res.reserve(label_map_.size());
    size_t no_label = component_indices_.size();
    for (size_t i = 0; i < label_map_.size(); i++) {
        if (label_map_[i] == no_label) res.emplace_back(i);
    }
    return res;
}

ConnectedComponentSegmentation& ConnectedComponentSegmentation::segment(std::vector<size_t> seeds_ind,
                                                                        float dist_thresh,
                                                                        float normal_angle_thresh,
                                                                        float color_diff_thresh,
                                                                        size_t min_segment_size,
                                                                        size_t max_segment_size)
{
    float radius_sq = dist_thresh*dist_thresh;

    normal_angle_thresh_ = normal_angle_thresh;
    color_diff_thresh_sq_ = color_diff_thresh*color_diff_thresh;

    const size_t unassigned = std::numeric_limits<size_t>::max();
    std::vector<size_t> current_label(points_->size(), unassigned);

    std::vector<size_t> frontier_set;
    frontier_set.reserve(points_->size());

//    std::vector<std::set<size_t> > ind_per_seed(seeds_ind.size());
    std::vector<std::vector<size_t> > ind_per_seed(seeds_ind.size());
    std::vector<std::set<size_t> > seeds_to_merge_with(seeds_ind.size());

    std::vector<size_t> neighbors;
    std::vector<float> distances;

#pragma omp parallel for shared (seeds_ind, current_label, ind_per_seed, seeds_to_merge_with) private (neighbors, distances, frontier_set)
    for (size_t i = 0; i < seeds_ind.size(); i++) {
        if (current_label[seeds_ind[i]] != unassigned) continue;

        seeds_to_merge_with[i].insert(i);

        frontier_set.clear();
        frontier_set.emplace_back(seeds_ind[i]);

        current_label[seeds_ind[i]] = i;

        while (!frontier_set.empty()) {
            size_t curr_seed = frontier_set[frontier_set.size()-1];
            frontier_set.resize(frontier_set.size()-1);

//            ind_per_seed[i].insert(curr_seed);
            ind_per_seed[i].emplace_back(curr_seed);

            kd_tree_->radiusSearch((*points_)[curr_seed], radius_sq, neighbors, distances);
            for (size_t j = 1; j < neighbors.size(); j++) {
                const size_t& curr_lbl = current_label[neighbors[j]];
                if (curr_lbl == i || is_similar_(curr_seed, neighbors[j])) {
                    if (curr_lbl == unassigned) {
                        frontier_set.emplace_back(neighbors[j]);
                        current_label[neighbors[j]] = i;
                    } else {
                        if (curr_lbl != i) seeds_to_merge_with[i].insert(curr_lbl);
                    }
                }
            }
        }

    }

    for (size_t i = 0; i < seeds_to_merge_with.size(); i++) {
        for (auto it = seeds_to_merge_with[i].begin(); it != seeds_to_merge_with[i].end(); ++it) {
            if (*it > i) seeds_to_merge_with[*it].insert(i);
        }
    }

    component_indices_.clear();
    for (size_t i = seeds_to_merge_with.size()-1; i < -1; i--) {
        if (seeds_to_merge_with[i].empty()) continue;
        size_t min_seed_ind = *seeds_to_merge_with[i].begin();
        if (min_seed_ind < i) {
            for (auto it = seeds_to_merge_with[i].begin(); it != seeds_to_merge_with[i].end(); ++it) {
                if (*it < i) seeds_to_merge_with[*it].insert(seeds_to_merge_with[i].begin(), seeds_to_merge_with[i].end());
            }
//            seeds_to_merge_with[i].clear();
        } else {
            std::set<size_t> curr_cc_ind;
            for (auto it = seeds_to_merge_with[i].begin(); it != seeds_to_merge_with[i].end(); ++it) {
                curr_cc_ind.insert(ind_per_seed[*it].begin(), ind_per_seed[*it].end());
            }
            if (curr_cc_ind.size() >= min_segment_size && curr_cc_ind.size() <= max_segment_size) {
                component_indices_.emplace_back(curr_cc_ind.begin(), curr_cc_ind.end());
            }
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
