#include <cilantro/connected_component_segmentation.hpp>
//#include <stack>

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

ConnectedComponentSegmentation& ConnectedComponentSegmentation::segment(std::vector<size_t> seeds_ind,
                                                                        float dist_thresh,
                                                                        float normal_angle_thresh,
                                                                        float color_diff_thresh,
                                                                        size_t min_segment_size,
                                                                        size_t max_segment_size)
{
    float radius_sq = dist_thresh*dist_thresh;

    normal_angle_thresh_ = normal_angle_thresh;
    color_diff_thresh_ = color_diff_thresh;

    std::vector<char> has_been_assigned(points_->size(), false);

    std::vector<size_t> neighbors;
    std::vector<float> distances;

    std::vector<size_t> seeds_stack;
    seeds_stack.reserve(points_->size());

    std::vector<size_t> curr_cc_ind;
    curr_cc_ind.reserve(points_->size());

    component_indices_.clear();
    for (size_t i = 0; i < seeds_ind.size(); i++) {
        if (has_been_assigned[seeds_ind[i]]) continue;

        seeds_stack.clear();
        seeds_stack.emplace_back(seeds_ind[i]);

        has_been_assigned[seeds_ind[i]] = true;

        curr_cc_ind.clear();

        while (!seeds_stack.empty()) {
            size_t curr_seed = seeds_stack[seeds_stack.size()-1];
            seeds_stack.resize(seeds_stack.size()-1);

            curr_cc_ind.emplace_back(curr_seed);

            kd_tree_->radiusSearch((*points_)[curr_seed], radius_sq, neighbors, distances);
            for (size_t j = 1; j < neighbors.size(); j++) {
                if (!has_been_assigned[neighbors[j]] && is_similar_(curr_seed,neighbors[j])) {
                    seeds_stack.emplace_back(neighbors[j]);
                    has_been_assigned[neighbors[j]] = true;
                }
            }
        }

        if (curr_cc_ind.size() >= min_segment_size && curr_cc_ind.size() <= max_segment_size) {
            component_indices_.emplace_back(curr_cc_ind);
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
