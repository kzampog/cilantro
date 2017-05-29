#include <cilantro/connected_component_segmentation.hpp>
#include <stack>

ConnectedComponentSegmentation::ConnectedComponentSegmentation(const std::vector<Eigen::Vector3f> &points, const std::vector<Eigen::Vector3f> &normals, const std::vector<Eigen::Vector3f> &colors)
        : points_(&points),
          normals_((normals.size() == points.size()) ? &normals : NULL),
          colors_((colors.size() == points.size()) ? &colors : NULL),
          kd_tree_ptr_(new KDTree(points)),
          kd_tree_owned_(true)
{}

ConnectedComponentSegmentation::ConnectedComponentSegmentation(const std::vector<Eigen::Vector3f> &points, const std::vector<Eigen::Vector3f> &normals, const std::vector<Eigen::Vector3f> &colors, const KDTree &kd_tree)
        : points_(&points),
          normals_((normals.size() == points.size()) ? &normals : NULL),
          colors_((colors.size() == points.size()) ? &colors : NULL),
          kd_tree_ptr_((KDTree*)&kd_tree),
          kd_tree_owned_(false)
{}

ConnectedComponentSegmentation::ConnectedComponentSegmentation(const PointCloud &cloud)
        : points_(&cloud.points),
          normals_((cloud.normals.size() == cloud.points.size()) ? &cloud.normals : NULL),
          colors_((cloud.colors.size() == cloud.points.size()) ? &cloud.colors : NULL),
          kd_tree_ptr_(new KDTree(cloud.points)),
          kd_tree_owned_(true)
{}

ConnectedComponentSegmentation::ConnectedComponentSegmentation(const PointCloud &cloud, const KDTree &kd_tree)
        : points_(&cloud.points),
          normals_((cloud.normals.size() == cloud.points.size()) ? &cloud.normals : NULL),
          colors_((cloud.colors.size() == cloud.points.size()) ? &cloud.colors : NULL),
          kd_tree_ptr_((KDTree*)&kd_tree),
          kd_tree_owned_(false)
{}

ConnectedComponentSegmentation::~ConnectedComponentSegmentation() {
    if (kd_tree_owned_) delete kd_tree_ptr_;
}

ConnectedComponentSegmentation& ConnectedComponentSegmentation::segment(std::vector<size_t> seeds_ind,
                                                                        float dist_thresh,
                                                                        float normal_angle_thresh,
                                                                        float color_diff_thresh,
                                                                        float min_neighbor_fraction)
{
    // TODO
    std::vector<char> visited(points_->size(), 0);
    component_indices_.clear();
    size_t curr_cc = 0;

    for (size_t s = 0; s < seeds_ind.size(); s++) {
        size_t si = seeds_ind[s];
        if (visited[si]) continue;

        // Stack!

        // Get all neighbors
        // Get visited and unvisited neighbors
        // Compute fraction of visited that comply with current seed
        // Add unvisited to stack

        // Mark current seed as visited

    }


    return *this;
}

ConnectedComponentSegmentation& ConnectedComponentSegmentation::segment(float dist_thresh,
                                                                        float normal_angle_thresh,
                                                                        float color_diff_thresh,
                                                                        float min_neighbor_fraction)
{
    std::vector<size_t> seeds_ind(points_->size());
    for (size_t i = 0; i < seeds_ind.size(); i++) {
        seeds_ind[i] = i;
    }
    return segment(seeds_ind, dist_thresh, normal_angle_thresh, color_diff_thresh, min_neighbor_fraction);
}

std::vector<std::vector<size_t> > ConnectedComponentSegmentation::getComponentInSizeRangePointIndices(size_t min_segment_size, size_t max_segment_size) const {
    std::vector<std::vector<size_t> > res;
    res.reserve(component_indices_.size());
    for (size_t i = 0; i < component_indices_.size(); i++) {
        if (component_indices_[i].size() >= min_segment_size && component_indices_.size() <= max_segment_size) {
            res.push_back(component_indices_[i]);
        }
    }
    return res;
}
