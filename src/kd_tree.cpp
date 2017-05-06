#include <sisyphus/kd_tree.hpp>

KDTree::KDTree(const std::vector<Eigen::Vector3f> &points, size_t max_leaf_size)
        : pcd_to_kd_(VectorOfEigenVectorsAdaptor_(points)),
          kd_tree_(new KDTree::TreeType_(3, pcd_to_kd_, nanoflann::KDTreeSingleIndexAdaptorParams(max_leaf_size)))
{
    kd_tree_->buildIndex();
}

KDTree::KDTree(const PointCloud &cloud, size_t max_leaf_size)
        : pcd_to_kd_(VectorOfEigenVectorsAdaptor_(cloud.points)),
          kd_tree_(new KDTree::TreeType_(3, pcd_to_kd_, nanoflann::KDTreeSingleIndexAdaptorParams(max_leaf_size)))
{
    kd_tree_->buildIndex();
}

KDTree::~KDTree() {
    delete kd_tree_;
}

void KDTree::kNearestNeighbors(const Eigen::Vector3f &query_pt, size_t k, std::vector<size_t> &neighbors, std::vector<float> &distances) {
    neighbors.resize(k);
    distances.resize(k);

    size_t num_results = kd_tree_->knnSearch(query_pt.data(), k, neighbors.data(), distances.data());

    neighbors.resize(num_results);
    distances.resize(num_results);
}

void KDTree::nearestNeighborsInRadius(const Eigen::Vector3f &query_pt, float radius, std::vector<size_t> &neighbors, std::vector<float> &distances) {
    std::vector<std::pair<size_t,float> > matches;

    nanoflann::SearchParams params;
    params.sorted = true;
    size_t num_results = kd_tree_->radiusSearch(query_pt.data(), radius*radius, matches, params);

    neighbors.resize(num_results);
    distances.resize(num_results);
    for (size_t i = 0; i < num_results; i++) {
        neighbors[i] = matches[i].first;
        distances[i] = matches[i].second;
    }
}

void KDTree::kNearestNeighborsInRadius(const Eigen::Vector3f &query_pt, size_t k, float radius, std::vector<size_t> &neighbors, std::vector<float> &distances) {
    KDTree::nearestNeighborsInRadius(query_pt, radius, neighbors, distances);
    if (neighbors.size() > k) {
        neighbors.resize(k);
        distances.resize(k);
    }
}
