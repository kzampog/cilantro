#include <sisyphus/normal_estimation.hpp>
#include <sisyphus/pca.hpp>

NormalEstimation::NormalEstimation(const PointCloud &cloud)
        : input_cloud_(cloud),
          kd_tree_ptr_(new KDTree(cloud)),
          kd_tree_owned_(true),
          view_point_(Eigen::Vector3f::Zero())
{
}

NormalEstimation::NormalEstimation(const PointCloud &cloud, const KDTree &kd_tree)
        : input_cloud_(cloud),
          kd_tree_ptr_((KDTree*)&kd_tree),
          kd_tree_owned_(false),
          view_point_(Eigen::Vector3f::Zero())
{
}

NormalEstimation::~NormalEstimation() {
    if (kd_tree_owned_) delete kd_tree_ptr_;
}

void NormalEstimation::computeNormalsKNN(PointCloud &cloud, size_t num_neighbors) const {

    Eigen::Vector3f nan(std::numeric_limits<float>::quiet_NaN(), std::numeric_limits<float>::quiet_NaN(), std::numeric_limits<float>::quiet_NaN());
    if (input_cloud_.points.size() < 3) {
        cloud.normals.resize(input_cloud_.points.size());
        for (size_t i = 0; i < cloud.normals.size(); i++) {
            cloud.normals[i] = nan;
        }
        return;
    }

    cloud.normals.resize(input_cloud_.points.size());
    for (size_t i = 0; i < input_cloud_.points.size(); i++) {
        std::vector<size_t> neighbors;
        std::vector<float> distances;
        kd_tree_ptr_->kNearestNeighbors(input_cloud_.points[i], num_neighbors, neighbors, distances);

        std::vector<Eigen::Vector3f> neighborhood(neighbors.size());
        for (size_t j = 0; j < neighbors.size(); j++) {
            neighborhood[j] = input_cloud_.points[neighbors[j]];
        }

        PCA pca(neighborhood);
        cloud.normals[i] = pca.getEigenVectors().col(2);

        if (cloud.normals[i].dot(view_point_ - input_cloud_.points[i]) < 0.0f) {
            cloud.normals[i] = -cloud.normals[i];
        }
    }
}

void NormalEstimation::computeNormalsRadius(PointCloud &cloud, float radius) const {

    Eigen::Vector3f nan(std::numeric_limits<float>::quiet_NaN(), std::numeric_limits<float>::quiet_NaN(), std::numeric_limits<float>::quiet_NaN());

    cloud.normals.resize(input_cloud_.points.size());
    for (size_t i = 0; i < input_cloud_.points.size(); i++) {
        std::vector<size_t> neighbors;
        std::vector<float> distances;
        kd_tree_ptr_->nearestNeighborsInRadius(input_cloud_.points[i], radius, neighbors, distances);

        if (neighbors.size() < 3) {
            cloud.normals[i] = nan;
            continue;
        }

        std::vector<Eigen::Vector3f> neighborhood(neighbors.size());
        for (size_t j = 0; j < neighbors.size(); j++) {
            neighborhood[j] = input_cloud_.points[neighbors[j]];
        }

        PCA pca(neighborhood);
        cloud.normals[i] = pca.getEigenVectors().col(2);

        if (cloud.normals[i].dot(view_point_ - input_cloud_.points[i]) < 0.0f) {
            cloud.normals[i] = -cloud.normals[i];
        }
    }
}
