#include <sisyphus/normal_estimation.hpp>
#include <sisyphus/pca.hpp>

NormalEstimation::NormalEstimation(const PointCloud &cloud)
        : input_cloud_((PointCloud&)cloud),
          kd_tree_ptr_(new KDTree(cloud)),
          kd_tree_owned_(true),
          view_point_(Eigen::Vector3f::Zero())
{
}

NormalEstimation::NormalEstimation(const PointCloud &cloud, const KDTree &kd_tree)
        : input_cloud_((PointCloud&)cloud),
          kd_tree_ptr_((KDTree*)&kd_tree),
          kd_tree_owned_(false),
          view_point_(Eigen::Vector3f::Zero())
{
}

NormalEstimation::~NormalEstimation() {
    if (kd_tree_owned_) delete kd_tree_ptr_;
}

void NormalEstimation::computeNormalsKNN(std::vector<Eigen::Vector3f> &normals, size_t num_neighbors) const {

    Eigen::Vector3f nan(std::numeric_limits<float>::quiet_NaN(), std::numeric_limits<float>::quiet_NaN(), std::numeric_limits<float>::quiet_NaN());
    if (input_cloud_.size() < 3) {
        normals.resize(input_cloud_.size());
        for (size_t i = 0; i < normals.size(); i++) {
            normals[i] = nan;
        }
        return;
    }

    normals.resize(input_cloud_.size());
    for (size_t i = 0; i < input_cloud_.size(); i++) {
        std::vector<size_t> neighbors;
        std::vector<float> distances;
        kd_tree_ptr_->kNearestNeighbors(input_cloud_.points[i], num_neighbors, neighbors, distances);

        std::vector<Eigen::Vector3f> neighborhood(neighbors.size());
        for (size_t j = 0; j < neighbors.size(); j++) {
            neighborhood[j] = input_cloud_.points[neighbors[j]];
        }

        PCA pca(neighborhood);
        normals[i] = pca.getEigenVectors().col(2);

        if (normals[i].dot(view_point_ - input_cloud_.points[i]) < 0.0f) {
            normals[i] *= -1.0f;
        }
    }
}

void NormalEstimation::computeNormalsKNN(PointCloud &cloud, size_t num_neighbors) const {
    computeNormalsKNN(cloud.normals, num_neighbors);
}

void NormalEstimation::computeNormalsKNN(size_t num_neighbors) const {
    computeNormalsKNN(input_cloud_.normals, num_neighbors);
}

void NormalEstimation::computeNormalsRadius(std::vector<Eigen::Vector3f> &normals, float radius) const {

    Eigen::Vector3f nan(std::numeric_limits<float>::quiet_NaN(), std::numeric_limits<float>::quiet_NaN(), std::numeric_limits<float>::quiet_NaN());

    normals.resize(input_cloud_.size());
    for (size_t i = 0; i < input_cloud_.size(); i++) {
        std::vector<size_t> neighbors;
        std::vector<float> distances;
        kd_tree_ptr_->nearestNeighborsInRadius(input_cloud_.points[i], radius, neighbors, distances);

        if (neighbors.size() < 3) {
            normals[i] = nan;
            continue;
        }

        std::vector<Eigen::Vector3f> neighborhood(neighbors.size());
        for (size_t j = 0; j < neighbors.size(); j++) {
            neighborhood[j] = input_cloud_.points[neighbors[j]];
        }

        PCA pca(neighborhood);
        normals[i] = pca.getEigenVectors().col(2);

        if (normals[i].dot(view_point_ - input_cloud_.points[i]) < 0.0f) {
            normals[i] *= -1.0f;
        }
    }
}

void NormalEstimation::computeNormalsRadius(PointCloud &cloud, float radius) const {
    computeNormalsRadius(cloud.normals, radius);
}

void NormalEstimation::computeNormalsRadius(float radius) const {
    computeNormalsRadius(input_cloud_.normals, radius);
}
