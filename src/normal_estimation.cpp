#include <cilantro/normal_estimation.hpp>
#include <cilantro/principal_component_analysis.hpp>

NormalEstimation::NormalEstimation(const std::vector<Eigen::Vector3f> &points)
        : input_cloud_(NULL),
          input_points_(&points),
          kd_tree_ptr_(new KDTree(points)),
          kd_tree_owned_(true),
          view_point_(Eigen::Vector3f::Zero())
{}

NormalEstimation::NormalEstimation(const std::vector<Eigen::Vector3f> &points, const KDTree &kd_tree)
        : input_cloud_(NULL),
          input_points_(&points),
          kd_tree_ptr_((KDTree*)&kd_tree),
          kd_tree_owned_(false),
          view_point_(Eigen::Vector3f::Zero())
{}

NormalEstimation::NormalEstimation(const PointCloud &cloud)
        : input_cloud_((PointCloud *)&cloud),
          input_points_(&cloud.points),
          kd_tree_ptr_(new KDTree(cloud)),
          kd_tree_owned_(true),
          view_point_(Eigen::Vector3f::Zero())
{}

NormalEstimation::NormalEstimation(const PointCloud &cloud, const KDTree &kd_tree)
        : input_cloud_((PointCloud *)&cloud),
          input_points_(&cloud.points),
          kd_tree_ptr_((KDTree*)&kd_tree),
          kd_tree_owned_(false),
          view_point_(Eigen::Vector3f::Zero())
{}

NormalEstimation::~NormalEstimation() {
    if (kd_tree_owned_) delete kd_tree_ptr_;
}

std::vector<Eigen::Vector3f> NormalEstimation::estimateNormalsKNN(size_t num_neighbors) const {
    std::vector<Eigen::Vector3f> normals;

    Eigen::Vector3f nan(std::numeric_limits<float>::quiet_NaN(), std::numeric_limits<float>::quiet_NaN(), std::numeric_limits<float>::quiet_NaN());
    if (input_points_->size() < 3) {
        normals.resize(input_points_->size());
        for (size_t i = 0; i < normals.size(); i++) {
            normals[i] = nan;
        }
        return normals;
    }

    normals.resize(input_points_->size());
    for (size_t i = 0; i < input_points_->size(); i++) {
        std::vector<size_t> neighbors;
        std::vector<float> distances;
        kd_tree_ptr_->kNNSearch((*input_points_)[i], num_neighbors, neighbors, distances);

        std::vector<Eigen::Vector3f> neighborhood(neighbors.size());
        for (size_t j = 0; j < neighbors.size(); j++) {
            neighborhood[j] = (*input_points_)[neighbors[j]];
        }

        PrincipalComponentAnalysis3D pca(neighborhood);
        normals[i] = pca.getEigenVectorsMatrix().col(2);

        if (normals[i].dot(view_point_ - (*input_points_)[i]) < 0.0f) {
            normals[i] *= -1.0f;
        }
    }

    return normals;
}

void NormalEstimation::estimateNormalsInPlaceKNN(size_t num_neighbors) const {
    if (input_cloud_ == NULL) return;
    input_cloud_->normals = estimateNormalsKNN(num_neighbors);
}

std::vector<Eigen::Vector3f> NormalEstimation::estimateNormalsRadius(float radius) const {
    std::vector<Eigen::Vector3f> normals;

    Eigen::Vector3f nan(std::numeric_limits<float>::quiet_NaN(), std::numeric_limits<float>::quiet_NaN(), std::numeric_limits<float>::quiet_NaN());

    normals.resize(input_points_->size());
    for (size_t i = 0; i < input_points_->size(); i++) {
        std::vector<size_t> neighbors;
        std::vector<float> distances;
        kd_tree_ptr_->radiusSearch((*input_points_)[i], radius, neighbors, distances);

        if (neighbors.size() < 3) {
            normals[i] = nan;
            continue;
        }

        std::vector<Eigen::Vector3f> neighborhood(neighbors.size());
        for (size_t j = 0; j < neighbors.size(); j++) {
            neighborhood[j] = (*input_points_)[neighbors[j]];
        }

        PrincipalComponentAnalysis3D pca(neighborhood);
        normals[i] = pca.getEigenVectorsMatrix().col(2);

        if (normals[i].dot(view_point_ - (*input_points_)[i]) < 0.0f) {
            normals[i] *= -1.0f;
        }
    }

    return normals;
}

void NormalEstimation::estimateNormalsInPlaceRadius(float radius) const {
    if (input_cloud_ == NULL) return;
    input_cloud_->normals = estimateNormalsRadius(radius);
}

std::vector<Eigen::Vector3f> NormalEstimation::estimateNormalsKNNInRadius(size_t k, float radius) const {
    std::vector<Eigen::Vector3f> normals;

    Eigen::Vector3f nan(std::numeric_limits<float>::quiet_NaN(), std::numeric_limits<float>::quiet_NaN(), std::numeric_limits<float>::quiet_NaN());

    normals.resize(input_points_->size());
    for (size_t i = 0; i < input_points_->size(); i++) {
        std::vector<size_t> neighbors;
        std::vector<float> distances;
        kd_tree_ptr_->kNNInRadiusSearch((*input_points_)[i], k, radius, neighbors, distances);

        if (neighbors.size() < 3) {
            normals[i] = nan;
            continue;
        }

        std::vector<Eigen::Vector3f> neighborhood(neighbors.size());
        for (size_t j = 0; j < neighbors.size(); j++) {
            neighborhood[j] = (*input_points_)[neighbors[j]];
        }

        PrincipalComponentAnalysis3D pca(neighborhood);
        normals[i] = pca.getEigenVectorsMatrix().col(2);

        if (normals[i].dot(view_point_ - (*input_points_)[i]) < 0.0f) {
            normals[i] *= -1.0f;
        }
    }

    return normals;
}

void NormalEstimation::estimateNormalsInPlaceKNNInRadius(size_t k, float radius) const {
    if (input_cloud_ == NULL) return;
    input_cloud_->normals = estimateNormalsKNNInRadius(k, radius);
}

std::vector<Eigen::Vector3f> NormalEstimation::estimateNormals(const KDTree::Neighborhood &nh) const {
    if (nh.type == KDTree::NeighborhoodType::KNN) {
        return estimateNormalsKNN(nh.maxNumberOfNeighbors);
    } else if (nh.type == KDTree::NeighborhoodType::RADIUS) {
        return estimateNormalsRadius(nh.radius);
    } else {
        return estimateNormalsKNNInRadius(nh.maxNumberOfNeighbors, nh.radius);
    }
}

void NormalEstimation::estimateNormalsInPlace(const KDTree::Neighborhood &nh) const {
    if (input_cloud_ == NULL) return;
    input_cloud_->normals = estimateNormals(nh);
}
