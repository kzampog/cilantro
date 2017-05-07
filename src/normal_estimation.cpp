#include <sisyphus/normal_estimation.hpp>
#include <sisyphus/kd_tree.hpp>
#include <sisyphus/pca.hpp>

void estimatePointCloudNormalsKNN(PointCloud &cloud, size_t num_neighbors, const Eigen::Vector3f &view_point) {
    KDTree kd_tree(cloud);

    Eigen::Vector3f nan(std::numeric_limits<float>::quiet_NaN(), std::numeric_limits<float>::quiet_NaN(), std::numeric_limits<float>::quiet_NaN());

    cloud.normals.resize(cloud.points.size());
    for (size_t i = 0; i < cloud.points.size(); i++) {
        std::vector<size_t> neighbors;
        std::vector<float> distances;
        kd_tree.kNearestNeighbors(cloud.points[i], num_neighbors, neighbors, distances);

        if (neighbors.size() < 3) {
            cloud.normals[i] = nan;
            continue;
        }

        std::vector<Eigen::Vector3f> neighborhood(neighbors.size());
        for (size_t j = 0; j < neighbors.size(); j++) {
            neighborhood[j] = cloud.points[neighbors[j]];
        }

        PCA pca(neighborhood);
        cloud.normals[i] = pca.getEigenVectors().col(2);

        if (cloud.normals[i].dot(view_point - cloud.points[i]) < 0.0f) {
            cloud.normals[i] = -cloud.normals[i];
        }
    }
}

void estimatePointCloudNormalsRadius(PointCloud &cloud, float radius, const Eigen::Vector3f &view_point) {
    KDTree kd_tree(cloud);

    Eigen::Vector3f nan(std::numeric_limits<float>::quiet_NaN(), std::numeric_limits<float>::quiet_NaN(), std::numeric_limits<float>::quiet_NaN());

    cloud.normals.resize(cloud.points.size());
    for (size_t i = 0; i < cloud.points.size(); i++) {
        std::vector<size_t> neighbors;
        std::vector<float> distances;
        kd_tree.nearestNeighborsInRadius(cloud.points[i], radius, neighbors, distances);

        if (neighbors.size() < 3) {
            cloud.normals[i] = nan;
            continue;
        }

        std::vector<Eigen::Vector3f> neighborhood(neighbors.size());
        for (size_t j = 0; j < neighbors.size(); j++) {
            neighborhood[j] = cloud.points[neighbors[j]];
        }

        PCA pca(neighborhood);
        cloud.normals[i] = pca.getEigenVectors().col(2);

        if (cloud.normals[i].dot(view_point - cloud.points[i]) < 0.0f) {
            cloud.normals[i] = -cloud.normals[i];
        }
    }
}
