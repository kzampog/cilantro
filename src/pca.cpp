#include <sisyphus/pca.hpp>

PCA::PCA(const PointCloud &pc) {
	num_points_ = pc.points.size();
	data_ = (float *)pc.points.data();
	compute_();
}

PCA::PCA(const std::vector<Eigen::Vector3f> &points) {
	num_points_ = points.size();
	data_ = (float *)points.data();
	compute_();
}

void PCA::compute_() {
	mean_ = Eigen::Map<Eigen::MatrixXf>(data_, 3, num_points_).rowwise().mean();

}
