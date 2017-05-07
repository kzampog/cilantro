#include <sisyphus/pca.hpp>

PCA::PCA(const PointCloud &pc)
        : num_points_(pc.points.size()),
          data_((float *)pc.points.data())
{
    compute_();
}

PCA::PCA(const std::vector<Eigen::Vector3f> &points)
        : num_points_(points.size()),
          data_((float *)points.data())
{
    compute_();
}

PCA::PCA(float * data, size_t num_points)
        : num_points_(num_points),
          data_(data)
{
    compute_();
}

void PCA::compute_() {
    mean_ = Eigen::Map<Eigen::MatrixXf>(data_, 3, num_points_).rowwise().mean();
    Eigen::MatrixXf centered = Eigen::Map<Eigen::MatrixXf>(data_, 3, num_points_).colwise() - mean_;

    Eigen::JacobiSVD<Eigen::Matrix<float, 3, Eigen::Dynamic> > svd(centered, Eigen::ComputeFullU | Eigen::ComputeThinV);
    eigenvectors_ = svd.matrixU();
    eigenvectors_.col(2) = eigenvectors_.col(0).cross(eigenvectors_.col(1));
    eigenvalues_ = svd.singularValues().array().square();
}
