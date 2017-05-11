#include <cilantro/pca.hpp>

PCA::PCA(const PointCloud &pc)
        : dim_(3),
          num_points_(pc.size()),
          data_((float *)pc.points.data())
{
    compute_();
}

PCA::PCA(const std::vector<Eigen::Vector3f> &points)
        : dim_(3),
          num_points_(points.size()),
          data_((float *)points.data())
{
    compute_();
}

PCA::PCA(float * data, size_t dim, size_t num_points)
        : dim_(dim),
          num_points_(num_points),
          data_(data)
{
    compute_();
}

void PCA::compute_() {
    mean_ = Eigen::Map<Eigen::MatrixXf>(data_, dim_, num_points_).rowwise().mean();
    Eigen::MatrixXf centered = Eigen::Map<Eigen::MatrixXf>(data_, dim_, num_points_).colwise() - mean_;

    Eigen::JacobiSVD<Eigen::MatrixXf> svd(centered, Eigen::ComputeFullU | Eigen::ComputeThinV);
    eigenvectors_ = svd.matrixU();
    if (eigenvectors_.determinant() < 0.0f) {
        eigenvectors_.col(dim_-1) *= -1.0f;
    }
    eigenvalues_ = svd.singularValues().array().square();
}
