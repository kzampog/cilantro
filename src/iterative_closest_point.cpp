#include <cilantro/iterative_closest_point.hpp>

bool estimateRigidTransformPointToPoint(const Eigen::Matrix<float,3,Eigen::Dynamic> &dst,
                                        const Eigen::Matrix<float,3,Eigen::Dynamic> &src,
                                        Eigen::Matrix3f &rot_mat,
                                        Eigen::Vector3f &t_vec)
{
    if (src.cols() < 3 || src.cols() != dst.cols()) {
        return false;
    }

    Eigen::Vector3f mu_dst(dst.rowwise().mean());
    Eigen::Vector3f mu_src(src.rowwise().mean());

    Eigen::Matrix<float,3,Eigen::Dynamic> dst_centered(dst.colwise() - mu_dst);
    Eigen::Matrix<float,3,Eigen::Dynamic> src_centered(src.colwise() - mu_src);

    Eigen::Matrix3f cov = dst_centered*(src_centered.transpose())/src.cols();

    Eigen::JacobiSVD<Eigen::Matrix3f> svd(cov, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Matrix3f U(svd.matrixU());
    Eigen::Matrix3f Vt(svd.matrixV().transpose());
    Eigen::Matrix3f tmp(U * Vt);
    if (tmp.determinant() < 0) {
        Eigen::Matrix3f S(Eigen::Matrix3f::Identity());
        S(2, 2) = -1;
        rot_mat = U * S * Vt;
    } else {
        rot_mat = tmp;
    }

    t_vec = mu_dst - rot_mat*mu_src;

    return true;
}
