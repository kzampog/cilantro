#include <cilantro/transform_estimation_utilities.hpp>

namespace cilantro {
    bool estimateRigidTransformPointToPoint(const Eigen::Ref<const Eigen::Matrix<float,3,Eigen::Dynamic> > &dst,
                                            const Eigen::Ref<const Eigen::Matrix<float,3,Eigen::Dynamic> > &src,
                                            Eigen::Ref<Eigen::Matrix3f> rot_mat,
                                            Eigen::Ref<Eigen::Vector3f> t_vec)
    {
        if (src.cols() < 3 || src.cols() != dst.cols()) {
            rot_mat.setIdentity();
            t_vec.setZero();
            return false;
        }

        Eigen::Vector3f mu_dst(dst.rowwise().mean());
        Eigen::Vector3f mu_src(src.rowwise().mean());

        Eigen::Matrix<float,3,Eigen::Dynamic> dst_centered(dst.colwise() - mu_dst);
        Eigen::Matrix<float,3,Eigen::Dynamic> src_centered(src.colwise() - mu_src);

        Eigen::Matrix3f cov = dst_centered*(src_centered.transpose())/src.cols();

        Eigen::JacobiSVD<Eigen::Matrix3f> svd(cov, Eigen::ComputeFullU | Eigen::ComputeFullV);
        if (svd.matrixU().determinant() * svd.matrixV().determinant() < 0.0f) {
            Eigen::Matrix3f U(svd.matrixU());
            U.col(2) *= -1.0f;
            rot_mat = U*svd.matrixV().transpose();
        } else {
            rot_mat = svd.matrixU()*svd.matrixV().transpose();
        }
        t_vec = mu_dst - rot_mat*mu_src;

        return true;
    }

    bool estimateRigidTransformPointToPoint(const Eigen::Ref<const Eigen::Matrix<float,3,Eigen::Dynamic> > &dst,
                                            const Eigen::Ref<const Eigen::Matrix<float,3,Eigen::Dynamic> > &src,
                                            const std::vector<size_t> &dst_ind,
                                            const std::vector<size_t> &src_ind,
                                            Eigen::Ref<Eigen::Matrix3f> rot_mat,
                                            Eigen::Ref<Eigen::Vector3f> t_vec)
    {
        if (dst_ind.size() != src_ind.size()) {
            rot_mat.setIdentity();
            t_vec.setZero();
            return false;
        }

        Eigen::Matrix<float,3,Eigen::Dynamic> dst_corr(3, dst_ind.size());
        Eigen::Matrix<float,3,Eigen::Dynamic> src_corr(3, src_ind.size());
        for (size_t i = 0; i < dst_ind.size(); i++) {
            dst_corr.col(i) = dst.col(dst_ind[i]);
            src_corr.col(i) = src.col(src_ind[i]);
        }

        return estimateRigidTransformPointToPoint(dst_corr, src_corr, rot_mat, t_vec);
    }

    bool estimateRigidTransformPointToPlane(const Eigen::Ref<const Eigen::Matrix<float,3,Eigen::Dynamic> > &dst_p,
                                            const Eigen::Ref<const Eigen::Matrix<float,3,Eigen::Dynamic> > &dst_n,
                                            const Eigen::Ref<const Eigen::Matrix<float,3,Eigen::Dynamic> > &src_p,
                                            Eigen::Ref<Eigen::Matrix3f> rot_mat,
                                            Eigen::Ref<Eigen::Vector3f> t_vec,
                                            size_t max_iter,
                                            float convergence_tol)
    {
        if (src_p.cols() < 6 || src_p.cols() != dst_p.cols() || dst_p.cols() != dst_n.cols()) {
            rot_mat.setIdentity();
            t_vec.setZero();
            return false;
        }

        Eigen::Matrix3f rot_mat_iter;
        Eigen::Matrix<float,6,1> d_theta;
        rot_mat.setIdentity();
        t_vec.setZero();
        Eigen::Matrix<float,3,Eigen::Dynamic> src_t(src_p);

        Eigen::Matrix<float,Eigen::Dynamic,6> A(dst_p.cols(),6);
        Eigen::Matrix<float,Eigen::Dynamic,1> b(dst_p.cols(),1);

        size_t iter = 0;
        while (iter < max_iter) {
            if (iter > 0) {
                src_t = (rot_mat*src_p).colwise() + t_vec;
            }

            // Compute differential
            for (size_t i = 0; i < A.rows(); i++) {
                const Eigen::Vector3f& d = dst_p.col(i);
                const Eigen::Vector3f& n = dst_n.col(i);
                const Eigen::Vector3f& s = src_t.col(i);
                A(i,0) = n[2]*s[1] - n[1]*s[2];
                A(i,1) = n[0]*s[2] - n[2]*s[0];
                A(i,2) = n[1]*s[0] - n[0]*s[1];
                A(i,3) = n[0];
                A(i,4) = n[1];
                A(i,5) = n[2];
                b[i] = n[0]*d[0] + n[1]*d[1] + n[2]*d[2] - n[0]*s[0] - n[1]*s[1] - n[2]*s[2];
            }

            d_theta = (A.transpose()*A).ldlt().solve(A.transpose()*b);

            // Update estimate
            rot_mat_iter = Eigen::AngleAxisf(d_theta[2],Eigen::Vector3f::UnitZ()) *
                           Eigen::AngleAxisf(d_theta[1],Eigen::Vector3f::UnitY()) *
                           Eigen::AngleAxisf(d_theta[0],Eigen::Vector3f::UnitX());

            rot_mat = rot_mat_iter*rot_mat;
            t_vec = rot_mat_iter*t_vec + d_theta.tail(3);

            // Orthonormalize rotation
            Eigen::JacobiSVD<Eigen::Matrix3f> svd(rot_mat, Eigen::ComputeFullU | Eigen::ComputeFullV);
            if (svd.matrixU().determinant() * svd.matrixV().determinant() < 0.0f) {
                Eigen::Matrix3f U(svd.matrixU());
                U.col(2) *= -1.0f;
                rot_mat = U*svd.matrixV().transpose();
            } else {
                rot_mat = svd.matrixU()*svd.matrixV().transpose();
            }

            iter++;

            // Check for convergence
            if (d_theta.norm() < convergence_tol) return true;
        }

        return false;
    }

    bool estimateRigidTransformPointToPlane(const Eigen::Ref<const Eigen::Matrix<float,3,Eigen::Dynamic> > &dst_p,
                                            const Eigen::Ref<const Eigen::Matrix<float,3,Eigen::Dynamic> > &dst_n,
                                            const Eigen::Ref<const Eigen::Matrix<float,3,Eigen::Dynamic> > &src_p,
                                            const std::vector<size_t> &dst_ind,
                                            const std::vector<size_t> &src_ind,
                                            Eigen::Ref<Eigen::Matrix3f> rot_mat,
                                            Eigen::Ref<Eigen::Vector3f> t_vec,
                                            size_t max_iter,
                                            float convergence_tol)
    {
        if (dst_ind.size() != src_ind.size()) {
            rot_mat.setIdentity();
            t_vec.setZero();
            return false;
        }

        Eigen::Matrix<float,3,Eigen::Dynamic> dst_p_corr(3, dst_ind.size());
        Eigen::Matrix<float,3,Eigen::Dynamic> dst_n_corr(3, dst_ind.size());
        Eigen::Matrix<float,3,Eigen::Dynamic> src_p_corr(3, src_ind.size());
        for (size_t i = 0; i < dst_ind.size(); i++) {
            dst_p_corr.col(i) = dst_p.col(dst_ind[i]);
            dst_n_corr.col(i) = dst_n.col(dst_ind[i]);
            src_p_corr.col(i) = src_p.col(src_ind[i]);
        }

        return estimateRigidTransformPointToPlane(dst_p_corr, dst_n_corr, src_p_corr, rot_mat, t_vec, max_iter, convergence_tol);
    }
}
