#pragma once

#include <cilantro/data_containers.hpp>

namespace cilantro {
    // Point-to-point (closed form, SVD)
    template <typename ScalarT>
    bool estimateRigidTransformPointToPointClosedForm(const ConstPointSetMatrixMap<ScalarT,3> &dst,
                                                      const ConstPointSetMatrixMap<ScalarT,3> &src,
                                                      Eigen::Ref<Eigen::Matrix<ScalarT,3,3>> rot_mat,
                                                      Eigen::Ref<Eigen::Matrix<ScalarT,3,1>> t_vec)
    {
        if (src.cols() != dst.cols() || src.cols() < 3) {
            rot_mat.setIdentity();
            t_vec.setZero();
            return false;
        }

        Point<ScalarT,3> mu_dst(dst.rowwise().mean());
        Point<ScalarT,3> mu_src(src.rowwise().mean());

        PointSet<ScalarT,3> dst_centered(dst.colwise() - mu_dst);
        PointSet<ScalarT,3> src_centered(src.colwise() - mu_src);

        Eigen::Matrix<ScalarT,3,3> cov = dst_centered*(src_centered.transpose())/src.cols();

        Eigen::JacobiSVD<Eigen::Matrix<ScalarT,3,3>> svd(cov, Eigen::ComputeFullU | Eigen::ComputeFullV);
        if (svd.matrixU().determinant() * svd.matrixV().determinant() < 0.0) {
            Eigen::Matrix<ScalarT,3,3> U(svd.matrixU());
            U.col(2) *= -1.0;
            rot_mat = U*svd.matrixV().transpose();
        } else {
            rot_mat = svd.matrixU()*svd.matrixV().transpose();
        }
        t_vec = mu_dst - rot_mat*mu_src;

        return true;
    }

    template <typename ScalarT>
    bool estimateRigidTransformPointToPointClosedForm(const ConstPointSetMatrixMap<ScalarT,3> &dst,
                                                      const ConstPointSetMatrixMap<ScalarT,3> &src,
                                                      const std::vector<size_t> &dst_ind,
                                                      const std::vector<size_t> &src_ind,
                                                      Eigen::Ref<Eigen::Matrix<ScalarT,3,3>> rot_mat,
                                                      Eigen::Ref<Eigen::Matrix<ScalarT,3,1>> t_vec)
    {
        if (dst_ind.size() != src_ind.size()) {
            rot_mat.setIdentity();
            t_vec.setZero();
            return false;
        }

        PointSet<ScalarT,3> dst_corr(3, dst_ind.size());
        PointSet<ScalarT,3> src_corr(3, src_ind.size());
        for (size_t i = 0; i < dst_ind.size(); i++) {
            dst_corr.col(i) = dst.col(dst_ind[i]);
            src_corr.col(i) = src.col(src_ind[i]);
        }

        return estimateRigidTransformPointToPointClosedForm<ScalarT>(dst_corr, src_corr, rot_mat, t_vec);
    }

    // Point-to-point (iterative)
    template <typename ScalarT>
    bool estimateRigidTransformPointToPointIterative(const ConstPointSetMatrixMap<ScalarT,3> &dst_p,
                                                     const ConstPointSetMatrixMap<ScalarT,3> &src_p,
                                                     Eigen::Ref<Eigen::Matrix<ScalarT,3,3>> rot_mat,
                                                     Eigen::Ref<Eigen::Matrix<ScalarT,3,1>> t_vec,
                                                     size_t max_iter = 1,
                                                     ScalarT convergence_tol = 1e-5)
    {
        if (src_p.cols() != dst_p.cols() || src_p.cols() < 3) {
            rot_mat.setIdentity();
            t_vec.setZero();
            return false;
        }

        size_t num_eq = 3*dst_p.cols();

        Eigen::Matrix<ScalarT,3,3> rot_mat_iter;
        Eigen::Matrix<ScalarT,6,1> d_theta;
        rot_mat.setIdentity();
        t_vec.setZero();
        PointSet<ScalarT,3> src_t(src_p);

//        Eigen::Matrix<ScalarT,Eigen::Dynamic,6> A(num_eq, 6);
        Eigen::Matrix<ScalarT,6,Eigen::Dynamic> At(6, num_eq);
        Eigen::Matrix<ScalarT,Eigen::Dynamic,1> b(num_eq, 1);

        size_t eq_ind;
        size_t iter = 0;
        while (iter < max_iter) {
            if (iter > 0) {
                src_t = (rot_mat*src_p).colwise() + t_vec;
            }

            // Compute differential
            eq_ind = 0;
            for (size_t i = 0; i < dst_p.cols(); i++) {
                const Eigen::Matrix<ScalarT,3,1>& d = dst_p.col(i);
                const Eigen::Matrix<ScalarT,3,1>& s = src_t.col(i);

//                A(eq_ind,0) = 0.0;
//                A(eq_ind,1) = s[2];
//                A(eq_ind,2) = -s[1];
//                A(eq_ind,3) = 1.0;
//                A(eq_ind,4) = 0.0;
//                A(eq_ind,5) = 0.0;
//                b[eq_ind++] = d[0] - s[0];
//
//                A(eq_ind,0) = -s[2];
//                A(eq_ind,1) = 0.0;
//                A(eq_ind,2) = s[0];
//                A(eq_ind,3) = 0.0;
//                A(eq_ind,4) = 1.0;
//                A(eq_ind,5) = 0.0;
//                b[eq_ind++] = d[1] - s[1];
//
//                A(eq_ind,0) = s[1];
//                A(eq_ind,1) = -s[0];
//                A(eq_ind,2) = 0.0;
//                A(eq_ind,3) = 0.0;
//                A(eq_ind,4) = 0.0;
//                A(eq_ind,5) = 1.0;
//                b[eq_ind++] = d[2] - s[2];

                At(0,eq_ind) = 0.0;
                At(1,eq_ind) = s[2];
                At(2,eq_ind) = -s[1];
                At(3,eq_ind) = 1.0;
                At(4,eq_ind) = 0.0;
                At(5,eq_ind) = 0.0;
                b[eq_ind++] = d[0] - s[0];

                At(0,eq_ind) = -s[2];
                At(1,eq_ind) = 0.0;
                At(2,eq_ind) = s[0];
                At(3,eq_ind) = 0.0;
                At(4,eq_ind) = 1.0;
                At(5,eq_ind) = 0.0;
                b[eq_ind++] = d[1] - s[1];

                At(0,eq_ind) = s[1];
                At(1,eq_ind) = -s[0];
                At(2,eq_ind) = 0.0;
                At(3,eq_ind) = 0.0;
                At(4,eq_ind) = 0.0;
                At(5,eq_ind) = 1.0;
                b[eq_ind++] = d[2] - s[2];
            }

//            d_theta = (A.transpose()*A).ldlt().solve(A.transpose()*b);
            d_theta = (At*At.transpose()).ldlt().solve(At*b);

            // Update estimate
            rot_mat_iter = Eigen::AngleAxis<ScalarT>(d_theta[2], Eigen::Matrix<ScalarT,3,1>::UnitZ()) *
                           Eigen::AngleAxis<ScalarT>(d_theta[1], Eigen::Matrix<ScalarT,3,1>::UnitY()) *
                           Eigen::AngleAxis<ScalarT>(d_theta[0], Eigen::Matrix<ScalarT,3,1>::UnitX());

            rot_mat = rot_mat_iter*rot_mat;
            t_vec = rot_mat_iter*t_vec + d_theta.tail(3);

            // Orthonormalize rotation
            Eigen::JacobiSVD<Eigen::Matrix<ScalarT,3,3>> svd(rot_mat, Eigen::ComputeFullU | Eigen::ComputeFullV);
            if (svd.matrixU().determinant() * svd.matrixV().determinant() < 0.0) {
                Eigen::Matrix<ScalarT,3,3> U(svd.matrixU());
                U.col(2) *= -1.0;
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

    template <typename ScalarT>
    bool estimateRigidTransformPointToPointIterative(const ConstPointSetMatrixMap<ScalarT,3> &dst_p,
                                                     const ConstPointSetMatrixMap<ScalarT,3> &src_p,
                                                     const std::vector<size_t> &dst_ind,
                                                     const std::vector<size_t> &src_ind,
                                                     Eigen::Ref<Eigen::Matrix<ScalarT,3,3>> rot_mat,
                                                     Eigen::Ref<Eigen::Matrix<ScalarT,3,1>> t_vec,
                                                     size_t max_iter = 1,
                                                     ScalarT convergence_tol = 1e-5)
    {
        if (dst_ind.size() != src_ind.size()) {
            rot_mat.setIdentity();
            t_vec.setZero();
            return false;
        }

        PointSet<ScalarT,3> dst_p_corr(3, dst_ind.size());
        PointSet<ScalarT,3> src_p_corr(3, src_ind.size());
        for (size_t i = 0; i < dst_ind.size(); i++) {
            dst_p_corr.col(i) = dst_p.col(dst_ind[i]);
            src_p_corr.col(i) = src_p.col(src_ind[i]);
        }

        return estimateRigidTransformPointToPointIterative<ScalarT>(dst_p_corr, src_p_corr, rot_mat, t_vec, max_iter, convergence_tol);
    }

    // Point-to-plane
    template <typename ScalarT>
    bool estimateRigidTransformPointToPlane(const ConstPointSetMatrixMap<ScalarT,3> &dst_p,
                                            const ConstPointSetMatrixMap<ScalarT,3> &dst_n,
                                            const ConstPointSetMatrixMap<ScalarT,3> &src_p,
                                            Eigen::Ref<Eigen::Matrix<ScalarT,3,3>> rot_mat,
                                            Eigen::Ref<Eigen::Matrix<ScalarT,3,1>> t_vec,
                                            size_t max_iter = 1,
                                            ScalarT convergence_tol = 1e-5)
    {
        if (src_p.cols() != dst_p.cols() || dst_p.cols() != dst_n.cols() || src_p.cols() < 6) {
            rot_mat.setIdentity();
            t_vec.setZero();
            return false;
        }

        Eigen::Matrix<ScalarT,3,3> rot_mat_iter;
        Eigen::Matrix<ScalarT,6,1> d_theta;
        rot_mat.setIdentity();
        t_vec.setZero();
        PointSet<ScalarT,3> src_t(src_p);

//        Eigen::Matrix<ScalarT,Eigen::Dynamic,6> A(dst_p.cols(),6);
        Eigen::Matrix<ScalarT,6,Eigen::Dynamic> At(6,dst_p.cols());
        Eigen::Matrix<ScalarT,Eigen::Dynamic,1> b(dst_p.cols(),1);

        size_t iter = 0;
        while (iter < max_iter) {
            if (iter > 0) {
                src_t = (rot_mat*src_p).colwise() + t_vec;
            }

            // Compute differential
            for (size_t i = 0; i < dst_p.cols(); i++) {
                const Eigen::Matrix<ScalarT,3,1>& d = dst_p.col(i);
                const Eigen::Matrix<ScalarT,3,1>& n = dst_n.col(i);
                const Eigen::Matrix<ScalarT,3,1>& s = src_t.col(i);

//                A(i,0) = n[2]*s[1] - n[1]*s[2];
//                A(i,1) = n[0]*s[2] - n[2]*s[0];
//                A(i,2) = n[1]*s[0] - n[0]*s[1];
//                A(i,3) = n[0];
//                A(i,4) = n[1];
//                A(i,5) = n[2];
//                b[i] = n[0]*d[0] + n[1]*d[1] + n[2]*d[2] - n[0]*s[0] - n[1]*s[1] - n[2]*s[2];

                At(0,i) = n[2]*s[1] - n[1]*s[2];
                At(1,i) = n[0]*s[2] - n[2]*s[0];
                At(2,i) = n[1]*s[0] - n[0]*s[1];
                At(3,i) = n[0];
                At(4,i) = n[1];
                At(5,i) = n[2];
                b[i] = n[0]*d[0] + n[1]*d[1] + n[2]*d[2] - n[0]*s[0] - n[1]*s[1] - n[2]*s[2];
            }

//            d_theta = (A.transpose()*A).ldlt().solve(A.transpose()*b);
            d_theta = (At*At.transpose()).ldlt().solve(At*b);

            // Update estimate
            rot_mat_iter = Eigen::AngleAxis<ScalarT>(d_theta[2], Eigen::Matrix<ScalarT,3,1>::UnitZ()) *
                           Eigen::AngleAxis<ScalarT>(d_theta[1], Eigen::Matrix<ScalarT,3,1>::UnitY()) *
                           Eigen::AngleAxis<ScalarT>(d_theta[0], Eigen::Matrix<ScalarT,3,1>::UnitX());

            rot_mat = rot_mat_iter*rot_mat;
            t_vec = rot_mat_iter*t_vec + d_theta.tail(3);

            // Orthonormalize rotation
            Eigen::JacobiSVD<Eigen::Matrix<ScalarT,3,3>> svd(rot_mat, Eigen::ComputeFullU | Eigen::ComputeFullV);
            if (svd.matrixU().determinant() * svd.matrixV().determinant() < 0.0) {
                Eigen::Matrix<ScalarT,3,3> U(svd.matrixU());
                U.col(2) *= -1.0;
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

    template <typename ScalarT>
    bool estimateRigidTransformPointToPlane(const ConstPointSetMatrixMap<ScalarT,3> &dst_p,
                                            const ConstPointSetMatrixMap<ScalarT,3> &dst_n,
                                            const ConstPointSetMatrixMap<ScalarT,3> &src_p,
                                            const std::vector<size_t> &dst_ind,
                                            const std::vector<size_t> &src_ind,
                                            Eigen::Ref<Eigen::Matrix<ScalarT,3,3>> rot_mat,
                                            Eigen::Ref<Eigen::Matrix<ScalarT,3,1>> t_vec,
                                            size_t max_iter = 1,
                                            ScalarT convergence_tol = 1e-5)
    {
        if (dst_ind.size() != src_ind.size()) {
            rot_mat.setIdentity();
            t_vec.setZero();
            return false;
        }

        PointSet<ScalarT,3> dst_p_corr(3, dst_ind.size());
        PointSet<ScalarT,3> dst_n_corr(3, dst_ind.size());
        PointSet<ScalarT,3> src_p_corr(3, src_ind.size());
        for (size_t i = 0; i < dst_ind.size(); i++) {
            dst_p_corr.col(i) = dst_p.col(dst_ind[i]);
            dst_n_corr.col(i) = dst_n.col(dst_ind[i]);
            src_p_corr.col(i) = src_p.col(src_ind[i]);
        }

        return estimateRigidTransformPointToPlane<ScalarT>(dst_p_corr, dst_n_corr, src_p_corr, rot_mat, t_vec, max_iter, convergence_tol);
    }

    // Point-to-point and point-to-plane combination
    template <typename ScalarT>
    bool estimateRigidTransformCombinedMetric(const ConstPointSetMatrixMap<ScalarT,3> &dst_p,
                                              const ConstPointSetMatrixMap<ScalarT,3> &dst_n,
                                              const ConstPointSetMatrixMap<ScalarT,3> &src_p,
                                              ScalarT point_to_point_weight,
                                              ScalarT point_to_plane_weight,
                                              Eigen::Ref<Eigen::Matrix<ScalarT,3,3>> rot_mat,
                                              Eigen::Ref<Eigen::Matrix<ScalarT,3,1>> t_vec,
                                              size_t max_iter = 1,
                                              ScalarT convergence_tol = 1e-5)
    {
        ScalarT point_weight = std::abs(point_to_point_weight);
        ScalarT plane_weight = std::abs(point_to_plane_weight);

        if (src_p.cols() != dst_p.cols() || dst_p.cols() != dst_n.cols() || src_p.cols() < 3 || (point_weight == 0.0 && plane_weight == 0.0)) {
            rot_mat.setIdentity();
            t_vec.setZero();
            return false;
        }

        if (point_weight == 0.0) {
            // Do point-to-plane
            return estimateRigidTransformPointToPlane<ScalarT>(dst_p, dst_n, src_p, rot_mat, t_vec, max_iter, convergence_tol);
        }

        if (plane_weight == 0.0) {
            // Do point-to-point
            return estimateRigidTransformPointToPointIterative<ScalarT>(dst_p, src_p, rot_mat, t_vec, max_iter, convergence_tol);
        }

        Eigen::Matrix<ScalarT,3,3> rot_mat_iter;
        Eigen::Matrix<ScalarT,6,1> d_theta;
        rot_mat.setIdentity();
        t_vec.setZero();
        PointSet<ScalarT,3> src_t(src_p);

        size_t num_eq = 4*dst_p.cols();

//        Eigen::Matrix<ScalarT,Eigen::Dynamic,6> A(num_eq, 6);
        Eigen::Matrix<ScalarT,6,Eigen::Dynamic> At(6, num_eq);
        Eigen::Matrix<ScalarT,Eigen::Dynamic,1> b(num_eq, 1);

        size_t eq_ind;
        size_t iter = 0;
        while (iter < max_iter) {
            if (iter > 0) {
                src_t = (rot_mat*src_p).colwise() + t_vec;
            }

            // Compute differential
            eq_ind = 0;
            for (size_t i = 0; i < dst_p.cols(); i++) {
                const Eigen::Matrix<ScalarT,3,1>& d = dst_p.col(i);
                const Eigen::Matrix<ScalarT,3,1>& n = dst_n.col(i);
                const Eigen::Matrix<ScalarT,3,1>& s = src_t.col(i);

//                A(eq_ind,0) = (n[2]*s[1] - n[1]*s[2])*plane_weight;
//                A(eq_ind,1) = (n[0]*s[2] - n[2]*s[0])*plane_weight;
//                A(eq_ind,2) = (n[1]*s[0] - n[0]*s[1])*plane_weight;
//                A(eq_ind,3) = n[0]*plane_weight;
//                A(eq_ind,4) = n[1]*plane_weight;
//                A(eq_ind,5) = n[2]*plane_weight;
//                b[eq_ind++] = (n[0]*d[0] + n[1]*d[1] + n[2]*d[2] - n[0]*s[0] - n[1]*s[1] - n[2]*s[2])*plane_weight;
//
//                A(eq_ind,0) = 0.0;
//                A(eq_ind,1) = s[2]*point_weight;
//                A(eq_ind,2) = -s[1]*point_weight;
//                A(eq_ind,3) = 1.0*point_weight;
//                A(eq_ind,4) = 0.0;
//                A(eq_ind,5) = 0.0;
//                b[eq_ind++] = (d[0] - s[0])*point_weight;
//
//                A(eq_ind,0) = -s[2]*point_weight;
//                A(eq_ind,1) = 0.0;
//                A(eq_ind,2) = s[0]*point_weight;
//                A(eq_ind,3) = 0.0;
//                A(eq_ind,4) = 1.0*point_weight;
//                A(eq_ind,5) = 0.0;
//                b[eq_ind++] = (d[1] - s[1])*point_weight;
//
//                A(eq_ind,0) = s[1]*point_weight;
//                A(eq_ind,1) = -s[0]*point_weight;
//                A(eq_ind,2) = 0.0;
//                A(eq_ind,3) = 0.0;
//                A(eq_ind,4) = 0.0;
//                A(eq_ind,5) = 1.0*point_weight;
//                b[eq_ind++] = (d[2] - s[2])*point_weight;

                At(0,eq_ind) = (n[2]*s[1] - n[1]*s[2])*plane_weight;
                At(1,eq_ind) = (n[0]*s[2] - n[2]*s[0])*plane_weight;
                At(2,eq_ind) = (n[1]*s[0] - n[0]*s[1])*plane_weight;
                At(3,eq_ind) = n[0]*plane_weight;
                At(4,eq_ind) = n[1]*plane_weight;
                At(5,eq_ind) = n[2]*plane_weight;
                b[eq_ind++] = (n[0]*d[0] + n[1]*d[1] + n[2]*d[2] - n[0]*s[0] - n[1]*s[1] - n[2]*s[2])*plane_weight;

                At(0,eq_ind) = 0.0;
                At(1,eq_ind) = s[2]*point_weight;
                At(2,eq_ind) = -s[1]*point_weight;
                At(3,eq_ind) = 1.0*point_weight;
                At(4,eq_ind) = 0.0;
                At(5,eq_ind) = 0.0;
                b[eq_ind++] = (d[0] - s[0])*point_weight;

                At(0,eq_ind) = -s[2]*point_weight;
                At(1,eq_ind) = 0.0;
                At(2,eq_ind) = s[0]*point_weight;
                At(3,eq_ind) = 0.0;
                At(4,eq_ind) = 1.0*point_weight;
                At(5,eq_ind) = 0.0;
                b[eq_ind++] = (d[1] - s[1])*point_weight;

                At(0,eq_ind) = s[1]*point_weight;
                At(1,eq_ind) = -s[0]*point_weight;
                At(2,eq_ind) = 0.0;
                At(3,eq_ind) = 0.0;
                At(4,eq_ind) = 0.0;
                At(5,eq_ind) = 1.0*point_weight;
                b[eq_ind++] = (d[2] - s[2])*point_weight;
            }

//            d_theta = (A.transpose()*A).ldlt().solve(A.transpose()*b);
            d_theta = (At*At.transpose()).ldlt().solve(At*b);

            // Update estimate
            rot_mat_iter = Eigen::AngleAxis<ScalarT>(d_theta[2], Eigen::Matrix<ScalarT,3,1>::UnitZ()) *
                           Eigen::AngleAxis<ScalarT>(d_theta[1], Eigen::Matrix<ScalarT,3,1>::UnitY()) *
                           Eigen::AngleAxis<ScalarT>(d_theta[0], Eigen::Matrix<ScalarT,3,1>::UnitX());

            rot_mat = rot_mat_iter*rot_mat;
            t_vec = rot_mat_iter*t_vec + d_theta.tail(3);

            // Orthonormalize rotation
            Eigen::JacobiSVD<Eigen::Matrix<ScalarT,3,3>> svd(rot_mat, Eigen::ComputeFullU | Eigen::ComputeFullV);
            if (svd.matrixU().determinant() * svd.matrixV().determinant() < 0.0) {
                Eigen::Matrix<ScalarT,3,3> U(svd.matrixU());
                U.col(2) *= -1.0;
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

    template <typename ScalarT>
    bool estimateRigidTransformCombinedMetric(const ConstPointSetMatrixMap<ScalarT,3> &dst_p,
                                              const ConstPointSetMatrixMap<ScalarT,3> &dst_n,
                                              const ConstPointSetMatrixMap<ScalarT,3> &src_p,
                                              const std::vector<size_t> &dst_ind,
                                              const std::vector<size_t> &src_ind,
                                              ScalarT point_to_point_weight,
                                              ScalarT point_to_plane_weight,
                                              Eigen::Ref<Eigen::Matrix<ScalarT,3,3>> rot_mat,
                                              Eigen::Ref<Eigen::Matrix<ScalarT,3,1>> t_vec,
                                              size_t max_iter = 1,
                                              ScalarT convergence_tol = 1e-5)
    {
        if (dst_ind.size() != src_ind.size()) {
            rot_mat.setIdentity();
            t_vec.setZero();
            return false;
        }

        PointSet<ScalarT,3> dst_p_corr(3, dst_ind.size());
        PointSet<ScalarT,3> dst_n_corr(3, dst_ind.size());
        PointSet<ScalarT,3> src_p_corr(3, src_ind.size());
        for (size_t i = 0; i < dst_ind.size(); i++) {
            dst_p_corr.col(i) = dst_p.col(dst_ind[i]);
            dst_n_corr.col(i) = dst_n.col(dst_ind[i]);
            src_p_corr.col(i) = src_p.col(src_ind[i]);
        }

        return estimateRigidTransformCombinedMetric<ScalarT>(dst_p_corr, dst_n_corr, src_p_corr, point_to_point_weight, point_to_plane_weight, rot_mat, t_vec, max_iter, convergence_tol);
    }
}
