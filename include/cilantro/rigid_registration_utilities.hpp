#pragma once

#include <cilantro/space_transformations.hpp>
#include <cilantro/correspondence.hpp>

namespace cilantro {
    // Point-to-point (closed form, SVD)
    template <typename ScalarT, ptrdiff_t EigenDim>
    bool estimateRigidTransformPointToPointClosedForm(const ConstVectorSetMatrixMap<ScalarT,EigenDim> &dst,
                                                      const ConstVectorSetMatrixMap<ScalarT,EigenDim> &src,
                                                      RigidTransformation<ScalarT,EigenDim> &tform)
    {
        if (src.cols() != dst.cols() || src.cols() < src.rows()) {
            tform.setIdentity();
            return false;
        }

        Vector<ScalarT,EigenDim> mu_dst(dst.rowwise().mean());
        Vector<ScalarT,EigenDim> mu_src(src.rowwise().mean());

        VectorSet<ScalarT,EigenDim> dst_centered(dst.colwise() - mu_dst);
        VectorSet<ScalarT,EigenDim> src_centered(src.colwise() - mu_src);

        Eigen::JacobiSVD<Eigen::Matrix<ScalarT,EigenDim,EigenDim>> svd(dst_centered*src_centered.transpose(), Eigen::ComputeFullU | Eigen::ComputeFullV);
        if ((svd.matrixU()*svd.matrixV()).determinant() < (ScalarT)0.0) {
            Eigen::Matrix<ScalarT,EigenDim,EigenDim> U(svd.matrixU());
            U.col(dst.rows()-1) *= (ScalarT)(-1.0);
            tform.linear() = U*svd.matrixV().transpose();
        } else {
            tform.linear() = svd.matrixU()*svd.matrixV().transpose();
        }
        tform.translation() = mu_dst - tform.linear()*mu_src;

        return true;
    }

    template <typename ScalarT, ptrdiff_t EigenDim, typename CorrValueT = ScalarT>
    bool estimateRigidTransformPointToPointClosedForm(const ConstVectorSetMatrixMap<ScalarT,EigenDim> &dst,
                                                      const ConstVectorSetMatrixMap<ScalarT,EigenDim> &src,
                                                      const CorrespondenceSet<CorrValueT> &corr,
                                                      RigidTransformation<ScalarT,EigenDim> &tform)
    {
        VectorSet<ScalarT,EigenDim> dst_corr, src_corr;
        selectCorrespondingPoints<ScalarT,EigenDim,CorrValueT>(corr, dst, src, dst_corr, src_corr);
        return estimateRigidTransformPointToPointClosedForm<ScalarT,EigenDim>(dst_corr, src_corr, tform);
    }

    // Point-to-point (iterative)
    template <typename ScalarT>
    bool estimateRigidTransformPointToPoint3DIterative(const ConstVectorSetMatrixMap<ScalarT,3> &dst_p,
                                                       const ConstVectorSetMatrixMap<ScalarT,3> &src_p,
                                                       RigidTransformation<ScalarT,3> &tform,
                                                       size_t max_iter = 1,
                                                       ScalarT convergence_tol = 1e-5)
    {
        if (src_p.cols() != dst_p.cols() || src_p.cols() < 3) {
            tform.setIdentity();
            return false;
        }

        const size_t num_eq = 3*dst_p.cols();

        Eigen::Matrix<ScalarT,3,3> rot_mat_iter;
        Eigen::Matrix<ScalarT,6,1> d_theta;
        tform.setIdentity();
        Vector<ScalarT,3> s;

        Eigen::Matrix<ScalarT,6,Eigen::Dynamic> At(6, num_eq);
        Eigen::Matrix<ScalarT,Eigen::Dynamic,1> b(num_eq, 1);

        size_t eq_ind;
        size_t iter = 0;
        while (iter < max_iter) {
            // Compute differential
            if (iter > 0) {
#pragma omp parallel for shared (At, b) private (s, eq_ind)
                for (size_t i = 0; i < dst_p.cols(); i++) {
                    const auto& d = dst_p.col(i);
                    s = tform*src_p.col(i);

                    eq_ind = 3*i;

                    At(0,eq_ind) = (ScalarT)0.0;
                    At(1,eq_ind) = s[2];
                    At(2,eq_ind) = -s[1];
                    At(3,eq_ind) = (ScalarT)1.0;
                    At(4,eq_ind) = (ScalarT)0.0;
                    At(5,eq_ind) = (ScalarT)0.0;
                    b[eq_ind++] = d[0] - s[0];

                    At(0,eq_ind) = -s[2];
                    At(1,eq_ind) = (ScalarT)0.0;
                    At(2,eq_ind) = s[0];
                    At(3,eq_ind) = (ScalarT)0.0;
                    At(4,eq_ind) = (ScalarT)1.0;
                    At(5,eq_ind) = (ScalarT)0.0;
                    b[eq_ind++] = d[1] - s[1];

                    At(0,eq_ind) = s[1];
                    At(1,eq_ind) = -s[0];
                    At(2,eq_ind) = (ScalarT)0.0;
                    At(3,eq_ind) = (ScalarT)0.0;
                    At(4,eq_ind) = (ScalarT)0.0;
                    At(5,eq_ind) = (ScalarT)1.0;
                    b[eq_ind++] = d[2] - s[2];
                }
            } else {
#pragma omp parallel for shared (At, b) private (eq_ind)
                for (size_t i = 0; i < dst_p.cols(); i++) {
                    const auto& d = dst_p.col(i);
                    const auto& s = src_p.col(i);

                    eq_ind = 3*i;

                    At(0,eq_ind) = (ScalarT)0.0;
                    At(1,eq_ind) = s[2];
                    At(2,eq_ind) = -s[1];
                    At(3,eq_ind) = (ScalarT)1.0;
                    At(4,eq_ind) = (ScalarT)0.0;
                    At(5,eq_ind) = (ScalarT)0.0;
                    b[eq_ind++] = d[0] - s[0];

                    At(0,eq_ind) = -s[2];
                    At(1,eq_ind) = (ScalarT)0.0;
                    At(2,eq_ind) = s[0];
                    At(3,eq_ind) = (ScalarT)0.0;
                    At(4,eq_ind) = (ScalarT)1.0;
                    At(5,eq_ind) = (ScalarT)0.0;
                    b[eq_ind++] = d[1] - s[1];

                    At(0,eq_ind) = s[1];
                    At(1,eq_ind) = -s[0];
                    At(2,eq_ind) = (ScalarT)0.0;
                    At(3,eq_ind) = (ScalarT)0.0;
                    At(4,eq_ind) = (ScalarT)0.0;
                    At(5,eq_ind) = (ScalarT)1.0;
                    b[eq_ind++] = d[2] - s[2];
                }
            }

            d_theta = (At*At.transpose()).ldlt().solve(At*b);

            // Update estimate
            rot_mat_iter = Eigen::AngleAxis<ScalarT>(d_theta[2], Eigen::Matrix<ScalarT,3,1>::UnitZ()) *
                           Eigen::AngleAxis<ScalarT>(d_theta[1], Eigen::Matrix<ScalarT,3,1>::UnitY()) *
                           Eigen::AngleAxis<ScalarT>(d_theta[0], Eigen::Matrix<ScalarT,3,1>::UnitX());

            tform.linear() = rot_mat_iter*tform.linear();
            tform.linear() = tform.rotation();
            tform.translation() = rot_mat_iter*tform.translation() + d_theta.tail(3);

            iter++;

            // Check for convergence
            if (d_theta.norm() < convergence_tol) return true;
        }

        return false;
    }

    template <typename ScalarT, typename CorrValueT = ScalarT>
    bool estimateRigidTransformPointToPoint3DIterative(const ConstVectorSetMatrixMap<ScalarT,3> &dst_p,
                                                       const ConstVectorSetMatrixMap<ScalarT,3> &src_p,
                                                       const CorrespondenceSet<CorrValueT> &corr,
                                                       RigidTransformation<ScalarT,3> &tform,
                                                       size_t max_iter = 1,
                                                       ScalarT convergence_tol = 1e-5)
    {
        VectorSet<ScalarT,3> dst_p_corr, src_p_corr;
        selectCorrespondingPoints<ScalarT,3,CorrValueT>(corr, dst_p, src_p, dst_p_corr, src_p_corr);
        return estimateRigidTransformPointToPoint3DIterative<ScalarT>(dst_p_corr, src_p_corr, tform, max_iter, convergence_tol);
    }

    // Point-to-plane
    template <typename ScalarT>
    bool estimateRigidTransformPointToPlane3D(const ConstVectorSetMatrixMap<ScalarT,3> &dst_p,
                                              const ConstVectorSetMatrixMap<ScalarT,3> &dst_n,
                                              const ConstVectorSetMatrixMap<ScalarT,3> &src_p,
                                              RigidTransformation<ScalarT,3> &tform,
                                              size_t max_iter = 1,
                                              ScalarT convergence_tol = 1e-5)
    {
        if (src_p.cols() != dst_p.cols() || dst_p.cols() != dst_n.cols() || src_p.cols() < 6) {
            tform.setIdentity();
            return false;
        }

        Eigen::Matrix<ScalarT,3,3> rot_mat_iter;
        Eigen::Matrix<ScalarT,6,1> d_theta;
        tform.setIdentity();
        Vector<ScalarT,3> s;

        Eigen::Matrix<ScalarT,6,Eigen::Dynamic> At(6,dst_p.cols());
        Eigen::Matrix<ScalarT,Eigen::Dynamic,1> b(dst_p.cols(),1);

        size_t iter = 0;
        while (iter < max_iter) {
            // Compute differential
            if (iter > 0) {
#pragma omp parallel for shared (At, b) private (s)
                for (size_t i = 0; i < dst_p.cols(); i++) {
                    const auto& d = dst_p.col(i);
                    const auto& n = dst_n.col(i);
                    s = tform*src_p.col(i);

                    At(0,i) = n[2]*s[1] - n[1]*s[2];
                    At(1,i) = n[0]*s[2] - n[2]*s[0];
                    At(2,i) = n[1]*s[0] - n[0]*s[1];
                    At(3,i) = n[0];
                    At(4,i) = n[1];
                    At(5,i) = n[2];
                    b[i] = n[0]*d[0] + n[1]*d[1] + n[2]*d[2] - n[0]*s[0] - n[1]*s[1] - n[2]*s[2];
                }
            } else {
#pragma omp parallel for shared (At, b)
                for (size_t i = 0; i < dst_p.cols(); i++) {
                    const auto& d = dst_p.col(i);
                    const auto& n = dst_n.col(i);
                    const auto& s = src_p.col(i);

                    At(0,i) = n[2]*s[1] - n[1]*s[2];
                    At(1,i) = n[0]*s[2] - n[2]*s[0];
                    At(2,i) = n[1]*s[0] - n[0]*s[1];
                    At(3,i) = n[0];
                    At(4,i) = n[1];
                    At(5,i) = n[2];
                    b[i] = n[0]*d[0] + n[1]*d[1] + n[2]*d[2] - n[0]*s[0] - n[1]*s[1] - n[2]*s[2];
                }
            }

            d_theta = (At*At.transpose()).ldlt().solve(At*b);

            // Update estimate
            rot_mat_iter = Eigen::AngleAxis<ScalarT>(d_theta[2], Eigen::Matrix<ScalarT,3,1>::UnitZ()) *
                           Eigen::AngleAxis<ScalarT>(d_theta[1], Eigen::Matrix<ScalarT,3,1>::UnitY()) *
                           Eigen::AngleAxis<ScalarT>(d_theta[0], Eigen::Matrix<ScalarT,3,1>::UnitX());

            tform.linear() = rot_mat_iter*tform.linear();
            tform.linear() = tform.rotation();
            tform.translation() = rot_mat_iter*tform.translation() + d_theta.tail(3);

            iter++;

            // Check for convergence
            if (d_theta.norm() < convergence_tol) return true;
        }

        return false;
    }

    template <typename ScalarT, typename CorrValueT = ScalarT>
    bool estimateRigidTransformPointToPlane3D(const ConstVectorSetMatrixMap<ScalarT,3> &dst_p,
                                              const ConstVectorSetMatrixMap<ScalarT,3> &dst_n,
                                              const ConstVectorSetMatrixMap<ScalarT,3> &src_p,
                                              const CorrespondenceSet<CorrValueT> &corr,
                                              RigidTransformation<ScalarT,3> &tform,
                                              size_t max_iter = 1,
                                              ScalarT convergence_tol = 1e-5)
    {
        VectorSet<ScalarT,3> dst_p_corr(3, corr.size());
        VectorSet<ScalarT,3> dst_n_corr(3, corr.size());
        VectorSet<ScalarT,3> src_p_corr(3, corr.size());
#pragma omp parallel for
        for (size_t i = 0; i < corr.size(); i++) {
            dst_p_corr.col(i) = dst_p.col(corr[i].indexInFirst);
            dst_n_corr.col(i) = dst_n.col(corr[i].indexInFirst);
            src_p_corr.col(i) = src_p.col(corr[i].indexInSecond);
        }
        return estimateRigidTransformPointToPlane3D<ScalarT>(dst_p_corr, dst_n_corr, src_p_corr, tform, max_iter, convergence_tol);
    }

    // Point-to-point and point-to-plane combination
    template <typename ScalarT>
    bool estimateRigidTransformCombinedMetric3D(const ConstVectorSetMatrixMap<ScalarT,3> &dst_p,
                                                const ConstVectorSetMatrixMap<ScalarT,3> &dst_n,
                                                const ConstVectorSetMatrixMap<ScalarT,3> &src_p,
                                                ScalarT point_to_point_weight,
                                                ScalarT point_to_plane_weight,
                                                RigidTransformation<ScalarT,3> &tform,
                                                size_t max_iter = 1,
                                                ScalarT convergence_tol = 1e-5)
    {
        const ScalarT point_to_point_weight_sqrt = std::sqrt(point_to_point_weight);
        const ScalarT point_to_plane_weight_sqrt = std::sqrt(point_to_plane_weight);

        if (src_p.cols() != dst_p.cols() || dst_p.cols() != dst_n.cols() || src_p.cols() < 3 || (point_to_point_weight_sqrt == (ScalarT)0.0 && point_to_plane_weight_sqrt == (ScalarT)0.0)) {
            tform.setIdentity();
            return false;
        }

        if (point_to_point_weight_sqrt == (ScalarT)0.0) {
            // Do point-to-plane
            return estimateRigidTransformPointToPlane3D<ScalarT>(dst_p, dst_n, src_p, tform, max_iter, convergence_tol);
        }

        if (point_to_plane_weight_sqrt == (ScalarT)0.0) {
            // Do point-to-point
            return estimateRigidTransformPointToPoint3DIterative<ScalarT>(dst_p, src_p, tform, max_iter, convergence_tol);
        }

        Eigen::Matrix<ScalarT,3,3> rot_mat_iter;
        Eigen::Matrix<ScalarT,6,1> d_theta;
        tform.setIdentity();
        Vector<ScalarT,3> s;

        const size_t num_eq = 4*dst_p.cols();

        Eigen::Matrix<ScalarT,6,Eigen::Dynamic> At(6, num_eq);
        Eigen::Matrix<ScalarT,Eigen::Dynamic,1> b(num_eq, 1);

        size_t eq_ind;
        size_t iter = 0;
        while (iter < max_iter) {
            // Compute differential
            if (iter > 0) {
#pragma omp parallel for shared (At, b) private (s, eq_ind)
                for (size_t i = 0; i < dst_p.cols(); i++) {
                    const auto& d = dst_p.col(i);
                    const auto& n = dst_n.col(i);
                    s = tform*src_p.col(i);

                    eq_ind = 4*i;

                    At(0,eq_ind) = (n[2]*s[1] - n[1]*s[2])*point_to_plane_weight_sqrt;
                    At(1,eq_ind) = (n[0]*s[2] - n[2]*s[0])*point_to_plane_weight_sqrt;
                    At(2,eq_ind) = (n[1]*s[0] - n[0]*s[1])*point_to_plane_weight_sqrt;
                    At(3,eq_ind) = n[0]*point_to_plane_weight_sqrt;
                    At(4,eq_ind) = n[1]*point_to_plane_weight_sqrt;
                    At(5,eq_ind) = n[2]*point_to_plane_weight_sqrt;
                    b[eq_ind++] = (n[0]*d[0] + n[1]*d[1] + n[2]*d[2] - n[0]*s[0] - n[1]*s[1] - n[2]*s[2])*point_to_plane_weight_sqrt;

                    At(0,eq_ind) = (ScalarT)0.0;
                    At(1,eq_ind) = s[2]*point_to_point_weight_sqrt;
                    At(2,eq_ind) = -s[1]*point_to_point_weight_sqrt;
                    At(3,eq_ind) = point_to_point_weight_sqrt;
                    At(4,eq_ind) = (ScalarT)0.0;
                    At(5,eq_ind) = (ScalarT)0.0;
                    b[eq_ind++] = (d[0] - s[0])*point_to_point_weight_sqrt;

                    At(0,eq_ind) = -s[2]*point_to_point_weight_sqrt;
                    At(1,eq_ind) = (ScalarT)0.0;
                    At(2,eq_ind) = s[0]*point_to_point_weight_sqrt;
                    At(3,eq_ind) = (ScalarT)0.0;
                    At(4,eq_ind) = point_to_point_weight_sqrt;
                    At(5,eq_ind) = (ScalarT)0.0;
                    b[eq_ind++] = (d[1] - s[1])*point_to_point_weight_sqrt;

                    At(0,eq_ind) = s[1]*point_to_point_weight_sqrt;
                    At(1,eq_ind) = -s[0]*point_to_point_weight_sqrt;
                    At(2,eq_ind) = (ScalarT)0.0;
                    At(3,eq_ind) = (ScalarT)0.0;
                    At(4,eq_ind) = (ScalarT)0.0;
                    At(5,eq_ind) = point_to_point_weight_sqrt;
                    b[eq_ind++] = (d[2] - s[2])*point_to_point_weight_sqrt;
                }
            } else {
#pragma omp parallel for shared (At, b) private (eq_ind)
                for (size_t i = 0; i < dst_p.cols(); i++) {
                    const auto& d = dst_p.col(i);
                    const auto& n = dst_n.col(i);
                    const auto& s = src_p.col(i);

                    eq_ind = 4*i;

                    At(0,eq_ind) = (n[2]*s[1] - n[1]*s[2])*point_to_plane_weight_sqrt;
                    At(1,eq_ind) = (n[0]*s[2] - n[2]*s[0])*point_to_plane_weight_sqrt;
                    At(2,eq_ind) = (n[1]*s[0] - n[0]*s[1])*point_to_plane_weight_sqrt;
                    At(3,eq_ind) = n[0]*point_to_plane_weight_sqrt;
                    At(4,eq_ind) = n[1]*point_to_plane_weight_sqrt;
                    At(5,eq_ind) = n[2]*point_to_plane_weight_sqrt;
                    b[eq_ind++] = (n[0]*d[0] + n[1]*d[1] + n[2]*d[2] - n[0]*s[0] - n[1]*s[1] - n[2]*s[2])*point_to_plane_weight_sqrt;

                    At(0,eq_ind) = (ScalarT)0.0;
                    At(1,eq_ind) = s[2]*point_to_point_weight_sqrt;
                    At(2,eq_ind) = -s[1]*point_to_point_weight_sqrt;
                    At(3,eq_ind) = point_to_point_weight_sqrt;
                    At(4,eq_ind) = (ScalarT)0.0;
                    At(5,eq_ind) = (ScalarT)0.0;
                    b[eq_ind++] = (d[0] - s[0])*point_to_point_weight_sqrt;

                    At(0,eq_ind) = -s[2]*point_to_point_weight_sqrt;
                    At(1,eq_ind) = (ScalarT)0.0;
                    At(2,eq_ind) = s[0]*point_to_point_weight_sqrt;
                    At(3,eq_ind) = (ScalarT)0.0;
                    At(4,eq_ind) = point_to_point_weight_sqrt;
                    At(5,eq_ind) = (ScalarT)0.0;
                    b[eq_ind++] = (d[1] - s[1])*point_to_point_weight_sqrt;

                    At(0,eq_ind) = s[1]*point_to_point_weight_sqrt;
                    At(1,eq_ind) = -s[0]*point_to_point_weight_sqrt;
                    At(2,eq_ind) = (ScalarT)0.0;
                    At(3,eq_ind) = (ScalarT)0.0;
                    At(4,eq_ind) = (ScalarT)0.0;
                    At(5,eq_ind) = point_to_point_weight_sqrt;
                    b[eq_ind++] = (d[2] - s[2])*point_to_point_weight_sqrt;
                }
            }

            d_theta = (At*At.transpose()).ldlt().solve(At*b);

            // Update estimate
            rot_mat_iter = Eigen::AngleAxis<ScalarT>(d_theta[2], Eigen::Matrix<ScalarT,3,1>::UnitZ()) *
                           Eigen::AngleAxis<ScalarT>(d_theta[1], Eigen::Matrix<ScalarT,3,1>::UnitY()) *
                           Eigen::AngleAxis<ScalarT>(d_theta[0], Eigen::Matrix<ScalarT,3,1>::UnitX());

            tform.linear() = rot_mat_iter*tform.linear();
            tform.linear() = tform.rotation();
            tform.translation() = rot_mat_iter*tform.translation() + d_theta.tail(3);

            iter++;

            // Check for convergence
            if (d_theta.norm() < convergence_tol) return true;
        }

        return false;
    }

    template <typename ScalarT, typename CorrValueT = ScalarT>
    bool estimateRigidTransformCombinedMetric3D(const ConstVectorSetMatrixMap<ScalarT,3> &dst_p,
                                                const ConstVectorSetMatrixMap<ScalarT,3> &dst_n,
                                                const ConstVectorSetMatrixMap<ScalarT,3> &src_p,
                                                const CorrespondenceSet<CorrValueT> &corr,
                                                ScalarT point_to_point_weight,
                                                ScalarT point_to_plane_weight,
                                                RigidTransformation<ScalarT,3> &tform,
                                                size_t max_iter = 1,
                                                ScalarT convergence_tol = 1e-5)
    {
        VectorSet<ScalarT,3> dst_p_corr(3, corr.size());
        VectorSet<ScalarT,3> dst_n_corr(3, corr.size());
        VectorSet<ScalarT,3> src_p_corr(3, corr.size());
#pragma omp parallel for
        for (size_t i = 0; i < corr.size(); i++) {
            dst_p_corr.col(i) = dst_p.col(corr[i].indexInFirst);
            dst_n_corr.col(i) = dst_n.col(corr[i].indexInFirst);
            src_p_corr.col(i) = src_p.col(corr[i].indexInSecond);
        }

        return estimateRigidTransformCombinedMetric3D<ScalarT>(dst_p_corr, dst_n_corr, src_p_corr, point_to_point_weight, point_to_plane_weight, tform, max_iter, convergence_tol);
    }
}
