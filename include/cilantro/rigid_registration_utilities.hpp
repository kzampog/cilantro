#pragma once

#include <cilantro/space_transformations.hpp>
#include <cilantro/correspondence.hpp>
#include <cilantro/common_pair_evaluators.hpp>

namespace cilantro {
    // Point-to-point (closed form, SVD)
    template <typename ScalarT, ptrdiff_t EigenDim>
    bool estimateRigidTransformPointToPointClosedForm(const ConstVectorSetMatrixMap<ScalarT,EigenDim> &dst,
                                                      const ConstVectorSetMatrixMap<ScalarT,EigenDim> &src,
                                                      RigidTransform<ScalarT,EigenDim> &tform)
    {
        if (src.cols() != dst.cols() || src.cols() == 0) {
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
            tform.linear().noalias() = U*svd.matrixV().transpose();
        } else {
            tform.linear().noalias() = svd.matrixU()*svd.matrixV().transpose();
        }
        tform.translation().noalias() = mu_dst - tform.linear()*mu_src;

        return true;
    }

    template <typename ScalarT, ptrdiff_t EigenDim, typename CorrValueT = ScalarT>
    bool estimateRigidTransformPointToPointClosedForm(const ConstVectorSetMatrixMap<ScalarT,EigenDim> &dst,
                                                      const ConstVectorSetMatrixMap<ScalarT,EigenDim> &src,
                                                      const CorrespondenceSet<CorrValueT> &corr,
                                                      RigidTransform<ScalarT,EigenDim> &tform)
    {
        VectorSet<ScalarT,EigenDim> dst_corr, src_corr;
        selectCorrespondingPoints<ScalarT,EigenDim,CorrValueT>(corr, dst, src, dst_corr, src_corr);
        return estimateRigidTransformPointToPointClosedForm<ScalarT,EigenDim>(dst_corr, src_corr, tform);
    }

    template <typename ScalarT, class PointCorrWeightEvaluatorT = UnityWeightEvaluator<ScalarT,ScalarT>, class PlaneCorrWeightEvaluatorT = UnityWeightEvaluator<ScalarT,ScalarT>>
    bool estimateRigidTransformCombinedMetric3(const ConstVectorSetMatrixMap<ScalarT,3> &dst_p,
                                               const ConstVectorSetMatrixMap<ScalarT,3> &dst_n,
                                               const ConstVectorSetMatrixMap<ScalarT,3> &src_p,
                                               const CorrespondenceSet<typename PointCorrWeightEvaluatorT::InputScalar> &point_to_point_correspondences,
                                               ScalarT point_to_point_weight,
                                               const CorrespondenceSet<typename PlaneCorrWeightEvaluatorT::InputScalar> &point_to_plane_correspondences,
                                               ScalarT point_to_plane_weight,
                                               RigidTransform<ScalarT,3> &tform,
                                               size_t max_iter = 1,
                                               ScalarT convergence_tol = (ScalarT)1e-5,
                                               const PointCorrWeightEvaluatorT &point_corr_evaluator = PointCorrWeightEvaluatorT(),
                                               const PlaneCorrWeightEvaluatorT &plane_corr_evaluator = PlaneCorrWeightEvaluatorT())
    {
        tform.setIdentity();

        const bool has_point_to_point_terms = !point_to_point_correspondences.empty() && (point_to_point_weight > (ScalarT)0.0);
        const bool has_point_to_plane_terms = !point_to_plane_correspondences.empty() && (point_to_plane_weight > (ScalarT)0.0);

        if ((!has_point_to_point_terms && !has_point_to_plane_terms) ||
            (has_point_to_plane_terms && dst_p.cols() != dst_n.cols()))
        {
            return false;
        }

        Eigen::Matrix<ScalarT,6,6> AtA;
        Eigen::Matrix<ScalarT,6,1> Atb;

        Eigen::Matrix<ScalarT,3,3> rot_mat_iter;
        Eigen::Matrix<ScalarT,6,1> d_theta;

#pragma omp declare reduction (+: Eigen::Matrix<ScalarT,6,6>: omp_out = omp_out + omp_in) initializer(omp_priv = Eigen::Matrix<ScalarT,6,6>::Zero())
#pragma omp declare reduction (+: Eigen::Matrix<ScalarT,6,1>: omp_out = omp_out + omp_in) initializer(omp_priv = Eigen::Matrix<ScalarT,6,1>::Zero())

        size_t iter = 0;
        while (iter < max_iter) {
            // Compute differential
            AtA.setZero();
            Atb.setZero();
#pragma omp parallel reduction (+: AtA) reduction (+: Atb)
            {
                if (has_point_to_point_terms) {
#pragma omp for nowait
                    for (size_t i = 0; i < point_to_point_correspondences.size(); i++) {
                        const auto& corr = point_to_point_correspondences[i];
                        const auto d = dst_p.col(corr.indexInFirst);
                        const ScalarT weight = point_to_point_weight*point_corr_evaluator(corr.indexInFirst, corr.indexInSecond, corr.value);
                        Vector<ScalarT,3> s = tform*src_p.col(corr.indexInSecond);

                        Eigen::Matrix<ScalarT,6,3> eq_vecs;

                        eq_vecs(0,0) = (ScalarT)0.0;
                        eq_vecs(1,0) = s[2];
                        eq_vecs(2,0) = -s[1];
                        eq_vecs(3,0) = (ScalarT)1.0;
                        eq_vecs(4,0) = (ScalarT)0.0;
                        eq_vecs(5,0) = (ScalarT)0.0;

                        eq_vecs(0,1) = -s[2];
                        eq_vecs(1,1) = (ScalarT)0.0;
                        eq_vecs(2,1) = s[0];
                        eq_vecs(3,1) = (ScalarT)0.0;
                        eq_vecs(4,1) = (ScalarT)1.0;
                        eq_vecs(5,1) = (ScalarT)0.0;

                        eq_vecs(0,2) = s[1];
                        eq_vecs(1,2) = -s[0];
                        eq_vecs(2,2) = (ScalarT)0.0;
                        eq_vecs(3,2) = (ScalarT)0.0;
                        eq_vecs(4,2) = (ScalarT)0.0;
                        eq_vecs(5,2) = (ScalarT)1.0;

                        Eigen::Matrix<ScalarT,6,6> AtA_priv((weight*eq_vecs)*eq_vecs.transpose());
                        Eigen::Matrix<ScalarT,6,1> Atb_priv(eq_vecs*(weight*Eigen::Matrix<ScalarT,3,1>(d[0]-s[0],d[1]-s[1],d[2]-s[2])));

                        AtA += AtA_priv;
                        Atb += Atb_priv;
                    }
                }
                if (has_point_to_plane_terms) {
#pragma omp for nowait
                    for (size_t i = 0; i < point_to_plane_correspondences.size(); i++) {
                        const auto& corr = point_to_plane_correspondences[i];
                        const auto d = dst_p.col(corr.indexInFirst);
                        const auto n = dst_n.col(corr.indexInFirst);
                        const ScalarT weight = point_to_plane_weight*plane_corr_evaluator(corr.indexInFirst, corr.indexInSecond, corr.value);
                        Vector<ScalarT,3> s = tform*src_p.col(corr.indexInSecond);;

                        Eigen::Matrix<ScalarT,6,1> eq_vec;
                        eq_vec(0) = (n[2]*s[1] - n[1]*s[2]);
                        eq_vec(1) = (n[0]*s[2] - n[2]*s[0]);
                        eq_vec(2) = (n[1]*s[0] - n[0]*s[1]);
                        eq_vec(3) = n[0];
                        eq_vec(4) = n[1];
                        eq_vec(5) = n[2];

                        Eigen::Matrix<ScalarT,6,6> AtA_priv((weight*eq_vec)*eq_vec.transpose());
                        Eigen::Matrix<ScalarT,6,1> Atb_priv((weight*(n[0]*d[0] + n[1]*d[1] + n[2]*d[2] - n[0]*s[0] - n[1]*s[1] - n[2]*s[2]))*eq_vec);

                        AtA += AtA_priv;
                        Atb += Atb_priv;
                    }
                }
            }

            d_theta.noalias() = AtA.ldlt().solve(Atb);

            // Update estimate
            rot_mat_iter.noalias() = (Eigen::AngleAxis<ScalarT>(d_theta[2], Eigen::Matrix<ScalarT,3,1>::UnitZ()) *
                                      Eigen::AngleAxis<ScalarT>(d_theta[1], Eigen::Matrix<ScalarT,3,1>::UnitY()) *
                                      Eigen::AngleAxis<ScalarT>(d_theta[0], Eigen::Matrix<ScalarT,3,1>::UnitX())).matrix();

            tform.linear() = rot_mat_iter*tform.linear();
            tform.linear() = tform.rotation();
            tform.translation() = rot_mat_iter*tform.translation() + d_theta.tail(3);

            iter++;

            // Check for convergence
            if (d_theta.norm() < convergence_tol) return true;
        }

        return false;
    }
}
