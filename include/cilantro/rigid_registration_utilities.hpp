#pragma once

#include <cilantro/space_transformations.hpp>
#include <cilantro/correspondence.hpp>
#include <cilantro/common_pair_evaluators.hpp>

namespace cilantro {
    // Point-to-point (closed form, SVD)
    template <typename ScalarT, ptrdiff_t EigenDim>
    bool estimateRigidTransformPointToPointClosedForm(const ConstVectorSetMatrixMap<ScalarT,EigenDim> &dst,
                                                      const ConstVectorSetMatrixMap<ScalarT,EigenDim> &src,
                                                      RigidTransformation<ScalarT,EigenDim> &tform)
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
                                                      RigidTransformation<ScalarT,EigenDim> &tform)
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
                                               RigidTransformation<ScalarT,3> &tform,
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

        const size_t num_point_to_point_equations = 3*has_point_to_point_terms*point_to_point_correspondences.size();
        const size_t num_point_to_plane_equations = has_point_to_plane_terms*point_to_plane_correspondences.size();
        const size_t num_equations = num_point_to_point_equations + num_point_to_plane_equations;

        Eigen::Matrix<ScalarT,6,Eigen::Dynamic> At(6, num_equations);
        Eigen::Matrix<ScalarT,Eigen::Dynamic,1> b(num_equations, 1);

        const ScalarT point_to_point_weight_sqrt = std::sqrt(point_to_point_weight);
        const ScalarT point_to_plane_weight_sqrt = std::sqrt(point_to_plane_weight);
        Eigen::Matrix<ScalarT,3,3> rot_mat_iter;
        Eigen::Matrix<ScalarT,6,1> d_theta;
        Vector<ScalarT,3> s;
        size_t eq_ind;

        size_t iter = 0;
        while (iter < max_iter) {
            // Compute differential
#pragma omp parallel shared (At, b) private (s, eq_ind)
            {
                if (has_point_to_point_terms) {
#pragma omp for nowait
                    for (size_t i = 0; i < point_to_point_correspondences.size(); i++) {
                        const auto& corr = point_to_point_correspondences[i];
                        const auto d = dst_p.col(corr.indexInFirst);
                        const ScalarT weight = point_to_point_weight_sqrt*std::sqrt(point_corr_evaluator(corr.indexInFirst, corr.indexInSecond, corr.value));
                        s.noalias() = tform*src_p.col(corr.indexInSecond);

                        eq_ind = 3*i;

                        At(0,eq_ind) = (ScalarT)0.0;
                        At(1,eq_ind) = s[2]*weight;
                        At(2,eq_ind) = -s[1]*weight;
                        At(3,eq_ind) = weight;
                        At(4,eq_ind) = (ScalarT)0.0;
                        At(5,eq_ind) = (ScalarT)0.0;
                        b[eq_ind++] = (d[0] - s[0])*weight;

                        At(0,eq_ind) = -s[2]*weight;
                        At(1,eq_ind) = (ScalarT)0.0;
                        At(2,eq_ind) = s[0]*weight;
                        At(3,eq_ind) = (ScalarT)0.0;
                        At(4,eq_ind) = weight;
                        At(5,eq_ind) = (ScalarT)0.0;
                        b[eq_ind++] = (d[1] - s[1])*weight;

                        At(0,eq_ind) = s[1]*weight;
                        At(1,eq_ind) = -s[0]*weight;
                        At(2,eq_ind) = (ScalarT)0.0;
                        At(3,eq_ind) = (ScalarT)0.0;
                        At(4,eq_ind) = (ScalarT)0.0;
                        At(5,eq_ind) = weight;
                        b[eq_ind] = (d[2] - s[2])*weight;
                    }
                }
                if (has_point_to_plane_terms) {
#pragma omp for nowait
                    for (size_t i = 0; i < point_to_plane_correspondences.size(); i++) {
                        const auto& corr = point_to_plane_correspondences[i];
                        const auto d = dst_p.col(corr.indexInFirst);
                        const auto n = dst_n.col(corr.indexInFirst);
                        const ScalarT weight = point_to_plane_weight_sqrt*std::sqrt(plane_corr_evaluator(corr.indexInFirst, corr.indexInSecond, corr.value));
                        s.noalias() = tform*src_p.col(corr.indexInSecond);

                        eq_ind = num_point_to_point_equations + i;

                        At(0,eq_ind) = (n[2]*s[1] - n[1]*s[2])*weight;
                        At(1,eq_ind) = (n[0]*s[2] - n[2]*s[0])*weight;
                        At(2,eq_ind) = (n[1]*s[0] - n[0]*s[1])*weight;
                        At(3,eq_ind) = n[0]*weight;
                        At(4,eq_ind) = n[1]*weight;
                        At(5,eq_ind) = n[2]*weight;
                        b[eq_ind] = (n[0]*d[0] + n[1]*d[1] + n[2]*d[2] - n[0]*s[0] - n[1]*s[1] - n[2]*s[2])*weight;
                    }
                }
            }

            d_theta.noalias() = (At*At.transpose()).ldlt().solve(At*b);

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
