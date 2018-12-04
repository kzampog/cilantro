#pragma once

#include <cilantro/space_transformations.hpp>
#include <cilantro/correspondence.hpp>
#include <cilantro/common_pair_evaluators.hpp>

namespace cilantro {
    // Point-to-point rigid (closed form, SVD)
    template <class TransformT>
    typename std::enable_if<int(TransformT::Mode) == int(Eigen::Isometry),bool>::type
    estimateTransformPointToPointMetric(const ConstVectorSetMatrixMap<typename TransformT::Scalar,TransformT::Dim> &dst,
                                        const ConstVectorSetMatrixMap<typename TransformT::Scalar,TransformT::Dim> &src,
                                        TransformT &tform)
    {
        typedef typename TransformT::Scalar ScalarT;

        if (src.cols() != dst.cols() || src.cols() == 0) {
            tform.setIdentity();
            return false;
        }

        Vector<ScalarT,TransformT::Dim> mu_dst(dst.rowwise().mean());
        Vector<ScalarT,TransformT::Dim> mu_src(src.rowwise().mean());

        Eigen::Matrix<ScalarT,TransformT::Dim,Eigen::Dynamic,Eigen::RowMajor> dst_centered(dst.colwise() - mu_dst);
        Eigen::Matrix<ScalarT,TransformT::Dim,Eigen::Dynamic,Eigen::RowMajor> src_centered(src.colwise() - mu_src);

        Eigen::Matrix<ScalarT,TransformT::Dim,TransformT::Dim> sigma((ScalarT(1)/ScalarT(dst.cols()))*(dst_centered*src_centered.transpose()));

        Eigen::JacobiSVD<Eigen::Matrix<ScalarT,TransformT::Dim,TransformT::Dim>> svd(sigma, Eigen::ComputeFullU | Eigen::ComputeFullV);
        if ((svd.matrixU()*svd.matrixV()).determinant() < (ScalarT)0.0) {
            Eigen::Matrix<ScalarT,TransformT::Dim,TransformT::Dim> U(svd.matrixU());
            U.col(dst.rows()-1) *= (ScalarT)(-1.0);
            tform.linear().noalias() = U*svd.matrixV().transpose();
        } else {
            tform.linear().noalias() = svd.matrixU()*svd.matrixV().transpose();
        }
        tform.translation().noalias() = mu_dst - tform.linear()*mu_src;

        return src.cols() >= TransformT::Dim;
    }

    template <class TransformT>
    typename std::enable_if<int(TransformT::Mode) == int(Eigen::Affine) || int(TransformT::Mode) == int(Eigen::AffineCompact),bool>::type
    estimateTransformPointToPointMetric(const ConstVectorSetMatrixMap<typename TransformT::Scalar,TransformT::Dim> &dst,
                                        const ConstVectorSetMatrixMap<typename TransformT::Scalar,TransformT::Dim> &src,
                                        TransformT &tform)
    {
        typedef typename TransformT::Scalar ScalarT;
        enum { Dim = TransformT::Dim, NumUnknowns = TransformT::Dim*(TransformT::Dim + 1) };

        if (src.cols() != dst.cols() || src.cols() == 0) {
            tform.setIdentity();
            return false;
        }

        Eigen::Matrix<ScalarT,NumUnknowns,NumUnknowns> AtA(Eigen::Matrix<ScalarT,NumUnknowns,NumUnknowns>::Zero());
        Eigen::Matrix<ScalarT,NumUnknowns,1> Atb(Eigen::Matrix<ScalarT,NumUnknowns,1>::Zero());

#pragma omp declare reduction (+: Eigen::Matrix<ScalarT,NumUnknowns,NumUnknowns>: omp_out = omp_out + omp_in) initializer(omp_priv = Eigen::Matrix<ScalarT,NumUnknowns,NumUnknowns>::Zero())
#pragma omp declare reduction (+: Eigen::Matrix<ScalarT,NumUnknowns,1>: omp_out = omp_out + omp_in) initializer(omp_priv = Eigen::Matrix<ScalarT,NumUnknowns,1>::Zero())

#pragma omp parallel for reduction (+: AtA) reduction (+: Atb)
        for (size_t i = 0; i < src.cols(); i++) {
            Eigen::Matrix<ScalarT,NumUnknowns,Dim> eq_vecs(Eigen::Matrix<ScalarT,NumUnknowns,Dim>::Zero());
            eq_vecs.template block<Dim,Dim>(Dim*Dim, 0).setIdentity();

            for (size_t j = 0; j < Dim; j++) {
                eq_vecs.template block<Dim,1>(j*Dim, j) = src.col(i);
            }

            Eigen::Matrix<ScalarT,NumUnknowns,NumUnknowns> AtA_priv(eq_vecs*eq_vecs.transpose());
            Eigen::Matrix<ScalarT,NumUnknowns,1> Atb_priv(eq_vecs*dst.col(i));

            AtA += AtA_priv;
            Atb += Atb_priv;
        }

        Eigen::Matrix<ScalarT,NumUnknowns,1> theta = AtA.ldlt().solve(Atb);

        tform.linear() = Eigen::Map<Eigen::Matrix<ScalarT,Dim,Dim,Eigen::RowMajor>>(theta.data(), Dim, Dim);
        tform.translation() = theta.template tail<Dim>();

        return src.cols() >= Dim + 1;
    }

    template <class TransformT, typename CorrValueT = typename TransformT::Scalar>
    bool estimateTransformPointToPointMetric(const ConstVectorSetMatrixMap<typename TransformT::Scalar,TransformT::Dim> &dst,
                                             const ConstVectorSetMatrixMap<typename TransformT::Scalar,TransformT::Dim> &src,
                                             const CorrespondenceSet<CorrValueT> &corr,
                                             TransformT &tform)
    {
        VectorSet<typename TransformT::Scalar,TransformT::Dim> dst_corr, src_corr;
        selectCorrespondingPoints<typename TransformT::Scalar,TransformT::Dim,CorrValueT>(corr, dst, src, dst_corr, src_corr);
        return estimateTransformPointToPointMetric(dst_corr, src_corr, tform);
    }

    template <class TransformT, class PointCorrWeightEvaluatorT = UnityWeightEvaluator<typename TransformT::Scalar,typename TransformT::Scalar>, class PlaneCorrWeightEvaluatorT = UnityWeightEvaluator<typename TransformT::Scalar,typename TransformT::Scalar>>
    typename std::enable_if<int(TransformT::Mode) == int(Eigen::Isometry) && TransformT::Dim == 3,bool>::type
    estimateTransformCombinedMetric(const ConstVectorSetMatrixMap<typename TransformT::Scalar,3> &dst_p,
                                    const ConstVectorSetMatrixMap<typename TransformT::Scalar,3> &dst_n,
                                    const ConstVectorSetMatrixMap<typename TransformT::Scalar,3> &src_p,
                                    const CorrespondenceSet<typename PointCorrWeightEvaluatorT::InputScalar> &point_to_point_correspondences,
                                    typename TransformT::Scalar point_to_point_weight,
                                    const CorrespondenceSet<typename PlaneCorrWeightEvaluatorT::InputScalar> &point_to_plane_correspondences,
                                    typename TransformT::Scalar point_to_plane_weight,
                                    TransformT &tform,
                                    size_t max_iter = 1,
                                    typename TransformT::Scalar convergence_tol = (typename TransformT::Scalar)1e-5,
                                    const PointCorrWeightEvaluatorT &point_corr_evaluator = PointCorrWeightEvaluatorT(),
                                    const PlaneCorrWeightEvaluatorT &plane_corr_evaluator = PlaneCorrWeightEvaluatorT())
    {
        typedef typename TransformT::Scalar ScalarT;

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

                        Eigen::Matrix<ScalarT,6,3,Eigen::RowMajor> eq_vecs;

                        eq_vecs(0,0) = (ScalarT)0.0;
                        eq_vecs(0,1) = -s[2];
                        eq_vecs(0,2) = s[1];

                        eq_vecs(1,0) = s[2];
                        eq_vecs(1,1) = (ScalarT)0.0;
                        eq_vecs(1,2) = -s[0];

                        eq_vecs(2,0) = -s[1];
                        eq_vecs(2,1) = s[0];
                        eq_vecs(2,2) = (ScalarT)0.0;

                        eq_vecs(3,0) = (ScalarT)1.0;
                        eq_vecs(3,1) = (ScalarT)0.0;
                        eq_vecs(3,2) = (ScalarT)0.0;

                        eq_vecs(4,0) = (ScalarT)0.0;
                        eq_vecs(4,1) = (ScalarT)1.0;
                        eq_vecs(4,2) = (ScalarT)0.0;

                        eq_vecs(5,0) = (ScalarT)0.0;
                        eq_vecs(5,1) = (ScalarT)0.0;
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
