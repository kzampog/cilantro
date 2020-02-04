#pragma once

#include <cilantro/config.hpp>
#include <cilantro/core/space_transformations.hpp>
#include <cilantro/core/correspondence.hpp>
#include <cilantro/core/common_pair_evaluators.hpp>
#include <cilantro/core/openmp_reductions.hpp>

namespace cilantro {
    // Rigid, point-to-point, general dimension, closed form, SVD
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

    // Affine, point-to-point, general dimension, closed form
    template <class TransformT>
    typename std::enable_if<int(TransformT::Mode) == int(Eigen::Affine) || int(TransformT::Mode) == int(Eigen::AffineCompact),bool>::type
    estimateTransformPointToPointMetric(const ConstVectorSetMatrixMap<typename TransformT::Scalar,TransformT::Dim> &dst,
                                        const ConstVectorSetMatrixMap<typename TransformT::Scalar,TransformT::Dim> &src,
                                        TransformT &tform)
    {
        typedef typename TransformT::Scalar ScalarT;
        enum {
            Dim = TransformT::Dim,
            NumUnknowns = TransformT::Dim*(TransformT::Dim + 1)
        };

        if (src.cols() != dst.cols() || src.cols() == 0) {
            tform.setIdentity();
            return false;
        }

        Eigen::Matrix<ScalarT,NumUnknowns,NumUnknowns> AtA(Eigen::Matrix<ScalarT,NumUnknowns,NumUnknowns>::Zero());
        Eigen::Matrix<ScalarT,NumUnknowns,1> Atb(Eigen::Matrix<ScalarT,NumUnknowns,1>::Zero());

#ifdef ENABLE_NON_DETERMINISTIC_PARALLELISM
DECLARE_MATRIX_SUM_REDUCTION(ScalarT,NumUnknowns,NumUnknowns)
DECLARE_MATRIX_SUM_REDUCTION(ScalarT,NumUnknowns,1)
#pragma omp parallel MATRIX_SUM_REDUCTION(ScalarT,NumUnknowns,NumUnknowns,AtA) MATRIX_SUM_REDUCTION(ScalarT,NumUnknowns,1,Atb)
//#pragma omp parallel reduction (internal::MatrixReductions<ScalarT,NumUnknowns,NumUnknowns>::operator+: AtA) reduction (internal::MatrixReductions<ScalarT,NumUnknowns,1>::operator+: Atb)
#endif
        {
            Eigen::Matrix<ScalarT,NumUnknowns,Dim> eq_vecs;
            eq_vecs.template block<Dim*Dim,Dim>(0, 0).setZero();
            eq_vecs.template block<Dim,Dim>(Dim*Dim, 0).setIdentity();

#ifdef ENABLE_NON_DETERMINISTIC_PARALLELISM
#pragma omp for nowait
#endif
            for (size_t i = 0; i < src.cols(); i++) {
                for (size_t j = 0; j < Dim; j++) {
                    eq_vecs.template block<Dim,1>(j*Dim, j) = src.col(i);
                }

                AtA += eq_vecs*eq_vecs.transpose();
                Atb += eq_vecs*dst.col(i);
            }
        }

        Eigen::Matrix<ScalarT,NumUnknowns,1> theta = AtA.ldlt().solve(Atb);

        tform.linear() = Eigen::Map<Eigen::Matrix<ScalarT,Dim,Dim,Eigen::RowMajor>>(theta.data(), Dim, Dim);
        tform.translation() = theta.template tail<Dim>();

        return src.cols() >= Dim + 1;
    }

    template <class TransformT, typename CorrSetT>
    bool estimateTransformPointToPointMetric(const ConstVectorSetMatrixMap<typename TransformT::Scalar,TransformT::Dim> &dst,
                                             const ConstVectorSetMatrixMap<typename TransformT::Scalar,TransformT::Dim> &src,
                                             const CorrSetT &corr,
                                             TransformT &tform)
    {
        VectorSet<typename TransformT::Scalar,TransformT::Dim> dst_corr, src_corr;
        selectCorrespondingPoints<typename TransformT::Scalar,TransformT::Dim,CorrSetT>(corr, dst, src, dst_corr, src_corr);
        return estimateTransformPointToPointMetric(dst_corr, src_corr, tform);
    }

    // Rigid, combined metric, 2D, iterative
    template <class TransformT, class PointCorrSetT, class PlaneCorrSetT, class PointCorrWeightEvaluatorT = UnityWeightEvaluator<typename TransformT::Scalar,typename TransformT::Scalar>, class PlaneCorrWeightEvaluatorT = UnityWeightEvaluator<typename TransformT::Scalar,typename TransformT::Scalar>>
    typename std::enable_if<int(TransformT::Mode) == int(Eigen::Isometry) && TransformT::Dim == 2,bool>::type
    estimateTransformCombinedMetric(const ConstVectorSetMatrixMap<typename TransformT::Scalar,2> &dst_p,
                                    const ConstVectorSetMatrixMap<typename TransformT::Scalar,2> &dst_n,
                                    const ConstVectorSetMatrixMap<typename TransformT::Scalar,2> &src_p,
                                    const PointCorrSetT &point_to_point_correspondences,
                                    typename TransformT::Scalar point_to_point_weight,
                                    const PlaneCorrSetT &point_to_plane_correspondences,
                                    typename TransformT::Scalar point_to_plane_weight,
                                    TransformT &tform,
                                    size_t max_iter = 1,
                                    typename TransformT::Scalar convergence_tol = (typename TransformT::Scalar)1e-5,
                                    const PointCorrWeightEvaluatorT &point_corr_evaluator = PointCorrWeightEvaluatorT(),
                                    const PlaneCorrWeightEvaluatorT &plane_corr_evaluator = PlaneCorrWeightEvaluatorT(),
                                    const Vector<typename TransformT::Scalar,2>& dst_mean = Vector<typename TransformT::Scalar,2>::Zero(),
                                    const Vector<typename TransformT::Scalar,2>& src_mean = Vector<typename TransformT::Scalar,2>::Zero())
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

        const Eigen::Translation<ScalarT,2> t_dst(dst_mean);
        const Eigen::Translation<ScalarT,2> t_src(-src_mean);
        Eigen::Matrix<ScalarT,2,2> flip;
        flip << 0, -1, 1, 0;

        Eigen::Matrix<ScalarT,3,3> AtA;
        Eigen::Matrix<ScalarT,3,1> Atb;

        Eigen::Matrix<ScalarT,2,2> rot_mat_iter;
        Eigen::Matrix<ScalarT,3,1> d_theta;

        for (size_t iter = 0; iter < max_iter; ++iter) {
            // Compute differential
            AtA.setZero();
            Atb.setZero();
#ifdef ENABLE_NON_DETERMINISTIC_PARALLELISM
DECLARE_MATRIX_SUM_REDUCTION(ScalarT,3,3)
DECLARE_MATRIX_SUM_REDUCTION(ScalarT,3,1)
#pragma omp parallel MATRIX_SUM_REDUCTION(ScalarT,3,3,AtA) MATRIX_SUM_REDUCTION(ScalarT,3,1,Atb)
//#pragma omp parallel reduction (internal::MatrixReductions<ScalarT,3,3>::operator+: AtA) reduction (internal::MatrixReductions<ScalarT,3,1>::operator+: Atb)
#endif
            {
                if (has_point_to_point_terms) {
                    Eigen::Matrix<ScalarT,3,2,Eigen::RowMajor> eq_vecs;
                    eq_vecs.template bottomRows<2>().setIdentity();
#ifdef ENABLE_NON_DETERMINISTIC_PARALLELISM
#pragma omp for nowait
#endif
                    for (size_t i = 0; i < point_to_point_correspondences.size(); i++) {
                        const auto& corr = point_to_point_correspondences[i];
                        const Vector<ScalarT,2> d = dst_p.col(corr.indexInFirst) - dst_mean;
                        const ScalarT weight = point_to_point_weight * point_corr_evaluator(corr.indexInFirst, corr.indexInSecond, corr.value);
                        const Vector<ScalarT,2> s = tform * (src_p.col(corr.indexInSecond) - src_mean);

                        eq_vecs(0,0) = -(s[1] + d[1]);
                        eq_vecs(0,1) = s[0] + d[0];

                        AtA += weight * eq_vecs * eq_vecs.transpose();
                        Atb += weight * eq_vecs * (d - s);
                    }
                }

                if (has_point_to_plane_terms) {
                    Eigen::Matrix<ScalarT,3,1> eq_vec;
#ifdef ENABLE_NON_DETERMINISTIC_PARALLELISM
#pragma omp for nowait
#endif
                    for (size_t i = 0; i < point_to_plane_correspondences.size(); i++) {
                        const auto& corr = point_to_plane_correspondences[i];
                        const ScalarT weight = point_to_plane_weight * plane_corr_evaluator(corr.indexInFirst, corr.indexInSecond, corr.value);
                        const Vector<ScalarT,2> d = dst_p.col(corr.indexInFirst) - dst_mean;
                        const Vector<ScalarT,2> n = dst_n.col(corr.indexInFirst);
                        const Vector<ScalarT,2> s = tform * (src_p.col(corr.indexInSecond) - src_mean);

                        eq_vec(0) = (d + s).dot(flip * n);
                        eq_vec.template tail<2>() = n;

                        AtA += weight * eq_vec * eq_vec.transpose();
                        Atb += weight * (n.dot(d - s)) * eq_vec;
                    }
                }
            }

            d_theta.noalias() = AtA.ldlt().solve(Atb);

            // Update estimate
            const ScalarT theta = std::atan(d_theta(0));
            const ScalarT denom = std::sqrt(1 + d_theta(0) * d_theta(0));
            const Eigen::Rotation2D<ScalarT> Ra(theta);
            const Eigen::Translation<ScalarT, 2> ta(std::cos(theta) * d_theta.template tail<2>());
            tform = Ra * ta * Ra * tform;

            // Check for convergence
            if (d_theta.norm() < convergence_tol) {
                tform = t_dst * tform * t_src;
                return true;
            }
        }
        tform = t_dst * tform * t_src;

        return false;
    }

    // Rigid, combined metric, 3D, iterative
    template <class TransformT, class PointCorrSetT, class PlaneCorrSetT, class PointCorrWeightEvaluatorT = UnityWeightEvaluator<typename TransformT::Scalar,typename TransformT::Scalar>, class PlaneCorrWeightEvaluatorT = UnityWeightEvaluator<typename TransformT::Scalar,typename TransformT::Scalar>>
    typename std::enable_if<int(TransformT::Mode) == int(Eigen::Isometry) && TransformT::Dim == 3,bool>::type
    estimateTransformCombinedMetric(const ConstVectorSetMatrixMap<typename TransformT::Scalar,3> &dst_p,
                                    const ConstVectorSetMatrixMap<typename TransformT::Scalar,3> &dst_n,
                                    const ConstVectorSetMatrixMap<typename TransformT::Scalar,3> &src_p,
                                    const PointCorrSetT &point_to_point_correspondences,
                                    typename TransformT::Scalar point_to_point_weight,
                                    const PlaneCorrSetT &point_to_plane_correspondences,
                                    typename TransformT::Scalar point_to_plane_weight,
                                    TransformT &tform,
                                    size_t max_iter = 1,
                                    typename TransformT::Scalar convergence_tol = (typename TransformT::Scalar)1e-5,
                                    const PointCorrWeightEvaluatorT &point_corr_evaluator = PointCorrWeightEvaluatorT(),
                                    const PlaneCorrWeightEvaluatorT &plane_corr_evaluator = PlaneCorrWeightEvaluatorT(),
                                    const Vector<typename TransformT::Scalar,3>& dst_mean = Vector<typename TransformT::Scalar,3>::Zero(),
                                    const Vector<typename TransformT::Scalar,3>& src_mean = Vector<typename TransformT::Scalar,3>::Zero())
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

        const Eigen::Translation<ScalarT,3> t_dst(dst_mean);
        const Eigen::Translation<ScalarT,3> t_src(-src_mean);

        Eigen::Matrix<ScalarT,6,6> AtA;
        Eigen::Matrix<ScalarT,6,1> Atb;

        Eigen::Matrix<ScalarT,3,3> rot_mat_iter;
        Eigen::Matrix<ScalarT,6,1> d_theta;

        for (size_t iter = 0; iter < max_iter; ++iter) {
            // Compute differential
            AtA.setZero();
            Atb.setZero();
#ifdef ENABLE_NON_DETERMINISTIC_PARALLELISM
DECLARE_MATRIX_SUM_REDUCTION(ScalarT,6,6)
DECLARE_MATRIX_SUM_REDUCTION(ScalarT,6,1)
#pragma omp parallel MATRIX_SUM_REDUCTION(ScalarT,6,6,AtA) MATRIX_SUM_REDUCTION(ScalarT,6,1,Atb)
//#pragma omp parallel reduction (internal::MatrixReductions<ScalarT,6,6>::operator+: AtA) reduction (internal::MatrixReductions<ScalarT,6,1>::operator+: Atb)
#endif
            {
                if (has_point_to_point_terms) {
                    Eigen::Matrix<ScalarT,6,3,Eigen::RowMajor> eq_vecs;
                    eq_vecs.template bottomRows<3>().setIdentity();
#ifdef ENABLE_NON_DETERMINISTIC_PARALLELISM
#pragma omp for nowait
#endif
                    for (size_t i = 0; i < point_to_point_correspondences.size(); i++) {
                        const auto& corr = point_to_point_correspondences[i];
                        const Vector<ScalarT,3> d = dst_p.col(corr.indexInFirst) - dst_mean;
                        const ScalarT weight = point_to_point_weight * point_corr_evaluator(corr.indexInFirst, corr.indexInSecond, corr.value);
                        const Vector<ScalarT,3> s = tform * (src_p.col(corr.indexInSecond) - src_mean);

                        eq_vecs(0,0) = (ScalarT)0.0;
                        eq_vecs(1,1) = (ScalarT)0.0;
                        eq_vecs(2,2) = (ScalarT)0.0;

                        eq_vecs(0,1) = -(d[2] + s[2]);
                        eq_vecs(0,2) = (d[1] + s[1]);
                        eq_vecs(1,2) = -(d[0] + s[0]);

                        eq_vecs(1,0) = -eq_vecs(0, 1);
                        eq_vecs(2,0) = -eq_vecs(0, 2);
                        eq_vecs(2,1) = -eq_vecs(1, 2);

                        AtA += weight * eq_vecs * eq_vecs.transpose();
                        Atb += weight * eq_vecs * (d - s);
                    }
                }

                if (has_point_to_plane_terms) {
                    Eigen::Matrix<ScalarT,6,1> eq_vec;
#ifdef ENABLE_NON_DETERMINISTIC_PARALLELISM
#pragma omp for nowait
#endif
                    for (size_t i = 0; i < point_to_plane_correspondences.size(); i++) {
                        const auto& corr = point_to_plane_correspondences[i];
                        const ScalarT weight = point_to_plane_weight*plane_corr_evaluator(corr.indexInFirst, corr.indexInSecond, corr.value);
                        const Vector<ScalarT,3> d = dst_p.col(corr.indexInFirst) - dst_mean;
                        const Vector<ScalarT,3> n = dst_n.col(corr.indexInFirst);
                        const Vector<ScalarT,3> s = tform*(src_p.col(corr.indexInSecond) - src_mean);

                        eq_vec.template head<3>() = (d + s).cross(n);
                        eq_vec.template tail<3>() = n;

                        AtA += weight*eq_vec*eq_vec.transpose();
                        Atb += weight*(n.dot(d - s))*eq_vec;
                    }
                }
            }

            d_theta.noalias() = AtA.ldlt().solve(Atb);

            // Update estimate
            ScalarT na = d_theta.template head<3>().norm();
            ScalarT theta = std::atan(na);
#if EIGEN_VERSION_AT_LEAST(3, 2, 93)
            const Eigen::AngleAxis<ScalarT> Ra(theta, d_theta.template head<3>().stableNormalized());
#else
            const Eigen::AngleAxis<ScalarT> Ra(theta, d_theta.template head<3>().normalized());
#endif
            const Eigen::Translation<ScalarT, 3> ta(std::cos(theta) * d_theta.template tail<3>());
            tform = Ra * ta * Ra * tform;

            // Check for convergence
            if (d_theta.norm() < convergence_tol) {
                tform = t_dst * tform * t_src;
                return true;
            }
        }
        tform = t_dst * tform * t_src;
        return false;
    }

    // Affine, combined metric, general dimension, closed form
    template <class TransformT, class PointCorrSetT, class PlaneCorrSetT, class PointCorrWeightEvaluatorT = UnityWeightEvaluator<typename TransformT::Scalar,typename TransformT::Scalar>, class PlaneCorrWeightEvaluatorT = UnityWeightEvaluator<typename TransformT::Scalar,typename TransformT::Scalar>>
    typename std::enable_if<int(TransformT::Mode) == int(Eigen::Affine) || int(TransformT::Mode) == int(Eigen::AffineCompact),bool>::type
    estimateTransformCombinedMetric(const ConstVectorSetMatrixMap<typename TransformT::Scalar,TransformT::Dim> &dst_p,
                                    const ConstVectorSetMatrixMap<typename TransformT::Scalar,TransformT::Dim> &dst_n,
                                    const ConstVectorSetMatrixMap<typename TransformT::Scalar,TransformT::Dim> &src_p,
                                    const PointCorrSetT &point_to_point_correspondences,
                                    typename TransformT::Scalar point_to_point_weight,
                                    const PlaneCorrSetT &point_to_plane_correspondences,
                                    typename TransformT::Scalar point_to_plane_weight,
                                    TransformT &tform,
                                    size_t /*max_iter = 1*/,
                                    typename TransformT::Scalar /*convergence_tol = (typename TransformT::Scalar)1e-5*/,
                                    const PointCorrWeightEvaluatorT &point_corr_evaluator = PointCorrWeightEvaluatorT(),
                                    const PlaneCorrWeightEvaluatorT &plane_corr_evaluator = PlaneCorrWeightEvaluatorT(),
                                    const Vector<typename TransformT::Scalar,TransformT::Dim>& dst_mean = Vector<typename TransformT::Scalar,TransformT::Dim>::Zero(),
                                    const Vector<typename TransformT::Scalar,TransformT::Dim>& src_mean = Vector<typename TransformT::Scalar,TransformT::Dim>::Zero())
    {
        typedef typename TransformT::Scalar ScalarT;
        enum {
            Dim = TransformT::Dim,
            NumUnknowns = TransformT::Dim*(TransformT::Dim + 1)
        };

        const bool has_point_to_point_terms = !point_to_point_correspondences.empty() && (point_to_point_weight > (ScalarT)0.0);
        const bool has_point_to_plane_terms = !point_to_plane_correspondences.empty() && (point_to_plane_weight > (ScalarT)0.0);

        if ((!has_point_to_point_terms && !has_point_to_plane_terms) ||
            (has_point_to_plane_terms && dst_p.cols() != dst_n.cols()))
        {
            tform.setIdentity();
            return false;
        }

        const Eigen::Translation<ScalarT,Dim> t_dst(dst_mean);
        const Eigen::Translation<ScalarT,Dim> t_src(-src_mean);

        Eigen::Matrix<ScalarT,NumUnknowns,NumUnknowns> AtA(Eigen::Matrix<ScalarT,NumUnknowns,NumUnknowns>::Zero());
        Eigen::Matrix<ScalarT,NumUnknowns,1> Atb(Eigen::Matrix<ScalarT,NumUnknowns,1>::Zero());

#ifdef ENABLE_NON_DETERMINISTIC_PARALLELISM
DECLARE_MATRIX_SUM_REDUCTION(ScalarT,NumUnknowns,NumUnknowns)
DECLARE_MATRIX_SUM_REDUCTION(ScalarT,NumUnknowns,1)
#pragma omp parallel MATRIX_SUM_REDUCTION(ScalarT,NumUnknowns,NumUnknowns,AtA) MATRIX_SUM_REDUCTION(ScalarT,NumUnknowns,1,Atb)
//#pragma omp parallel reduction (internal::MatrixReductions<ScalarT,NumUnknowns,NumUnknowns>::operator+: AtA) reduction (internal::MatrixReductions<ScalarT,NumUnknowns,1>::operator+: Atb)
#endif
        {
            if (has_point_to_point_terms) {
                Eigen::Matrix<ScalarT,NumUnknowns,Dim> eq_vecs;
                eq_vecs.template block<Dim*Dim,Dim>(0, 0).setZero();
                eq_vecs.template block<Dim,Dim>(Dim*Dim, 0).setIdentity();
#ifdef ENABLE_NON_DETERMINISTIC_PARALLELISM
#pragma omp for nowait
#endif
                for (size_t i = 0; i < point_to_point_correspondences.size(); i++) {
                    const auto& corr = point_to_point_correspondences[i];
                    const ScalarT weight = point_to_point_weight*point_corr_evaluator(corr.indexInFirst, corr.indexInSecond, corr.value);

                    for (size_t j = 0; j < Dim; j++) {
                        eq_vecs.template block<Dim,1>(j*Dim, j) = src_p.col(corr.indexInSecond) - src_mean;
                    }

                    AtA += (weight*eq_vecs)*eq_vecs.transpose();
                    Atb += eq_vecs*(weight*(dst_p.col(corr.indexInFirst) - dst_mean));
                }
            }

            if (has_point_to_plane_terms) {
                Eigen::Matrix<ScalarT,NumUnknowns,1> eq_vec;
#ifdef ENABLE_NON_DETERMINISTIC_PARALLELISM
#pragma omp for nowait
#endif
                for (size_t i = 0; i < point_to_plane_correspondences.size(); i++) {
                    const auto& corr = point_to_plane_correspondences[i];
                    const auto n = dst_n.col(corr.indexInFirst);
                    const ScalarT weight = point_to_plane_weight*plane_corr_evaluator(corr.indexInFirst, corr.indexInSecond, corr.value);

                    for (size_t j = 0; j < Dim; j++) {
                        eq_vec.template segment<Dim>(j*Dim) = n[j]*(src_p.col(corr.indexInSecond) - src_mean);
                    }
                    eq_vec.template tail<Dim>() = n;

                    AtA += (weight*eq_vec)*eq_vec.transpose();
                    Atb += (weight*(n.dot(dst_p.col(corr.indexInFirst) - dst_mean)))*eq_vec;
                }
            }
        }

        Eigen::Matrix<ScalarT,NumUnknowns,1> theta = AtA.ldlt().solve(Atb);

        tform.linear() = Eigen::Map<Eigen::Matrix<ScalarT,Dim,Dim,Eigen::RowMajor>>(theta.data(), Dim, Dim);
        tform.translation() = theta.template tail<Dim>();
        tform = t_dst * tform * t_src;

        return has_point_to_point_terms*point_to_point_correspondences.size() + has_point_to_plane_terms*point_to_plane_correspondences.size() >= Dim + 1;
    }

    // Rigid, combined symmetric metric, 2D, iterative
    // For more details about this method see
    // Szymon Rusinkiewicz. A Symmetric Objective Function for ICP. ACM Transactions on Graphics (Proc. SIGGRAPH) 38(4), July 2019.
    template <class TransformT, class PointCorrSetT, class PlaneCorrSetT, class PointCorrWeightEvaluatorT = UnityWeightEvaluator<typename TransformT::Scalar,typename TransformT::Scalar>, class PlaneCorrWeightEvaluatorT = UnityWeightEvaluator<typename TransformT::Scalar,typename TransformT::Scalar>>
    typename std::enable_if<int(TransformT::Mode) == int(Eigen::Isometry) && TransformT::Dim == 2,bool>::type
    estimateTransformSymmetricMetric(const ConstVectorSetMatrixMap<typename TransformT::Scalar,2> &dst_p,
                                     const ConstVectorSetMatrixMap<typename TransformT::Scalar,2> &dst_n,
                                     const ConstVectorSetMatrixMap<typename TransformT::Scalar,2> &src_p,
                                     const ConstVectorSetMatrixMap<typename TransformT::Scalar,2> &src_n,
                                     const PointCorrSetT &point_to_point_correspondences,
                                     typename TransformT::Scalar point_to_point_weight,
                                     const PlaneCorrSetT &point_to_plane_correspondences,
                                     typename TransformT::Scalar point_to_plane_weight,
                                     TransformT &tform,
                                     size_t max_iter = 1,
                                     typename TransformT::Scalar convergence_tol = (typename TransformT::Scalar)1e-5,
                                     const PointCorrWeightEvaluatorT &point_corr_evaluator = PointCorrWeightEvaluatorT(),
                                     const PlaneCorrWeightEvaluatorT &plane_corr_evaluator = PlaneCorrWeightEvaluatorT(),
                                     const Vector<typename TransformT::Scalar,2>& dst_mean = Vector<typename TransformT::Scalar,2>::Zero(),
                                     const Vector<typename TransformT::Scalar,2>& src_mean = Vector<typename TransformT::Scalar,2>::Zero())
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

        const Eigen::Translation<ScalarT,2> t_dst(dst_mean);
        const Eigen::Translation<ScalarT,2> t_src(-src_mean);
        Eigen::Matrix<ScalarT,2,2> flip;
        flip << 0, -1, 1, 0;

        Eigen::Matrix<ScalarT,3,3> AtA;
        Eigen::Matrix<ScalarT,3,1> Atb;

        Eigen::Matrix<ScalarT,3,1> d_theta;

        for (size_t iter = 0; iter < max_iter; ++iter) {
            // Compute differential
            AtA.setZero();
            Atb.setZero();
#ifdef ENABLE_NON_DETERMINISTIC_PARALLELISM
DECLARE_MATRIX_SUM_REDUCTION(ScalarT,3,3)
DECLARE_MATRIX_SUM_REDUCTION(ScalarT,3,1)
#pragma omp parallel MATRIX_SUM_REDUCTION(ScalarT,3,3,AtA) MATRIX_SUM_REDUCTION(ScalarT,3,1,Atb)
//#pragma omp parallel reduction (internal::MatrixReductions<ScalarT,3,3>::operator+: AtA) reduction (internal::MatrixReductions<ScalarT,3,1>::operator+: Atb)
#endif
            {
                if (has_point_to_point_terms) {
                    Eigen::Matrix<ScalarT,3,2,Eigen::RowMajor> eq_vecs;
                    eq_vecs.template bottomRows<2>().setIdentity();
#ifdef ENABLE_NON_DETERMINISTIC_PARALLELISM
#pragma omp for nowait
#endif
                    for (size_t i = 0; i < point_to_point_correspondences.size(); i++) {
                        const auto& corr = point_to_point_correspondences[i];
                        const Vector<ScalarT,2> d = dst_p.col(corr.indexInFirst) - dst_mean;
                        const ScalarT weight = point_to_point_weight * point_corr_evaluator(corr.indexInFirst, corr.indexInSecond, corr.value);
                        const Vector<ScalarT,2> s = tform * (src_p.col(corr.indexInSecond) - src_mean);

                        eq_vecs(0,0) = -(s[1] + d[1]);
                        eq_vecs(0,1) = s[0] + d[0];

                        AtA += weight * eq_vecs * eq_vecs.transpose();
                        Atb += weight * eq_vecs * (d - s);
                    }
                }

                if (has_point_to_plane_terms) {
                    Eigen::Matrix<ScalarT,3,1> eq_vec;
#ifdef ENABLE_NON_DETERMINISTIC_PARALLELISM
#pragma omp for nowait
#endif
                    for (size_t i = 0; i < point_to_plane_correspondences.size(); i++) {
                        const auto& corr = point_to_plane_correspondences[i];
                        const ScalarT weight = point_to_plane_weight * plane_corr_evaluator(corr.indexInFirst, corr.indexInSecond, corr.value);
                        const Vector<ScalarT,2> d = dst_p.col(corr.indexInFirst) - dst_mean;
                        const Vector<ScalarT,2> n = dst_n.col(corr.indexInFirst) + src_n.col(corr.indexInSecond);
                        const Vector<ScalarT,2> s = tform * (src_p.col(corr.indexInSecond) - src_mean);

                        eq_vec(0) = (flip * (d + s)).dot(n);
                        eq_vec.template tail<2>() = n;

                        AtA += weight * eq_vec * eq_vec.transpose();
                        Atb += weight * (n.dot(d - s)) * eq_vec;
                    }
                }
            }

            d_theta.noalias() = AtA.ldlt().solve(Atb);

            // Update estimate
            const ScalarT theta = std::atan(d_theta(0));
            const Eigen::Rotation2D<ScalarT> Ra(theta);
            const Eigen::Translation<ScalarT, 2> ta(std::cos(theta) * d_theta.template tail<2>());
            tform = Ra * ta * Ra * tform;

            // Check for convergence
            if (d_theta.norm() < convergence_tol) {
                tform = t_dst * tform * t_src;
                return true;
            }
        }
        tform = t_dst * tform * t_src;
        return false;
    }

    // Rigid, combined symmetric metric, 3D, iterative
    // For more details about this method see
    // Szymon Rusinkiewicz. A Symmetric Objective Function for ICP. ACM Transactions on Graphics (Proc. SIGGRAPH) 38(4), July 2019.
    template <class TransformT, class PointCorrSetT, class PlaneCorrSetT, class PointCorrWeightEvaluatorT = UnityWeightEvaluator<typename TransformT::Scalar,typename TransformT::Scalar>, class PlaneCorrWeightEvaluatorT = UnityWeightEvaluator<typename TransformT::Scalar,typename TransformT::Scalar>>
    typename std::enable_if<int(TransformT::Mode) == int(Eigen::Isometry) && TransformT::Dim == 3,bool>::type
    estimateTransformSymmetricMetric(const ConstVectorSetMatrixMap<typename TransformT::Scalar,3> &dst_p,
                                     const ConstVectorSetMatrixMap<typename TransformT::Scalar,3> &dst_n,
                                     const ConstVectorSetMatrixMap<typename TransformT::Scalar,3> &src_p,
                                     const ConstVectorSetMatrixMap<typename TransformT::Scalar,3> &src_n,
                                     const PointCorrSetT &point_to_point_correspondences,
                                     typename TransformT::Scalar point_to_point_weight,
                                     const PlaneCorrSetT &point_to_plane_correspondences,
                                     typename TransformT::Scalar point_to_plane_weight,
                                     TransformT &tform,
                                     size_t max_iter = 1,
                                     typename TransformT::Scalar convergence_tol = (typename TransformT::Scalar)1e-5,
                                     const PointCorrWeightEvaluatorT &point_corr_evaluator = PointCorrWeightEvaluatorT(),
                                     const PlaneCorrWeightEvaluatorT &plane_corr_evaluator = PlaneCorrWeightEvaluatorT(),
                                     const Vector<typename TransformT::Scalar,3>& dst_mean = Vector<typename TransformT::Scalar,3>::Zero(),
                                     const Vector<typename TransformT::Scalar,3>& src_mean = Vector<typename TransformT::Scalar,3>::Zero())
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

        const Eigen::Translation<ScalarT,3> t_dst(dst_mean);
        const Eigen::Translation<ScalarT,3> t_src(-src_mean);

        Eigen::Matrix<ScalarT,6,6> AtA;
        Eigen::Matrix<ScalarT,6,1> Atb;

        Eigen::Matrix<ScalarT,3,3> rot_mat_iter;
        Eigen::Matrix<ScalarT,6,1> d_theta;

        for (size_t iter = 0; iter < max_iter; ++iter) {
            // Compute differential
            AtA.setZero();
            Atb.setZero();
#ifdef ENABLE_NON_DETERMINISTIC_PARALLELISM
DECLARE_MATRIX_SUM_REDUCTION(ScalarT,6,6)
DECLARE_MATRIX_SUM_REDUCTION(ScalarT,6,1)
#pragma omp parallel MATRIX_SUM_REDUCTION(ScalarT,6,6,AtA) MATRIX_SUM_REDUCTION(ScalarT,6,1,Atb)
//#pragma omp parallel reduction (internal::MatrixReductions<ScalarT,6,6>::operator+: AtA) reduction (internal::MatrixReductions<ScalarT,6,1>::operator+: Atb)
#endif
            {
                if (has_point_to_point_terms) {
                    Eigen::Matrix<ScalarT,6,3,Eigen::RowMajor> eq_vecs;
                    eq_vecs.template bottomRows<3>().setIdentity();
#ifdef ENABLE_NON_DETERMINISTIC_PARALLELISM
#pragma omp for nowait
#endif
                    for (size_t i = 0; i < point_to_point_correspondences.size(); i++) {
                        const auto& corr = point_to_point_correspondences[i];
                        const Vector<ScalarT,3> d = dst_p.col(corr.indexInFirst) - dst_mean;
                        const ScalarT weight = point_to_point_weight * point_corr_evaluator(corr.indexInFirst, corr.indexInSecond, corr.value);
                        const Vector<ScalarT,3> s = tform * (src_p.col(corr.indexInSecond) - src_mean);

                        eq_vecs(0,0) = (ScalarT)0.0;
                        eq_vecs(1,1) = (ScalarT)0.0;
                        eq_vecs(2,2) = (ScalarT)0.0;

                        eq_vecs(0,1) = -(d[2] + s[2]);
                        eq_vecs(0,2) = (d[1] + s[1]);
                        eq_vecs(1,2) = -(d[0] + s[0]);

                        eq_vecs(1,0) = -eq_vecs(0, 1);
                        eq_vecs(2,0) = -eq_vecs(0, 2);
                        eq_vecs(2,1) = -eq_vecs(1, 2);

                        AtA += weight * eq_vecs * eq_vecs.transpose();
                        Atb += weight * eq_vecs * (d - s);
                    }
                }

                if (has_point_to_plane_terms) {
                    Eigen::Matrix<ScalarT,6,1> eq_vec;
#ifdef ENABLE_NON_DETERMINISTIC_PARALLELISM
#pragma omp for nowait
#endif
                    for (size_t i = 0; i < point_to_plane_correspondences.size(); i++) {
                        const auto& corr = point_to_plane_correspondences[i];
                        const ScalarT weight = point_to_plane_weight * plane_corr_evaluator(corr.indexInFirst, corr.indexInSecond, corr.value);
                        const Vector<ScalarT,3> d = dst_p.col(corr.indexInFirst) - dst_mean;
                        const Vector<ScalarT,3> n = dst_n.col(corr.indexInFirst) + src_n.col(corr.indexInSecond);
                        const Vector<ScalarT,3> s = tform * (src_p.col(corr.indexInSecond) - src_mean);

                        eq_vec.template head<3>() = (d + s).cross(n);
                        eq_vec.template tail<3>() = n;

                        AtA += weight * eq_vec * eq_vec.transpose();
                        Atb += weight * (n.dot(d - s)) * eq_vec;
                    }
                }
            }

            d_theta.noalias() = AtA.ldlt().solve(Atb);

            // Update estimate
            ScalarT na = d_theta.template head<3>().norm();
            ScalarT theta = std::atan(na);
#if EIGEN_VERSION_AT_LEAST(3, 2, 93)
            const Eigen::AngleAxis<ScalarT> Ra(theta, d_theta.template head<3>().stableNormalized());
#else
            const Eigen::AngleAxis<ScalarT> Ra(theta, d_theta.template head<3>().normalized());
#endif
            const Eigen::Translation<ScalarT, 3> ta(std::cos(theta) * d_theta.template tail<3>());
            tform = Ra * ta * Ra * tform;

            // Check for convergence
            if (d_theta.norm() < convergence_tol) {
                tform = t_dst * tform * t_src;
                return true;
            }
        }
        tform = t_dst * tform * t_src;
        return false;
    }
}
