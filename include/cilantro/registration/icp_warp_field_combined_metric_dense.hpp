#pragma once

#include <cilantro/registration/icp_base.hpp>
#include <cilantro/registration/warp_field_estimation.hpp>
#include <cilantro/registration/correspondence_search_combined_metric_adaptor.hpp>
#include <cilantro/core/kd_tree.hpp>

namespace cilantro {
    // TransformT is the local motion model
    template <class TransformT, class CorrespondenceSearchEngineT, class RegNeighborhoodSetT, class PointToPointCorrWeightEvaluatorT = UnityWeightEvaluator<typename TransformT::Scalar,typename TransformT::Scalar>, class PointToPlaneCorrWeightEvaluatorT = UnityWeightEvaluator<typename TransformT::Scalar,typename TransformT::Scalar>, class RegularizationWeightEvaluatorT = RBFKernelWeightEvaluator<typename TransformT::Scalar,typename TransformT::Scalar,true>>
    class CombinedMetricDenseWarpFieldICP : public IterativeClosestPointBase<CombinedMetricDenseWarpFieldICP<TransformT,CorrespondenceSearchEngineT,RegNeighborhoodSetT,PointToPointCorrWeightEvaluatorT,PointToPlaneCorrWeightEvaluatorT,RegularizationWeightEvaluatorT>,TransformSet<TransformT>,CorrespondenceSearchEngineT,VectorSet<typename TransformT::Scalar,1>> {
        typedef IterativeClosestPointBase<CombinedMetricDenseWarpFieldICP<TransformT,CorrespondenceSearchEngineT,RegNeighborhoodSetT,PointToPointCorrWeightEvaluatorT,PointToPlaneCorrWeightEvaluatorT,RegularizationWeightEvaluatorT>,TransformSet<TransformT>,CorrespondenceSearchEngineT,VectorSet<typename TransformT::Scalar,1>> Base;
        friend Base;
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        typedef PointToPointCorrWeightEvaluatorT PointToPointCorrespondenceWeightEvaluator;

        typedef PointToPlaneCorrWeightEvaluatorT PointToPlaneCorrespondenceWeightEvaluator;

        typedef RegularizationWeightEvaluatorT RegularizationWeightEvaluator;

        CombinedMetricDenseWarpFieldICP(const ConstVectorSetMatrixMap<typename TransformT::Scalar,TransformT::Dim> &dst_p,
                                        const ConstVectorSetMatrixMap<typename TransformT::Scalar,TransformT::Dim> &dst_n,
                                        const ConstVectorSetMatrixMap<typename TransformT::Scalar,TransformT::Dim> &src_p,
                                        CorrespondenceSearchEngineT &corr_engine,
                                        const RegNeighborhoodSetT &regularization_neighborhoods,
                                        PointToPointCorrWeightEvaluatorT &point_corr_eval,
                                        PointToPlaneCorrWeightEvaluatorT &plane_corr_eval,
                                        RegularizationWeightEvaluatorT &reg_eval)
                : Base(corr_engine),
                  dst_points_(dst_p), dst_normals_(dst_n), src_points_(src_p),
                  regularization_neighborhoods_(regularization_neighborhoods),
                  point_to_point_weight_((typename TransformT::Scalar)0.0), point_to_plane_weight_((typename TransformT::Scalar)1.0),
                  stiffness_weight_((typename TransformT::Scalar)1.0), huber_boundary_((typename TransformT::Scalar)1e-4),
                  max_gauss_newton_iterations_(10), gauss_newton_convergence_tol_((typename TransformT::Scalar)1e-5),
                  max_conjugate_gradient_iterations_(1000), conjugate_gradient_convergence_tol_((typename TransformT::Scalar)1e-5),
                  point_corr_eval_(point_corr_eval), plane_corr_eval_(plane_corr_eval), reg_eval_(reg_eval),
                  src_points_trans_(src_points_.rows(), src_points_.cols()), tforms_iter_(src_points_.cols())
        {
            this->transform_init_.resize(src_p.cols());
            this->transform_init_.setIdentity();
        }

        inline PointToPointCorrespondenceWeightEvaluator& pointToPointCorrespondenceWeightEvaluator() {
            return point_corr_eval_;
        }

        inline PointToPlaneCorrespondenceWeightEvaluator& pointToPlaneCorrespondenceWeightEvaluator() {
            return plane_corr_eval_;
        }

        inline RegularizationWeightEvaluator& regularizationWeightEvaluator() { return reg_eval_; }

        inline typename TransformT::Scalar getPointToPointMetricWeight() const { return point_to_point_weight_; }

        inline CombinedMetricDenseWarpFieldICP& setPointToPointMetricWeight(typename TransformT::Scalar weight) {
            point_to_point_weight_ = weight;
            return *this;
        }

        inline typename TransformT::Scalar getPointToPlaneMetricWeight() const { return point_to_plane_weight_; }

        inline CombinedMetricDenseWarpFieldICP& setPointToPlaneMetricWeight(typename TransformT::Scalar weight) {
            point_to_plane_weight_ = weight;
            return *this;
        }

        inline typename TransformT::Scalar getStiffnessRegularizationWeight() const { return stiffness_weight_; }

        inline CombinedMetricDenseWarpFieldICP& setStiffnessRegularizationWeight(typename TransformT::Scalar weight) {
            stiffness_weight_ = weight;
            return *this;
        }

        inline size_t getMaxNumberOfGaussNewtonIterations() const { return max_gauss_newton_iterations_; }

        inline CombinedMetricDenseWarpFieldICP& setMaxNumberOfGaussNewtonIterations(size_t max_iter) {
            max_gauss_newton_iterations_ = max_iter;
            return *this;
        }

        inline typename TransformT::Scalar getGaussNewtonConvergenceTolerance() const { return gauss_newton_convergence_tol_; }

        inline CombinedMetricDenseWarpFieldICP& setGaussNewtonConvergenceTolerance(typename TransformT::Scalar conv_tol) {
            gauss_newton_convergence_tol_ = conv_tol;
            return *this;
        }

        inline size_t getMaxNumberOfConjugateGradientIterations() const { return max_conjugate_gradient_iterations_; }

        inline CombinedMetricDenseWarpFieldICP& setMaxNumberOfConjugateGradientIterations(size_t max_iter) {
            max_conjugate_gradient_iterations_ = max_iter;
            return *this;
        }

        inline typename TransformT::Scalar getConjugateGradientConvergenceTolerance() const { return conjugate_gradient_convergence_tol_; }

        inline CombinedMetricDenseWarpFieldICP& setConjugateGradientConvergenceTolerance(typename TransformT::Scalar conv_tol) {
            conjugate_gradient_convergence_tol_ = conv_tol;
            return *this;
        }

        inline typename TransformT::Scalar getHuberLossBoundary() const { return huber_boundary_; }

        inline CombinedMetricDenseWarpFieldICP& setHuberLossBoundary(typename TransformT::Scalar huber_boundary) {
            huber_boundary_ = huber_boundary;
            return *this;
        }

    private:
        // Input data holders
        ConstVectorSetMatrixMap<typename TransformT::Scalar,TransformT::Dim> dst_points_;
        ConstVectorSetMatrixMap<typename TransformT::Scalar,TransformT::Dim> dst_normals_;
        ConstVectorSetMatrixMap<typename TransformT::Scalar,TransformT::Dim> src_points_;
        const RegNeighborhoodSetT& regularization_neighborhoods_;

        typename TransformT::Scalar point_to_point_weight_;
        typename TransformT::Scalar point_to_plane_weight_;
        typename TransformT::Scalar stiffness_weight_;
        typename TransformT::Scalar huber_boundary_;
        size_t max_gauss_newton_iterations_;
        typename TransformT::Scalar gauss_newton_convergence_tol_;
        size_t max_conjugate_gradient_iterations_;
        typename TransformT::Scalar conjugate_gradient_convergence_tol_;

        PointToPointCorrespondenceWeightEvaluator& point_corr_eval_;
        PointToPlaneCorrespondenceWeightEvaluator& plane_corr_eval_;
        RegularizationWeightEvaluator& reg_eval_;

        VectorSet<typename TransformT::Scalar,TransformT::Dim> src_points_trans_;
        TransformSet<TransformT> tforms_iter_;

        // ICP interface
        inline void initializeComputation() {}

        // ICP interface
        inline void updateCorrespondences() {
            this->correspondence_search_engine_.findCorrespondences(this->transform_);
        }

        // ICP interface
        void updateEstimate() {
            transformPoints(this->transform_, src_points_, src_points_trans_);

            CorrespondenceSearchCombinedMetricAdaptor<CorrespondenceSearchEngineT> corr_getter_proxy(this->correspondence_search_engine_);
            estimateDenseWarpFieldCombinedMetric(dst_points_, dst_normals_, src_points_trans_, corr_getter_proxy.getPointToPointCorrespondences(), point_to_point_weight_, corr_getter_proxy.getPointToPlaneCorrespondences(), point_to_plane_weight_, regularization_neighborhoods_, stiffness_weight_, tforms_iter_, huber_boundary_, max_gauss_newton_iterations_, gauss_newton_convergence_tol_, max_conjugate_gradient_iterations_, conjugate_gradient_convergence_tol_, point_corr_eval_, plane_corr_eval_, reg_eval_);
            this->transform_.preApply(tforms_iter_);

            typename TransformT::Scalar max_delta_norm_sq = (typename TransformT::Scalar)0.0;
#pragma omp parallel for reduction (max: max_delta_norm_sq)
            for (size_t i = 0; i < tforms_iter_.size(); i++) {
                typename TransformT::Scalar last_delta_norm_sq = (tforms_iter_[i].linear() - TransformT::LinearMatrixType::Identity()).squaredNorm() + tforms_iter_[i].translation().squaredNorm();
                if (last_delta_norm_sq > max_delta_norm_sq) max_delta_norm_sq = last_delta_norm_sq;
            }
            this->last_delta_norm_ = std::sqrt(max_delta_norm_sq);
        }

        // ICP interface
        VectorSet<typename TransformT::Scalar,1> computeResiduals() {
            if (dst_points_.cols() == 0) {
                return VectorSet<typename TransformT::Scalar,1>::Constant(1, src_points_.cols(), std::numeric_limits<typename TransformT::Scalar>::quiet_NaN());
            }
            VectorSet<typename TransformT::Scalar,1> res(1, src_points_.cols());
            KDTree<typename TransformT::Scalar,TransformT::Dim,KDTreeDistanceAdaptors::L2> dst_tree(dst_points_);
            Neighbor<typename TransformT::Scalar> nn;
            Vector<typename TransformT::Scalar,TransformT::Dim> src_p_trans;
#pragma omp parallel for shared (res) private (nn, src_p_trans)
            for (size_t i = 0; i < src_points_.cols(); i++) {
                src_p_trans.noalias() = this->transform_[i]*src_points_.col(i);
                dst_tree.nearestNeighborSearch(src_p_trans, nn);
                typename TransformT::Scalar point_to_plane_dist = dst_normals_.col(nn.index).dot(dst_points_.col(nn.index) - src_p_trans);
                res[i] = point_to_point_weight_*(dst_points_.col(nn.index) - src_p_trans).squaredNorm() + point_to_plane_weight_*point_to_plane_dist*point_to_plane_dist;
            }
            return res;
        }
    };

    template <class CorrespondenceSearchEngineT, class PointToPointCorrWeightEvaluatorT = UnityWeightEvaluator<float,float>, class PointToPlaneCorrWeightEvaluatorT = UnityWeightEvaluator<float,float>, class RegularizationWeightEvaluatorT = RBFKernelWeightEvaluator<float,float,true>>
    using CombinedMetricDenseRigidWarpFieldICP2f = CombinedMetricDenseWarpFieldICP<RigidTransform<float,2>,CorrespondenceSearchEngineT,PointToPointCorrWeightEvaluatorT,PointToPlaneCorrWeightEvaluatorT,RegularizationWeightEvaluatorT>;

    template <class CorrespondenceSearchEngineT, class PointToPointCorrWeightEvaluatorT = UnityWeightEvaluator<double,double>, class PointToPlaneCorrWeightEvaluatorT = UnityWeightEvaluator<double,double>, class RegularizationWeightEvaluatorT = RBFKernelWeightEvaluator<double,double,true>>
    using CombinedMetricDenseRigidWarpFieldICP2d = CombinedMetricDenseWarpFieldICP<RigidTransform<double,2>,CorrespondenceSearchEngineT,PointToPointCorrWeightEvaluatorT,PointToPlaneCorrWeightEvaluatorT,RegularizationWeightEvaluatorT>;

    template <class CorrespondenceSearchEngineT, class PointToPointCorrWeightEvaluatorT = UnityWeightEvaluator<float,float>, class PointToPlaneCorrWeightEvaluatorT = UnityWeightEvaluator<float,float>, class RegularizationWeightEvaluatorT = RBFKernelWeightEvaluator<float,float,true>>
    using CombinedMetricDenseRigidWarpFieldICP3f = CombinedMetricDenseWarpFieldICP<RigidTransform<float,3>,CorrespondenceSearchEngineT,PointToPointCorrWeightEvaluatorT,PointToPlaneCorrWeightEvaluatorT,RegularizationWeightEvaluatorT>;

    template <class CorrespondenceSearchEngineT, class PointToPointCorrWeightEvaluatorT = UnityWeightEvaluator<double,double>, class PointToPlaneCorrWeightEvaluatorT = UnityWeightEvaluator<double,double>, class RegularizationWeightEvaluatorT = RBFKernelWeightEvaluator<double,double,true>>
    using CombinedMetricDenseRigidWarpFieldICP3d = CombinedMetricDenseWarpFieldICP<RigidTransform<double,3>,CorrespondenceSearchEngineT,PointToPointCorrWeightEvaluatorT,PointToPlaneCorrWeightEvaluatorT,RegularizationWeightEvaluatorT>;

    template <class CorrespondenceSearchEngineT, class PointToPointCorrWeightEvaluatorT = UnityWeightEvaluator<float,float>, class PointToPlaneCorrWeightEvaluatorT = UnityWeightEvaluator<float,float>, class RegularizationWeightEvaluatorT = RBFKernelWeightEvaluator<float,float,true>>
    using CombinedMetricDenseAffineWarpFieldICP2f = CombinedMetricDenseWarpFieldICP<AffineTransform<float,2>,CorrespondenceSearchEngineT,PointToPointCorrWeightEvaluatorT,PointToPlaneCorrWeightEvaluatorT,RegularizationWeightEvaluatorT>;

    template <class CorrespondenceSearchEngineT, class PointToPointCorrWeightEvaluatorT = UnityWeightEvaluator<double,double>, class PointToPlaneCorrWeightEvaluatorT = UnityWeightEvaluator<double,double>, class RegularizationWeightEvaluatorT = RBFKernelWeightEvaluator<double,double,true>>
    using CombinedMetricDenseAffineWarpFieldICP2d = CombinedMetricDenseWarpFieldICP<AffineTransform<double,2>,CorrespondenceSearchEngineT,PointToPointCorrWeightEvaluatorT,PointToPlaneCorrWeightEvaluatorT,RegularizationWeightEvaluatorT>;

    template <class CorrespondenceSearchEngineT, class PointToPointCorrWeightEvaluatorT = UnityWeightEvaluator<float,float>, class PointToPlaneCorrWeightEvaluatorT = UnityWeightEvaluator<float,float>, class RegularizationWeightEvaluatorT = RBFKernelWeightEvaluator<float,float,true>>
    using CombinedMetricDenseAffineWarpFieldICP3f = CombinedMetricDenseWarpFieldICP<AffineTransform<float,3>,CorrespondenceSearchEngineT,PointToPointCorrWeightEvaluatorT,PointToPlaneCorrWeightEvaluatorT,RegularizationWeightEvaluatorT>;

    template <class CorrespondenceSearchEngineT, class PointToPointCorrWeightEvaluatorT = UnityWeightEvaluator<double,double>, class PointToPlaneCorrWeightEvaluatorT = UnityWeightEvaluator<double,double>, class RegularizationWeightEvaluatorT = RBFKernelWeightEvaluator<double,double,true>>
    using CombinedMetricDenseAffineWarpFieldICP3d = CombinedMetricDenseWarpFieldICP<AffineTransform<double,3>,CorrespondenceSearchEngineT,PointToPointCorrWeightEvaluatorT,PointToPlaneCorrWeightEvaluatorT,RegularizationWeightEvaluatorT>;
}
