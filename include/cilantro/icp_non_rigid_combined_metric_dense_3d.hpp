#pragma once

#include <cilantro/icp_base.hpp>
#include <cilantro/non_rigid_registration_utilities.hpp>
#include <cilantro/correspondence_search_combined_metric_adaptor.hpp>
#include <cilantro/kd_tree.hpp>

namespace cilantro {
    template <typename ScalarT, class CorrespondenceSearchEngineT, class PointToPointCorrWeightEvaluatorT = UnityWeightEvaluator<ScalarT,ScalarT>, class PointToPlaneCorrWeightEvaluatorT = UnityWeightEvaluator<ScalarT,ScalarT>, class RegularizationWeightEvaluatorT = RBFKernelWeightEvaluator<ScalarT,ScalarT,true>>
    class DenseCombinedMetricNonRigidICP3 : public IterativeClosestPointBase<DenseCombinedMetricNonRigidICP3<ScalarT,CorrespondenceSearchEngineT,PointToPointCorrWeightEvaluatorT,PointToPlaneCorrWeightEvaluatorT,RegularizationWeightEvaluatorT>,RigidTransformationSet<ScalarT,3>,CorrespondenceSearchEngineT,VectorSet<ScalarT,1>> {
        friend class IterativeClosestPointBase<DenseCombinedMetricNonRigidICP3<ScalarT,CorrespondenceSearchEngineT,PointToPointCorrWeightEvaluatorT,PointToPlaneCorrWeightEvaluatorT,RegularizationWeightEvaluatorT>,RigidTransformationSet<ScalarT,3>,CorrespondenceSearchEngineT,VectorSet<ScalarT,1>>;
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        typedef PointToPointCorrWeightEvaluatorT PointToPointCorrespondenceWeightEvaluator;

        typedef PointToPlaneCorrWeightEvaluatorT PointToPlaneCorrespondenceWeightEvaluator;

        typedef RegularizationWeightEvaluatorT RegularizationWeightEvaluator;

        DenseCombinedMetricNonRigidICP3(const ConstVectorSetMatrixMap<ScalarT,3> &dst_p,
                                        const ConstVectorSetMatrixMap<ScalarT,3> &dst_n,
                                        const ConstVectorSetMatrixMap<ScalarT,3> &src_p,
                                        CorrespondenceSearchEngineT &corr_engine,
                                        PointToPointCorrWeightEvaluatorT &point_corr_eval,
                                        PointToPlaneCorrWeightEvaluatorT &plane_corr_eval,
                                        const std::vector<NeighborSet<ScalarT>> &regularization_neighborhoods,
                                        RegularizationWeightEvaluatorT &reg_eval)
                : IterativeClosestPointBase<DenseCombinedMetricNonRigidICP3<ScalarT,CorrespondenceSearchEngineT,PointToPointCorrWeightEvaluatorT,PointToPlaneCorrWeightEvaluatorT,RegularizationWeightEvaluatorT>,RigidTransformationSet<ScalarT,3>,CorrespondenceSearchEngineT,VectorSet<ScalarT,1>>(corr_engine),
                  dst_points_(dst_p), dst_normals_(dst_n), src_points_(src_p),
                  regularization_neighborhoods_(regularization_neighborhoods),
                  point_to_point_weight_((ScalarT)0.0), point_to_plane_weight_((ScalarT)1.0),
                  stiffness_weight_((ScalarT)1.0), huber_boundary_((ScalarT)1e-4),
                  max_gauss_newton_iterations_(10), gauss_newton_convergence_tol_((ScalarT)1e-5),
                  max_conjugate_gradient_iterations_(1000), conjugate_gradient_convergence_tol_((ScalarT)1e-5),
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

        inline ScalarT getPointToPointMetricWeight() const { return point_to_point_weight_; }

        inline DenseCombinedMetricNonRigidICP3& setPointToPointMetricWeight(ScalarT weight) {
            point_to_point_weight_ = weight;
            return *this;
        }

        inline ScalarT getPointToPlaneMetricWeight() const { return point_to_plane_weight_; }

        inline DenseCombinedMetricNonRigidICP3& setPointToPlaneMetricWeight(ScalarT weight) {
            point_to_plane_weight_ = weight;
            return *this;
        }

        inline ScalarT getStiffnessRegularizationWeight() const { return stiffness_weight_; }

        inline DenseCombinedMetricNonRigidICP3& setStiffnessRegularizationWeight(ScalarT weight) {
            stiffness_weight_ = weight;
            return *this;
        }

        inline size_t getMaxNumberOfGaussNewtonIterations() const { return max_gauss_newton_iterations_; }

        inline DenseCombinedMetricNonRigidICP3& setMaxNumberOfGaussNewtonIterations(size_t max_iter) {
            max_gauss_newton_iterations_ = max_iter;
            return *this;
        }

        inline ScalarT getGaussNewtonConvergenceTolerance() const { return gauss_newton_convergence_tol_; }

        inline DenseCombinedMetricNonRigidICP3& setGaussNewtonConvergenceTolerance(ScalarT conv_tol) {
            gauss_newton_convergence_tol_ = conv_tol;
            return *this;
        }

        inline size_t getMaxNumberOfConjugateGradientIterations() const { return max_conjugate_gradient_iterations_; }

        inline DenseCombinedMetricNonRigidICP3& setMaxNumberOfConjugateGradientIterations(size_t max_iter) {
            max_conjugate_gradient_iterations_ = max_iter;
            return *this;
        }

        inline ScalarT getConjugateGradientConvergenceTolerance() const { return conjugate_gradient_convergence_tol_; }

        inline DenseCombinedMetricNonRigidICP3& setConjugateGradientConvergenceTolerance(ScalarT conv_tol) {
            conjugate_gradient_convergence_tol_ = conv_tol;
            return *this;
        }

        inline ScalarT getHuberLossBoundary() const { return huber_boundary_; }

        inline DenseCombinedMetricNonRigidICP3& setHuberLossBoundary(ScalarT huber_boundary) {
            huber_boundary_ = huber_boundary;
            return *this;
        }

    private:
        // Input data holders
        ConstVectorSetMatrixMap<ScalarT,3> dst_points_;
        ConstVectorSetMatrixMap<ScalarT,3> dst_normals_;
        ConstVectorSetMatrixMap<ScalarT,3> src_points_;
        const std::vector<NeighborSet<ScalarT>>& regularization_neighborhoods_;

        ScalarT point_to_point_weight_;
        ScalarT point_to_plane_weight_;
        ScalarT stiffness_weight_;
        ScalarT huber_boundary_;
        size_t max_gauss_newton_iterations_;
        ScalarT gauss_newton_convergence_tol_;
        size_t max_conjugate_gradient_iterations_;
        ScalarT conjugate_gradient_convergence_tol_;

        PointToPointCorrespondenceWeightEvaluator& point_corr_eval_;
        PointToPlaneCorrespondenceWeightEvaluator& plane_corr_eval_;
        RegularizationWeightEvaluator& reg_eval_;

        VectorSet<ScalarT,3> src_points_trans_;
        RigidTransformationSet<ScalarT,3> tforms_iter_;

        // ICP interface
        inline void initializeComputation() {}

        // ICP interface
        inline void updateCorrespondences() {
            this->correspondence_search_engine_.findCorrespondences(this->transform_);
        }

        // ICP interface
        void updateEstimate() {
#pragma omp parallel for
            for (size_t i = 0; i < src_points_.cols(); i++) {
                src_points_trans_.col(i).noalias() = this->transform_[i]*src_points_.col(i);
            }

            CorrespondenceSearchCombinedMetricAdaptor<CorrespondenceSearchEngineT> corr_getter_proxy(this->correspondence_search_engine_);
            estimateDenseWarpFieldCombinedMetric3<ScalarT,PointToPointCorrespondenceWeightEvaluator,PointToPlaneCorrespondenceWeightEvaluator,RegularizationWeightEvaluator>(dst_points_, dst_normals_, src_points_trans_, corr_getter_proxy.getPointToPointCorrespondences(), point_to_point_weight_, corr_getter_proxy.getPointToPlaneCorrespondences(), point_to_plane_weight_, regularization_neighborhoods_, stiffness_weight_, tforms_iter_, huber_boundary_, max_gauss_newton_iterations_, gauss_newton_convergence_tol_, max_conjugate_gradient_iterations_, conjugate_gradient_convergence_tol_, point_corr_eval_, plane_corr_eval_, reg_eval_);
            this->transform_.preApply(tforms_iter_);

            ScalarT max_delta_norm_sq = (ScalarT)0.0;
#pragma omp parallel for reduction (max: max_delta_norm_sq)
            for (size_t i = 0; i < tforms_iter_.size(); i++) {
                ScalarT last_delta_norm_sq = (tforms_iter_[i].linear() - Eigen::Matrix<ScalarT,3,3>::Identity()).squaredNorm() + tforms_iter_[i].translation().squaredNorm();
                if (last_delta_norm_sq > max_delta_norm_sq) max_delta_norm_sq = last_delta_norm_sq;
            }
            this->last_delta_norm_ = std::sqrt(max_delta_norm_sq);
        }

        // ICP interface
        VectorSet<ScalarT,1> computeResiduals() {
            if (dst_points_.cols() == 0) {
                return VectorSet<ScalarT,1>::Constant(1, src_points_.cols(), std::numeric_limits<ScalarT>::quiet_NaN());
            }
            VectorSet<ScalarT,1> res(1, src_points_.cols());
            KDTree<ScalarT,3,KDTreeDistanceAdaptors::L2> dst_tree(dst_points_);
            Neighbor<ScalarT> nn;
            Vector<ScalarT,3> src_p_trans;
#pragma omp parallel for shared (res) private (nn, src_p_trans)
            for (size_t i = 0; i < src_points_.cols(); i++) {
                src_p_trans.noalias() = this->transform_[i]*src_points_.col(i);
                dst_tree.nearestNeighborSearch(src_p_trans, nn);
                ScalarT point_to_plane_dist = dst_normals_.col(nn.index).dot(dst_points_.col(nn.index) - src_p_trans);
                res[i] = point_to_point_weight_*(dst_points_.col(nn.index) - src_p_trans).squaredNorm() + point_to_plane_weight_*point_to_plane_dist*point_to_plane_dist;
            }
            return res;
        }
    };

    template <class CorrespondenceSearchEngineT, class PointToPointCorrWeightEvaluatorT = UnityWeightEvaluator<float,float>, class PointToPlaneCorrWeightEvaluatorT = UnityWeightEvaluator<float,float>, class RegularizationWeightEvaluatorT = RBFKernelWeightEvaluator<float,float,true>>
    using DenseCombinedMetricNonRigidICP3f = DenseCombinedMetricNonRigidICP3<float,CorrespondenceSearchEngineT,PointToPointCorrWeightEvaluatorT,PointToPlaneCorrWeightEvaluatorT,RegularizationWeightEvaluatorT>;

    template <class CorrespondenceSearchEngineT, class PointToPointCorrWeightEvaluatorT = UnityWeightEvaluator<double,double>, class PointToPlaneCorrWeightEvaluatorT = UnityWeightEvaluator<double,double>, class RegularizationWeightEvaluatorT = RBFKernelWeightEvaluator<double,double,true>>
    using DenseCombinedMetricNonRigidICP3d = DenseCombinedMetricNonRigidICP3<double,CorrespondenceSearchEngineT,PointToPointCorrWeightEvaluatorT,PointToPlaneCorrWeightEvaluatorT,RegularizationWeightEvaluatorT>;
}
