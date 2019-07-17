#pragma once

#include <cilantro/icp_base.hpp>
#include <cilantro/transform_estimation.hpp>
#include <cilantro/correspondence_search_combined_metric_adaptor.hpp>
#include <cilantro/kd_tree.hpp>

namespace cilantro {
    template <class TransformT, class CorrespondenceSearchEngineT, class PointToPointCorrWeightEvaluatorT = UnityWeightEvaluator<typename TransformT::Scalar,typename TransformT::Scalar>, class PointToPlaneCorrWeightEvaluatorT = UnityWeightEvaluator<typename TransformT::Scalar,typename TransformT::Scalar>>
    class CombinedMetricSingleTransformICP : public IterativeClosestPointBase<CombinedMetricSingleTransformICP<TransformT,CorrespondenceSearchEngineT,PointToPointCorrWeightEvaluatorT,PointToPlaneCorrWeightEvaluatorT>,TransformT,CorrespondenceSearchEngineT,VectorSet<typename TransformT::Scalar,1>> {

        typedef IterativeClosestPointBase<CombinedMetricSingleTransformICP<TransformT,CorrespondenceSearchEngineT,PointToPointCorrWeightEvaluatorT,PointToPlaneCorrWeightEvaluatorT>,TransformT,CorrespondenceSearchEngineT,VectorSet<typename TransformT::Scalar,1>> Base;

        friend Base;

    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        typedef PointToPointCorrWeightEvaluatorT PointToPointCorrespondenceWeightEvaluator;

        typedef PointToPlaneCorrWeightEvaluatorT PointToPlaneCorrespondenceWeightEvaluator;

        CombinedMetricSingleTransformICP(const ConstVectorSetMatrixMap<typename TransformT::Scalar,TransformT::Dim> &dst_p,
                                         const ConstVectorSetMatrixMap<typename TransformT::Scalar,TransformT::Dim> &dst_n,
                                         const ConstVectorSetMatrixMap<typename TransformT::Scalar,TransformT::Dim> &src_p,
                                         CorrespondenceSearchEngineT &corr_engine,
                                         PointToPointCorrWeightEvaluatorT &point_corr_eval,
                                         PointToPlaneCorrWeightEvaluatorT &plane_corr_eval)
                : Base(corr_engine),
                  dst_points_(dst_p), dst_normals_(dst_n), src_points_(src_p), src_normals_(0),
                  max_optimization_iterations_(1), optimization_convergence_tol_((typename TransformT::Scalar)1e-5),
                  point_to_point_weight_((typename TransformT::Scalar)0.0), point_to_plane_weight_((typename TransformT::Scalar)1.0),
                  point_corr_eval_(point_corr_eval), plane_corr_eval_(plane_corr_eval),
                  src_points_trans_(src_points_.rows(), src_points_.cols()),
                  dst_mean_(dst_points_.rowwise().mean()), src_mean_(src_points_.rowwise().mean()),
                  has_source_normals_(false)
        {
            this->transform_init_.setIdentity();
        }

        CombinedMetricSingleTransformICP(const ConstVectorSetMatrixMap<typename TransformT::Scalar,TransformT::Dim> &dst_p,
                                         const ConstVectorSetMatrixMap<typename TransformT::Scalar,TransformT::Dim> &dst_n,
                                         const ConstVectorSetMatrixMap<typename TransformT::Scalar,TransformT::Dim> &src_p,
                                         const ConstVectorSetMatrixMap<typename TransformT::Scalar,TransformT::Dim> &src_n,
                                         CorrespondenceSearchEngineT &corr_engine,
                                         PointToPointCorrWeightEvaluatorT &point_corr_eval,
                                         PointToPlaneCorrWeightEvaluatorT &plane_corr_eval)
                : Base(corr_engine),
                  dst_points_(dst_p), dst_normals_(dst_n), src_points_(src_p), src_normals_(src_n),
                  max_optimization_iterations_(1), optimization_convergence_tol_((typename TransformT::Scalar)1e-5),
                  point_to_point_weight_((typename TransformT::Scalar)0.0), point_to_plane_weight_((typename TransformT::Scalar)1.0),
                  point_corr_eval_(point_corr_eval), plane_corr_eval_(plane_corr_eval),
                  src_points_trans_(src_points_.rows(), src_points_.cols()),
                  src_normals_trans_(src_points_.rows(), src_points_.cols()),
                  dst_mean_(dst_points_.rowwise().mean()), src_mean_(src_points_.rowwise().mean()),
                  has_source_normals_(true)
        {
            this->transform_init_.setIdentity();
        }

        inline PointToPointCorrespondenceWeightEvaluator& pointToPointCorrespondenceWeightEvaluator() {
            return point_corr_eval_;
        }

        inline PointToPlaneCorrespondenceWeightEvaluator& pointToPlaneCorrespondenceWeightEvaluator() {
            return plane_corr_eval_;
        }

        inline typename TransformT::Scalar getPointToPointMetricWeight() const { return point_to_point_weight_; }

        inline CombinedMetricSingleTransformICP& setPointToPointMetricWeight(typename TransformT::Scalar weight) {
            point_to_point_weight_ = weight;
            return *this;
        }

        inline typename TransformT::Scalar getPointToPlaneMetricWeight() const { return point_to_plane_weight_; }

        inline CombinedMetricSingleTransformICP& setPointToPlaneMetricWeight(typename TransformT::Scalar weight) {
            point_to_plane_weight_ = weight;
            return *this;
        }

        inline size_t getMaxNumberOfOptimizationStepIterations() const { return max_optimization_iterations_; }

        inline CombinedMetricSingleTransformICP& setMaxNumberOfOptimizationStepIterations(size_t max_iter) {
            max_optimization_iterations_ = max_iter;
            return *this;
        }

        inline typename TransformT::Scalar getOptimizationStepConvergenceTolerance() const { return optimization_convergence_tol_; }

        inline CombinedMetricSingleTransformICP& setOptimizationStepConvergenceTolerance(typename TransformT::Scalar conv_tol) {
            optimization_convergence_tol_ = conv_tol;
            return *this;
        }

    private:
        ConstVectorSetMatrixMap<typename TransformT::Scalar,TransformT::Dim> dst_points_;
        ConstVectorSetMatrixMap<typename TransformT::Scalar,TransformT::Dim> dst_normals_;
        ConstVectorSetMatrixMap<typename TransformT::Scalar,TransformT::Dim> src_points_;
        ConstVectorSetMatrixMap<typename TransformT::Scalar,TransformT::Dim> src_normals_;

        size_t max_optimization_iterations_;
        typename TransformT::Scalar optimization_convergence_tol_;
        typename TransformT::Scalar point_to_point_weight_;
        typename TransformT::Scalar point_to_plane_weight_;

        PointToPointCorrespondenceWeightEvaluator& point_corr_eval_;
        PointToPlaneCorrespondenceWeightEvaluator& plane_corr_eval_;

        VectorSet<typename TransformT::Scalar,TransformT::Dim> src_points_trans_;
        VectorSet<typename TransformT::Scalar,TransformT::Dim> src_normals_trans_;

        Vector<typename TransformT::Scalar,TransformT::Dim> dst_mean_;
        Vector<typename TransformT::Scalar,TransformT::Dim> src_mean_;

        bool has_source_normals_;

        // ICP interface
        inline void initializeComputation() {}

        // ICP interface
        inline void updateCorrespondences() {
            this->correspondence_search_engine_.findCorrespondences(this->transform_);
        }

        void updateEstimate() {
            transformPoints(this->transform_, src_points_, src_points_trans_);
            CorrespondenceSearchCombinedMetricAdaptor<CorrespondenceSearchEngineT> corr_getter_proxy(this->correspondence_search_engine_);
            TransformT tform_iter;

            if (has_source_normals_) {
                transformNormals(this->transform_, src_normals_, src_normals_trans_);
                estimateTransformSymmetricMetric(dst_points_, dst_normals_, src_points_trans_, src_normals_trans_, corr_getter_proxy.getPointToPointCorrespondences(), point_to_point_weight_, corr_getter_proxy.getPointToPlaneCorrespondences(), point_to_plane_weight_, tform_iter, max_optimization_iterations_, optimization_convergence_tol_, point_corr_eval_, plane_corr_eval_, dst_mean_, this->transform_ * src_mean_);
            } else {
                estimateTransformCombinedMetric(dst_points_, dst_normals_, src_points_trans_, corr_getter_proxy.getPointToPointCorrespondences(), point_to_point_weight_, corr_getter_proxy.getPointToPlaneCorrespondences(), point_to_plane_weight_, tform_iter, max_optimization_iterations_, optimization_convergence_tol_, point_corr_eval_, plane_corr_eval_, dst_mean_, this->transform_ * src_mean_);
            }

            this->transform_ = tform_iter*this->transform_;
            if (int(Base::Transform::Mode) == int(Eigen::Isometry)) {
                this->transform_.linear() = this->transform_.rotation();
            }
            this->last_delta_norm_ = std::sqrt((tform_iter.linear() - TransformT::LinearMatrixType::Identity()).squaredNorm() + tform_iter.translation().squaredNorm());
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
            Vector<typename TransformT::Scalar,TransformT::Dim> normal;
#pragma omp parallel for shared (res) private (nn, src_p_trans)
            for (size_t i = 0; i < src_points_.cols(); i++) {
                src_p_trans.noalias() = this->transform_*src_points_.col(i);
                dst_tree.nearestNeighborSearch(src_p_trans, nn);
                normal = dst_normals_.col(nn.index);
                if (has_source_normals_) normal += src_normals_.col(i);
                typename TransformT::Scalar point_to_plane_dist = normal.dot(dst_points_.col(nn.index) - src_p_trans);
                res[i] = point_to_point_weight_*(dst_points_.col(nn.index) - src_p_trans).squaredNorm() + point_to_plane_weight_*point_to_plane_dist*point_to_plane_dist;
            }
            return res;
        }
    };

    template <class CorrespondenceSearchEngineT, class PointToPointCorrWeightEvaluatorT = UnityWeightEvaluator<float,float>, class PointToPlaneCorrWeightEvaluatorT = UnityWeightEvaluator<float,float>>
    using CombinedMetricRigidTransformICP2f = CombinedMetricSingleTransformICP<RigidTransform<float,2>,CorrespondenceSearchEngineT,PointToPointCorrWeightEvaluatorT,PointToPlaneCorrWeightEvaluatorT>;

    template <class CorrespondenceSearchEngineT, class PointToPointCorrWeightEvaluatorT = UnityWeightEvaluator<double,double>, class PointToPlaneCorrWeightEvaluatorT = UnityWeightEvaluator<double,double>>
    using CombinedMetricRigidTransformICP2d = CombinedMetricSingleTransformICP<RigidTransform<double,2>,CorrespondenceSearchEngineT,PointToPointCorrWeightEvaluatorT,PointToPlaneCorrWeightEvaluatorT>;

    template <class CorrespondenceSearchEngineT, class PointToPointCorrWeightEvaluatorT = UnityWeightEvaluator<float,float>, class PointToPlaneCorrWeightEvaluatorT = UnityWeightEvaluator<float,float>>
    using CombinedMetricRigidTransformICP3f = CombinedMetricSingleTransformICP<RigidTransform<float,3>,CorrespondenceSearchEngineT,PointToPointCorrWeightEvaluatorT,PointToPlaneCorrWeightEvaluatorT>;

    template <class CorrespondenceSearchEngineT, class PointToPointCorrWeightEvaluatorT = UnityWeightEvaluator<double,double>, class PointToPlaneCorrWeightEvaluatorT = UnityWeightEvaluator<double,double>>
    using CombinedMetricRigidTransformICP3d = CombinedMetricSingleTransformICP<RigidTransform<double,3>,CorrespondenceSearchEngineT,PointToPointCorrWeightEvaluatorT,PointToPlaneCorrWeightEvaluatorT>;

    template <class CorrespondenceSearchEngineT, class PointToPointCorrWeightEvaluatorT = UnityWeightEvaluator<float,float>, class PointToPlaneCorrWeightEvaluatorT = UnityWeightEvaluator<float,float>>
    using CombinedMetricAffineTransformICP2f = CombinedMetricSingleTransformICP<AffineTransform<float,2>,CorrespondenceSearchEngineT,PointToPointCorrWeightEvaluatorT,PointToPlaneCorrWeightEvaluatorT>;

    template <class CorrespondenceSearchEngineT, class PointToPointCorrWeightEvaluatorT = UnityWeightEvaluator<double,double>, class PointToPlaneCorrWeightEvaluatorT = UnityWeightEvaluator<double,double>>
    using CombinedMetricAffineTransformICP2d = CombinedMetricSingleTransformICP<AffineTransform<double,2>,CorrespondenceSearchEngineT,PointToPointCorrWeightEvaluatorT,PointToPlaneCorrWeightEvaluatorT>;

    template <class CorrespondenceSearchEngineT, class PointToPointCorrWeightEvaluatorT = UnityWeightEvaluator<float,float>, class PointToPlaneCorrWeightEvaluatorT = UnityWeightEvaluator<float,float>>
    using CombinedMetricAffineTransformICP3f = CombinedMetricSingleTransformICP<AffineTransform<float,3>,CorrespondenceSearchEngineT,PointToPointCorrWeightEvaluatorT,PointToPlaneCorrWeightEvaluatorT>;

    template <class CorrespondenceSearchEngineT, class PointToPointCorrWeightEvaluatorT = UnityWeightEvaluator<double,double>, class PointToPlaneCorrWeightEvaluatorT = UnityWeightEvaluator<double,double>>
    using CombinedMetricAffineTransformICP3d = CombinedMetricSingleTransformICP<AffineTransform<double,3>,CorrespondenceSearchEngineT,PointToPointCorrWeightEvaluatorT,PointToPlaneCorrWeightEvaluatorT>;
}
