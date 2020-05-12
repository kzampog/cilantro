#pragma once

#include <cilantro/correspondence_search/common_transformable_feature_adaptors.hpp>
#include <cilantro/correspondence_search/correspondence_search_kd_tree.hpp>
#include <cilantro/correspondence_search/correspondence_search_projective.hpp>
#include <cilantro/registration/icp_single_transform_point_to_point_metric.hpp>
#include <cilantro/registration/icp_single_transform_combined_metric.hpp>
#include <cilantro/registration/icp_warp_field_combined_metric_dense.hpp>
#include <cilantro/registration/icp_warp_field_combined_metric_sparse.hpp>

namespace cilantro {
    namespace internal {
        template <class TransformT, class CorrSearchT>
        using DefaultPointToPointMetricICP = PointToPointMetricSingleTransformICP<TransformT,CorrSearchT>;

        template <class TransformT, class CorrSearchT>
        class DefaultPointToPointMetricICPEntities {
        public:
            EIGEN_MAKE_ALIGNED_OPERATOR_NEW

            DefaultPointToPointMetricICPEntities(const ConstVectorSetMatrixMap<typename TransformT::Scalar,TransformT::Dim> &dst_points,
                                                 const ConstVectorSetMatrixMap<typename TransformT::Scalar,TransformT::Dim> &src_points)
                    : dst_feat_(dst_points), src_feat_(src_points),
                      corr_search_(dst_feat_, src_feat_, corr_dist_evaluator_)
            {}

        protected:
            PointFeaturesAdaptor<typename TransformT::Scalar,TransformT::Dim> dst_feat_;
            PointFeaturesAdaptor<typename TransformT::Scalar,TransformT::Dim> src_feat_;
            DistanceEvaluator<typename TransformT::Scalar,typename TransformT::Scalar> corr_dist_evaluator_;
            CorrSearchT corr_search_;
        };

        template <class TransformT, class CorrSearchT>
        class SimplePointToPointMetricICPWrapper : private DefaultPointToPointMetricICPEntities<TransformT,CorrSearchT>,
                                                   public DefaultPointToPointMetricICP<TransformT,CorrSearchT>
        {
        public:
            EIGEN_MAKE_ALIGNED_OPERATOR_NEW

            SimplePointToPointMetricICPWrapper(const ConstVectorSetMatrixMap<typename TransformT::Scalar,TransformT::Dim> &dst_points,
                                               const ConstVectorSetMatrixMap<typename TransformT::Scalar,TransformT::Dim> &src_points)
                    : DefaultPointToPointMetricICPEntities<TransformT,CorrSearchT>(dst_points, src_points),
                      DefaultPointToPointMetricICP<TransformT,CorrSearchT>(dst_points, src_points, this->corr_search_)
            {}
        };

        template <class TransformT, class CorrSearchT>
        using DefaultCombinedMetricICP = CombinedMetricSingleTransformICP<TransformT,CorrSearchT,UnityWeightEvaluator<typename TransformT::Scalar,typename TransformT::Scalar>,UnityWeightEvaluator<typename TransformT::Scalar,typename TransformT::Scalar>>;

        template <class TransformT, class CorrSearchT>
        class DefaultCombinedMetricICPEntities {
        public:
            EIGEN_MAKE_ALIGNED_OPERATOR_NEW

            DefaultCombinedMetricICPEntities(const ConstVectorSetMatrixMap<typename TransformT::Scalar,TransformT::Dim> &dst_points,
                                             const ConstVectorSetMatrixMap<typename TransformT::Scalar,TransformT::Dim> &src_points)
                    : dst_feat_(dst_points), src_feat_(src_points),
                      corr_search_(dst_feat_, src_feat_, corr_dist_evaluator_)
            {}

        protected:
            PointFeaturesAdaptor<typename TransformT::Scalar,TransformT::Dim> dst_feat_;
            PointFeaturesAdaptor<typename TransformT::Scalar,TransformT::Dim> src_feat_;
            DistanceEvaluator<typename TransformT::Scalar,typename TransformT::Scalar> corr_dist_evaluator_;
            CorrSearchT corr_search_;
            UnityWeightEvaluator<typename TransformT::Scalar,typename TransformT::Scalar> point_corr_weight_eval_;
            UnityWeightEvaluator<typename TransformT::Scalar,typename TransformT::Scalar> plane_corr_weight_eval_;
        };

        template <class TransformT, class CorrSearchT>
        class SimpleCombinedMetricICPWrapper : private DefaultCombinedMetricICPEntities<TransformT,CorrSearchT>,
                                               public DefaultCombinedMetricICP<TransformT,CorrSearchT>
        {
        public:
            EIGEN_MAKE_ALIGNED_OPERATOR_NEW

            SimpleCombinedMetricICPWrapper(const ConstVectorSetMatrixMap<typename TransformT::Scalar,TransformT::Dim> &dst_points,
                                           const ConstVectorSetMatrixMap<typename TransformT::Scalar,TransformT::Dim> &dst_normals,
                                           const ConstVectorSetMatrixMap<typename TransformT::Scalar,TransformT::Dim> &src_points)
                    : DefaultCombinedMetricICPEntities<TransformT,CorrSearchT>(dst_points, src_points),
                      DefaultCombinedMetricICP<TransformT,CorrSearchT>(dst_points, dst_normals, src_points, this->corr_search_, this->point_corr_weight_eval_, this->plane_corr_weight_eval_)
            {}

            SimpleCombinedMetricICPWrapper(const ConstVectorSetMatrixMap<typename TransformT::Scalar,TransformT::Dim> &dst_points,
                                           const ConstVectorSetMatrixMap<typename TransformT::Scalar,TransformT::Dim> &dst_normals,
                                           const ConstVectorSetMatrixMap<typename TransformT::Scalar,TransformT::Dim> &src_points,
                                           const ConstVectorSetMatrixMap<typename TransformT::Scalar,TransformT::Dim> &src_normals)
                    : DefaultCombinedMetricICPEntities<TransformT,CorrSearchT>(dst_points, src_points),
                      DefaultCombinedMetricICP<TransformT,CorrSearchT>(dst_points, dst_normals, src_points, src_normals, this->corr_search_, this->point_corr_weight_eval_, this->plane_corr_weight_eval_)
            {}
        };

        template <class TransformT, class CorrSearchT>
        using DefaultCombinedMetricDenseWarpFieldICP = CombinedMetricDenseWarpFieldICP<TransformT,CorrSearchT,NeighborhoodSet<typename TransformT::Scalar>,UnityWeightEvaluator<typename TransformT::Scalar,typename TransformT::Scalar>,UnityWeightEvaluator<typename TransformT::Scalar,typename TransformT::Scalar>,RBFKernelWeightEvaluator<typename TransformT::Scalar,typename TransformT::Scalar,true>>;

        template <class TransformT, class CorrSearchT>
        class DefaultCombinedMetricDenseWarpFieldICPEntities {
        public:
            EIGEN_MAKE_ALIGNED_OPERATOR_NEW

            DefaultCombinedMetricDenseWarpFieldICPEntities(const ConstVectorSetMatrixMap<typename TransformT::Scalar,TransformT::Dim> &dst_points,
                                                           const ConstVectorSetMatrixMap<typename TransformT::Scalar,TransformT::Dim> &src_points)
                    : dst_feat_(dst_points), src_feat_(src_points),
                      corr_search_(dst_feat_, src_feat_, corr_dist_evaluator_)
            {}

        protected:
            PointFeaturesAdaptor<typename TransformT::Scalar,TransformT::Dim> dst_feat_;
            PointFeaturesAdaptor<typename TransformT::Scalar,TransformT::Dim> src_feat_;
            DistanceEvaluator<typename TransformT::Scalar,typename TransformT::Scalar> corr_dist_evaluator_;
            CorrSearchT corr_search_;
            UnityWeightEvaluator<typename TransformT::Scalar,typename TransformT::Scalar> point_corr_weight_eval_;
            UnityWeightEvaluator<typename TransformT::Scalar,typename TransformT::Scalar> plane_corr_weight_eval_;
            RBFKernelWeightEvaluator<typename TransformT::Scalar,typename TransformT::Scalar,true> reg_weight_eval_;
        };

        template <class TransformT, class CorrSearchT>
        class SimpleCombinedMetricDenseWarpFieldICPWrapper : private DefaultCombinedMetricDenseWarpFieldICPEntities<TransformT,CorrSearchT>,
                                                             public DefaultCombinedMetricDenseWarpFieldICP<TransformT,CorrSearchT>
        {
        public:
            EIGEN_MAKE_ALIGNED_OPERATOR_NEW
            SimpleCombinedMetricDenseWarpFieldICPWrapper(const ConstVectorSetMatrixMap<typename TransformT::Scalar,TransformT::Dim> &dst_points,
                                                         const ConstVectorSetMatrixMap<typename TransformT::Scalar,TransformT::Dim> &dst_normals,
                                                         const ConstVectorSetMatrixMap<typename TransformT::Scalar,TransformT::Dim> &src_points,
                                                         const std::vector<NeighborSet<typename TransformT::Scalar>> &regularization_neighborhoods)
                    : DefaultCombinedMetricDenseWarpFieldICPEntities<TransformT,CorrSearchT>(dst_points, src_points),
                      DefaultCombinedMetricDenseWarpFieldICP<TransformT,CorrSearchT>(dst_points, dst_normals, src_points, this->corr_search_, regularization_neighborhoods, this->point_corr_weight_eval_, this->plane_corr_weight_eval_, this->reg_weight_eval_)
            {}
        };

        template <class TransformT, class CorrSearchT>
        using DefaultCombinedMetricSparseWarpFieldICP = CombinedMetricSparseWarpFieldICP<TransformT,CorrSearchT,NeighborhoodSet<typename TransformT::Scalar>,NeighborhoodSet<typename TransformT::Scalar>,UnityWeightEvaluator<typename TransformT::Scalar,typename TransformT::Scalar>,UnityWeightEvaluator<typename TransformT::Scalar,typename TransformT::Scalar>,RBFKernelWeightEvaluator<typename TransformT::Scalar,typename TransformT::Scalar,true>,RBFKernelWeightEvaluator<typename TransformT::Scalar,typename TransformT::Scalar,true>>;

        template <class TransformT, class CorrSearchT>
        class DefaultCombinedMetricSparseWarpFieldICPEntities {
        public:
            EIGEN_MAKE_ALIGNED_OPERATOR_NEW

            DefaultCombinedMetricSparseWarpFieldICPEntities(const ConstVectorSetMatrixMap<typename TransformT::Scalar,TransformT::Dim> &dst_points,
                                                            const ConstVectorSetMatrixMap<typename TransformT::Scalar,TransformT::Dim> &src_points)
                    : dst_feat_(dst_points), src_feat_(src_points),
                      corr_search_(dst_feat_, src_feat_, corr_dist_evaluator_)
            {}

        protected:
            PointFeaturesAdaptor<typename TransformT::Scalar,TransformT::Dim> dst_feat_;
            PointFeaturesAdaptor<typename TransformT::Scalar,TransformT::Dim> src_feat_;
            DistanceEvaluator<typename TransformT::Scalar,typename TransformT::Scalar> corr_dist_evaluator_;
            CorrSearchT corr_search_;
            UnityWeightEvaluator<typename TransformT::Scalar,typename TransformT::Scalar> point_corr_weight_eval_;
            UnityWeightEvaluator<typename TransformT::Scalar,typename TransformT::Scalar> plane_corr_weight_eval_;
            RBFKernelWeightEvaluator<typename TransformT::Scalar,typename TransformT::Scalar,true> control_weight_eval_;
            RBFKernelWeightEvaluator<typename TransformT::Scalar,typename TransformT::Scalar,true> reg_weight_eval_;
        };

        template <class TransformT, class CorrSearchT>
        class SimpleCombinedMetricSparseWarpFieldICPWrapper : private DefaultCombinedMetricSparseWarpFieldICPEntities<TransformT,CorrSearchT>,
                                                              public DefaultCombinedMetricSparseWarpFieldICP<TransformT,CorrSearchT>
        {
        public:
            EIGEN_MAKE_ALIGNED_OPERATOR_NEW

            SimpleCombinedMetricSparseWarpFieldICPWrapper(const ConstVectorSetMatrixMap<typename TransformT::Scalar,TransformT::Dim> &dst_points,
                                                          const ConstVectorSetMatrixMap<typename TransformT::Scalar,TransformT::Dim> &dst_normals,
                                                          const ConstVectorSetMatrixMap<typename TransformT::Scalar,TransformT::Dim> &src_points,
                                                          const std::vector<NeighborSet<typename TransformT::Scalar>> &src_to_control_neighborhoods,
                                                          size_t num_control_nodes,
                                                          const std::vector<NeighborSet<typename TransformT::Scalar>> &control_regularization_neighborhoods)
                    : DefaultCombinedMetricSparseWarpFieldICPEntities<TransformT,CorrSearchT>(dst_points, src_points),
                      DefaultCombinedMetricSparseWarpFieldICP<TransformT,CorrSearchT>(dst_points, dst_normals, src_points, this->corr_search_, src_to_control_neighborhoods, num_control_nodes, control_regularization_neighborhoods, this->point_corr_weight_eval_, this->plane_corr_weight_eval_, this->control_weight_eval_, this->reg_weight_eval_)
            {}
        };

        template <class TransformT>
        using DefaultKDTreeSearch = CorrespondenceSearchKDTree<PointFeaturesAdaptor<typename TransformT::Scalar,TransformT::Dim>,KDTreeDistanceAdaptors::L2,PointFeaturesAdaptor<typename TransformT::Scalar,TransformT::Dim>,DistanceEvaluator<typename TransformT::Scalar,typename TransformT::Scalar>>;

        template <class TransformT>
        using DefaultProjectiveSearch = CorrespondenceSearchProjective<typename TransformT::Scalar,PointFeaturesAdaptor<typename TransformT::Scalar,3>,DistanceEvaluator<typename TransformT::Scalar,typename TransformT::Scalar>>;
    } // namespace internal

    template <class TransformT>
    using SimplePointToPointMetricICP = internal::SimplePointToPointMetricICPWrapper<TransformT,internal::DefaultKDTreeSearch<TransformT>>;

    // template <class TransformT>
    // using SimplePointToPointMetricProjectiveICP = internal::SimplePointToPointMetricICPWrapper<TransformT,internal::DefaultProjectiveSearch<TransformT>>;

    template <class TransformT>
    using SimpleCombinedMetricICP = internal::SimpleCombinedMetricICPWrapper<TransformT,internal::DefaultKDTreeSearch<TransformT>>;

    template <class TransformT>
    using SimpleCombinedMetricProjectiveICP = internal::SimpleCombinedMetricICPWrapper<TransformT,internal::DefaultProjectiveSearch<TransformT>>;

    template <class TransformT>
    using SimpleCombinedMetricDenseWarpFieldICP = internal::SimpleCombinedMetricDenseWarpFieldICPWrapper<TransformT,internal::DefaultKDTreeSearch<TransformT>>;

    template <class TransformT>
    using SimpleCombinedMetricDenseWarpFieldProjectiveICP = internal::SimpleCombinedMetricDenseWarpFieldICPWrapper<TransformT,internal::DefaultProjectiveSearch<TransformT>>;

    template <class TransformT>
    using SimpleCombinedMetricSparseWarpFieldICP = internal::SimpleCombinedMetricSparseWarpFieldICPWrapper<TransformT,internal::DefaultKDTreeSearch<TransformT>>;

    template <class TransformT>
    using SimpleCombinedMetricSparseWarpFieldProjectiveICP = internal::SimpleCombinedMetricSparseWarpFieldICPWrapper<TransformT,internal::DefaultProjectiveSearch<TransformT>>;

    // Point to point
    typedef SimplePointToPointMetricICP<RigidTransform<float,2>> SimplePointToPointMetricRigidICP2f;
    typedef SimplePointToPointMetricICP<RigidTransform<double,2>> SimplePointToPointMetricRigidICP2d;
    typedef SimplePointToPointMetricICP<RigidTransform<float,3>> SimplePointToPointMetricRigidICP3f;
    typedef SimplePointToPointMetricICP<RigidTransform<double,3>> SimplePointToPointMetricRigidICP3d;

    typedef SimplePointToPointMetricICP<AffineTransform<float,2>> SimplePointToPointMetricAffineICP2f;
    typedef SimplePointToPointMetricICP<AffineTransform<double,2>> SimplePointToPointMetricAffineICP2d;
    typedef SimplePointToPointMetricICP<AffineTransform<float,3>> SimplePointToPointMetricAffineICP3f;
    typedef SimplePointToPointMetricICP<AffineTransform<double,3>> SimplePointToPointMetricAffineICP3d;

    // typedef SimplePointToPointMetricProjectiveICP<RigidTransform<float,3>> SimplePointToPointMetricRigidProjectiveICP3f;
    // typedef SimplePointToPointMetricProjectiveICP<RigidTransform<double,3>> SimplePointToPointMetricRigidProjectiveICP3d;

    // typedef SimplePointToPointMetricProjectiveICP<AffineTransform<float,3>> SimplePointToPointMetricAffineProjectiveICP3f;
    // typedef SimplePointToPointMetricProjectiveICP<AffineTransform<double,3>> SimplePointToPointMetricAffineProjectiveICP3d;

    // Combined metric
    typedef SimpleCombinedMetricICP<RigidTransform<float,2>> SimpleCombinedMetricRigidICP2f;
    typedef SimpleCombinedMetricICP<RigidTransform<double,2>> SimpleCombinedMetricRigidICP2d;
    typedef SimpleCombinedMetricICP<RigidTransform<float,3>> SimpleCombinedMetricRigidICP3f;
    typedef SimpleCombinedMetricICP<RigidTransform<double,3>> SimpleCombinedMetricRigidICP3d;

    typedef SimpleCombinedMetricICP<AffineTransform<float,2>> SimpleCombinedMetricAffineICP2f;
    typedef SimpleCombinedMetricICP<AffineTransform<double,2>> SimpleCombinedMetricAffineICP2d;
    typedef SimpleCombinedMetricICP<AffineTransform<float,3>> SimpleCombinedMetricAffineICP3f;
    typedef SimpleCombinedMetricICP<AffineTransform<double,3>> SimpleCombinedMetricAffineICP3d;

    typedef SimpleCombinedMetricProjectiveICP<RigidTransform<float,3>> SimpleCombinedMetricRigidProjectiveICP3f;
    typedef SimpleCombinedMetricProjectiveICP<RigidTransform<double,3>> SimpleCombinedMetricRigidProjectiveICP3d;

    typedef SimpleCombinedMetricProjectiveICP<AffineTransform<float,3>> SimpleCombinedMetricAffineProjectiveICP3f;
    typedef SimpleCombinedMetricProjectiveICP<AffineTransform<double,3>> SimpleCombinedMetricAffineProjectiveICP3d;

    // Dense warp field
    typedef SimpleCombinedMetricDenseWarpFieldICP<RigidTransform<float,2>> SimpleCombinedMetricDenseRigidWarpFieldICP2f;
    typedef SimpleCombinedMetricDenseWarpFieldICP<RigidTransform<double,2>> SimpleCombinedMetricDenseRigidWarpFieldICP2d;
    typedef SimpleCombinedMetricDenseWarpFieldICP<RigidTransform<float,3>> SimpleCombinedMetricDenseRigidWarpFieldICP3f;
    typedef SimpleCombinedMetricDenseWarpFieldICP<RigidTransform<double,3>> SimpleCombinedMetricDenseRigidWarpFieldICP3d;

    typedef SimpleCombinedMetricDenseWarpFieldICP<AffineTransform<float,2>> SimpleCombinedMetricDenseAffineWarpFieldICP2f;
    typedef SimpleCombinedMetricDenseWarpFieldICP<AffineTransform<double,2>> SimpleCombinedMetricDenseAffineWarpFieldICP2d;
    typedef SimpleCombinedMetricDenseWarpFieldICP<AffineTransform<float,3>> SimpleCombinedMetricDenseAffineWarpFieldICP3f;
    typedef SimpleCombinedMetricDenseWarpFieldICP<AffineTransform<double,3>> SimpleCombinedMetricDenseAffineWarpFieldICP3d;

    typedef SimpleCombinedMetricDenseWarpFieldProjectiveICP<RigidTransform<float,3>> SimpleCombinedMetricDenseRigidWarpFieldProjectiveICP3f;
    typedef SimpleCombinedMetricDenseWarpFieldProjectiveICP<RigidTransform<double,3>> SimpleCombinedMetricDenseRigidWarpFieldProjectiveICP3d;

    typedef SimpleCombinedMetricDenseWarpFieldProjectiveICP<AffineTransform<float,3>> SimpleCombinedMetricDenseAffineWarpFieldProjectiveICP3f;
    typedef SimpleCombinedMetricDenseWarpFieldProjectiveICP<AffineTransform<double,3>> SimpleCombinedMetricDenseAffineWarpFieldProjectiveICP3d;

    // Sparse warp field
    typedef SimpleCombinedMetricSparseWarpFieldICP<RigidTransform<float,2>> SimpleCombinedMetricSparseRigidWarpFieldICP2f;
    typedef SimpleCombinedMetricSparseWarpFieldICP<RigidTransform<double,2>> SimpleCombinedMetricSparseRigidWarpFieldICP2d;
    typedef SimpleCombinedMetricSparseWarpFieldICP<RigidTransform<float,3>> SimpleCombinedMetricSparseRigidWarpFieldICP3f;
    typedef SimpleCombinedMetricSparseWarpFieldICP<RigidTransform<double,3>> SimpleCombinedMetricSparseRigidWarpFieldICP3d;

    typedef SimpleCombinedMetricSparseWarpFieldICP<AffineTransform<float,2>> SimpleCombinedMetricSparseAffineWarpFieldICP2f;
    typedef SimpleCombinedMetricSparseWarpFieldICP<AffineTransform<double,2>> SimpleCombinedMetricSparseAffineWarpFieldICP2d;
    typedef SimpleCombinedMetricSparseWarpFieldICP<AffineTransform<float,3>> SimpleCombinedMetricSparseAffineWarpFieldICP3f;
    typedef SimpleCombinedMetricSparseWarpFieldICP<AffineTransform<double,3>> SimpleCombinedMetricSparseAffineWarpFieldICP3d;

    typedef SimpleCombinedMetricSparseWarpFieldProjectiveICP<RigidTransform<float,3>> SimpleCombinedMetricSparseRigidWarpFieldProjectiveICP3f;
    typedef SimpleCombinedMetricSparseWarpFieldProjectiveICP<RigidTransform<double,3>> SimpleCombinedMetricSparseRigidWarpFieldProjectiveICP3d;

    typedef SimpleCombinedMetricSparseWarpFieldProjectiveICP<AffineTransform<float,3>> SimpleCombinedMetricSparseAffineWarpFieldProjectiveICP3f;
    typedef SimpleCombinedMetricSparseWarpFieldProjectiveICP<AffineTransform<double,3>> SimpleCombinedMetricSparseAffineWarpFieldProjectiveICP3d;
}
