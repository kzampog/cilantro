#pragma once

#include <cilantro/common_transformable_feature_adaptors.hpp>
#include <cilantro/correspondence_search_kd_tree.hpp>
#include <cilantro/correspondence_search_projective.hpp>
#include <cilantro/icp_rigid_point_to_point.hpp>
#include <cilantro/icp_rigid_combined_metric_3d.hpp>
#include <cilantro/icp_non_rigid_combined_metric_dense_3d.hpp>
#include <cilantro/icp_non_rigid_combined_metric_sparse_3d.hpp>

namespace cilantro {
    namespace internal {
        template <typename ScalarT, ptrdiff_t EigenDim, class CorrSearchT>
        class SimpleICPEntitiesContainer {
        public:
            EIGEN_MAKE_ALIGNED_OPERATOR_NEW

            SimpleICPEntitiesContainer(const ConstVectorSetMatrixMap<ScalarT,EigenDim> &dst_points,
                                       const ConstVectorSetMatrixMap<ScalarT,EigenDim> &src_points)
                    : dst_feat_(dst_points), src_feat_(src_points),
                      corr_search_(dst_feat_, src_feat_, corr_dist_evaluator_)
            {}

        protected:
            PointFeaturesAdaptor<ScalarT,EigenDim> dst_feat_;
            PointFeaturesAdaptor<ScalarT,EigenDim> src_feat_;
            DistanceEvaluator<ScalarT,ScalarT> corr_dist_evaluator_;
            CorrSearchT corr_search_;
            UnityWeightEvaluator<ScalarT,ScalarT> point_corr_weight_eval_;
            UnityWeightEvaluator<ScalarT,ScalarT> plane_corr_weight_eval_;
            RBFKernelWeightEvaluator<ScalarT,ScalarT,true> control_weight_eval_;
            RBFKernelWeightEvaluator<ScalarT,ScalarT,true> reg_weight_eval_;
        };

        template <typename ScalarT, ptrdiff_t EigenDim>
        using DefaultKDTreeCorrespondenceSearch = CorrespondenceSearchKDTree<PointFeaturesAdaptor<ScalarT,EigenDim>,KDTreeDistanceAdaptors::L2,PointFeaturesAdaptor<ScalarT,EigenDim>,DistanceEvaluator<ScalarT,ScalarT>>;

        template <typename ScalarT>
        using DefaultProjectiveCorrespondenceSearch3 = CorrespondenceSearchProjective3<ScalarT,PointFeaturesAdaptor<ScalarT,3>,DistanceEvaluator<ScalarT,ScalarT>>;
    }

    template <typename ScalarT, ptrdiff_t EigenDim>
    class SimplePointToPointMetricRigidICP : private internal::SimpleICPEntitiesContainer<ScalarT,EigenDim,internal::DefaultKDTreeCorrespondenceSearch<ScalarT,EigenDim>>,
                                             public PointToPointMetricRigidICP<ScalarT,EigenDim,internal::DefaultKDTreeCorrespondenceSearch<ScalarT,EigenDim>>
    {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        SimplePointToPointMetricRigidICP(const ConstVectorSetMatrixMap<ScalarT,EigenDim> &dst_points,
                                         const ConstVectorSetMatrixMap<ScalarT,EigenDim> &src_points)
                : internal::SimpleICPEntitiesContainer<ScalarT,EigenDim,internal::DefaultKDTreeCorrespondenceSearch<ScalarT,EigenDim>>(dst_points, src_points),
                  PointToPointMetricRigidICP<ScalarT,EigenDim,internal::DefaultKDTreeCorrespondenceSearch<ScalarT,EigenDim>>(dst_points, src_points, this->corr_search_)
        {}
    };

    typedef SimplePointToPointMetricRigidICP<float,2> SimplePointToPointMetricRigidICP2f;
    typedef SimplePointToPointMetricRigidICP<double,2> SimplePointToPointMetricRigidICP2d;
    typedef SimplePointToPointMetricRigidICP<float,3> SimplePointToPointMetricRigidICP3f;
    typedef SimplePointToPointMetricRigidICP<double,3> SimplePointToPointMetricRigidICP3d;

    template <typename ScalarT>
    class SimplePointToPointMetricRigidProjectiveICP3 : private internal::SimpleICPEntitiesContainer<ScalarT,3,internal::DefaultProjectiveCorrespondenceSearch3<ScalarT>>,
                                                        public PointToPointMetricRigidICP<ScalarT,3,internal::DefaultProjectiveCorrespondenceSearch3<ScalarT>>
    {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        SimplePointToPointMetricRigidProjectiveICP3(const ConstVectorSetMatrixMap<ScalarT,3> &dst_points,
                                                    const ConstVectorSetMatrixMap<ScalarT,3> &src_points)
                : internal::SimpleICPEntitiesContainer<ScalarT,3,internal::DefaultProjectiveCorrespondenceSearch3<ScalarT>>(dst_points, src_points),
                  PointToPointMetricRigidICP<ScalarT,3,internal::DefaultProjectiveCorrespondenceSearch3<ScalarT>>(dst_points, src_points, this->corr_search_)
        {}
    };

    typedef SimplePointToPointMetricRigidProjectiveICP3<float> SimplePointToPointMetricRigidProjectiveICP3f;
    typedef SimplePointToPointMetricRigidProjectiveICP3<double> SimplePointToPointMetricRigidProjectiveICP3d;

    template <typename ScalarT>
    class SimpleCombinedMetricRigidICP3 : private internal::SimpleICPEntitiesContainer<ScalarT,3,internal::DefaultKDTreeCorrespondenceSearch<ScalarT,3>>,
                                          public CombinedMetricRigidICP3<ScalarT,internal::DefaultKDTreeCorrespondenceSearch<ScalarT,3>,UnityWeightEvaluator<ScalarT,ScalarT>,UnityWeightEvaluator<ScalarT,ScalarT>>
    {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        SimpleCombinedMetricRigidICP3(const ConstVectorSetMatrixMap<ScalarT,3> &dst_points,
                                      const ConstVectorSetMatrixMap<ScalarT,3> &dst_normals,
                                      const ConstVectorSetMatrixMap<ScalarT,3> &src_points)
                : internal::SimpleICPEntitiesContainer<ScalarT,3,internal::DefaultKDTreeCorrespondenceSearch<ScalarT,3>>(dst_points, src_points),
                  CombinedMetricRigidICP3<ScalarT,internal::DefaultKDTreeCorrespondenceSearch<ScalarT,3>,UnityWeightEvaluator<ScalarT,ScalarT>,UnityWeightEvaluator<ScalarT,ScalarT>>(dst_points, dst_normals, src_points, this->corr_search_, this->point_corr_weight_eval_, this->plane_corr_weight_eval_)
        {}
    };

    typedef SimpleCombinedMetricRigidICP3<float> SimpleCombinedMetricRigidICP3f;
    typedef SimpleCombinedMetricRigidICP3<double> SimpleCombinedMetricRigidICP3d;

    template <typename ScalarT>
    class SimpleCombinedMetricRigidProjectiveICP3 : private internal::SimpleICPEntitiesContainer<ScalarT,3,internal::DefaultProjectiveCorrespondenceSearch3<ScalarT>>,
                                                    public CombinedMetricRigidICP3<ScalarT,internal::DefaultProjectiveCorrespondenceSearch3<ScalarT>,UnityWeightEvaluator<ScalarT,ScalarT>,UnityWeightEvaluator<ScalarT,ScalarT>>
    {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        SimpleCombinedMetricRigidProjectiveICP3(const ConstVectorSetMatrixMap<ScalarT,3> &dst_points,
                                                const ConstVectorSetMatrixMap<ScalarT,3> &dst_normals,
                                                const ConstVectorSetMatrixMap<ScalarT,3> &src_points)
                : internal::SimpleICPEntitiesContainer<ScalarT,3,internal::DefaultProjectiveCorrespondenceSearch3<ScalarT>>(dst_points, src_points),
                  CombinedMetricRigidICP3<ScalarT,internal::DefaultProjectiveCorrespondenceSearch3<ScalarT>,UnityWeightEvaluator<ScalarT,ScalarT>,UnityWeightEvaluator<ScalarT,ScalarT>>(dst_points, dst_normals, src_points, this->corr_search_, this->point_corr_weight_eval_, this->plane_corr_weight_eval_)
        {}
    };

    typedef SimpleCombinedMetricRigidProjectiveICP3<float> SimpleCombinedMetricRigidProjectiveICP3f;
    typedef SimpleCombinedMetricRigidProjectiveICP3<double> SimpleCombinedMetricRigidProjectiveICP3d;

    template <typename ScalarT>
    class SimpleDenseCombinedMetricNonRigidICP3 : private internal::SimpleICPEntitiesContainer<ScalarT,3,internal::DefaultKDTreeCorrespondenceSearch<ScalarT,3>>,
                                                  public DenseCombinedMetricNonRigidICP3<ScalarT,internal::DefaultKDTreeCorrespondenceSearch<ScalarT,3>,UnityWeightEvaluator<ScalarT,ScalarT>,UnityWeightEvaluator<ScalarT,ScalarT>,RBFKernelWeightEvaluator<ScalarT,ScalarT,true>>
    {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        SimpleDenseCombinedMetricNonRigidICP3(const ConstVectorSetMatrixMap<ScalarT,3> &dst_points,
                                              const ConstVectorSetMatrixMap<ScalarT,3> &dst_normals,
                                              const ConstVectorSetMatrixMap<ScalarT,3> &src_points,
                                              const std::vector<NeighborSet<ScalarT>> &regularization_neighborhoods)
                : internal::SimpleICPEntitiesContainer<ScalarT,3,internal::DefaultKDTreeCorrespondenceSearch<ScalarT,3>>(dst_points, src_points),
                  DenseCombinedMetricNonRigidICP3<ScalarT,internal::DefaultKDTreeCorrespondenceSearch<ScalarT,3>,UnityWeightEvaluator<ScalarT,ScalarT>,UnityWeightEvaluator<ScalarT,ScalarT>,RBFKernelWeightEvaluator<ScalarT,ScalarT,true>>(dst_points, dst_normals, src_points, this->corr_search_, this->point_corr_weight_eval_, this->plane_corr_weight_eval_, regularization_neighborhoods, this->reg_weight_eval_)
        {}
    };

    typedef SimpleDenseCombinedMetricNonRigidICP3<float> SimpleDenseCombinedMetricNonRigidICP3f;
    typedef SimpleDenseCombinedMetricNonRigidICP3<double> SimpleDenseCombinedMetricNonRigidICP3d;

    template <typename ScalarT>
    class SimpleDenseCombinedMetricNonRigidProjectiveICP3 : private internal::SimpleICPEntitiesContainer<ScalarT,3,internal::DefaultProjectiveCorrespondenceSearch3<ScalarT>>,
                                                            public DenseCombinedMetricNonRigidICP3<ScalarT,internal::DefaultProjectiveCorrespondenceSearch3<ScalarT>,UnityWeightEvaluator<ScalarT,ScalarT>,UnityWeightEvaluator<ScalarT,ScalarT>,RBFKernelWeightEvaluator<ScalarT,ScalarT,true>>
    {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        SimpleDenseCombinedMetricNonRigidProjectiveICP3(const ConstVectorSetMatrixMap<ScalarT,3> &dst_points,
                                                        const ConstVectorSetMatrixMap<ScalarT,3> &dst_normals,
                                                        const ConstVectorSetMatrixMap<ScalarT,3> &src_points,
                                                        const std::vector<NeighborSet<ScalarT>> &regularization_neighborhoods)
                : internal::SimpleICPEntitiesContainer<ScalarT,3,internal::DefaultProjectiveCorrespondenceSearch3<ScalarT>>(dst_points, src_points),
                  DenseCombinedMetricNonRigidICP3<ScalarT,internal::DefaultProjectiveCorrespondenceSearch3<ScalarT>,UnityWeightEvaluator<ScalarT,ScalarT>,UnityWeightEvaluator<ScalarT,ScalarT>,RBFKernelWeightEvaluator<ScalarT,ScalarT,true>>(dst_points, dst_normals, src_points, this->corr_search_, this->point_corr_weight_eval_, this->plane_corr_weight_eval_, regularization_neighborhoods, this->reg_weight_eval_)
        {}
    };

    typedef SimpleDenseCombinedMetricNonRigidProjectiveICP3<float> SimpleDenseCombinedMetricNonRigidProjectiveICP3f;
    typedef SimpleDenseCombinedMetricNonRigidProjectiveICP3<double> SimpleDenseCombinedMetricNonRigidProjectiveICP3d;

    template <typename ScalarT>
    class SimpleSparseCombinedMetricNonRigidICP3 : private internal::SimpleICPEntitiesContainer<ScalarT,3,internal::DefaultKDTreeCorrespondenceSearch<ScalarT,3>>,
                                                   public SparseCombinedMetricNonRigidICP3<ScalarT,internal::DefaultKDTreeCorrespondenceSearch<ScalarT,3>,UnityWeightEvaluator<ScalarT,ScalarT>,UnityWeightEvaluator<ScalarT,ScalarT>,RBFKernelWeightEvaluator<ScalarT,ScalarT,true>,RBFKernelWeightEvaluator<ScalarT,ScalarT,true>>
    {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        SimpleSparseCombinedMetricNonRigidICP3(const ConstVectorSetMatrixMap<ScalarT,3> &dst_points,
                                               const ConstVectorSetMatrixMap<ScalarT,3> &dst_normals,
                                               const ConstVectorSetMatrixMap<ScalarT,3> &src_points,
                                               const std::vector<NeighborSet<ScalarT>> &src_to_control_neighborhoods,
                                               size_t num_control_nodes,
                                               const std::vector<NeighborSet<ScalarT>> &control_regularization_neighborhoods)
                : internal::SimpleICPEntitiesContainer<ScalarT,3,internal::DefaultKDTreeCorrespondenceSearch<ScalarT,3>>(dst_points, src_points),
                  SparseCombinedMetricNonRigidICP3<ScalarT,internal::DefaultKDTreeCorrespondenceSearch<ScalarT,3>,UnityWeightEvaluator<ScalarT,ScalarT>,UnityWeightEvaluator<ScalarT,ScalarT>,RBFKernelWeightEvaluator<ScalarT,ScalarT,true>,RBFKernelWeightEvaluator<ScalarT,ScalarT,true>>(dst_points, dst_normals, src_points, this->corr_search_, this->point_corr_weight_eval_, this->plane_corr_weight_eval_, src_to_control_neighborhoods, num_control_nodes, this->control_weight_eval_, control_regularization_neighborhoods, this->reg_weight_eval_)
        {}
    };

    typedef SimpleSparseCombinedMetricNonRigidICP3<float> SimpleSparseCombinedMetricNonRigidICP3f;
    typedef SimpleSparseCombinedMetricNonRigidICP3<double> SimpleSparseCombinedMetricNonRigidICP3d;

    template <typename ScalarT>
    class SimpleSparseCombinedMetricNonRigidProjectiveICP3 : private internal::SimpleICPEntitiesContainer<ScalarT,3,internal::DefaultProjectiveCorrespondenceSearch3<ScalarT>>,
                                                             public SparseCombinedMetricNonRigidICP3<ScalarT,internal::DefaultProjectiveCorrespondenceSearch3<ScalarT>,UnityWeightEvaluator<ScalarT,ScalarT>,UnityWeightEvaluator<ScalarT,ScalarT>,RBFKernelWeightEvaluator<ScalarT,ScalarT,true>,RBFKernelWeightEvaluator<ScalarT,ScalarT,true>>
    {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        SimpleSparseCombinedMetricNonRigidProjectiveICP3(const ConstVectorSetMatrixMap<ScalarT,3> &dst_points,
                                                         const ConstVectorSetMatrixMap<ScalarT,3> &dst_normals,
                                                         const ConstVectorSetMatrixMap<ScalarT,3> &src_points,
                                                         const std::vector<NeighborSet<ScalarT>> &src_to_control_neighborhoods,
                                                         size_t num_control_nodes,
                                                         const std::vector<NeighborSet<ScalarT>> &control_regularization_neighborhoods)
                : internal::SimpleICPEntitiesContainer<ScalarT,3,internal::DefaultProjectiveCorrespondenceSearch3<ScalarT>>(dst_points, src_points),
                  SparseCombinedMetricNonRigidICP3<ScalarT,internal::DefaultProjectiveCorrespondenceSearch3<ScalarT>,UnityWeightEvaluator<ScalarT,ScalarT>,UnityWeightEvaluator<ScalarT,ScalarT>,RBFKernelWeightEvaluator<ScalarT,ScalarT,true>,RBFKernelWeightEvaluator<ScalarT,ScalarT,true>>(dst_points, dst_normals, src_points, this->corr_search_, this->point_corr_weight_eval_, this->plane_corr_weight_eval_, src_to_control_neighborhoods, num_control_nodes, this->control_weight_eval_, control_regularization_neighborhoods, this->reg_weight_eval_)
        {}
    };

    typedef SimpleSparseCombinedMetricNonRigidProjectiveICP3<float> SimpleSparseCombinedMetricNonRigidProjectiveICP3f;
    typedef SimpleSparseCombinedMetricNonRigidProjectiveICP3<double> SimpleSparseCombinedMetricNonRigidProjectiveICP3d;
}
