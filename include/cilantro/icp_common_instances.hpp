#pragma once

#include <cilantro/icp_core.hpp>
#include <cilantro/icp_rigid_point_to_point_metric_optimizer.hpp>
#include <cilantro/icp_rigid_combined_metric_optimizer_3d.hpp>
#include <cilantro/icp_common_feature_adaptors.hpp>
#include <cilantro/icp_correspondence_search_kd_tree.hpp>
#include <cilantro/icp_correspondence_search_projective.hpp>

namespace cilantro {
    template <typename ScalarT, ptrdiff_t EigenDim>
    class PointToPointMetricRigidICP : private RigidICPPointToPointMetricOptimizer<ScalarT,EigenDim>,
                                       private std::pair<PointFeaturesAdaptor<ScalarT,EigenDim>,PointFeaturesAdaptor<ScalarT,EigenDim>>,
                                       private CorrespondenceDistanceEvaluator<ScalarT>,
                                       private ICPCorrespondenceSearchKDTree<PointFeaturesAdaptor<ScalarT,EigenDim>,KDTreeDistanceAdaptors::L2,CorrespondenceDistanceEvaluator<ScalarT>>,
                                       public IterativeClosestPoint<RigidICPPointToPointMetricOptimizer<ScalarT,EigenDim>,ICPCorrespondenceSearchKDTree<PointFeaturesAdaptor<ScalarT,EigenDim>,KDTreeDistanceAdaptors::L2,CorrespondenceDistanceEvaluator<ScalarT>>>
    {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        PointToPointMetricRigidICP(const ConstVectorSetMatrixMap<ScalarT,EigenDim> &dst_points,
                                   const ConstVectorSetMatrixMap<ScalarT,EigenDim> &src_points)
                : RigidICPPointToPointMetricOptimizer<ScalarT,EigenDim>(dst_points, src_points),
                  std::pair<PointFeaturesAdaptor<ScalarT,EigenDim>,PointFeaturesAdaptor<ScalarT,EigenDim>>(PointFeaturesAdaptor<ScalarT,EigenDim>(dst_points), PointFeaturesAdaptor<ScalarT,EigenDim>(src_points)),
                  CorrespondenceDistanceEvaluator<ScalarT>(),
                  ICPCorrespondenceSearchKDTree<PointFeaturesAdaptor<ScalarT,EigenDim>,KDTreeDistanceAdaptors::L2,CorrespondenceDistanceEvaluator<ScalarT>>(this->first, this->second, *this),
                  IterativeClosestPoint<RigidICPPointToPointMetricOptimizer<ScalarT,EigenDim>,ICPCorrespondenceSearchKDTree<PointFeaturesAdaptor<ScalarT,EigenDim>,KDTreeDistanceAdaptors::L2,CorrespondenceDistanceEvaluator<ScalarT>>>(*this, *this)
        {}
    };

    typedef PointToPointMetricRigidICP<float,2> PointToPointMetricRigidICP2f;
    typedef PointToPointMetricRigidICP<double,2> PointToPointMetricRigidICP2d;
    typedef PointToPointMetricRigidICP<float,3> PointToPointMetricRigidICP3f;
    typedef PointToPointMetricRigidICP<double,3> PointToPointMetricRigidICP3d;

    template <typename ScalarT>
    class CombinedMetricRigidICP3 : private RigidICPCombinedMetricOptimizer3<ScalarT>,
                                    private std::pair<PointFeaturesAdaptor<ScalarT,3>,PointFeaturesAdaptor<ScalarT,3>>,
                                    private CorrespondenceDistanceEvaluator<ScalarT>,
                                    private ICPCorrespondenceSearchKDTree<PointFeaturesAdaptor<ScalarT,3>,KDTreeDistanceAdaptors::L2,CorrespondenceDistanceEvaluator<ScalarT>>,
                                    public IterativeClosestPoint<RigidICPCombinedMetricOptimizer3<ScalarT>,ICPCorrespondenceSearchKDTree<PointFeaturesAdaptor<ScalarT,3>,KDTreeDistanceAdaptors::L2,CorrespondenceDistanceEvaluator<ScalarT>>>
    {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        CombinedMetricRigidICP3(const ConstVectorSetMatrixMap<ScalarT,3> &dst_points,
                                const ConstVectorSetMatrixMap<ScalarT,3> &dst_normals,
                                const ConstVectorSetMatrixMap<ScalarT,3> &src_points)
                : RigidICPCombinedMetricOptimizer3<ScalarT>(dst_points, dst_normals, src_points),
                  std::pair<PointFeaturesAdaptor<ScalarT,3>,PointFeaturesAdaptor<ScalarT,3>>(PointFeaturesAdaptor<ScalarT,3>(dst_points), PointFeaturesAdaptor<ScalarT,3>(src_points)),
                  CorrespondenceDistanceEvaluator<ScalarT>(),
                  ICPCorrespondenceSearchKDTree<PointFeaturesAdaptor<ScalarT,3>,KDTreeDistanceAdaptors::L2,CorrespondenceDistanceEvaluator<ScalarT>>(this->first, this->second, *this),
                  IterativeClosestPoint<RigidICPCombinedMetricOptimizer3<ScalarT>,ICPCorrespondenceSearchKDTree<PointFeaturesAdaptor<ScalarT,3>,KDTreeDistanceAdaptors::L2,CorrespondenceDistanceEvaluator<ScalarT>>>(*this, *this)
        {}
    };

    typedef CombinedMetricRigidICP3<float> CombinedMetricRigidICP3f;
    typedef CombinedMetricRigidICP3<double> CombinedMetricRigidICP3d;

    template <typename ScalarT>
    class PointToPointMetricRigidProjectiveICP3 : private RigidICPPointToPointMetricOptimizer<ScalarT,3>,
                                                  private std::pair<PointFeaturesAdaptor<ScalarT,3>,PointFeaturesAdaptor<ScalarT,3>>,
                                                  private CorrespondenceDistanceEvaluator<ScalarT>,
                                                  private ICPCorrespondenceSearchProjective3<ScalarT,CorrespondenceDistanceEvaluator<ScalarT>>,
                                                  public IterativeClosestPoint<RigidICPPointToPointMetricOptimizer<ScalarT,3>,ICPCorrespondenceSearchProjective3<ScalarT,CorrespondenceDistanceEvaluator<ScalarT>>>
    {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        PointToPointMetricRigidProjectiveICP3(const ConstVectorSetMatrixMap<ScalarT,3> &dst_points,
                                              const ConstVectorSetMatrixMap<ScalarT,3> &src_points)
                : RigidICPPointToPointMetricOptimizer<ScalarT,3>(dst_points, src_points),
                  std::pair<PointFeaturesAdaptor<ScalarT,3>,PointFeaturesAdaptor<ScalarT,3>>(PointFeaturesAdaptor<ScalarT,3>(dst_points), PointFeaturesAdaptor<ScalarT,3>(src_points)),
                  CorrespondenceDistanceEvaluator<ScalarT>(),
                  ICPCorrespondenceSearchProjective3<ScalarT,CorrespondenceDistanceEvaluator<ScalarT>>(this->first, this->second, *this),
                  IterativeClosestPoint<RigidICPPointToPointMetricOptimizer<ScalarT,3>,ICPCorrespondenceSearchProjective3<ScalarT,CorrespondenceDistanceEvaluator<ScalarT>>>(*this, *this)
        {}
    };

    typedef PointToPointMetricRigidProjectiveICP3<float> PointToPointMetricRigidProjectiveICP3f;
    typedef PointToPointMetricRigidProjectiveICP3<double> PointToPointMetricRigidProjectiveICP3d;

    template <typename ScalarT>
    class CombinedMetricRigidProjectiveICP3 : private RigidICPCombinedMetricOptimizer3<ScalarT>,
                                              private std::pair<PointFeaturesAdaptor<ScalarT,3>,PointFeaturesAdaptor<ScalarT,3>>,
                                              private CorrespondenceDistanceEvaluator<ScalarT>,
                                              private ICPCorrespondenceSearchProjective3<ScalarT,CorrespondenceDistanceEvaluator<ScalarT>>,
                                              public IterativeClosestPoint<RigidICPCombinedMetricOptimizer3<ScalarT>,ICPCorrespondenceSearchProjective3<ScalarT,CorrespondenceDistanceEvaluator<ScalarT>>>
    {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        CombinedMetricRigidProjectiveICP3(const ConstVectorSetMatrixMap<ScalarT,3> &dst_points,
                                          const ConstVectorSetMatrixMap<ScalarT,3> &dst_normals,
                                          const ConstVectorSetMatrixMap<ScalarT,3> &src_points)
                : RigidICPCombinedMetricOptimizer3<ScalarT>(dst_points, dst_normals, src_points),
                  std::pair<PointFeaturesAdaptor<ScalarT,3>,PointFeaturesAdaptor<ScalarT,3>>(PointFeaturesAdaptor<ScalarT,3>(dst_points), PointFeaturesAdaptor<ScalarT,3>(src_points)),
                  CorrespondenceDistanceEvaluator<ScalarT>(),
                  ICPCorrespondenceSearchProjective3<ScalarT,CorrespondenceDistanceEvaluator<ScalarT>>(this->first, this->second, *this),
                  IterativeClosestPoint<RigidICPCombinedMetricOptimizer3<ScalarT>,ICPCorrespondenceSearchProjective3<ScalarT,CorrespondenceDistanceEvaluator<ScalarT>>>(*this, *this)
        {}
    };

    typedef CombinedMetricRigidProjectiveICP3<float> CombinedMetricRigidProjectiveICP3f;
    typedef CombinedMetricRigidProjectiveICP3<double> CombinedMetricRigidProjectiveICP3d;
}
