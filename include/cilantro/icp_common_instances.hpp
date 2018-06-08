#pragma once

#include <cilantro/icp_common_feature_adaptors.hpp>
#include <cilantro/icp_correspondence_search_kd_tree.hpp>
#include <cilantro/icp_correspondence_search_projective.hpp>
#include <cilantro/icp_rigid_point_to_point.hpp>
#include <cilantro/icp_rigid_combined_metric_3d.hpp>

namespace cilantro {
    template <typename ScalarT, ptrdiff_t EigenDim>
    class SimplePointToPointMetricRigidICP : private std::pair<PointFeaturesAdaptor<ScalarT,EigenDim>,PointFeaturesAdaptor<ScalarT,EigenDim>>,
                                             private CorrespondenceDistanceEvaluator<ScalarT>,
                                             private ICPCorrespondenceSearchKDTree<PointFeaturesAdaptor<ScalarT,EigenDim>,KDTreeDistanceAdaptors::L2,CorrespondenceDistanceEvaluator<ScalarT>>,
                                             public PointToPointMetricRigidICP<ScalarT,EigenDim,ICPCorrespondenceSearchKDTree<PointFeaturesAdaptor<ScalarT,EigenDim>,KDTreeDistanceAdaptors::L2,CorrespondenceDistanceEvaluator<ScalarT>>>
    {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        SimplePointToPointMetricRigidICP(const ConstVectorSetMatrixMap<ScalarT,EigenDim> &dst_points,
                                         const ConstVectorSetMatrixMap<ScalarT,EigenDim> &src_points)
                : std::pair<PointFeaturesAdaptor<ScalarT,EigenDim>,PointFeaturesAdaptor<ScalarT,EigenDim>>(PointFeaturesAdaptor<ScalarT,EigenDim>(dst_points), PointFeaturesAdaptor<ScalarT,EigenDim>(src_points)),
                  CorrespondenceDistanceEvaluator<ScalarT>(),
                  ICPCorrespondenceSearchKDTree<PointFeaturesAdaptor<ScalarT,EigenDim>,KDTreeDistanceAdaptors::L2,CorrespondenceDistanceEvaluator<ScalarT>>(this->first, this->second, *this),
                  PointToPointMetricRigidICP<ScalarT,EigenDim,ICPCorrespondenceSearchKDTree<PointFeaturesAdaptor<ScalarT,EigenDim>,KDTreeDistanceAdaptors::L2,CorrespondenceDistanceEvaluator<ScalarT>>>(dst_points, src_points, *this)
        {}
    };

    typedef SimplePointToPointMetricRigidICP<float,2> SimplePointToPointMetricRigidICP2f;
    typedef SimplePointToPointMetricRigidICP<double,2> SimplePointToPointMetricRigidICP2d;
    typedef SimplePointToPointMetricRigidICP<float,3> SimplePointToPointMetricRigidICP3f;
    typedef SimplePointToPointMetricRigidICP<double,3> SimplePointToPointMetricRigidICP3d;

    template <typename ScalarT>
    class SimpleCombinedMetricRigidICP3 : private std::pair<PointFeaturesAdaptor<ScalarT,3>,PointFeaturesAdaptor<ScalarT,3>>,
                                          private CorrespondenceDistanceEvaluator<ScalarT>,
                                          private ICPCorrespondenceSearchKDTree<PointFeaturesAdaptor<ScalarT,3>,KDTreeDistanceAdaptors::L2,CorrespondenceDistanceEvaluator<ScalarT>>,
                                          public CombinedMetricRigidICP3<ScalarT,ICPCorrespondenceSearchKDTree<PointFeaturesAdaptor<ScalarT,3>,KDTreeDistanceAdaptors::L2,CorrespondenceDistanceEvaluator<ScalarT>>>
    {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        SimpleCombinedMetricRigidICP3(const ConstVectorSetMatrixMap<ScalarT,3> &dst_points,
                                      const ConstVectorSetMatrixMap<ScalarT,3> &dst_normals,
                                      const ConstVectorSetMatrixMap<ScalarT,3> &src_points)
                : std::pair<PointFeaturesAdaptor<ScalarT,3>,PointFeaturesAdaptor<ScalarT,3>>(PointFeaturesAdaptor<ScalarT,3>(dst_points), PointFeaturesAdaptor<ScalarT,3>(src_points)),
                  CorrespondenceDistanceEvaluator<ScalarT>(),
                  ICPCorrespondenceSearchKDTree<PointFeaturesAdaptor<ScalarT,3>,KDTreeDistanceAdaptors::L2,CorrespondenceDistanceEvaluator<ScalarT>>(this->first, this->second, *this),
                  CombinedMetricRigidICP3<ScalarT,ICPCorrespondenceSearchKDTree<PointFeaturesAdaptor<ScalarT,3>,KDTreeDistanceAdaptors::L2,CorrespondenceDistanceEvaluator<ScalarT>>>(dst_points, dst_normals, src_points, *this)
        {}
    };

    typedef SimpleCombinedMetricRigidICP3<float> SimpleCombinedMetricRigidICP3f;
    typedef SimpleCombinedMetricRigidICP3<double> SimpleCombinedMetricRigidICP3d;

    template <typename ScalarT>
    class SimplePointToPointMetricRigidProjectiveICP3 : private std::pair<PointFeaturesAdaptor<ScalarT,3>,PointFeaturesAdaptor<ScalarT,3>>,
                                                        private CorrespondenceDistanceEvaluator<ScalarT>,
                                                        private ICPCorrespondenceSearchProjective3<ScalarT,CorrespondenceDistanceEvaluator<ScalarT>>,
                                                        public PointToPointMetricRigidICP<ScalarT,3,ICPCorrespondenceSearchProjective3<ScalarT,CorrespondenceDistanceEvaluator<ScalarT>>>
    {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        SimplePointToPointMetricRigidProjectiveICP3(const ConstVectorSetMatrixMap<ScalarT,3> &dst_points,
                                                    const ConstVectorSetMatrixMap<ScalarT,3> &src_points)
                : std::pair<PointFeaturesAdaptor<ScalarT,3>,PointFeaturesAdaptor<ScalarT,3>>(PointFeaturesAdaptor<ScalarT,3>(dst_points), PointFeaturesAdaptor<ScalarT,3>(src_points)),
                  CorrespondenceDistanceEvaluator<ScalarT>(),
                  ICPCorrespondenceSearchProjective3<ScalarT,CorrespondenceDistanceEvaluator<ScalarT>>(this->first, this->second, *this),
                  PointToPointMetricRigidICP<ScalarT,3,ICPCorrespondenceSearchProjective3<ScalarT,CorrespondenceDistanceEvaluator<ScalarT>>>(dst_points, src_points, *this)
        {}
    };

    typedef SimplePointToPointMetricRigidProjectiveICP3<float> SimplePointToPointMetricRigidProjectiveICP3f;
    typedef SimplePointToPointMetricRigidProjectiveICP3<double> SimplePointToPointMetricRigidProjectiveICP3d;

    template <typename ScalarT>
    class SimpleCombinedMetricRigidProjectiveICP3 : private std::pair<PointFeaturesAdaptor<ScalarT,3>,PointFeaturesAdaptor<ScalarT,3>>,
                                                    private CorrespondenceDistanceEvaluator<ScalarT>,
                                                    private ICPCorrespondenceSearchProjective3<ScalarT,CorrespondenceDistanceEvaluator<ScalarT>>,
                                                    public CombinedMetricRigidICP3<ScalarT,ICPCorrespondenceSearchProjective3<ScalarT,CorrespondenceDistanceEvaluator<ScalarT>>>
    {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        SimpleCombinedMetricRigidProjectiveICP3(const ConstVectorSetMatrixMap<ScalarT,3> &dst_points,
                                                const ConstVectorSetMatrixMap<ScalarT,3> &dst_normals,
                                                const ConstVectorSetMatrixMap<ScalarT,3> &src_points)
                : std::pair<PointFeaturesAdaptor<ScalarT,3>,PointFeaturesAdaptor<ScalarT,3>>(PointFeaturesAdaptor<ScalarT,3>(dst_points), PointFeaturesAdaptor<ScalarT,3>(src_points)),
                  CorrespondenceDistanceEvaluator<ScalarT>(),
                  ICPCorrespondenceSearchProjective3<ScalarT,CorrespondenceDistanceEvaluator<ScalarT>>(this->first, this->second, *this),
                  CombinedMetricRigidICP3<ScalarT,ICPCorrespondenceSearchProjective3<ScalarT,CorrespondenceDistanceEvaluator<ScalarT>>>(dst_points, dst_normals, src_points, *this)
        {}
    };

    typedef SimpleCombinedMetricRigidProjectiveICP3<float> SimpleCombinedMetricRigidProjectiveICP3f;
    typedef SimpleCombinedMetricRigidProjectiveICP3<double> SimpleCombinedMetricRigidProjectiveICP3d;
}
