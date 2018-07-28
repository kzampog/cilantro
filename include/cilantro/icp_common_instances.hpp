#pragma once

#include <cilantro/common_transformable_feature_adaptors.hpp>
#include <cilantro/correspondence_search_kd_tree.hpp>
#include <cilantro/correspondence_search_projective.hpp>
#include <cilantro/icp_rigid_point_to_point.hpp>
#include <cilantro/icp_rigid_combined_metric_3d.hpp>
#include <cilantro/icp_non_rigid_combined_metric_dense_3d.hpp>
#include <cilantro/icp_non_rigid_combined_metric_sparse_3d.hpp>

namespace cilantro {
    template <typename ScalarT, ptrdiff_t EigenDim>
    class SimplePointToPointMetricRigidICP : private std::pair<PointFeaturesAdaptor<ScalarT,EigenDim>,PointFeaturesAdaptor<ScalarT,EigenDim>>,
                                             private DistanceEvaluator<ScalarT,ScalarT>,
                                             private CorrespondenceSearchKDTree<PointFeaturesAdaptor<ScalarT,EigenDim>,KDTreeDistanceAdaptors::L2,DistanceEvaluator<ScalarT,ScalarT>>,
                                             public PointToPointMetricRigidICP<ScalarT,EigenDim,CorrespondenceSearchKDTree<PointFeaturesAdaptor<ScalarT,EigenDim>,KDTreeDistanceAdaptors::L2,DistanceEvaluator<ScalarT,ScalarT>>>
    {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        SimplePointToPointMetricRigidICP(const ConstVectorSetMatrixMap<ScalarT,EigenDim> &dst_points,
                                         const ConstVectorSetMatrixMap<ScalarT,EigenDim> &src_points)
                : std::pair<PointFeaturesAdaptor<ScalarT,EigenDim>,PointFeaturesAdaptor<ScalarT,EigenDim>>(PointFeaturesAdaptor<ScalarT,EigenDim>(dst_points), PointFeaturesAdaptor<ScalarT,EigenDim>(src_points)),
                  DistanceEvaluator<ScalarT,ScalarT>(),
                  CorrespondenceSearchKDTree<PointFeaturesAdaptor<ScalarT,EigenDim>,KDTreeDistanceAdaptors::L2,DistanceEvaluator<ScalarT,ScalarT>>(this->first, this->second, *this),
                  PointToPointMetricRigidICP<ScalarT,EigenDim,CorrespondenceSearchKDTree<PointFeaturesAdaptor<ScalarT,EigenDim>,KDTreeDistanceAdaptors::L2,DistanceEvaluator<ScalarT,ScalarT>>>(dst_points, src_points, *this)
        {}
    };

    typedef SimplePointToPointMetricRigidICP<float,2> SimplePointToPointMetricRigidICP2f;
    typedef SimplePointToPointMetricRigidICP<double,2> SimplePointToPointMetricRigidICP2d;
    typedef SimplePointToPointMetricRigidICP<float,3> SimplePointToPointMetricRigidICP3f;
    typedef SimplePointToPointMetricRigidICP<double,3> SimplePointToPointMetricRigidICP3d;

    template <typename ScalarT>
    class SimplePointToPointMetricRigidProjectiveICP3 : private std::pair<PointFeaturesAdaptor<ScalarT,3>,PointFeaturesAdaptor<ScalarT,3>>,
                                                        private DistanceEvaluator<ScalarT,ScalarT>,
                                                        private CorrespondenceSearchProjective3<ScalarT,DistanceEvaluator<ScalarT,ScalarT>>,
                                                        public PointToPointMetricRigidICP<ScalarT,3,CorrespondenceSearchProjective3<ScalarT,DistanceEvaluator<ScalarT,ScalarT>>>
    {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        SimplePointToPointMetricRigidProjectiveICP3(const ConstVectorSetMatrixMap<ScalarT,3> &dst_points,
                                                    const ConstVectorSetMatrixMap<ScalarT,3> &src_points)
                : std::pair<PointFeaturesAdaptor<ScalarT,3>,PointFeaturesAdaptor<ScalarT,3>>(PointFeaturesAdaptor<ScalarT,3>(dst_points), PointFeaturesAdaptor<ScalarT,3>(src_points)),
                  DistanceEvaluator<ScalarT,ScalarT>(),
                  CorrespondenceSearchProjective3<ScalarT,DistanceEvaluator<ScalarT,ScalarT>>(this->first, this->second, *this),
                  PointToPointMetricRigidICP<ScalarT,3,CorrespondenceSearchProjective3<ScalarT,DistanceEvaluator<ScalarT,ScalarT>>>(dst_points, src_points, *this)
        {}
    };

    typedef SimplePointToPointMetricRigidProjectiveICP3<float> SimplePointToPointMetricRigidProjectiveICP3f;
    typedef SimplePointToPointMetricRigidProjectiveICP3<double> SimplePointToPointMetricRigidProjectiveICP3d;

    template <typename ScalarT>
    class SimpleCombinedMetricRigidICP3 : private std::pair<PointFeaturesAdaptor<ScalarT,3>,PointFeaturesAdaptor<ScalarT,3>>,
                                          private DistanceEvaluator<ScalarT,ScalarT>,
                                          private CorrespondenceSearchKDTree<PointFeaturesAdaptor<ScalarT,3>,KDTreeDistanceAdaptors::L2,DistanceEvaluator<ScalarT,ScalarT>>,
                                          public CombinedMetricRigidICP3<ScalarT,CorrespondenceSearchKDTree<PointFeaturesAdaptor<ScalarT,3>,KDTreeDistanceAdaptors::L2,DistanceEvaluator<ScalarT,ScalarT>>>
    {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        SimpleCombinedMetricRigidICP3(const ConstVectorSetMatrixMap<ScalarT,3> &dst_points,
                                      const ConstVectorSetMatrixMap<ScalarT,3> &dst_normals,
                                      const ConstVectorSetMatrixMap<ScalarT,3> &src_points)
                : std::pair<PointFeaturesAdaptor<ScalarT,3>,PointFeaturesAdaptor<ScalarT,3>>(PointFeaturesAdaptor<ScalarT,3>(dst_points), PointFeaturesAdaptor<ScalarT,3>(src_points)),
                  DistanceEvaluator<ScalarT,ScalarT>(),
                  CorrespondenceSearchKDTree<PointFeaturesAdaptor<ScalarT,3>,KDTreeDistanceAdaptors::L2,DistanceEvaluator<ScalarT,ScalarT>>(this->first, this->second, *this),
                  CombinedMetricRigidICP3<ScalarT,CorrespondenceSearchKDTree<PointFeaturesAdaptor<ScalarT,3>,KDTreeDistanceAdaptors::L2,DistanceEvaluator<ScalarT,ScalarT>>>(dst_points, dst_normals, src_points, *this)
        {}
    };

    typedef SimpleCombinedMetricRigidICP3<float> SimpleCombinedMetricRigidICP3f;
    typedef SimpleCombinedMetricRigidICP3<double> SimpleCombinedMetricRigidICP3d;

    template <typename ScalarT>
    class SimpleCombinedMetricRigidProjectiveICP3 : private std::pair<PointFeaturesAdaptor<ScalarT,3>,PointFeaturesAdaptor<ScalarT,3>>,
                                                    private DistanceEvaluator<ScalarT,ScalarT>,
                                                    private CorrespondenceSearchProjective3<ScalarT,DistanceEvaluator<ScalarT,ScalarT>>,
                                                    public CombinedMetricRigidICP3<ScalarT,CorrespondenceSearchProjective3<ScalarT,DistanceEvaluator<ScalarT,ScalarT>>>
    {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        SimpleCombinedMetricRigidProjectiveICP3(const ConstVectorSetMatrixMap<ScalarT,3> &dst_points,
                                                const ConstVectorSetMatrixMap<ScalarT,3> &dst_normals,
                                                const ConstVectorSetMatrixMap<ScalarT,3> &src_points)
                : std::pair<PointFeaturesAdaptor<ScalarT,3>,PointFeaturesAdaptor<ScalarT,3>>(PointFeaturesAdaptor<ScalarT,3>(dst_points), PointFeaturesAdaptor<ScalarT,3>(src_points)),
                  DistanceEvaluator<ScalarT,ScalarT>(),
                  CorrespondenceSearchProjective3<ScalarT,DistanceEvaluator<ScalarT,ScalarT>>(this->first, this->second, *this),
                  CombinedMetricRigidICP3<ScalarT,CorrespondenceSearchProjective3<ScalarT,DistanceEvaluator<ScalarT,ScalarT>>>(dst_points, dst_normals, src_points, *this)
        {}
    };

    typedef SimpleCombinedMetricRigidProjectiveICP3<float> SimpleCombinedMetricRigidProjectiveICP3f;
    typedef SimpleCombinedMetricRigidProjectiveICP3<double> SimpleCombinedMetricRigidProjectiveICP3d;

    template <typename ScalarT>
    class SimpleDenseCombinedMetricNonRigidICP3 : private std::pair<PointFeaturesAdaptor<ScalarT,3>,PointFeaturesAdaptor<ScalarT,3>>,
                                                  private DistanceEvaluator<ScalarT,ScalarT>,
                                                  private CorrespondenceSearchKDTree<PointFeaturesAdaptor<ScalarT,3>,KDTreeDistanceAdaptors::L2,DistanceEvaluator<ScalarT,ScalarT>>,
                                                  public DenseCombinedMetricNonRigidICP3<ScalarT,CorrespondenceSearchKDTree<PointFeaturesAdaptor<ScalarT,3>,KDTreeDistanceAdaptors::L2,DistanceEvaluator<ScalarT,ScalarT>>>
    {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        SimpleDenseCombinedMetricNonRigidICP3(const ConstVectorSetMatrixMap<ScalarT,3> &dst_points,
                                              const ConstVectorSetMatrixMap<ScalarT,3> &dst_normals,
                                              const ConstVectorSetMatrixMap<ScalarT,3> &src_points,
                                              const std::vector<NeighborSet<ScalarT>> &regularization_neighborhoods)
                : std::pair<PointFeaturesAdaptor<ScalarT,3>,PointFeaturesAdaptor<ScalarT,3>>(PointFeaturesAdaptor<ScalarT,3>(dst_points), PointFeaturesAdaptor<ScalarT,3>(src_points)),
                  DistanceEvaluator<ScalarT,ScalarT>(),
                  CorrespondenceSearchKDTree<PointFeaturesAdaptor<ScalarT,3>,KDTreeDistanceAdaptors::L2,DistanceEvaluator<ScalarT,ScalarT>>(this->first, this->second, *this),
                  DenseCombinedMetricNonRigidICP3<ScalarT,CorrespondenceSearchKDTree<PointFeaturesAdaptor<ScalarT,3>,KDTreeDistanceAdaptors::L2,DistanceEvaluator<ScalarT,ScalarT>>>(dst_points, dst_normals, src_points, regularization_neighborhoods, *this)
        {}
    };

    typedef SimpleDenseCombinedMetricNonRigidICP3<float> SimpleDenseCombinedMetricNonRigidICP3f;
    typedef SimpleDenseCombinedMetricNonRigidICP3<double> SimpleDenseCombinedMetricNonRigidICP3d;

    template <typename ScalarT>
    class SimpleDenseCombinedMetricNonRigidProjectiveICP3 : private std::pair<PointFeaturesAdaptor<ScalarT,3>,PointFeaturesAdaptor<ScalarT,3>>,
                                                            private DistanceEvaluator<ScalarT,ScalarT>,
                                                            private CorrespondenceSearchProjective3<ScalarT,DistanceEvaluator<ScalarT,ScalarT>>,
                                                            public DenseCombinedMetricNonRigidICP3<ScalarT,CorrespondenceSearchProjective3<ScalarT,DistanceEvaluator<ScalarT,ScalarT>>>
    {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        SimpleDenseCombinedMetricNonRigidProjectiveICP3(const ConstVectorSetMatrixMap<ScalarT,3> &dst_points,
                                                        const ConstVectorSetMatrixMap<ScalarT,3> &dst_normals,
                                                        const ConstVectorSetMatrixMap<ScalarT,3> &src_points,
                                                        const std::vector<NeighborSet<ScalarT>> &regularization_neighborhoods)
                : std::pair<PointFeaturesAdaptor<ScalarT,3>,PointFeaturesAdaptor<ScalarT,3>>(PointFeaturesAdaptor<ScalarT,3>(dst_points), PointFeaturesAdaptor<ScalarT,3>(src_points)),
                  DistanceEvaluator<ScalarT,ScalarT>(),
                  CorrespondenceSearchProjective3<ScalarT,DistanceEvaluator<ScalarT,ScalarT>>(this->first, this->second, *this),
                  DenseCombinedMetricNonRigidICP3<ScalarT,CorrespondenceSearchProjective3<ScalarT,DistanceEvaluator<ScalarT,ScalarT>>>(dst_points, dst_normals, src_points, regularization_neighborhoods, *this)
        {}
    };

    typedef SimpleDenseCombinedMetricNonRigidProjectiveICP3<float> SimpleDenseCombinedMetricNonRigidProjectiveICP3f;
    typedef SimpleDenseCombinedMetricNonRigidProjectiveICP3<double> SimpleDenseCombinedMetricNonRigidProjectiveICP3d;

    template <typename ScalarT>
    class SimpleSparseCombinedMetricNonRigidICP3 : private std::pair<PointFeaturesAdaptor<ScalarT,3>,PointFeaturesAdaptor<ScalarT,3>>,
                                                   private DistanceEvaluator<ScalarT,ScalarT>,
                                                   private CorrespondenceSearchKDTree<PointFeaturesAdaptor<ScalarT,3>,KDTreeDistanceAdaptors::L2,DistanceEvaluator<ScalarT,ScalarT>>,
                                                   public SparseCombinedMetricNonRigidICP3<ScalarT,CorrespondenceSearchKDTree<PointFeaturesAdaptor<ScalarT,3>,KDTreeDistanceAdaptors::L2,DistanceEvaluator<ScalarT,ScalarT>>>
    {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        SimpleSparseCombinedMetricNonRigidICP3(const ConstVectorSetMatrixMap<ScalarT,3> &dst_points,
                                               const ConstVectorSetMatrixMap<ScalarT,3> &dst_normals,
                                               const ConstVectorSetMatrixMap<ScalarT,3> &src_points,
                                               size_t num_control_nodes,
                                               const std::vector<NeighborSet<ScalarT>> &src_to_control_neighborhoods,
                                               const std::vector<NeighborSet<ScalarT>> &control_regularization_neighborhoods)
                : std::pair<PointFeaturesAdaptor<ScalarT,3>,PointFeaturesAdaptor<ScalarT,3>>(PointFeaturesAdaptor<ScalarT,3>(dst_points), PointFeaturesAdaptor<ScalarT,3>(src_points)),
                  DistanceEvaluator<ScalarT,ScalarT>(),
                  CorrespondenceSearchKDTree<PointFeaturesAdaptor<ScalarT,3>,KDTreeDistanceAdaptors::L2,DistanceEvaluator<ScalarT,ScalarT>>(this->first, this->second, *this),
                  SparseCombinedMetricNonRigidICP3<ScalarT,CorrespondenceSearchKDTree<PointFeaturesAdaptor<ScalarT,3>,KDTreeDistanceAdaptors::L2,DistanceEvaluator<ScalarT,ScalarT>>>(dst_points, dst_normals, src_points, num_control_nodes, src_to_control_neighborhoods, control_regularization_neighborhoods, *this)
        {}
    };

    typedef SimpleSparseCombinedMetricNonRigidICP3<float> SimpleSparseCombinedMetricNonRigidICP3f;
    typedef SimpleSparseCombinedMetricNonRigidICP3<double> SimpleSparseCombinedMetricNonRigidICP3d;

    template <typename ScalarT>
    class SimpleSparseCombinedMetricNonRigidProjectiveICP3 : private std::pair<PointFeaturesAdaptor<ScalarT,3>,PointFeaturesAdaptor<ScalarT,3>>,
                                                             private DistanceEvaluator<ScalarT,ScalarT>,
                                                             private CorrespondenceSearchProjective3<ScalarT,DistanceEvaluator<ScalarT,ScalarT>>,
                                                             public SparseCombinedMetricNonRigidICP3<ScalarT,CorrespondenceSearchProjective3<ScalarT,DistanceEvaluator<ScalarT,ScalarT>>>
    {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        SimpleSparseCombinedMetricNonRigidProjectiveICP3(const ConstVectorSetMatrixMap<ScalarT,3> &dst_points,
                                                         const ConstVectorSetMatrixMap<ScalarT,3> &dst_normals,
                                                         const ConstVectorSetMatrixMap<ScalarT,3> &src_points,
                                                         size_t num_control_nodes,
                                                         const std::vector<NeighborSet<ScalarT>> &src_to_control_neighborhoods,
                                                         const std::vector<NeighborSet<ScalarT>> &control_regularization_neighborhoods)
                : std::pair<PointFeaturesAdaptor<ScalarT,3>,PointFeaturesAdaptor<ScalarT,3>>(PointFeaturesAdaptor<ScalarT,3>(dst_points), PointFeaturesAdaptor<ScalarT,3>(src_points)),
                  DistanceEvaluator<ScalarT,ScalarT>(),
                  CorrespondenceSearchProjective3<ScalarT,DistanceEvaluator<ScalarT,ScalarT>>(this->first, this->second, *this),
                  SparseCombinedMetricNonRigidICP3<ScalarT,CorrespondenceSearchProjective3<ScalarT,DistanceEvaluator<ScalarT,ScalarT>>>(dst_points, dst_normals, src_points, num_control_nodes, src_to_control_neighborhoods, control_regularization_neighborhoods, *this)
        {}
    };

    typedef SimpleSparseCombinedMetricNonRigidProjectiveICP3<float> SimpleSparseCombinedMetricNonRigidProjectiveICP3f;
    typedef SimpleSparseCombinedMetricNonRigidProjectiveICP3<double> SimpleSparseCombinedMetricNonRigidProjectiveICP3d;
}
