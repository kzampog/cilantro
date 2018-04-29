#pragma once

#include <cilantro/icp_rigid_point_to_point.hpp>
#include <cilantro/icp_rigid_combined_metric_3d.hpp>
#include <cilantro/icp_common_feature_adaptors.hpp>

namespace cilantro {
    template <typename ScalarT, ptrdiff_t EigenDim, template <class> class DistAdaptor = KDTreeDistanceAdaptors::L2>
    class PointToPointMetricPointFeaturesRigidICP : private std::pair<PointFeaturesAdaptor<ScalarT,EigenDim>,PointFeaturesAdaptor<ScalarT,EigenDim>>, public PointToPointMetricRigidICP<ScalarT,EigenDim,PointFeaturesAdaptor<ScalarT,EigenDim>,DistAdaptor> {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        PointToPointMetricPointFeaturesRigidICP(const ConstVectorSetMatrixMap<ScalarT,EigenDim> &dst_points,
                                                const ConstVectorSetMatrixMap<ScalarT,EigenDim> &src_points)
                : std::pair<PointFeaturesAdaptor<ScalarT,EigenDim>,PointFeaturesAdaptor<ScalarT,EigenDim>>(PointFeaturesAdaptor<ScalarT,EigenDim>(dst_points), PointFeaturesAdaptor<ScalarT,EigenDim>(src_points)),
                  PointToPointMetricRigidICP<ScalarT,EigenDim,PointFeaturesAdaptor<ScalarT,EigenDim>,DistAdaptor>(dst_points, src_points, this->first, this->second)
        {}
    };

    typedef PointToPointMetricPointFeaturesRigidICP<float,2> PointToPointMetricPointFeaturesRigidICP2f;
    typedef PointToPointMetricPointFeaturesRigidICP<double,2> PointToPointMetricPointFeaturesRigidICP2d;
    typedef PointToPointMetricPointFeaturesRigidICP<float,3> PointToPointMetricPointFeaturesRigidICP3f;
    typedef PointToPointMetricPointFeaturesRigidICP<double,3> PointToPointMetricPointFeaturesRigidICP3d;

    template <typename ScalarT, ptrdiff_t EigenDim, template <class> class DistAdaptor = KDTreeDistanceAdaptors::L2>
    class PointToPointMetricPointNormalFeaturesRigidICP : private std::pair<PointNormalFeaturesAdaptor<ScalarT,EigenDim>,PointNormalFeaturesAdaptor<ScalarT,EigenDim>>, public PointToPointMetricRigidICP<ScalarT,EigenDim,PointNormalFeaturesAdaptor<ScalarT,EigenDim>,DistAdaptor> {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        PointToPointMetricPointNormalFeaturesRigidICP(const ConstVectorSetMatrixMap<ScalarT,EigenDim> &dst_points,
                                                      const ConstVectorSetMatrixMap<ScalarT,EigenDim> &dst_normals,
                                                      const ConstVectorSetMatrixMap<ScalarT,EigenDim> &src_points,
                                                      const ConstVectorSetMatrixMap<ScalarT,EigenDim> &src_normals,
                                                      ScalarT normal_weight)
                : std::pair<PointNormalFeaturesAdaptor<ScalarT,EigenDim>,PointNormalFeaturesAdaptor<ScalarT,EigenDim>>(PointNormalFeaturesAdaptor<ScalarT,EigenDim>(dst_points, dst_normals, normal_weight), PointNormalFeaturesAdaptor<ScalarT,EigenDim>(src_points, src_normals, normal_weight)),
                  PointToPointMetricRigidICP<ScalarT,EigenDim,PointNormalFeaturesAdaptor<ScalarT,EigenDim>,DistAdaptor>(dst_points, src_points, this->first, this->second)
        {}
    };

    typedef PointToPointMetricPointNormalFeaturesRigidICP<float,2> PointToPointMetricPointNormalFeaturesRigidICP2f;
    typedef PointToPointMetricPointNormalFeaturesRigidICP<double,2> PointToPointMetricPointNormalFeaturesRigidICP2d;
    typedef PointToPointMetricPointNormalFeaturesRigidICP<float,3> PointToPointMetricPointNormalFeaturesRigidICP3f;
    typedef PointToPointMetricPointNormalFeaturesRigidICP<double,3> PointToPointMetricPointNormalFeaturesRigidICP3d;

    template <typename ScalarT, ptrdiff_t EigenDim, template <class> class DistAdaptor = KDTreeDistanceAdaptors::L2>
    class PointToPointMetricPointColorFeaturesRigidICP : private std::pair<PointColorFeaturesAdaptor<ScalarT,EigenDim>,PointColorFeaturesAdaptor<ScalarT,EigenDim>>, public PointToPointMetricRigidICP<ScalarT,EigenDim,PointColorFeaturesAdaptor<ScalarT,EigenDim>,DistAdaptor> {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        PointToPointMetricPointColorFeaturesRigidICP(const ConstVectorSetMatrixMap<ScalarT,EigenDim> &dst_points,
                                                     const ConstVectorSetMatrixMap<float,3> &dst_colors,
                                                     const ConstVectorSetMatrixMap<ScalarT,EigenDim> &src_points,
                                                     const ConstVectorSetMatrixMap<float,3> &src_colors,
                                                     ScalarT color_weight)
                : std::pair<PointColorFeaturesAdaptor<ScalarT,EigenDim>,PointColorFeaturesAdaptor<ScalarT,EigenDim>>(PointColorFeaturesAdaptor<ScalarT,EigenDim>(dst_points, dst_colors, color_weight), PointColorFeaturesAdaptor<ScalarT,EigenDim>(src_points, src_colors, color_weight)),
                  PointToPointMetricRigidICP<ScalarT,EigenDim,PointColorFeaturesAdaptor<ScalarT,EigenDim>,DistAdaptor>(dst_points, src_points, this->first, this->second)
        {}
    };

    typedef PointToPointMetricPointColorFeaturesRigidICP<float,2> PointToPointMetricPointColorFeaturesRigidICP2f;
    typedef PointToPointMetricPointColorFeaturesRigidICP<double,2> PointToPointMetricPointColorFeaturesRigidICP2d;
    typedef PointToPointMetricPointColorFeaturesRigidICP<float,3> PointToPointMetricPointColorFeaturesRigidICP3f;
    typedef PointToPointMetricPointColorFeaturesRigidICP<double,3> PointToPointMetricPointColorFeaturesRigidICP3d;

    template <typename ScalarT, ptrdiff_t EigenDim, template <class> class DistAdaptor = KDTreeDistanceAdaptors::L2>
    class PointToPointMetricPointNormalColorFeaturesRigidICP : private std::pair<PointNormalColorFeaturesAdaptor<ScalarT,EigenDim>,PointNormalColorFeaturesAdaptor<ScalarT,EigenDim>>, public PointToPointMetricRigidICP<ScalarT,EigenDim,PointNormalColorFeaturesAdaptor<ScalarT,EigenDim>,DistAdaptor> {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        PointToPointMetricPointNormalColorFeaturesRigidICP(const ConstVectorSetMatrixMap<ScalarT,EigenDim> &dst_points,
                                                           const ConstVectorSetMatrixMap<ScalarT,EigenDim> &dst_normals,
                                                           const ConstVectorSetMatrixMap<float,3> &dst_colors,
                                                           const ConstVectorSetMatrixMap<ScalarT,EigenDim> &src_points,
                                                           const ConstVectorSetMatrixMap<ScalarT,EigenDim> &src_normals,
                                                           const ConstVectorSetMatrixMap<float,3> &src_colors,
                                                           ScalarT normal_weight,
                                                           ScalarT color_weight)
                : std::pair<PointNormalColorFeaturesAdaptor<ScalarT,EigenDim>,PointNormalColorFeaturesAdaptor<ScalarT,EigenDim>>(PointNormalColorFeaturesAdaptor<ScalarT,EigenDim>(dst_points, dst_normals, dst_colors, normal_weight, color_weight), PointNormalColorFeaturesAdaptor<ScalarT,EigenDim>(src_points, src_normals, src_colors, normal_weight, color_weight)),
                  PointToPointMetricRigidICP<ScalarT,EigenDim,PointNormalColorFeaturesAdaptor<ScalarT,EigenDim>,DistAdaptor>(dst_points, src_points, this->first, this->second)
        {}
    };

    typedef PointToPointMetricPointNormalColorFeaturesRigidICP<float,2> PointToPointMetricPointNormalColorFeaturesRigidICP2f;
    typedef PointToPointMetricPointNormalColorFeaturesRigidICP<double,2> PointToPointMetricPointNormalColorFeaturesRigidICP2d;
    typedef PointToPointMetricPointNormalColorFeaturesRigidICP<float,3> PointToPointMetricPointNormalColorFeaturesRigidICP3f;
    typedef PointToPointMetricPointNormalColorFeaturesRigidICP<double,3> PointToPointMetricPointNormalColorFeaturesRigidICP3d;

    template <typename ScalarT, template <class> class DistAdaptor = KDTreeDistanceAdaptors::L2>
    class CombinedMetricPointFeaturesRigidICP3 : private std::pair<PointFeaturesAdaptor<ScalarT,3>,PointFeaturesAdaptor<ScalarT,3>>, public CombinedMetricRigidICP3<ScalarT,PointFeaturesAdaptor<ScalarT,3>,DistAdaptor> {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        CombinedMetricPointFeaturesRigidICP3(const ConstVectorSetMatrixMap<ScalarT,3> &dst_points,
                                             const ConstVectorSetMatrixMap<ScalarT,3> &dst_normals,
                                             const ConstVectorSetMatrixMap<ScalarT,3> &src_points)
                : std::pair<PointFeaturesAdaptor<ScalarT,3>,PointFeaturesAdaptor<ScalarT,3>>(PointFeaturesAdaptor<ScalarT,3>(dst_points), PointFeaturesAdaptor<ScalarT,3>(src_points)),
                  CombinedMetricRigidICP3<ScalarT,PointFeaturesAdaptor<ScalarT,3>,DistAdaptor>(dst_points, dst_normals, src_points, this->first, this->second)
        {}
    };

    typedef CombinedMetricPointFeaturesRigidICP3<float> CombinedMetricPointFeaturesRigidICP3f;
    typedef CombinedMetricPointFeaturesRigidICP3<double> CombinedMetricPointFeaturesRigidICP3d;

    template <typename ScalarT, template <class> class DistAdaptor = KDTreeDistanceAdaptors::L2>
    class CombinedMetricPointNormalFeaturesRigidICP3 : private std::pair<PointNormalFeaturesAdaptor<ScalarT,3>,PointNormalFeaturesAdaptor<ScalarT,3>>, public CombinedMetricRigidICP3<ScalarT,PointNormalFeaturesAdaptor<ScalarT,3>,DistAdaptor> {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        CombinedMetricPointNormalFeaturesRigidICP3(const ConstVectorSetMatrixMap<ScalarT,3> &dst_points,
                                                   const ConstVectorSetMatrixMap<ScalarT,3> &dst_normals,
                                                   const ConstVectorSetMatrixMap<ScalarT,3> &src_points,
                                                   const ConstVectorSetMatrixMap<ScalarT,3> &src_normals,
                                                   ScalarT normal_weight)
                : std::pair<PointNormalFeaturesAdaptor<ScalarT,3>,PointNormalFeaturesAdaptor<ScalarT,3>>(PointNormalFeaturesAdaptor<ScalarT,3>(dst_points, dst_normals, normal_weight), PointNormalFeaturesAdaptor<ScalarT,3>(src_points, src_normals, normal_weight)),
                  CombinedMetricRigidICP3<ScalarT,PointNormalFeaturesAdaptor<ScalarT,3>,DistAdaptor>(dst_points, dst_normals, src_points, this->first, this->second)
        {}
    };

    typedef CombinedMetricPointNormalFeaturesRigidICP3<float> CombinedMetricPointNormalFeaturesRigidICP3f;
    typedef CombinedMetricPointNormalFeaturesRigidICP3<double> CombinedMetricPointNormalFeaturesRigidICP3d;

    template <typename ScalarT, template <class> class DistAdaptor = KDTreeDistanceAdaptors::L2>
    class CombinedMetricPointColorFeaturesRigidICP3 : private std::pair<PointColorFeaturesAdaptor<ScalarT,3>,PointColorFeaturesAdaptor<ScalarT,3>>, public CombinedMetricRigidICP3<ScalarT,PointColorFeaturesAdaptor<ScalarT,3>,DistAdaptor> {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        CombinedMetricPointColorFeaturesRigidICP3(const ConstVectorSetMatrixMap<ScalarT,3> &dst_points,
                                                  const ConstVectorSetMatrixMap<ScalarT,3> &dst_normals,
                                                  const ConstVectorSetMatrixMap<float,3> &dst_colors,
                                                  const ConstVectorSetMatrixMap<ScalarT,3> &src_points,
                                                  const ConstVectorSetMatrixMap<float,3> &src_colors,
                                                  ScalarT color_weight)
                : std::pair<PointColorFeaturesAdaptor<ScalarT,3>,PointColorFeaturesAdaptor<ScalarT,3>>(PointColorFeaturesAdaptor<ScalarT,3>(dst_points, dst_colors, color_weight), PointColorFeaturesAdaptor<ScalarT,3>(src_points, src_colors, color_weight)),
                  CombinedMetricRigidICP3<ScalarT,PointColorFeaturesAdaptor<ScalarT,3>,DistAdaptor>(dst_points, dst_normals, src_points, this->first, this->second)
        {}
    };

    typedef CombinedMetricPointColorFeaturesRigidICP3<float> CombinedMetricPointColorFeaturesRigidICP3f;
    typedef CombinedMetricPointColorFeaturesRigidICP3<double> CombinedMetricPointColorFeaturesRigidICP3d;

    template <typename ScalarT, template <class> class DistAdaptor = KDTreeDistanceAdaptors::L2>
    class CombinedMetricPointNormalColorFeaturesRigidICP3 : private std::pair<PointNormalColorFeaturesAdaptor<ScalarT,3>,PointNormalColorFeaturesAdaptor<ScalarT,3>>, public CombinedMetricRigidICP3<ScalarT,PointNormalColorFeaturesAdaptor<ScalarT,3>,DistAdaptor> {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        CombinedMetricPointNormalColorFeaturesRigidICP3(const ConstVectorSetMatrixMap<ScalarT,3> &dst_points,
                                                        const ConstVectorSetMatrixMap<ScalarT,3> &dst_normals,
                                                        const ConstVectorSetMatrixMap<float,3> &dst_colors,
                                                        const ConstVectorSetMatrixMap<ScalarT,3> &src_points,
                                                        const ConstVectorSetMatrixMap<ScalarT,3> &src_normals,
                                                        const ConstVectorSetMatrixMap<float,3> &src_colors,
                                                        ScalarT normal_weight,
                                                        ScalarT color_weight)
                : std::pair<PointNormalColorFeaturesAdaptor<ScalarT,3>,PointNormalColorFeaturesAdaptor<ScalarT,3>>(PointNormalColorFeaturesAdaptor<ScalarT,3>(dst_points, dst_normals, dst_colors, normal_weight, color_weight), PointNormalColorFeaturesAdaptor<ScalarT,3>(src_points, src_normals, src_colors, normal_weight, color_weight)),
                  CombinedMetricRigidICP3<ScalarT,PointNormalColorFeaturesAdaptor<ScalarT,3>,DistAdaptor>(dst_points, dst_normals, src_points, this->first, this->second)
        {}
    };

    typedef CombinedMetricPointNormalColorFeaturesRigidICP3<float> CombinedMetricPointNormalColorFeaturesRigidICP3f;
    typedef CombinedMetricPointNormalColorFeaturesRigidICP3<double> CombinedMetricPointNormalColorFeaturesRigidICP3d;
}
