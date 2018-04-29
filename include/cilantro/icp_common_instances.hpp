#pragma once

#include <cilantro/icp_rigid_point_to_point.hpp>
#include <cilantro/icp_rigid_combined_metric_3d.hpp>
#include <cilantro/icp_common_feature_adaptors.hpp>

namespace cilantro {
    template <typename ScalarT, ptrdiff_t EigenDim, template <class> class DistAdaptor = KDTreeDistanceAdaptors::L2>
    class PointToPointMetricPointFeaturesRigidICP : private std::pair<PointFeaturesAdaptor<ScalarT,EigenDim>,PointFeaturesAdaptor<ScalarT,EigenDim>>, public PointToPointMetricRigidICP<ScalarT,EigenDim,PointFeaturesAdaptor<ScalarT,EigenDim>,DistAdaptor> {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        PointToPointMetricPointFeaturesRigidICP(const ConstVectorSetMatrixMap<ScalarT,EigenDim> &dst,
                                                const ConstVectorSetMatrixMap<ScalarT,EigenDim> &src)
                : std::pair<PointFeaturesAdaptor<ScalarT,EigenDim>,PointFeaturesAdaptor<ScalarT,EigenDim>>(PointFeaturesAdaptor<ScalarT,EigenDim>(dst), PointFeaturesAdaptor<ScalarT,EigenDim>(src)),
                  PointToPointMetricRigidICP<ScalarT,EigenDim,PointFeaturesAdaptor<ScalarT,EigenDim>,DistAdaptor>(dst, src, this->first, this->second)
        {}
    };
}
