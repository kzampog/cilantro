#pragma once

#include <cilantro/icp_base.hpp>
#include <cilantro/rigid_registration_utilities.hpp>
#include <cilantro/correspondence_search_combined_metric_adaptor.hpp>
#include <cilantro/kd_tree.hpp>

namespace cilantro {
    template <typename ScalarT, ptrdiff_t EigenDim, class CorrespondenceSearchEngineT>
    class PointToPointMetricRigidICP : public IterativeClosestPointBase<PointToPointMetricRigidICP<ScalarT,EigenDim,CorrespondenceSearchEngineT>,RigidTransformation<ScalarT,EigenDim>,CorrespondenceSearchEngineT,VectorSet<ScalarT,1>> {
        friend class IterativeClosestPointBase<PointToPointMetricRigidICP<ScalarT,EigenDim,CorrespondenceSearchEngineT>,RigidTransformation<ScalarT,EigenDim>,CorrespondenceSearchEngineT,VectorSet<ScalarT,1>>;
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        PointToPointMetricRigidICP(const ConstVectorSetMatrixMap<ScalarT,EigenDim> &dst,
                                   const ConstVectorSetMatrixMap<ScalarT,EigenDim> &src,
                                   CorrespondenceSearchEngineT &corr_engine)
                : IterativeClosestPointBase<PointToPointMetricRigidICP<ScalarT,EigenDim,CorrespondenceSearchEngineT>,RigidTransformation<ScalarT,EigenDim>,CorrespondenceSearchEngineT,VectorSet<ScalarT,1>>(corr_engine),
                  dst_points_(dst), src_points_(src),
                  src_points_trans_(src_points_.rows(), src_points_.cols())
        {
            this->transform_init_.setIdentity();
        }

    private:
        ConstVectorSetMatrixMap<ScalarT,EigenDim> dst_points_;
        ConstVectorSetMatrixMap<ScalarT,EigenDim> src_points_;

        VectorSet<ScalarT,EigenDim> src_points_trans_;

        // ICP interface
        inline void initializeComputation() {}

        // ICP interface
        inline void updateCorrespondences() {
            this->correspondence_search_engine_.findCorrespondences(this->transform_);
        }

        void updateEstimate() {
#pragma omp parallel for
            for (size_t i = 0; i < src_points_.cols(); i++) {
                src_points_trans_.col(i).noalias() = this->transform_*src_points_.col(i);
            }

            CorrespondenceSearchCombinedMetricAdaptor<CorrespondenceSearchEngineT> corr_getter_proxy(this->correspondence_search_engine_);
            RigidTransformation<ScalarT,EigenDim> tform_iter;
            estimateRigidTransformPointToPointClosedForm<ScalarT,EigenDim,typename CorrespondenceSearchCombinedMetricAdaptor<CorrespondenceSearchEngineT>::PointToPointCorrespondenceScalar>(dst_points_, src_points_trans_, corr_getter_proxy.getPointToPointCorrespondences(), tform_iter);

            this->transform_ = tform_iter*this->transform_;
            this->transform_.linear() = this->transform_.rotation();
            this->last_delta_norm_ = std::sqrt((tform_iter.linear() - Eigen::Matrix<ScalarT,EigenDim,EigenDim>::Identity(src_points_.rows(),src_points_.rows())).squaredNorm() + tform_iter.translation().squaredNorm());
        }

        // ICP interface
        VectorSet<ScalarT,1> computeResiduals() {
            if (dst_points_.cols() == 0) {
                return VectorSet<ScalarT,1>::Constant(1, src_points_.cols(), std::numeric_limits<ScalarT>::quiet_NaN());
            }
            VectorSet<ScalarT,1> res(1, src_points_.cols());
            KDTree<ScalarT,EigenDim,KDTreeDistanceAdaptors::L2> dst_tree(dst_points_);
            Neighbor<ScalarT> nn;
            Vector<ScalarT,3> src_p_trans;
#pragma omp parallel for shared (res) private (nn, src_p_trans)
            for (size_t i = 0; i < src_points_.cols(); i++) {
                src_p_trans.noalias() = this->transform_*src_points_.col(i);
                dst_tree.nearestNeighborSearch(src_p_trans, nn);
                res[i] = (dst_points_.col(nn.index) - src_p_trans).squaredNorm();
            }
            return res;
        }
    };

    template <class CorrespondenceSearchEngineT>
    using PointToPointMetricRigidICP2f = PointToPointMetricRigidICP<float,2,CorrespondenceSearchEngineT>;

    template <class CorrespondenceSearchEngineT>
    using PointToPointMetricRigidICP2d = PointToPointMetricRigidICP<double,2,CorrespondenceSearchEngineT>;

    template <class CorrespondenceSearchEngineT>
    using PointToPointMetricRigidICP3f = PointToPointMetricRigidICP<float,3,CorrespondenceSearchEngineT>;

    template <class CorrespondenceSearchEngineT>
    using PointToPointMetricRigidICP3d = PointToPointMetricRigidICP<double,3,CorrespondenceSearchEngineT>;
}
