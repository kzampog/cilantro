#pragma once

#include <memory>
#include <cilantro/icp_base.hpp>
#include <cilantro/rigid_registration_utilities.hpp>

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
                  dst_points_(dst), src_points_(src), src_points_trans_(src_points_.rows(), src_points_.cols())
        {
            this->transform_init_.setIdentity();
        }

    protected:
        ConstVectorSetMatrixMap<ScalarT,EigenDim> dst_points_;
        ConstVectorSetMatrixMap<ScalarT,EigenDim> src_points_;
        VectorSet<ScalarT,EigenDim> src_points_trans_;

        void initializeComputation() {
            this->correspondences_.reserve(std::max(dst_points_.cols(), src_points_.cols()));
        }

        bool updateCorrespondences() {
            this->correspondence_search_engine_.findCorrespondences(this->transform_, this->correspondences_);
            return this->correspondences_.size() >= src_points_.rows();
        }

        bool updateEstimate() {
            RigidTransformation<ScalarT,EigenDim> tform_iter;
#pragma omp parallel for
            for (size_t i = 0; i < src_points_.cols(); i++) {
                src_points_trans_.col(i) = this->transform_*src_points_.col(i);
            }
            if (estimateRigidTransformPointToPointClosedForm<ScalarT,EigenDim,typename CorrespondenceSearchEngineT::CorrespondenceScalar>(dst_points_, src_points_trans_, this->correspondences_, tform_iter)) {
                this->transform_ = tform_iter*this->transform_;
                this->transform_.linear() = this->transform_.rotation();
                this->last_delta_norm_ = std::sqrt((tform_iter.linear() - Eigen::Matrix<ScalarT,EigenDim,EigenDim>::Identity(src_points_.rows(),src_points_.rows())).squaredNorm() + tform_iter.translation().squaredNorm());
                return true;
            } else {
                return false;
            }
        }

        VectorSet<ScalarT,1> computeResiduals() {
            VectorSet<ScalarT,1> res(1, src_points_.cols());
            KDTree<ScalarT,EigenDim,KDTreeDistanceAdaptors::L2> dst_tree(dst_points_);
            NearestNeighborSearchResult<ScalarT> nn;
            Vector<ScalarT,3> src_p_trans;
#pragma omp parallel for shared (res) private (nn, src_p_trans)
            for (size_t i = 0; i < src_points_.cols(); i++) {
                src_p_trans = this->transform_*src_points_.col(i);
                dst_tree.nearestNeighborSearch(src_p_trans, nn);
                res[i] = (dst_points_.col(nn.index) - this->transform_*src_points_.col(i)).squaredNorm();
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
