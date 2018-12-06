#pragma once

#include <cilantro/icp_base.hpp>
#include <cilantro/transform_estimation.hpp>
#include <cilantro/correspondence_search_combined_metric_adaptor.hpp>
#include <cilantro/kd_tree.hpp>

namespace cilantro {
    template <class TransformT, class CorrespondenceSearchEngineT>
    class PointToPointMetricSingleTransformICP : public IterativeClosestPointBase<PointToPointMetricSingleTransformICP<TransformT,CorrespondenceSearchEngineT>,TransformT,CorrespondenceSearchEngineT,VectorSet<typename TransformT::Scalar,1>> {

        typedef IterativeClosestPointBase<PointToPointMetricSingleTransformICP<TransformT,CorrespondenceSearchEngineT>,TransformT,CorrespondenceSearchEngineT,VectorSet<typename TransformT::Scalar,1>> Base;

        friend Base;

    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        PointToPointMetricSingleTransformICP(const ConstVectorSetMatrixMap<typename TransformT::Scalar,TransformT::Dim> &dst,
                                             const ConstVectorSetMatrixMap<typename TransformT::Scalar,TransformT::Dim> &src,
                                             CorrespondenceSearchEngineT &corr_engine)
                : Base(corr_engine),
                  dst_points_(dst), src_points_(src),
                  src_points_trans_(src_points_.rows(), src_points_.cols())
        {
            this->transform_init_.setIdentity();
        }

    private:
        ConstVectorSetMatrixMap<typename TransformT::Scalar,TransformT::Dim> dst_points_;
        ConstVectorSetMatrixMap<typename TransformT::Scalar,TransformT::Dim> src_points_;

        VectorSet<typename TransformT::Scalar,TransformT::Dim> src_points_trans_;

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
            estimateTransformPointToPointMetric(dst_points_, src_points_trans_, corr_getter_proxy.getPointToPointCorrespondences(), tform_iter);

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
    using PointToPointMetricRigidTransformICP2f = PointToPointMetricSingleTransformICP<RigidTransform<float,2>,CorrespondenceSearchEngineT>;

    template <class CorrespondenceSearchEngineT>
    using PointToPointMetricRigidTransformICP2d = PointToPointMetricSingleTransformICP<RigidTransform<double,2>,CorrespondenceSearchEngineT>;

    template <class CorrespondenceSearchEngineT>
    using PointToPointMetricRigidTransformICP3f = PointToPointMetricSingleTransformICP<RigidTransform<float,3>,CorrespondenceSearchEngineT>;

    template <class CorrespondenceSearchEngineT>
    using PointToPointMetricRigidTransformICP3d = PointToPointMetricSingleTransformICP<RigidTransform<double,3>,CorrespondenceSearchEngineT>;

    template <class CorrespondenceSearchEngineT>
    using PointToPointMetricAffineTransformICP2f = PointToPointMetricSingleTransformICP<AffineTransform<float,2>,CorrespondenceSearchEngineT>;

    template <class CorrespondenceSearchEngineT>
    using PointToPointMetricAffineTransformICP2d = PointToPointMetricSingleTransformICP<AffineTransform<double,2>,CorrespondenceSearchEngineT>;

    template <class CorrespondenceSearchEngineT>
    using PointToPointMetricAffineTransformICP3f = PointToPointMetricSingleTransformICP<AffineTransform<float,3>,CorrespondenceSearchEngineT>;

    template <class CorrespondenceSearchEngineT>
    using PointToPointMetricAffineTransformICP3d = PointToPointMetricSingleTransformICP<AffineTransform<double,3>,CorrespondenceSearchEngineT>;
}
