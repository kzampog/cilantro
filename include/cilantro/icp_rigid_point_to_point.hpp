#pragma once

#include <memory>
#include <cilantro/icp_base.hpp>
#include <cilantro/rigid_registration_utilities.hpp>

namespace cilantro {
    template <typename ScalarT, ptrdiff_t EigenDim, class FeatureAdaptorT, template <class> class DistAdaptor = KDTreeDistanceAdaptors::L2>
    class PointToPointRigidICP : public IterativeClosestPointBase<PointToPointRigidICP<ScalarT,EigenDim,FeatureAdaptorT,DistAdaptor>,RigidTransformation<ScalarT,EigenDim>,VectorSet<ScalarT,1>,ScalarT,typename FeatureAdaptorT::Scalar> {
        friend class IterativeClosestPointBase<PointToPointRigidICP<ScalarT,EigenDim,FeatureAdaptorT,DistAdaptor>,RigidTransformation<ScalarT,EigenDim>,VectorSet<ScalarT,1>,ScalarT,typename FeatureAdaptorT::Scalar>;
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        PointToPointRigidICP(const ConstVectorSetMatrixMap<ScalarT,EigenDim> &dst,
                             const ConstVectorSetMatrixMap<ScalarT,EigenDim> &src,
                             FeatureAdaptorT &dst_feat,
                             FeatureAdaptorT &src_feat)
                : dst_points_(dst), src_points_(src), src_points_trans_(src_points_.rows(), src_points_.cols()),
                  dst_feat_adaptor_(dst_feat), src_feat_adaptor_(src_feat),
                  dst_tree_ptr_(NULL), src_tree_ptr_(NULL)
        {
            this->transform_init_.setIdentity();
            correspondences_.reserve(dst_points_.cols());
        }

    protected:
        ConstVectorSetMatrixMap<ScalarT,EigenDim> dst_points_;
        ConstVectorSetMatrixMap<ScalarT,EigenDim> src_points_;
        VectorSet<ScalarT,EigenDim> src_points_trans_;
        CorrespondenceSet<typename FeatureAdaptorT::Scalar> correspondences_;

        FeatureAdaptorT& dst_feat_adaptor_;
        FeatureAdaptorT& src_feat_adaptor_;
        std::shared_ptr<KDTree<typename FeatureAdaptorT::Scalar,FeatureAdaptorT::FeatureDimension,DistAdaptor>> dst_tree_ptr_;
        std::shared_ptr<KDTree<typename FeatureAdaptorT::Scalar,FeatureAdaptorT::FeatureDimension,DistAdaptor>> src_tree_ptr_;

        void initializeComputation() {
            if (!dst_tree_ptr_ && (this->corr_search_dir_ == CorrespondenceSearchDirection::SECOND_TO_FIRST || this->corr_search_dir_ == CorrespondenceSearchDirection::BOTH)) {
                dst_tree_ptr_.reset(new KDTree<typename FeatureAdaptorT::Scalar,FeatureAdaptorT::FeatureDimension,DistAdaptor>(dst_feat_adaptor_.getFeatureData()));
            }
            if (!src_tree_ptr_ && (this->corr_search_dir_ == CorrespondenceSearchDirection::FIRST_TO_SECOND || this->corr_search_dir_ == CorrespondenceSearchDirection::BOTH)) {
                src_tree_ptr_.reset(new KDTree<typename FeatureAdaptorT::Scalar,FeatureAdaptorT::FeatureDimension,DistAdaptor>(src_feat_adaptor_.getFeatureData()));
            }
        }

        bool updateCorrespondences() {
            switch (this->corr_search_dir_) {
                case CorrespondenceSearchDirection::FIRST_TO_SECOND: {
                    const ConstVectorSetMatrixMap<typename FeatureAdaptorT::Scalar,FeatureAdaptorT::FeatureDimension>& dst_feat_trans(dst_feat_adaptor_.getTransformedFeatureData(this->transform_.inverse()));
                    findNNCorrespondencesUnidirectional(dst_feat_trans, *src_tree_ptr_, false, correspondences_, this->corr_max_distance_);
                    break;
                }
                case CorrespondenceSearchDirection::SECOND_TO_FIRST: {
                    const ConstVectorSetMatrixMap<typename FeatureAdaptorT::Scalar,FeatureAdaptorT::FeatureDimension>& src_feat_trans(src_feat_adaptor_.getTransformedFeatureData(this->transform_));
                    findNNCorrespondencesUnidirectional(src_feat_trans, *dst_tree_ptr_, true, correspondences_, this->corr_max_distance_);
                    break;
                }
                case CorrespondenceSearchDirection::BOTH: {
                    const ConstVectorSetMatrixMap<typename FeatureAdaptorT::Scalar,FeatureAdaptorT::FeatureDimension>& dst_feat_trans(dst_feat_adaptor_.getTransformedFeatureData(this->transform_.inverse()));
                    const ConstVectorSetMatrixMap<typename FeatureAdaptorT::Scalar,FeatureAdaptorT::FeatureDimension>& src_feat_trans(src_feat_adaptor_.getTransformedFeatureData(this->transform_));
                    findNNCorrespondencesBidirectional(dst_feat_trans, src_feat_trans, *dst_tree_ptr_, *src_tree_ptr_, correspondences_, this->corr_max_distance_, this->corr_require_reciprocal_);
                    break;
                }
            }

            filterCorrespondencesFraction(correspondences_, this->corr_inlier_fraction_);

            return correspondences_.size() >= 3;
        }

        bool updateEstimate() {
            if (this->iterations_ > 0) {
                RigidTransformation<ScalarT,EigenDim> tform_iter;
#pragma omp parallel for
                for (size_t i = 0; i < src_points_.cols(); i++) {
                    src_points_trans_.col(i) = this->transform_*src_points_.col(i);
                }
                if (estimateRigidTransformPointToPointClosedForm<ScalarT,EigenDim,typename FeatureAdaptorT::Scalar>(dst_points_, src_points_trans_, correspondences_, tform_iter)) {
                    this->transform_ = tform_iter*this->transform_;
                    this->transform_.linear() = this->transform_.rotation();
                    this->last_delta_norm_ = std::sqrt((tform_iter.linear() - Eigen::Matrix<ScalarT,EigenDim,EigenDim>::Identity(src_points_.rows(),src_points_.rows())).squaredNorm() + tform_iter.translation().squaredNorm());
                    return true;
                } else {
                    return false;
                }
            } else {
                RigidTransformation<ScalarT,EigenDim> tform_iter;
                if (estimateRigidTransformPointToPointClosedForm<ScalarT,EigenDim,typename FeatureAdaptorT::Scalar>(dst_points_, src_points_, correspondences_, tform_iter)) {
                    this->transform_ = tform_iter*this->transform_;
                    this->transform_.linear() = this->transform_.rotation();
                    this->last_delta_norm_ = std::sqrt((tform_iter.linear() - Eigen::Matrix<ScalarT,EigenDim,EigenDim>::Identity(src_points_.rows(),src_points_.rows())).squaredNorm() + tform_iter.translation().squaredNorm());
                    return true;
                } else {
                    return false;
                }
            }
        }

        VectorSet<ScalarT,1> computeResiduals() {
            VectorSet<ScalarT,1> res(1,src_points_.cols());
            if (!dst_tree_ptr_) {
                dst_tree_ptr_.reset(new KDTree<typename FeatureAdaptorT::Scalar,FeatureAdaptorT::FeatureDimension,DistAdaptor>(dst_feat_adaptor_.getFeatureData()));
            }
            const ConstVectorSetMatrixMap<typename FeatureAdaptorT::Scalar,FeatureAdaptorT::FeatureDimension>& src_feat_trans = src_feat_adaptor_.getTransformedFeatureData(this->transform_);

            size_t neighbor;
            typename FeatureAdaptorT::Scalar distance;

#pragma omp parallel for private (neighbor, distance)
            for (size_t i = 0; i < src_feat_trans.cols(); i++) {
                dst_tree_ptr_->nearestNeighborSearch(src_feat_trans.col(i), neighbor, distance);
                res[i] = (dst_points_.col(neighbor) - this->transform_*src_points_.col(i)).squaredNorm();
            }
            return res;
        }
    };
}
