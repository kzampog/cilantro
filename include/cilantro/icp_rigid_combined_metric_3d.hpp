#pragma once

#include <memory>
#include <cilantro/icp_base.hpp>
#include <cilantro/rigid_registration_utilities.hpp>

namespace cilantro {
    template <typename ScalarT, class FeatureAdaptorT, template <class> class DistAdaptor = KDTreeDistanceAdaptors::L2>
    class CombinedMetricRigidICP3D : public IterativeClosestPointBase<CombinedMetricRigidICP3D<ScalarT,FeatureAdaptorT,DistAdaptor>,RigidTransformation<ScalarT,3>,VectorSet<ScalarT,1>,ScalarT,typename FeatureAdaptorT::Scalar> {
        friend class IterativeClosestPointBase<CombinedMetricRigidICP3D<ScalarT,FeatureAdaptorT,DistAdaptor>,RigidTransformation<ScalarT,3>,VectorSet<ScalarT,1>,ScalarT,typename FeatureAdaptorT::Scalar>;
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        CombinedMetricRigidICP3D(const ConstVectorSetMatrixMap<ScalarT,3> &dst_p,
                                 const ConstVectorSetMatrixMap<ScalarT,3> &dst_n,
                                 const ConstVectorSetMatrixMap<ScalarT,3> &src_p,
                                 FeatureAdaptorT &dst_feat,
                                 FeatureAdaptorT &src_feat)
                : dst_points_(dst_p), dst_normals_(dst_n), src_points_(src_p),
                  src_points_trans_(src_points_.rows(), src_points_.cols()),
                  dst_feat_adaptor_(dst_feat), src_feat_adaptor_(src_feat),
                  dst_feat_tree_ptr_(NULL), src_feat_tree_ptr_(NULL),
                  max_estimation_iterations_(1), point_to_point_weight_(0.1), point_to_plane_weight_(1.0)
        {
            this->transform_init_.setIdentity();
            correspondences_.reserve(dst_points_.cols());
        }

        inline ScalarT getPointToPointMetricWeight() const { return point_to_point_weight_; }

        inline CombinedMetricRigidICP3D& setPointToPointMetricWeight(ScalarT weight) { point_to_point_weight_ = weight; return *this; }

        inline ScalarT getPointToPlaneMetricWeight() const { return point_to_plane_weight_; }

        inline CombinedMetricRigidICP3D& setPointToPlaneMetricWeight(ScalarT weight) { point_to_plane_weight_ = weight; return *this; }

        inline size_t getMaxNumberOfOptimizationStepIterations() const { return max_estimation_iterations_; }

        inline CombinedMetricRigidICP3D& setMaxNumberOfOptimizationStepIterations(size_t max_iter) { max_estimation_iterations_ = max_iter; return *this; }

    protected:
        ConstVectorSetMatrixMap<ScalarT,3> dst_points_;
        ConstVectorSetMatrixMap<ScalarT,3> dst_normals_;
        ConstVectorSetMatrixMap<ScalarT,3> src_points_;
        VectorSet<ScalarT,3> src_points_trans_;
        CorrespondenceSet<typename FeatureAdaptorT::Scalar> correspondences_;

        FeatureAdaptorT& dst_feat_adaptor_;
        FeatureAdaptorT& src_feat_adaptor_;
        std::shared_ptr<KDTree<typename FeatureAdaptorT::Scalar,FeatureAdaptorT::FeatureDimension,DistAdaptor>> dst_feat_tree_ptr_;
        std::shared_ptr<KDTree<typename FeatureAdaptorT::Scalar,FeatureAdaptorT::FeatureDimension,DistAdaptor>> src_feat_tree_ptr_;

        size_t max_estimation_iterations_;
        ScalarT point_to_point_weight_;
        ScalarT point_to_plane_weight_;

        void initializeComputation() {
            if (!dst_feat_tree_ptr_ && (this->corr_search_dir_ == CorrespondenceSearchDirection::SECOND_TO_FIRST || this->corr_search_dir_ == CorrespondenceSearchDirection::BOTH)) {
                dst_feat_tree_ptr_.reset(new KDTree<typename FeatureAdaptorT::Scalar,FeatureAdaptorT::FeatureDimension,DistAdaptor>(dst_feat_adaptor_.getFeatureData()));
            }
            if (!src_feat_tree_ptr_ && (this->corr_search_dir_ == CorrespondenceSearchDirection::FIRST_TO_SECOND || this->corr_search_dir_ == CorrespondenceSearchDirection::BOTH)) {
                src_feat_tree_ptr_.reset(new KDTree<typename FeatureAdaptorT::Scalar,FeatureAdaptorT::FeatureDimension,DistAdaptor>(src_feat_adaptor_.getFeatureData()));
            }
        }

        bool updateCorrespondences() {
            switch (this->corr_search_dir_) {
                case CorrespondenceSearchDirection::FIRST_TO_SECOND: {
                    const ConstVectorSetMatrixMap<typename FeatureAdaptorT::Scalar,FeatureAdaptorT::FeatureDimension>& dst_feat_trans(dst_feat_adaptor_.getTransformedFeatureData(this->transform_.inverse()));
                    findNNCorrespondencesUnidirectional<typename FeatureAdaptorT::Scalar,FeatureAdaptorT::FeatureDimension,DistAdaptor>(dst_feat_trans, *src_feat_tree_ptr_, false, correspondences_, this->corr_max_distance_);
                    break;
                }
                case CorrespondenceSearchDirection::SECOND_TO_FIRST: {
                    const ConstVectorSetMatrixMap<typename FeatureAdaptorT::Scalar,FeatureAdaptorT::FeatureDimension>& src_feat_trans(src_feat_adaptor_.getTransformedFeatureData(this->transform_));
                    findNNCorrespondencesUnidirectional<typename FeatureAdaptorT::Scalar,FeatureAdaptorT::FeatureDimension,DistAdaptor>(src_feat_trans, *dst_feat_tree_ptr_, true, correspondences_, this->corr_max_distance_);
                    break;
                }
                case CorrespondenceSearchDirection::BOTH: {
                    const ConstVectorSetMatrixMap<typename FeatureAdaptorT::Scalar,FeatureAdaptorT::FeatureDimension>& dst_feat_trans(dst_feat_adaptor_.getTransformedFeatureData(this->transform_.inverse()));
                    const ConstVectorSetMatrixMap<typename FeatureAdaptorT::Scalar,FeatureAdaptorT::FeatureDimension>& src_feat_trans(src_feat_adaptor_.getTransformedFeatureData(this->transform_));
                    findNNCorrespondencesBidirectional<typename FeatureAdaptorT::Scalar,FeatureAdaptorT::FeatureDimension,DistAdaptor>(dst_feat_trans, src_feat_trans, *dst_feat_tree_ptr_, *src_feat_tree_ptr_, correspondences_, this->corr_max_distance_, this->corr_require_reciprocal_);
                    break;
                }
            }

            filterCorrespondencesFraction(correspondences_, this->corr_inlier_fraction_);

            return correspondences_.size() >= 3;
        }

        bool updateEstimate() {
            RigidTransformation<ScalarT,3> tform_iter;
            if (this->iterations_ > 0) {
#pragma omp parallel for
                for (size_t i = 0; i < src_points_.cols(); i++) {
                    src_points_trans_.col(i) = this->transform_*src_points_.col(i);
                }
                estimateRigidTransformCombinedMetric3D<ScalarT,typename FeatureAdaptorT::Scalar>(dst_points_, dst_normals_, src_points_trans_, correspondences_, point_to_point_weight_, point_to_plane_weight_, tform_iter, max_estimation_iterations_, this->convergence_tol_);
            } else {
                estimateRigidTransformCombinedMetric3D<ScalarT,typename FeatureAdaptorT::Scalar>(dst_points_, dst_normals_, src_points_, correspondences_, point_to_point_weight_, point_to_plane_weight_, tform_iter, max_estimation_iterations_, this->convergence_tol_);
            }
            this->transform_ = tform_iter*this->transform_;
            this->transform_.linear() = this->transform_.rotation();
            this->last_delta_norm_ = std::sqrt((tform_iter.linear() - Eigen::Matrix<ScalarT,3,3>::Identity()).squaredNorm() + tform_iter.translation().squaredNorm());
            return true;
        }

        VectorSet<ScalarT,1> computeResiduals() {
            if (!dst_feat_tree_ptr_) {
                dst_feat_tree_ptr_.reset(new KDTree<typename FeatureAdaptorT::Scalar,FeatureAdaptorT::FeatureDimension,DistAdaptor>(dst_feat_adaptor_.getFeatureData()));
            }

            const ConstVectorSetMatrixMap<typename FeatureAdaptorT::Scalar,FeatureAdaptorT::FeatureDimension>& src_feat_trans = src_feat_adaptor_.getTransformedFeatureData(this->transform_);

            NearestNeighborSearchResult<typename FeatureAdaptorT::Scalar> nn;
            Vector<ScalarT,3> src_p_trans;

            VectorSet<ScalarT,1> res(1,src_points_.cols());
#pragma omp parallel for shared (res) private (nn, src_p_trans)
            for (size_t i = 0; i < src_feat_trans.cols(); i++) {
                dst_feat_tree_ptr_->nearestNeighborSearch(src_feat_trans.col(i), nn);
                src_p_trans = this->transform_*src_points_.col(i);
                ScalarT point_to_plane_dist = dst_normals_.col(nn.index).dot(dst_points_.col(nn.index) - src_p_trans);
                res[i] = point_to_point_weight_*(dst_points_.col(nn.index) - src_p_trans).squaredNorm() + point_to_plane_weight_*point_to_plane_dist*point_to_plane_dist;
            }
            return res;
        }
    };
}
