#pragma once

#include <cilantro/core/correspondence.hpp>
#include <cilantro/core/common_pair_evaluators.hpp>
#include <cilantro/core/image_point_cloud_conversions.hpp>
#include <cilantro/correspondence_search/common_transformable_feature_adaptors.hpp>

namespace cilantro {
    template <class ScalarT, class EvaluationFeatureAdaptorT = PointFeaturesAdaptor<ScalarT,3>, class EvaluatorT = DistanceEvaluator<ScalarT,typename EvaluationFeatureAdaptorT::Scalar>, typename IndexT = size_t>
    class CorrespondenceSearchProjective {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        typedef EvaluatorT Evaluator;

        typedef typename EvaluatorT::OutputScalar CorrespondenceScalar;

        typedef IndexT CorrespondenceIndex;

        typedef CorrespondenceSet<CorrespondenceScalar,CorrespondenceIndex> SearchResult;

        template <class EvalFeatAdaptorT = EvaluationFeatureAdaptorT, class = typename std::enable_if<std::is_same<EvalFeatAdaptorT,PointFeaturesAdaptor<ScalarT,3>>::value>::type>
        CorrespondenceSearchProjective(PointFeaturesAdaptor<ScalarT,3> &dst_points,
                                       PointFeaturesAdaptor<ScalarT,3> &src_points,
                                       EvaluatorT &evaluator)
                : dst_search_features_adaptor_(dst_points), src_search_features_adaptor_(src_points),
                  src_evaluation_features_adaptor_(src_points), evaluator_(evaluator),
                  projection_image_width_(640), projection_image_height_(480),
                  projection_extrinsics_(RigidTransform<ScalarT,3>::Identity()),
                  projection_extrinsics_inv_(RigidTransform<ScalarT,3>::Identity()),
                  max_distance_((CorrespondenceScalar)(0.01*0.01)), inlier_fraction_(1.0)
        {
            // "Kinect"-like defaults
            projection_intrinsics_ << 528, 0, 320, 0, 528, 240, 0, 0, 1;
        }

        CorrespondenceSearchProjective(PointFeaturesAdaptor<ScalarT,3> &dst_points,
                                       PointFeaturesAdaptor<ScalarT,3> &src_points,
                                       EvaluationFeatureAdaptorT &src_eval_features,
                                       EvaluatorT &evaluator)
                : dst_search_features_adaptor_(dst_points), src_search_features_adaptor_(src_points),
                  src_evaluation_features_adaptor_(src_eval_features), evaluator_(evaluator),
                  projection_image_width_(640), projection_image_height_(480),
                  projection_extrinsics_(RigidTransform<ScalarT,3>::Identity()),
                  projection_extrinsics_inv_(RigidTransform<ScalarT,3>::Identity()),
                  max_distance_((CorrespondenceScalar)(0.01*0.01)), inlier_fraction_(1.0)
        {
            // "Kinect"-like defaults
            projection_intrinsics_ << 528, 0, 320, 0, 528, 240, 0, 0, 1;
        }

        inline CorrespondenceSearchProjective& findCorrespondences() {
            find_correspondences_(src_search_features_adaptor_.getFeaturesMatrixMap(), correspondences_);
            return *this;
        }

        // Interface for ICP use
        template <class TransformT>
        inline CorrespondenceSearchProjective& findCorrespondences(const TransformT &tform) {
            if (!std::is_same<PointFeaturesAdaptor<ScalarT,3>,EvaluationFeatureAdaptorT>::value ||
                &src_search_features_adaptor_ != (PointFeaturesAdaptor<ScalarT,3> *)(&src_evaluation_features_adaptor_))
            {
                src_evaluation_features_adaptor_.transformFeatures(tform);
            }
            find_correspondences_(src_search_features_adaptor_.transformFeatures(tform).getTransformedFeaturesMatrixMap(), correspondences_);
            return *this;
        }

        inline const SearchResult& getCorrespondences() const { return correspondences_; }

        inline Evaluator& evaluator() { return evaluator_; }

        inline const Eigen::Matrix<ScalarT,3,3>& getProjectionIntrinsicMatrix() const { return projection_intrinsics_; }

        inline CorrespondenceSearchProjective& setProjectionIntrinsicMatrix(const Eigen::Ref<const Eigen::Matrix<ScalarT,3,3>> &mat) {
            projection_intrinsics_ = mat;
            index_map_.resize(0,0);
            return *this;
        }

        inline size_t getProjectionImageWidth() const { return projection_image_width_; }

        inline CorrespondenceSearchProjective& setProjectionImageWidth(size_t w) {
            projection_image_width_ = w;
            index_map_.resize(0,0);
            return *this;
        }

        inline size_t getProjectionImageHeight() const { return projection_image_height_; }

        inline CorrespondenceSearchProjective& setProjectionImageHeight(size_t h) {
            projection_image_height_ = h;
            index_map_.resize(0,0);
            return *this;
        }

        inline const RigidTransform<ScalarT,3>& getProjectionExtrinsicMatrix() const {
            return projection_extrinsics_;
        }

        inline CorrespondenceSearchProjective& setProjectionExtrinsicMatrix(const RigidTransform<ScalarT,3> &mat) {
            projection_extrinsics_ = mat;
            projection_extrinsics_inv_ = mat.inverse();
            index_map_.resize(0,0);
            return *this;
        }

        inline CorrespondenceScalar getMaxDistance() const { return max_distance_; }

        inline CorrespondenceSearchProjective& setMaxDistance(CorrespondenceScalar dist_thresh) {
            max_distance_ = dist_thresh;
            return *this;
        }

        inline double getInlierFraction() const { return inlier_fraction_; }

        inline CorrespondenceSearchProjective& setInlierFraction(double fraction) {
            inlier_fraction_ = fraction;
            return *this;
        }

    private:
        PointFeaturesAdaptor<ScalarT,3>& dst_search_features_adaptor_;
        PointFeaturesAdaptor<ScalarT,3>& src_search_features_adaptor_;

        EvaluationFeatureAdaptorT& src_evaluation_features_adaptor_;
        Evaluator& evaluator_;

        Eigen::Matrix<IndexT,Eigen::Dynamic,Eigen::Dynamic> index_map_;
        Eigen::Matrix<ScalarT,3,3> projection_intrinsics_;
        size_t projection_image_width_;
        size_t projection_image_height_;
        RigidTransform<ScalarT,3> projection_extrinsics_;
        RigidTransform<ScalarT,3> projection_extrinsics_inv_;

        CorrespondenceScalar max_distance_;
        double inlier_fraction_;

        SearchResult correspondences_;

        void find_correspondences_(const ConstVectorSetMatrixMap<ScalarT,3>& src_points_trans, SearchResult &correspondences) {
            const ConstVectorSetMatrixMap<ScalarT,3>& dst_points(dst_search_features_adaptor_.getFeaturesMatrixMap());

            if (index_map_.rows() != projection_image_width_ || index_map_.cols() != projection_image_height_) {
                index_map_.resize(projection_image_width_, projection_image_height_);
                pointsToIndexMap<ScalarT,IndexT>(dst_points, projection_extrinsics_, projection_intrinsics_, index_map_.data(), projection_image_width_, projection_image_height_);
            }

            const IndexT empty = std::numeric_limits<IndexT>::max();
            SearchResult corr_tmp(src_points_trans.cols());
            const CorrespondenceScalar value_to_reject = max_distance_ + (CorrespondenceScalar)1.0;

#pragma omp parallel
            {
#pragma omp for
                for (size_t i = 0; i < corr_tmp.size(); i++) {
                    corr_tmp[i].value = value_to_reject;
                }

                Vector<ScalarT,3> src_pt_trans_cam;
#pragma omp for schedule(dynamic, 256)
                for (IndexT i = 0; i < src_points_trans.cols(); i++) {
                    src_pt_trans_cam.noalias() = projection_extrinsics_inv_*src_points_trans.col(i);
                    if (src_pt_trans_cam(2) <= (ScalarT)0.0) continue;
                    size_t x = (size_t)std::llround(src_pt_trans_cam(0)*projection_intrinsics_(0,0)/src_pt_trans_cam(2) + projection_intrinsics_(0,2));
                    size_t y = (size_t)std::llround(src_pt_trans_cam(1)*projection_intrinsics_(1,1)/src_pt_trans_cam(2) + projection_intrinsics_(1,2));
                    if (x >= projection_image_width_ || y >= projection_image_height_) continue;
                    IndexT ind = index_map_(x,y);
                    if (ind == empty) continue;
                    corr_tmp[i].indexInFirst = ind;
                    corr_tmp[i].indexInSecond = i;
                    corr_tmp[i].value = evaluator_(ind, i, (src_points_trans.col(i) - dst_points.col(ind)).squaredNorm());
                }
            }

            correspondences.resize(corr_tmp.size());
            size_t count = 0;
            for (size_t i = 0; i < corr_tmp.size(); i++) {
                if (corr_tmp[i].value < max_distance_) correspondences[count++] = corr_tmp[i];
            }
            correspondences.resize(count);

            filterCorrespondencesFraction(correspondences, inlier_fraction_);
        }
    };
}
