#pragma once

#include <memory>
#include <cilantro/core/correspondence.hpp>
#include <cilantro/core/kd_tree.hpp>
#include <cilantro/correspondence_search/correspondence_search_kd_tree_utilities.hpp>

namespace cilantro {
//    template <typename T, typename = int>
//    struct IsIsometry : std::false_type {};
//
//    template <typename T>
//    struct IsIsometry<T, decltype((void) T::Mode, 0)> : std::conditional<T::Mode == Eigen::Isometry, std::true_type, std::false_type>::type {};

    template <class SearchFeatureAdaptorT, template <class> class DistAdaptor = KDTreeDistanceAdaptors::L2, class EvaluationFeatureAdaptorT = SearchFeatureAdaptorT, class EvaluatorT = DistanceEvaluator<typename SearchFeatureAdaptorT::Scalar,typename EvaluationFeatureAdaptorT::Scalar>, typename IndexT = size_t>
    class CorrespondenceSearchKDTree {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        typedef EvaluatorT Evaluator;

        typedef typename EvaluatorT::OutputScalar CorrespondenceScalar;

        typedef IndexT CorrespondenceIndex;

        typedef CorrespondenceSet<CorrespondenceScalar,CorrespondenceIndex> SearchResult;

        typedef typename SearchFeatureAdaptorT::Scalar SearchFeatureScalar;

        typedef KDTree<SearchFeatureScalar,SearchFeatureAdaptorT::FeatureDimension,DistAdaptor,CorrespondenceIndex> SearchTree;

        template <class EvalFeatAdaptorT = EvaluationFeatureAdaptorT, class = typename std::enable_if<std::is_same<EvalFeatAdaptorT,SearchFeatureAdaptorT>::value>::type>
        CorrespondenceSearchKDTree(SearchFeatureAdaptorT &dst_features,
                                   SearchFeatureAdaptorT &src_features,
                                   EvaluatorT &evaluator)
                : dst_search_features_adaptor_(dst_features), src_search_features_adaptor_(src_features),
                  src_evaluation_features_adaptor_(src_features), evaluator_(evaluator),
                  search_dir_(CorrespondenceSearchDirection::SECOND_TO_FIRST),
                  max_distance_((CorrespondenceScalar)(0.01*0.01)),
                  inlier_fraction_(1.0), require_reciprocality_(false), one_to_one_(false)
        {}

        CorrespondenceSearchKDTree(SearchFeatureAdaptorT &dst_search_features,
                                   SearchFeatureAdaptorT &src_search_features,
                                   EvaluationFeatureAdaptorT &src_eval_features,
                                   EvaluatorT &evaluator)
                : dst_search_features_adaptor_(dst_search_features), src_search_features_adaptor_(src_search_features),
                  src_evaluation_features_adaptor_(src_eval_features), evaluator_(evaluator),
                  search_dir_(CorrespondenceSearchDirection::SECOND_TO_FIRST),
                  max_distance_((CorrespondenceScalar)(0.01*0.01)),
                  inlier_fraction_(1.0), require_reciprocality_(false), one_to_one_(false)
        {}

        CorrespondenceSearchKDTree& findCorrespondences() {
            switch (search_dir_) {
                case CorrespondenceSearchDirection::FIRST_TO_SECOND: {
                    if (!src_tree_ptr_) src_tree_ptr_.reset(new SearchTree(src_search_features_adaptor_.getFeaturesMatrixMap()));
                    findNNCorrespondencesUnidirectional<SearchFeatureScalar,SearchFeatureAdaptorT::FeatureDimension>(dst_search_features_adaptor_.getFeaturesMatrixMap(), *src_tree_ptr_, false, correspondences_, max_distance_, evaluator_);
                    break;
                }
                case CorrespondenceSearchDirection::SECOND_TO_FIRST: {
                    if (!dst_tree_ptr_) dst_tree_ptr_.reset(new SearchTree(dst_search_features_adaptor_.getFeaturesMatrixMap()));
                    findNNCorrespondencesUnidirectional<SearchFeatureScalar,SearchFeatureAdaptorT::FeatureDimension>(src_search_features_adaptor_.getFeaturesMatrixMap(), *dst_tree_ptr_, true, correspondences_, max_distance_, evaluator_);
                    break;
                }
                case CorrespondenceSearchDirection::BOTH: {
                    if (!dst_tree_ptr_) dst_tree_ptr_.reset(new SearchTree(dst_search_features_adaptor_.getFeaturesMatrixMap()));
                    if (!src_tree_ptr_) src_tree_ptr_.reset(new SearchTree(src_search_features_adaptor_.getFeaturesMatrixMap()));
                    findNNCorrespondencesBidirectional<SearchFeatureScalar,SearchFeatureAdaptorT::FeatureDimension>(dst_search_features_adaptor_.getFeaturesMatrixMap(), src_search_features_adaptor_.getFeaturesMatrixMap(), *dst_tree_ptr_, *src_tree_ptr_, correspondences_, max_distance_, require_reciprocality_, evaluator_);
                    break;
                }
            }

            filterCorrespondencesFraction(correspondences_, inlier_fraction_);
            if (one_to_one_)
                filterCorrespondencesOneToOne(correspondences_, search_dir_);

            return *this;
        }

        // Interface for ICP use
        template <class TransformT>
        CorrespondenceSearchKDTree& findCorrespondences(const TransformT &tform) {

//            if (IsIsometry<TransformT>::value && std::is_same<SearchTree,KDTree<FeatureScalar,FeatureAdaptorT::FeatureDimension,KDTreeDistanceAdaptors::L2>>::value) {
//                // Avoid re-building tree for src if transformation is rigid and metric is L2
//                switch (search_dir_) {
//                    case CorrespondenceSearchDirection::FIRST_TO_SECOND: {
//                        if (!src_tree_ptr_) src_tree_ptr_.reset(new SearchTree(src_features_adaptor_.getFeaturesMatrixMap()));
//                        findNNCorrespondencesUnidirectional<FeatureScalar,FeatureAdaptorT::FeatureDimension,DistAdaptor,EvaluatorT>(dst_features_adaptor_.transformFeatures(tform.inverse()).getTransformedFeaturesMatrixMap(), *src_tree_ptr_, false, correspondences_, max_distance_, evaluator_);
//                        break;
//                    }
//                    case CorrespondenceSearchDirection::SECOND_TO_FIRST: {
//                        if (!dst_tree_ptr_) dst_tree_ptr_.reset(new SearchTree(dst_features_adaptor_.getFeaturesMatrixMap()));
//                        findNNCorrespondencesUnidirectional<FeatureScalar,FeatureAdaptorT::FeatureDimension,DistAdaptor,EvaluatorT>(src_features_adaptor_.transformFeatures(tform).getTransformedFeaturesMatrixMap(), *dst_tree_ptr_, true, correspondences_, max_distance_, evaluator_);
//                        break;
//                    }
//                    case CorrespondenceSearchDirection::BOTH: {
//                        if (!dst_tree_ptr_) dst_tree_ptr_.reset(new SearchTree(dst_features_adaptor_.getFeaturesMatrixMap()));
//                        if (!src_tree_ptr_) src_tree_ptr_.reset(new SearchTree(src_features_adaptor_.getFeaturesMatrixMap()));
//                        findNNCorrespondencesBidirectional<FeatureScalar,FeatureAdaptorT::FeatureDimension,DistAdaptor,EvaluatorT>(dst_features_adaptor_.transformFeatures(tform.inverse()).getTransformedFeaturesMatrixMap(), src_features_adaptor_.transformFeatures(tform).getTransformedFeaturesMatrixMap(), *dst_tree_ptr_, *src_tree_ptr_, correspondences_, max_distance_, require_reciprocality_, evaluator_);
//                        break;
//                    }
//                }
//            } else {
//                // General case
//                switch (search_dir_) {
//                    case CorrespondenceSearchDirection::FIRST_TO_SECOND: {
//                        src_trans_tree_ptr_.reset(new SearchTree(src_features_adaptor_.transformFeatures(tform).getTransformedFeaturesMatrixMap()));
//                        findNNCorrespondencesUnidirectional<FeatureScalar,FeatureAdaptorT::FeatureDimension,DistAdaptor,EvaluatorT>(dst_features_adaptor_.getFeaturesMatrixMap(), *src_trans_tree_ptr_, false, correspondences_, max_distance_, evaluator_);
//                        break;
//                    }
//                    case CorrespondenceSearchDirection::SECOND_TO_FIRST: {
//                        if (!dst_tree_ptr_) dst_tree_ptr_.reset(new SearchTree(dst_features_adaptor_.getFeaturesMatrixMap()));
//                        findNNCorrespondencesUnidirectional<FeatureScalar,FeatureAdaptorT::FeatureDimension,DistAdaptor,EvaluatorT>(src_features_adaptor_.transformFeatures(tform).getTransformedFeaturesMatrixMap(), *dst_tree_ptr_, true, correspondences_, max_distance_, evaluator_);
//                        break;
//                    }
//                    case CorrespondenceSearchDirection::BOTH: {
//                        if (!dst_tree_ptr_) dst_tree_ptr_.reset(new SearchTree(dst_features_adaptor_.getFeaturesMatrixMap()));
//                        src_trans_tree_ptr_.reset(new SearchTree(src_features_adaptor_.transformFeatures(tform).getTransformedFeaturesMatrixMap()));
//                        findNNCorrespondencesBidirectional<FeatureScalar,FeatureAdaptorT::FeatureDimension,DistAdaptor,EvaluatorT>(dst_features_adaptor_.getFeaturesMatrixMap(), src_features_adaptor_.getTransformedFeaturesMatrixMap(), *dst_tree_ptr_, *src_trans_tree_ptr_, correspondences_, max_distance_, require_reciprocality_, evaluator_);
//                        break;
//                    }
//                }
//            }

            if (!std::is_same<SearchFeatureAdaptorT,EvaluationFeatureAdaptorT>::value ||
                &src_search_features_adaptor_ != (SearchFeatureAdaptorT *)(&src_evaluation_features_adaptor_))
            {
                src_evaluation_features_adaptor_.transformFeatures(tform);
            }

            switch (search_dir_) {
                case CorrespondenceSearchDirection::FIRST_TO_SECOND: {
                    src_trans_tree_ptr_.reset(new SearchTree(src_search_features_adaptor_.transformFeatures(tform).getTransformedFeaturesMatrixMap()));
                    findNNCorrespondencesUnidirectional<SearchFeatureScalar,SearchFeatureAdaptorT::FeatureDimension>(dst_search_features_adaptor_.getFeaturesMatrixMap(), *src_trans_tree_ptr_, false, correspondences_, max_distance_, evaluator_);
                    break;
                }
                case CorrespondenceSearchDirection::SECOND_TO_FIRST: {
                    if (!dst_tree_ptr_) dst_tree_ptr_.reset(new SearchTree(dst_search_features_adaptor_.getFeaturesMatrixMap()));
                    findNNCorrespondencesUnidirectional<SearchFeatureScalar,SearchFeatureAdaptorT::FeatureDimension>(src_search_features_adaptor_.transformFeatures(tform).getTransformedFeaturesMatrixMap(), *dst_tree_ptr_, true, correspondences_, max_distance_, evaluator_);
                    break;
                }
                case CorrespondenceSearchDirection::BOTH: {
                    if (!dst_tree_ptr_) dst_tree_ptr_.reset(new SearchTree(dst_search_features_adaptor_.getFeaturesMatrixMap()));
                    src_trans_tree_ptr_.reset(new SearchTree(src_search_features_adaptor_.transformFeatures(tform).getTransformedFeaturesMatrixMap()));
                    findNNCorrespondencesBidirectional<SearchFeatureScalar,SearchFeatureAdaptorT::FeatureDimension>(dst_search_features_adaptor_.getFeaturesMatrixMap(), src_search_features_adaptor_.getTransformedFeaturesMatrixMap(), *dst_tree_ptr_, *src_trans_tree_ptr_, correspondences_, max_distance_, require_reciprocality_, evaluator_);
                    break;
                }
            }

            filterCorrespondencesFraction(correspondences_, inlier_fraction_);
            if (one_to_one_)
                filterCorrespondencesOneToOne(correspondences_, search_dir_);

            return *this;
        }

        inline const SearchResult& getCorrespondences() const { return correspondences_; }

        inline Evaluator& evaluator() { return evaluator_; }

        inline const CorrespondenceSearchDirection& getSearchDirection() const { return search_dir_; }

        inline CorrespondenceSearchKDTree& setSearchDirection(const CorrespondenceSearchDirection &search_dir) {
            search_dir_ = search_dir;
            return *this;
        }

        inline CorrespondenceScalar getMaxDistance() const { return max_distance_; }

        inline CorrespondenceSearchKDTree& setMaxDistance(CorrespondenceScalar dist_thresh) {
            max_distance_ = dist_thresh;
            return *this;
        }

        inline double getInlierFraction() const { return inlier_fraction_; }

        inline CorrespondenceSearchKDTree& setInlierFraction(double fraction) {
            inlier_fraction_ = fraction;
            return *this;
        }

        inline bool getRequireReciprocality() const { return require_reciprocality_; }

        inline CorrespondenceSearchKDTree& setRequireReciprocality(bool require_reciprocal) {
            require_reciprocality_ = require_reciprocal;
            return *this;
        }

        inline bool getOneToOne() const { return one_to_one_;  }

        inline CorrespondenceSearchKDTree& setOneToOne(bool one_to_one) {
            one_to_one_ = one_to_one;
            return *this;
        }

       private:
        SearchFeatureAdaptorT& dst_search_features_adaptor_;
        SearchFeatureAdaptorT& src_search_features_adaptor_;

        EvaluationFeatureAdaptorT& src_evaluation_features_adaptor_;
        Evaluator& evaluator_;

        std::shared_ptr<SearchTree> dst_tree_ptr_;
        std::shared_ptr<SearchTree> src_tree_ptr_;
        std::shared_ptr<SearchTree> src_trans_tree_ptr_;

        CorrespondenceSearchDirection search_dir_;
        CorrespondenceScalar max_distance_;
        double inlier_fraction_;
        bool require_reciprocality_;
        bool one_to_one_;

        SearchResult correspondences_;
    };
}
