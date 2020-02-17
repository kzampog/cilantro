#pragma once

#include <cilantro/core/correspondence.hpp>
#include <cilantro/core/common_pair_evaluators.hpp>

namespace cilantro {
    template <class ScalarT, class EvaluationFeatureAdaptorT, class EvaluatorT = DistanceEvaluator<ScalarT,typename EvaluationFeatureAdaptorT::Scalar>, typename IndexT = size_t>
    class CorrespondenceSearchOracle {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        typedef EvaluatorT Evaluator;

        typedef typename EvaluatorT::OutputScalar CorrespondenceScalar;

        typedef IndexT CorrespondenceIndex;

        typedef CorrespondenceSet<CorrespondenceScalar,CorrespondenceIndex> SearchResult;

        typedef CorrespondenceSet<ScalarT,CorrespondenceIndex> OracleCorrespondences;

        CorrespondenceSearchOracle(const OracleCorrespondences &correspondences,
                                   EvaluationFeatureAdaptorT &src_eval_features,
                                   EvaluatorT &evaluator)
                : oracle_correspondences_(correspondences),
                  src_evaluation_features_adaptor_(src_eval_features),
                  evaluator_(evaluator),
                  max_distance_((CorrespondenceScalar)(0.01*0.01)), inlier_fraction_(1.0)
        {}

        CorrespondenceSearchOracle& findCorrespondences() {
            SearchResult corr_tmp(oracle_correspondences_.size());
#pragma omp parallel for shared (corr_tmp)
            for (size_t i = 0; i < oracle_correspondences_.size(); i++) {
                corr_tmp[i].indexInFirst = oracle_correspondences_[i].indexInFirst;
                corr_tmp[i].indexInSecond = oracle_correspondences_[i].indexInSecond;
                corr_tmp[i].value = evaluator_(oracle_correspondences_[i].indexInFirst, oracle_correspondences_[i].indexInSecond, oracle_correspondences_[i].value);
            }

            correspondences_.resize(corr_tmp.size());
            size_t count = 0;
            for (size_t i = 0; i < corr_tmp.size(); i++) {
                if (corr_tmp[i].value < max_distance_) correspondences_[count++] = corr_tmp[i];
            }
            correspondences_.resize(count);

            filterCorrespondencesFraction(correspondences_, inlier_fraction_);

            return *this;
        }

        // Interface for ICP use
        template <class TransformT>
        inline CorrespondenceSearchOracle& findCorrespondences(const TransformT &tform) {
            src_evaluation_features_adaptor_.transformFeatures(tform);
            return findCorrespondences();
        }

        inline const OracleCorrespondences& getOracleCorrespondences() const { return oracle_correspondences_; }

        inline const SearchResult& getCorrespondences() const { return correspondences_; }

        inline Evaluator& evaluator() { return evaluator_; }

        inline CorrespondenceScalar getMaxDistance() const { return max_distance_; }

        inline CorrespondenceSearchOracle& setMaxDistance(CorrespondenceScalar dist_thresh) {
            max_distance_ = dist_thresh;
            return *this;
        }

        inline double getInlierFraction() const { return inlier_fraction_; }

        inline CorrespondenceSearchOracle& setInlierFraction(double fraction) {
            inlier_fraction_ = fraction;
            return *this;
        }

    private:
        const OracleCorrespondences& oracle_correspondences_;

        EvaluationFeatureAdaptorT& src_evaluation_features_adaptor_;
        Evaluator& evaluator_;

        CorrespondenceScalar max_distance_;
        double inlier_fraction_;

        SearchResult correspondences_;
    };
}
