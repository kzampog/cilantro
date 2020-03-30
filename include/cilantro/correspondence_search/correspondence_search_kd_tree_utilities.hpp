#pragma once

#include <cilantro/core/common_pair_evaluators.hpp>

namespace cilantro {
    template <typename ScalarT, ptrdiff_t EigenDim, typename TreeT, typename CorrSetT, class EvaluatorT = DistanceEvaluator<ScalarT,ScalarT>>
    void findNNCorrespondencesUnidirectional(const ConstVectorSetMatrixMap<ScalarT,EigenDim> &query_pts,
                                             const TreeT &ref_tree,
                                             bool ref_is_first,
                                             CorrSetT &correspondences,
                                             typename EvaluatorT::OutputScalar max_distance,
                                             const EvaluatorT &evaluator = EvaluatorT())
    {
        using CorrIndexT = typename CorrSetT::value_type::Index;
        using CorrScalarT = typename CorrSetT::value_type::Scalar;

        if (ref_tree.getPointsMatrixMap().cols() == 0) {
            correspondences.clear();
            return;
        }

        CorrSetT corr_tmp(query_pts.cols());
        std::vector<bool> keep(query_pts.cols());
        typename TreeT::NeighborhoodResult nn;
        typename EvaluatorT::OutputScalar dist;
        if (ref_is_first) {
#pragma omp parallel for shared(corr_tmp) private(nn, dist) schedule(dynamic, 256)
            for (size_t i = 0; i < query_pts.cols(); i++) {
                ref_tree.kNNInRadiusSearch(query_pts.col(i), 1, max_distance, nn);
                keep[i] = !nn.empty() && (dist = evaluator(nn[0].index, i, nn[0].value)) < max_distance;
                if (keep[i]) corr_tmp[i] = {static_cast<CorrIndexT>(nn[0].index), static_cast<CorrIndexT>(i), static_cast<CorrScalarT>(dist)};
            }
        } else {
#pragma omp parallel for shared(corr_tmp) private(nn, dist) schedule(dynamic, 256)
            for (size_t i = 0; i < query_pts.cols(); i++) {
                ref_tree.kNNInRadiusSearch(query_pts.col(i), 1, max_distance, nn);
                keep[i] = !nn.empty() && (dist = evaluator(i, nn[0].index, nn[0].value)) < max_distance;
                if (keep[i]) corr_tmp[i] = {static_cast<CorrIndexT>(i), static_cast<CorrIndexT>(nn[0].index), static_cast<CorrScalarT>(dist)};
            }
        }

        correspondences.resize(corr_tmp.size());
        size_t count = 0;
        for (size_t i = 0; i < corr_tmp.size(); i++) {
            if (keep[i]) correspondences[count++] = corr_tmp[i];
        }
        correspondences.resize(count);
    }

    template <typename ScalarT, ptrdiff_t EigenDim, typename TreeT, typename CorrSetT, class EvaluatorT = DistanceEvaluator<ScalarT,ScalarT>>
    inline CorrSetT findNNCorrespondencesUnidirectional(const ConstVectorSetMatrixMap<ScalarT,EigenDim> &query_pts,
                                                        const TreeT &ref_tree,
                                                        bool ref_is_first,
                                                        typename EvaluatorT::OutputScalar max_distance,
                                                        const EvaluatorT &evaluator = EvaluatorT())
    {
        CorrSetT corr_set;
        findNNCorrespondencesUnidirectional<ScalarT,EigenDim>(query_pts, ref_tree, ref_is_first, corr_set, max_distance, evaluator);
        return corr_set;
    }

    template <typename ScalarT, ptrdiff_t EigenDim, typename FirstTreeT, typename SecondTreeT, typename CorrSetT, class EvaluatorT = DistanceEvaluator<ScalarT,ScalarT>>
    void findNNCorrespondencesBidirectional(const ConstVectorSetMatrixMap<ScalarT,EigenDim> &first_points,
                                            const ConstVectorSetMatrixMap<ScalarT,EigenDim> &second_points,
                                            const FirstTreeT &first_tree,
                                            const SecondTreeT &second_tree,
                                            CorrSetT &correspondences,
                                            typename EvaluatorT::OutputScalar max_distance,
                                            bool require_reciprocal = false,
                                            const EvaluatorT &evaluator = EvaluatorT())
    {
        CorrSetT corr_first_to_second, corr_second_to_first;
        findNNCorrespondencesUnidirectional<ScalarT,EigenDim>(first_points, second_tree, false, corr_first_to_second, max_distance, evaluator);
        findNNCorrespondencesUnidirectional<ScalarT,EigenDim>(second_points, first_tree, true, corr_second_to_first, max_distance, evaluator);

        typename CorrSetT::value_type::IndicesLexicographicalComparator comparator;

#pragma omp parallel sections
        {
#pragma omp section
            std::sort(corr_first_to_second.begin(), corr_first_to_second.end(), comparator);
#pragma omp section
            std::sort(corr_second_to_first.begin(), corr_second_to_first.end(), comparator);
        }

        correspondences.clear();
        correspondences.reserve(corr_first_to_second.size() + corr_second_to_first.size());

        if (require_reciprocal) {
            std::set_intersection(corr_first_to_second.begin(), corr_first_to_second.end(),
                                  corr_second_to_first.begin(), corr_second_to_first.end(),
                                  std::back_inserter(correspondences), comparator);
        } else {
            std::set_union(corr_first_to_second.begin(), corr_first_to_second.end(),
                           corr_second_to_first.begin(), corr_second_to_first.end(),
                           std::back_inserter(correspondences), comparator);
        }
    }

    template <typename ScalarT, ptrdiff_t EigenDim, typename FirstTreeT, typename SecondTreeT, typename CorrSetT, class EvaluatorT = DistanceEvaluator<ScalarT,ScalarT>>
    inline CorrSetT findNNCorrespondencesBidirectional(const ConstVectorSetMatrixMap<ScalarT,EigenDim> &first_points,
                                                       const ConstVectorSetMatrixMap<ScalarT,EigenDim> &second_points,
                                                       const FirstTreeT &first_tree,
                                                       const SecondTreeT &second_tree,
                                                       typename EvaluatorT::OutputScalar max_distance,
                                                       bool require_reciprocal = false,
                                                       const EvaluatorT &evaluator = EvaluatorT())
    {
        CorrSetT corr_set;
        findNNCorrespondencesBidirectional<ScalarT,EigenDim>(first_points, second_points, first_tree, second_tree, corr_set, max_distance, require_reciprocal, evaluator);
        return corr_set;
    }
}
