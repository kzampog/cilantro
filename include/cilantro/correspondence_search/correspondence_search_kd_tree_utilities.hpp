#pragma once

#include <cilantro/core/correspondence.hpp>
#include <cilantro/core/common_pair_evaluators.hpp>
#include <cilantro/core/kd_tree.hpp>

namespace cilantro {
    template <typename ScalarT, ptrdiff_t EigenDim, template <class> class DistAdaptor = KDTreeDistanceAdaptors::L2, class EvaluatorT = DistanceEvaluator<ScalarT,ScalarT>, typename CorrValueT = typename EvaluatorT::OutputScalar>
    void findNNCorrespondencesUnidirectional(const ConstVectorSetMatrixMap<ScalarT,EigenDim> &query_pts,
                                             const KDTree<ScalarT,EigenDim,DistAdaptor> &ref_tree,
                                             bool ref_is_first,
                                             CorrespondenceSet<CorrValueT> &correspondences,
                                             CorrValueT max_distance,
                                             const EvaluatorT &evaluator = EvaluatorT())
    {
        if (ref_tree.getPointsMatrixMap().cols() == 0) {
            correspondences.clear();
            return;
        }

        CorrespondenceSet<CorrValueT> corr_tmp(query_pts.cols());
        std::vector<bool> keep(query_pts.cols());
        Neighborhood<ScalarT> nn;
        CorrValueT dist;
        if (ref_is_first) {
#pragma omp parallel for shared(corr_tmp) private(nn, dist) schedule(dynamic, 256)
            for (size_t i = 0; i < query_pts.cols(); i++) {
                ref_tree.kNNInRadiusSearch(query_pts.col(i), 1, max_distance, nn);
                keep[i] = !nn.empty() && (dist = evaluator(nn[0].index, i, nn[0].value)) < max_distance;
                if (keep[i]) corr_tmp[i] = {nn[0].index, i, dist};
            }
        } else {
#pragma omp parallel for shared(corr_tmp) private(nn, dist) schedule(dynamic, 256)
            for (size_t i = 0; i < query_pts.cols(); i++) {
                ref_tree.kNNInRadiusSearch(query_pts.col(i), 1, max_distance, nn);
                keep[i] = !nn.empty() && (dist = evaluator(i, nn[0].index, nn[0].value)) < max_distance;
                if (keep[i]) corr_tmp[i] = {i, nn[0].index, dist};
            }
        }

        correspondences.resize(corr_tmp.size());
        size_t count = 0;
        for (size_t i = 0; i < corr_tmp.size(); i++) {
            if (keep[i]) correspondences[count++] = corr_tmp[i];
        }
        correspondences.resize(count);
    }

    template <typename ScalarT, ptrdiff_t EigenDim, template <class> class DistAdaptor = KDTreeDistanceAdaptors::L2, class EvaluatorT = DistanceEvaluator<ScalarT,ScalarT>, typename CorrValueT = typename EvaluatorT::OutputScalar>
    inline CorrespondenceSet<CorrValueT> findNNCorrespondencesUnidirectional(const ConstVectorSetMatrixMap<ScalarT,EigenDim> &query_pts,
                                                                             const KDTree<ScalarT,EigenDim,DistAdaptor> &ref_tree,
                                                                             bool ref_is_first,
                                                                             CorrValueT max_distance,
                                                                             const EvaluatorT &evaluator = EvaluatorT())
    {
        CorrespondenceSet<CorrValueT> corr_set;
        findNNCorrespondencesUnidirectional<ScalarT,EigenDim,DistAdaptor,EvaluatorT,CorrValueT>(query_pts, ref_tree, ref_is_first, corr_set, max_distance, evaluator);
        return corr_set;
    }

    template <typename ScalarT, ptrdiff_t EigenDim, template <class> class DistAdaptor = KDTreeDistanceAdaptors::L2, class EvaluatorT = DistanceEvaluator<ScalarT,ScalarT>, typename CorrValueT = typename EvaluatorT::OutputScalar>
    void findNNCorrespondencesBidirectional(const ConstVectorSetMatrixMap<ScalarT,EigenDim> &first_points,
                                            const ConstVectorSetMatrixMap<ScalarT,EigenDim> &second_points,
                                            const KDTree<ScalarT,EigenDim,DistAdaptor> &first_tree,
                                            const KDTree<ScalarT,EigenDim,DistAdaptor> &second_tree,
                                            CorrespondenceSet<CorrValueT> &correspondences,
                                            CorrValueT max_distance,
                                            bool require_reciprocal = false,
                                            const EvaluatorT &evaluator = EvaluatorT())
    {
        CorrespondenceSet<CorrValueT> corr_first_to_second, corr_second_to_first;
        findNNCorrespondencesUnidirectional<ScalarT,EigenDim,DistAdaptor,EvaluatorT,CorrValueT>(first_points, second_tree, false, corr_first_to_second, max_distance, evaluator);
        findNNCorrespondencesUnidirectional<ScalarT,EigenDim,DistAdaptor,EvaluatorT,CorrValueT>(second_points, first_tree, true, corr_second_to_first, max_distance, evaluator);

        typename Correspondence<CorrValueT>::IndicesLexicographicalComparator comparator;

#pragma omp parallel sections
        {
#pragma omp section
            std::sort(corr_first_to_second.begin(), corr_first_to_second.end(), comparator);
#pragma omp section
            std::sort(corr_second_to_first.begin(), corr_second_to_first.end(), comparator);
        }

        correspondences.clear();
        correspondences.reserve(corr_first_to_second.size()+corr_second_to_first.size());

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

    template <typename ScalarT, ptrdiff_t EigenDim, template <class> class DistAdaptor = KDTreeDistanceAdaptors::L2, class EvaluatorT = DistanceEvaluator<ScalarT,ScalarT>, typename CorrValueT = typename EvaluatorT::OutputScalar>
    inline CorrespondenceSet<CorrValueT> findNNCorrespondencesBidirectional(const ConstVectorSetMatrixMap<ScalarT,EigenDim> &first_points,
                                                                            const ConstVectorSetMatrixMap<ScalarT,EigenDim> &second_points,
                                                                            const KDTree<ScalarT,EigenDim,DistAdaptor> &first_tree,
                                                                            const KDTree<ScalarT,EigenDim,DistAdaptor> &second_tree,
                                                                            CorrValueT max_distance,
                                                                            bool require_reciprocal = false,
                                                                            const EvaluatorT &evaluator = EvaluatorT())
    {
        CorrespondenceSet<CorrValueT> corr_set;
        findNNCorrespondencesBidirectional<ScalarT,EigenDim,DistAdaptor,EvaluatorT,CorrValueT>(first_points, second_points, first_tree, second_tree, corr_set, max_distance, require_reciprocal, evaluator);
        return corr_set;
    }
}
