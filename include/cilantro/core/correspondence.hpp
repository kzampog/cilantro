#pragma once

#include <cilantro/core/data_containers.hpp>

namespace cilantro {
    enum struct CorrespondenceSearchDirection {FIRST_TO_SECOND, SECOND_TO_FIRST, BOTH};

    template <typename ScalarT, typename IndexT = size_t>
    struct Correspondence {
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        typedef ScalarT Scalar;

        IndexT indexInFirst;
        IndexT indexInSecond;
        ScalarT value;

        Correspondence() {}

        Correspondence(IndexT i, IndexT j, ScalarT val) : indexInFirst(i), indexInSecond(j), value(val) {}

        struct ValueLessComparator {
            inline bool operator()(const Correspondence &c1, const Correspondence &c2) const {
                return c1.value < c2.value;
            }
        };

        struct ValueGreaterComparator {
            inline bool operator()(const Correspondence &c1, const Correspondence &c2) const {
                return c1.value > c2.value;
            }
        };

        struct IndicesLexicographicalComparator {
            inline bool operator()(const Correspondence &c1, const Correspondence &c2) const {
                return std::tie(c1.indexInFirst, c1.indexInSecond) < std::tie(c2.indexInFirst, c2.indexInSecond);
            }
        };

        struct IndexInFirstAndValueLessComparator {
            inline bool operator()(const Correspondence &c1, const Correspondence &c2) const {
                return std::tie(c1.indexInFirst, c1.value) < std::tie(c2.indexInFirst, c2.value);
            }
        };

        struct IndexInSecondAndValueLessComparator {
            inline bool operator()(const Correspondence &c1, const Correspondence &c2) const {
                return std::tie(c1.indexInSecond, c1.value) < std::tie(c2.indexInSecond, c2.value);
            }
        };
    };

    template <typename ScalarT>
    using CorrespondenceSet = std::vector<Correspondence<ScalarT>>;

    template <typename CorrSetT, class ComparatorT = typename CorrSetT::value_type::ValueLessComparator>
    void filterCorrespondencesFraction(CorrSetT &correspondences,
                                       double fraction_to_keep,
                                       const ComparatorT &comparator = ComparatorT())
    {
        if (fraction_to_keep > 0.0 && fraction_to_keep < 1.0) {
            std::sort(correspondences.begin(), correspondences.end(), comparator);
            correspondences.erase(correspondences.begin() + std::llround(fraction_to_keep*correspondences.size()), correspondences.end());
        }
    }

    template <typename CorrSetT>
    void filterCorrespondencesOneToOne(CorrSetT &correspondences,
                                       const CorrespondenceSearchDirection &search_dir)
    {
        if (correspondences.empty()) return;

        CorrSetT correspondences_copy = correspondences;
        switch (search_dir) {
            case CorrespondenceSearchDirection::FIRST_TO_SECOND:
                correspondences.clear();
                std::sort(correspondences_copy.begin(), correspondences_copy.end(),
                          typename CorrSetT::value_type::IndexInSecondAndValueLessComparator());
                correspondences.push_back(correspondences_copy.front());
                for (const auto &corr : correspondences_copy) {
                    if (corr.indexInSecond != correspondences.back().indexInSecond) {
                        correspondences.push_back(corr);
                    }
                }
                break;
            case CorrespondenceSearchDirection::SECOND_TO_FIRST:
                correspondences.clear();
                std::sort(correspondences_copy.begin(), correspondences_copy.end(),
                          typename CorrSetT::value_type::IndexInFirstAndValueLessComparator());
                correspondences.push_back(correspondences_copy.front());
                for (const auto &corr : correspondences_copy) {
                    if (corr.indexInFirst != correspondences.back().indexInFirst) {
                        correspondences.push_back(corr);
                    }
                }
                break;
            default:
                break;
        }
    }

    template <typename ScalarT, ptrdiff_t EigenDim, typename CorrSetT>
    void selectFirstSetCorrespondingPoints(const CorrSetT &correspondences,
                                           const ConstVectorSetMatrixMap<ScalarT,EigenDim> &first,
                                           VectorSet<ScalarT,EigenDim> &first_corr)
    {
        first_corr.resize(first.rows(), correspondences.size());
#pragma omp parallel for
        for (size_t i = 0; i < correspondences.size(); i++) {
            first_corr.col(i) = first.col(correspondences[i].indexInFirst);
        }
    }

    template <typename ScalarT, ptrdiff_t EigenDim, typename CorrSetT>
    VectorSet<ScalarT,EigenDim> selectFirstSetCorrespondingPoints(const CorrSetT &correspondences,
                                                                  const ConstVectorSetMatrixMap<ScalarT,EigenDim> &first)
    {
        VectorSet<ScalarT,EigenDim> first_corr(first.rows(), correspondences.size());
#pragma omp parallel for
        for (size_t i = 0; i < correspondences.size(); i++) {
            first_corr.col(i) = first.col(correspondences[i].indexInFirst);
        }
        return first_corr;
    }

    template <typename ScalarT, ptrdiff_t EigenDim, typename CorrSetT>
    void selectSecondSetCorrespondingPoints(const CorrSetT &correspondences,
                                            const ConstVectorSetMatrixMap<ScalarT,EigenDim> &second,
                                            VectorSet<ScalarT,EigenDim> &second_corr)
    {
        second_corr.resize(second.rows(), correspondences.size());
#pragma omp parallel for
        for (size_t i = 0; i < correspondences.size(); i++) {
            second_corr.col(i) = second.col(correspondences[i].indexInSecond);
        }
    }

    template <typename ScalarT, ptrdiff_t EigenDim, typename CorrSetT>
    VectorSet<ScalarT,EigenDim> selectSecondSetCorrespondingPoints(const CorrSetT &correspondences,
                                                                   const ConstVectorSetMatrixMap<ScalarT,EigenDim> &second)
    {
        VectorSet<ScalarT,EigenDim> second_corr(second.rows(), correspondences.size());
#pragma omp parallel for
        for (size_t i = 0; i < correspondences.size(); i++) {
            second_corr.col(i) = second.col(correspondences[i].indexInSecond);
        }
        return second_corr;
    }

    template <typename ScalarT, ptrdiff_t EigenDim, typename CorrSetT>
    void selectCorrespondingPoints(const CorrSetT &correspondences,
                                   const ConstVectorSetMatrixMap<ScalarT,EigenDim> &first,
                                   const ConstVectorSetMatrixMap<ScalarT,EigenDim> &second,
                                   VectorSet<ScalarT,EigenDim> &first_corr,
                                   VectorSet<ScalarT,EigenDim> &second_corr)
    {
        first_corr.resize(first.rows(), correspondences.size());
        second_corr.resize(second.rows(), correspondences.size());
#pragma omp parallel for
        for (size_t i = 0; i < correspondences.size(); i++) {
            first_corr.col(i) = first.col(correspondences[i].indexInFirst);
            second_corr.col(i) = second.col(correspondences[i].indexInSecond);
        }
    }
}
