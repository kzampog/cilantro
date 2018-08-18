#pragma once

#include <Eigen/Dense>

namespace cilantro {
    namespace internal {
        template <typename T, typename = int>
        struct HasPointToPointCorrespondencesGetter : std::false_type {};

        template <typename T>
        struct HasPointToPointCorrespondencesGetter<T, decltype((void) std::declval<T>().getPointToPointCorrespondences(), 0)> : std::true_type {};

        template <typename T, typename = int>
        struct HasPointToPlaneCorrespondencesGetter : std::false_type {};

        template <typename T>
        struct HasPointToPlaneCorrespondencesGetter<T, decltype((void) std::declval<T>().getPointToPlaneCorrespondences(), 0)> : std::true_type {};
    }

    template <class CorrespondenceSearchT>
    class CorrespondenceSearchCombinedMetricProxy {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        typedef typename CorrespondenceSearchT::SearchResult SearchResult;

        CorrespondenceSearchCombinedMetricProxy(const CorrespondenceSearchT &corr_engine) : corr_engine_(corr_engine) {}

        template <class CorrSearchT = CorrespondenceSearchT>
        inline const typename std::enable_if<internal::HasPointToPointCorrespondencesGetter<CorrSearchT>::value,SearchResult>::type& getPointToPointCorrespondences() const {
            return corr_engine_.getPointToPointCorrespondences();
        }

        template <class CorrSearchT = CorrespondenceSearchT>
        inline const typename std::enable_if<!internal::HasPointToPointCorrespondencesGetter<CorrSearchT>::value,SearchResult>::type& getPointToPointCorrespondences() const {
            return corr_engine_.getCorrespondences();
        }

        template <class CorrSearchT = CorrespondenceSearchT>
        inline const typename std::enable_if<internal::HasPointToPlaneCorrespondencesGetter<CorrSearchT>::value,SearchResult>::type& getPointToPlaneCorrespondences() const {
            return corr_engine_.getPointToPlaneCorrespondences();
        }

        template <class CorrSearchT = CorrespondenceSearchT>
        inline const typename std::enable_if<!internal::HasPointToPlaneCorrespondencesGetter<CorrSearchT>::value,SearchResult>::type& getPointToPlaneCorrespondences() const {
            return corr_engine_.getCorrespondences();
        }

    private:
        const CorrespondenceSearchT& corr_engine_;
    };
}
