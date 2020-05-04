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

        template <typename T, typename = int>
        struct PointToPointCorrespondenceScalar { typedef typename T::CorrespondenceScalar Scalar; };

        template <typename T>
        struct PointToPointCorrespondenceScalar<T, decltype((void) std::declval<typename T::PointToPointCorrespondenceScalar>(), 0)> { typedef typename T::PointToPointCorrespondenceScalar Scalar; };

        template <typename T, typename = int>
        struct PointToPointCorrespondenceSearchResult { typedef typename T::SearchResult SearchResult; };

        template <typename T>
        struct PointToPointCorrespondenceSearchResult<T, decltype((void) std::declval<typename T::PointToPointCorrespondenceSearchResult>(), 0)> { typedef typename T::PointToPointCorrespondenceSearchResult SearchResult; };

        template <typename T, typename = int>
        struct PointToPlaneCorrespondenceScalar { typedef typename T::CorrespondenceScalar Scalar; };

        template <typename T>
        struct PointToPlaneCorrespondenceScalar<T, decltype((void) std::declval<typename T::PointToPlaneCorrespondenceScalar>(), 0)> { typedef typename T::PointToPlaneCorrespondenceScalar Scalar; };

        template <typename T, typename = int>
        struct PointToPlaneCorrespondenceSearchResult { typedef typename T::SearchResult SearchResult; };

        template <typename T>
        struct PointToPlaneCorrespondenceSearchResult<T, decltype((void) std::declval<typename T::PointToPlaneCorrespondenceSearchResult>(), 0)> { typedef typename T::PointToPlaneCorrespondenceSearchResult SearchResult; };
    } // namespace internal

    template <class CorrespondenceSearchT>
    class CorrespondenceSearchCombinedMetricAdaptor {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        typedef typename internal::PointToPointCorrespondenceScalar<CorrespondenceSearchT>::Scalar PointToPointCorrespondenceScalar;

        typedef typename internal::PointToPointCorrespondenceSearchResult<CorrespondenceSearchT>::SearchResult PointToPointCorrespondenceSearchResult;

        typedef typename internal::PointToPlaneCorrespondenceScalar<CorrespondenceSearchT>::Scalar PointToPlaneCorrespondenceScalar;

        typedef typename internal::PointToPlaneCorrespondenceSearchResult<CorrespondenceSearchT>::SearchResult PointToPlaneCorrespondenceSearchResult;

        typedef CorrespondenceSearchT BaseCorrespondenceSearch;

        inline CorrespondenceSearchCombinedMetricAdaptor(CorrespondenceSearchT &corr_engine)
                : corr_engine_(corr_engine)
        {}

        inline CorrespondenceSearchCombinedMetricAdaptor& findCorrespondences() {
            corr_engine_.findCorrespondences();
            return *this;
        }

        template <class TransformT>
        inline CorrespondenceSearchCombinedMetricAdaptor& findCorrespondences(const TransformT& tform) {
            corr_engine_.findCorrespondences(tform);
            return *this;
        }

        template <class CorrSearchT = CorrespondenceSearchT>
        inline const typename std::enable_if<internal::HasPointToPointCorrespondencesGetter<CorrSearchT>::value,PointToPointCorrespondenceSearchResult>::type& getPointToPointCorrespondences() const {
            return corr_engine_.getPointToPointCorrespondences();
        }

        template <class CorrSearchT = CorrespondenceSearchT>
        inline const typename std::enable_if<!internal::HasPointToPointCorrespondencesGetter<CorrSearchT>::value,PointToPointCorrespondenceSearchResult>::type& getPointToPointCorrespondences() const {
            return corr_engine_.getCorrespondences();
        }

        template <class CorrSearchT = CorrespondenceSearchT>
        inline const typename std::enable_if<internal::HasPointToPlaneCorrespondencesGetter<CorrSearchT>::value,PointToPlaneCorrespondenceSearchResult>::type& getPointToPlaneCorrespondences() const {
            return corr_engine_.getPointToPlaneCorrespondences();
        }

        template <class CorrSearchT = CorrespondenceSearchT>
        inline const typename std::enable_if<!internal::HasPointToPlaneCorrespondencesGetter<CorrSearchT>::value,PointToPlaneCorrespondenceSearchResult>::type& getPointToPlaneCorrespondences() const {
            return corr_engine_.getCorrespondences();
        }

        inline BaseCorrespondenceSearch& baseCorrespondenceSearchEngine() {
            return corr_engine_;
        }

    private:
        CorrespondenceSearchT& corr_engine_;
    };
}
