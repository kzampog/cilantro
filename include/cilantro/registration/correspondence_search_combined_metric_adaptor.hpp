#pragma once

#include <type_traits>
#include <utility>

namespace cilantro {

namespace internal {
template <typename T, typename = int>
struct HasPointToPointCorrespondencesGetter : std::false_type {};

template <typename T>
struct HasPointToPointCorrespondencesGetter<
    T, decltype((void)std::declval<T>().getPointToPointCorrespondences(), 0)> : std::true_type {};

template <typename T, typename = int>
struct HasPointToPlaneCorrespondencesGetter : std::false_type {};

template <typename T>
struct HasPointToPlaneCorrespondencesGetter<
    T, decltype((void)std::declval<T>().getPointToPlaneCorrespondences(), 0)> : std::true_type {};

template <typename T, typename = int>
struct PointToPointCorrespondenceScalar {
  using Scalar = typename T::CorrespondenceScalar;
};

template <typename T>
struct PointToPointCorrespondenceScalar<
    T, decltype((void)std::declval<typename T::PointToPointCorrespondenceScalar>(), 0)> {
  using Scalar = typename T::PointToPointCorrespondenceScalar;
};

template <typename T, typename = int>
struct PointToPointCorrespondenceSearchResult {
  using SearchResult = typename T::SearchResult;
};

template <typename T>
struct PointToPointCorrespondenceSearchResult<
    T, decltype((void)std::declval<typename T::PointToPointCorrespondenceSearchResult>(), 0)> {
  using SearchResult = typename T::PointToPointCorrespondenceSearchResult;
};

template <typename T, typename = int>
struct PointToPlaneCorrespondenceScalar {
  using Scalar = typename T::CorrespondenceScalar;
};

template <typename T>
struct PointToPlaneCorrespondenceScalar<
    T, decltype((void)std::declval<typename T::PointToPlaneCorrespondenceScalar>(), 0)> {
  using Scalar = typename T::PointToPlaneCorrespondenceScalar;
};

template <typename T, typename = int>
struct PointToPlaneCorrespondenceSearchResult {
  using SearchResult = typename T::SearchResult;
};

template <typename T>
struct PointToPlaneCorrespondenceSearchResult<
    T, decltype((void)std::declval<typename T::PointToPlaneCorrespondenceSearchResult>(), 0)> {
  using SearchResult = typename T::PointToPlaneCorrespondenceSearchResult;
};
}  // namespace internal

template <class CorrespondenceSearchT>
class CorrespondenceSearchCombinedMetricAdaptor {
public:
  using PointToPointCorrespondenceScalar =
      typename internal::PointToPointCorrespondenceScalar<CorrespondenceSearchT>::Scalar;

  using PointToPointCorrespondenceSearchResult =
      typename internal::PointToPointCorrespondenceSearchResult<
          CorrespondenceSearchT>::SearchResult;

  using PointToPlaneCorrespondenceScalar =
      typename internal::PointToPlaneCorrespondenceScalar<CorrespondenceSearchT>::Scalar;

  using PointToPlaneCorrespondenceSearchResult =
      typename internal::PointToPlaneCorrespondenceSearchResult<
          CorrespondenceSearchT>::SearchResult;

  using BaseCorrespondenceSearch = CorrespondenceSearchT;

  inline CorrespondenceSearchCombinedMetricAdaptor(CorrespondenceSearchT& corr_engine)
      : corr_engine_(corr_engine) {}

  inline CorrespondenceSearchCombinedMetricAdaptor& findCorrespondences() {
    corr_engine_.findCorrespondences();
    return *this;
  }

  template <class TransformT>
  inline CorrespondenceSearchCombinedMetricAdaptor& findCorrespondences(const TransformT& tform) {
    corr_engine_.findCorrespondences(tform);
    return *this;
  }

  inline const PointToPointCorrespondenceSearchResult& getPointToPointCorrespondences() const {
    if constexpr (internal::HasPointToPointCorrespondencesGetter<CorrespondenceSearchT>::value) {
      return corr_engine_.getPointToPointCorrespondences();
    } else {
      return corr_engine_.getCorrespondences();
    }
  }

  inline const PointToPlaneCorrespondenceSearchResult& getPointToPlaneCorrespondences() const {
    if constexpr (internal::HasPointToPlaneCorrespondencesGetter<CorrespondenceSearchT>::value) {
      return corr_engine_.getPointToPlaneCorrespondences();
    } else {
      return corr_engine_.getCorrespondences();
    }
  }

  inline BaseCorrespondenceSearch& baseCorrespondenceSearchEngine() { return corr_engine_; }

private:
  CorrespondenceSearchT& corr_engine_;
};

}  // namespace cilantro
