#pragma once

#include <vector>

namespace cilantro {

template <typename ScalarT, typename IndexT = size_t>
struct Neighbor {
  using Scalar = ScalarT;
  using Index = IndexT;

  IndexT index;
  ScalarT value;

  inline Neighbor() {}

  inline Neighbor(IndexT ind, ScalarT dist) : index(ind), value(dist) {}

  template <typename IndT = IndexT>
  inline operator IndT() const {
    return static_cast<IndT>(index);
  }

  struct IndexLessComparator {
    inline bool operator()(const Neighbor& nn1, const Neighbor& nn2) const {
      return nn1.index < nn2.index;
    }
  };

  struct IndexGreaterComparator {
    inline bool operator()(const Neighbor& nn1, const Neighbor& nn2) const {
      return nn1.index > nn2.index;
    }
  };

  struct ValueLessComparator {
    inline bool operator()(const Neighbor& nn1, const Neighbor& nn2) const {
      return nn1.value < nn2.value;
    }
  };

  struct ValueGreaterComparator {
    inline bool operator()(const Neighbor& nn1, const Neighbor& nn2) const {
      return nn1.value > nn2.value;
    }
  };
};

template <typename ScalarT, typename IndexT = size_t>
using NeighborSet = std::vector<Neighbor<ScalarT, IndexT>>;

template <typename ScalarT, typename IndexT = size_t>
using Neighborhood = std::vector<Neighbor<ScalarT, IndexT>>;

template <typename ScalarT, typename IndexT = size_t>
using NeighborhoodSet = std::vector<Neighborhood<ScalarT, IndexT>>;

template <typename CountT = size_t>
struct KNNNeighborhoodSpecification {
  using Size = CountT;

  inline KNNNeighborhoodSpecification(CountT k = (CountT)0) : maxNumberOfNeighbors(k) {}

  CountT maxNumberOfNeighbors;
};

template <typename ScalarT>
struct RadiusNeighborhoodSpecification {
  using Scalar = ScalarT;

  inline RadiusNeighborhoodSpecification(ScalarT r = (ScalarT)0) : radius(r) {}

  ScalarT radius;
};

template <typename ScalarT, typename CountT = size_t>
struct KNNInRadiusNeighborhoodSpecification {
  using Scalar = ScalarT;
  using Size = CountT;

  inline KNNInRadiusNeighborhoodSpecification(CountT k = 0, ScalarT r = (ScalarT)0)
      : maxNumberOfNeighbors(k), radius(r) {}

  CountT maxNumberOfNeighbors;
  ScalarT radius;
};

}  // namespace cilantro
